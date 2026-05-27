## MagiAttention 技术架构分析：从分布式 Attention 到底层 Kernel 实现

---

### 一、整体架构概览

MagiAttention 是一个面向**超长上下文、异构 Mask** 场景的分布式 Attention 系统（Context Parallel, CP），核心目标是在多 GPU 上实现 Attention 计算的**线性可扩展性**。

**系统分层架构（自顶向下）：**

```
┌─────────────────────────────────────────────────────────┐
│  User API Layer                                         │
│  magi_attn_varlen_key / magi_attn_flex_key              │
│  dispatch / undispatch / calc_attn                      │
├─────────────────────────────────────────────────────────┤
│  Runtime Management Layer                               │
│  DistAttnRuntimeKey → DistAttnRuntimeMgr (LRU cached)   │
├─────────────────────────────────────────────────────────┤
│  Meta Solver Layer                                      │
│  DispatchSolver → OverlapSolver → DistAttnSolver        │
│  (负载均衡分片 → 多阶段重叠编排 → 通信/计算元数据生成)     │
├─────────────────────────────────────────────────────────┤
│  Distributed Attention Execution Layer                  │
│  DistAttnRuntime (多阶段 pipeline)                       │
│  GroupCast (KV分发) + Partial Attn + GroupReduce (结果归约)│
├─────────────────────────────────────────────────────────┤
│  Communication Primitives Layer                         │
│  group_cast / group_reduce                              │
│  ├─ a2av_impl (基于 AllToAllV)                           │
│  ├─ native_grpcoll_impl (CUDA kernel 直接通信)           │
│  └─ hier_impl (分层: intranode NVLink + internode RDMA)  │
├─────────────────────────────────────────────────────────┤
│  Attention Kernel Backend Layer                         │
│  ├─ FFA (Flex-Flash-Attention, sm90 Hopper)             │
│  ├─ FA4 (Flash-Attention 4, sm100 Blackwell / sm80)     │
│  ├─ SDPA (PyTorch native, 全精度 fallback)               │
│  └─ SDPA_OL (Online Softmax SDPA)                       │
├─────────────────────────────────────────────────────────┤
│  CUDA Kernel Layer                                      │
│  ├─ csrc/flexible_flash_attention/ (FFA CUTLASS kernel) │
│  ├─ csrc/comm/grpcoll/ (GroupCast/Reduce CUDA kernels)  │
│  └─ csrc/extensions/ (KernelBarrier, sparse preprocess) │
└─────────────────────────────────────────────────────────┘
```

---

### 二、用户 API 层

入口位于 `magi_attention/api/magi_attn_interface.py`，提供两套 Key 构造接口和三个核心操作函数。

#### 2.1 Key 构造接口

| API | 说明 |
|-----|------|
| `magi_attn_varlen_key()` | FlashAttention 风格接口，接受 `cu_seqlens_q/k` + `causal` + `window_size` |
| `magi_attn_flex_key()` | 灵活接口，直接接受 `q_ranges`, `k_ranges`, `attn_mask_type` |

两者最终都会生成一个 `DistAttnRuntimeKey`，它：
1. 封装了所有 Attention 配置（ranges、mask 类型、head 信息、chunk 大小等）
2. 作为 hashable key 索引到 `DistAttnRuntimeMgr`（LRU 缓存，per-cp_group 隔离）
3. 内部触发 `DistAttnSolver.solve()` 完成所有元数据预计算

#### 2.2 核心操作函数

```python
# 1. 将全局序列按负载均衡策略分片到各 rank
local_tensor = dispatch(global_tensor, runtime_key)

# 2. 在分片后的本地数据上执行分布式 Attention
local_out, meta = calc_attn(local_q, local_k, local_v, runtime_key)

# 3. 将本地结果聚合回全局序列顺序
global_out = undispatch(local_out, runtime_key)
```

**典型使用流程：**
```
Key构造 → dispatch(x, rope, label, ...) → QKV投影 → calc_attn → undispatch(out)
```

---

### 三、Mask 抽象体系 —— AttnSlice

这是 MagiAttention 最核心的设计创新之一。

#### 3.1 AttnMaskType（四种基础 Mask 类型）

定义于 `magi_attention/common/enum.py`：

| 类型 | 语义 | 面积计算 |
|------|------|---------|
| **FULL** | 全连接，Q 可看到所有 K | `q_len × k_len` |
| **CAUSAL** | 因果 Mask，Q 只看到之前的 K | 梯形/三角形面积 |
| **INVCAUSAL** | 反因果 Mask | 同 CAUSAL 对称 |
| **BICAUSAL** | 双因果（平行四边形） | `(k_len - q_len + 1) × q_len` |

#### 3.2 AttnSlice

定义于 `magi_attention/meta/container/slice.py`：

```python
@dataclass
class AttnSlice:
    slice_id: int
    mask_type: AttnMaskType      # FULL / CAUSAL / BICAUSAL / INVCAUSAL
    q_range: AttnRange           # [q_start, q_end)
    k_range: AttnRange           # [k_start, k_end)
    area: int                    # 有效计算面积（用于负载均衡）
```

**设计思想**：将任意复杂的异构 Mask 分解为若干 `AttnSlice` 的集合，每个 Slice 是一个 `(q_range, k_range, mask_type)` 三元组。这使得：
- Mask 可以**紧凑表达**任意组合（不需要 O(N²) 的显式 Mask 矩阵）
- 分布式分片时可以在 **Slice 粒度**进行切割和调度
- 面积计算可以精确衡量每个 Slice 的计算量

#### 3.3 AttnMask

定义于 `magi_attention/common/mask.py`，是 AttnSlice 的上层聚合：

```python
class AttnMask:
    q_ranges: AttnRanges         # 所有 Q 区间
    k_ranges: AttnRanges         # 所有 K 区间
    attn_mask_type: list[AttnMaskType]  # 每对 (Q,K) 的 Mask 类型
    mask_tensor: Tensor          # 离散化的 Mask 矩阵（用于可视化/验证）
```

---

### 四、Meta Solver 层 —— 调度与编排

这是 MagiAttention 实现线性扩展的核心算法层，位于 `magi_attention/meta/solver/`。

#### 4.1 DistAttnSolver（总调度器）

`magi_attention/meta/solver/dist_attn_solver.py` 中的 `DistAttnSolver` 是整个求解流程的编排者：

```
输入: q_ranges, k_ranges, attn_mask_type, cp_group, config
  │
  ├─ Step 1: Chunking（将序列切分为 chunk）
  │   └─ 按 chunk_size 将 Q/K 切成若干 AttnChunk
  │
  ├─ Step 2: DispatchSolver.solve()（负载均衡分片）
  │   └─ 将 chunk 分配到各 CP rank，使计算量均衡
  │
  ├─ Step 3: OverlapSolver.solve()（多阶段重叠编排）
  │   └─ 将远程计算分为多个 stage，编排通信与计算的 overlap
  │
  ├─ Step 4: make_comm_meta()（生成通信元数据）
  │   └─ 为每个 stage 构造 GroupCast/GroupReduce 的参数
  │
  └─ Step 5: make_calc_meta()（生成计算元数据）
      └─ 为每个 stage 构造 AttnArg（ffa kernel 的调用参数）
```

#### 4.2 DispatchSolver（负载均衡调度）

`magi_attention/meta/solver/dispatch_solver.py` 实现了多种调度算法：

| 算法 | 类 | 特点 |
|------|-----|------|
| **Lower Bound** | `LBDispatchAlg` | 快速估算下界，不返回具体分配 |
| **Dynamic Programming** | `DPDispatchAlg` | 最优解，O(N²·W) 复杂度 |
| **Binary Search** | `BSDispatchAlg` | 二分搜索 + 贪心，近似最优 |
| **Min Heap** | `MinHeapDispatchAlg` | 堆贪心，默认推荐算法 |
| **Sequential** | `SequentialDispatchAlg` | 顺序分配，低开销 |
| **Sorted Sequential** | `SortedSequentialSelectAlg` | 排序后顺序分配 |

**调度目标**：将 N 个 AttnChunk 分配到 W 个 CP rank，最小化最大 rank 的总计算面积（即 makespan）。

**DispatchMeta 输出**：
```python
@dataclass
class DispatchMeta:
    partitions: list[list[int]]       # 每个 rank 分到的 chunk 索引
    partitions_perm_idxs: list[int]   # chunk 的全局排列顺序
    partitions_unperm_idxs: list[int] # 逆排列（用于 undispatch）
    chunk_actual_sizes: list[int]     # 每个 chunk 的实际 token 数
    split_sizes: list[int]            # 每个 rank 的总 token 数
```

#### 4.3 OverlapSolver（多阶段重叠）

`magi_attention/meta/solver/overlap_solver.py`：

**OverlapConfig 关键参数**：
- `degree=0`：无重叠，阻塞通信 + 合并 AttnArg
- `degree=1`：local + 1 remote stage，无多阶段切分
- `degree=N (N≥2)`：local + N 个 remote stage（静态多阶段）
- `degree=None`：动态模式，自动确定最优 degree

**重叠策略**：
- **UniformOverlapAlg**：均匀划分远程计算到各 stage
- **GreedyOverlapAlg**：贪心策略，按通信/计算代价比调整

---

### 五、分布式 Attention 执行层

#### 5.1 DistAttnRuntime —— Pipeline 执行引擎

`magi_attention/functional/dist_attn.py` 中的 `DistAttnRuntime` 是分布式 Attention 的执行核心。

**Forward 执行流程（多阶段 pipeline）：**

```
Stage 0 (Host/Local):
  ├─ 发起所有远程 KV 的 GroupCast（异步）
  ├─ 计算本地 Attention（local_q × local_kv）
  └─ 得到 partial_local_out, partial_local_lse

Stage 1..N (Remote):
  ├─ 等待当前 stage 的远程 KV 到达
  ├─ 计算远程 Attention（local_q × remote_kv_stage_i）
  ├─ GroupReduce 归约 partial_out/lse 到对应 rank
  └─ （可选）预取下一 stage 的远程 KV

Final:
  ├─ 等待所有 GroupReduce 完成
  ├─ correct_attn_out_lse() 合并所有 partial 结果
  └─ 返回最终 local_out
```

**关键优化点**：
- **Prefetch 策略**：支持 stage-by-stage 预取或一次性预取所有 stage
- **KernelBarrier**：CUDA kernel 级别的同步屏障，实现 compute-comm overlap 时的精确控制
- **concat_kv**：将 K、V 拼接为单个 tensor 进行通信，减少通信次数
- **accumulative buffer**：FFA 后端支持 out/lse 的累加式 buffer，避免显式 `correct_attn_out_lse`

#### 5.2 GroupCast 与 GroupReduce —— 零冗余通信原语

定义于 `magi_attention/comm/primitive/grpcoll/`，这是 MagiAttention 区别于 Ring Attention 的核心通信设计。

**GroupCast（前向 KV 分发）：**
- 每个 rank 将自己持有的 KV chunk **仅发送给需要它的 rank**（非广播）
- 通过 `dst_indices` 精确指定每个 split 的目标 rank 列表
- 实现**零冗余通信**：没有任何 rank 收到不需要的数据

**GroupReduce（反向梯度归约 / 前向 partial out 归约）：**
- 与 GroupCast 对称：每个 rank 将 partial 结果**仅发送给持有对应 Q 的 rank**
- 支持 `sum`/`avg`/`lse` 三种归约操作

**三种实现后端**：

| 后端 | 路径 | 适用场景 |
|------|------|---------|
| **A2AV** | `_a2av_grpcoll_impl.py` | 默认实现，基于 `alltoall_v` |
| **Native GrpColl** | `_native_grpcoll_impl.py` | CUDA kernel 直通，低延迟 |
| **Hierarchical** | `_group_collective_hier.py` | 分层通信（节点内 NVLink + 节点间 RDMA） |

#### 5.3 Dispatch / Undispatch

`magi_attention/functional/dispatch.py`：

- **dispatch（前向）**：根据 `DispatchMeta.partitions` 从全局序列中选取当前 rank 对应的 chunk，拼接为本地序列
- **undispatch（反向/恢复）**：`all_gather_v` 收集所有 rank 的本地结果，再按原始顺序重排

```python
class _DispatchFunc(torch.autograd.Function):
    forward:  select_local_chunks(x, meta, rank)    # O(shard_seqlen) 分配
    backward: gather_and_unpermute(grad, group, meta) # all_gather + 重排
```

---

### 六、Attention Kernel 后端层

#### 6.1 Flex-Flash-Attention（FFA）—— Hopper sm90

**代码位置**：`magi_attention/csrc/flexible_flash_attention/`

**核心设计**：在 Flash-Attention 3 基础上扩展，原生支持 AttnSlice 语义。

**CUDA Kernel 架构**：
```
flash_fwd_kernel_sm90.h          # FWD kernel 主体
├── mainloop_fwd_sm90_tma_gmma_ws.hpp  # 主循环（TMA + WGMMA + Warpgroup Specialization）
├── block_meta.h                 # DenseBlockMeta —— 根据 AttnSlice 计算 tile 边界
├── mask.h                       # AttnType 到 mask 逻辑的映射
├── seqlen.h                     # SeqlenInfo —— 管理变长序列的偏移
├── epilogue_fwd.hpp             # 输出写回（支持累加模式）
├── fwd_tile_scheduler.hpp       # Persistent Tile Scheduler
└── softmax.h                    # Online Softmax
```

**关键技术点**：

1. **TMA (Tensor Memory Accelerator)**：Hopper 硬件加速的异步内存拷贝，用于 Q/K/V 从 Global Memory 到 Shared Memory 的预取
2. **WGMMA (Warpgroup Matrix Multiply-Accumulate)**：Hopper TensorCore 指令，用于 QK^T 和 PV 的矩阵乘法
3. **Persistent Kernel**：kernel 启动后持续运行，通过 `fwd_tile_scheduler` 动态领取 tile，避免 kernel launch overhead
4. **AttnSlice-aware Block Iteration**：`DenseBlockMeta` 根据 `q_ranges`, `k_ranges`, `attn_type_map` 计算每个 tile 的有效 K 范围（`n_block_min`, `n_block_max`），跳过空白区域
5. **Accumulative Out/LSE**：支持将 partial attention 结果直接累加到 buffer，避免后续显式合并
6. **RangeMerge 优化**：将相邻的同类 AttnSlice 合并，减少 kernel 内部的循环开销

**Kernel 参数传递**：
```cpp
// FFA 接收 AttnSlice 信息
int2 const* q_ranges;      // [batch, 2] Q 区间
int2 const* k_ranges;      // [batch, 2] K 区间
int const* attn_type_map;   // [batch] mask 类型 (0=FULL, 1=CAUSAL, 2=INVCAUSAL, 3=BICAUSAL)
```

#### 6.2 Flash-Attention 4（FA4）—— Blackwell sm100 / Ampere sm80

**代码位置**：`magi_attention/functional/fa4.py` + `magi_attention/functional/flash-attention/`

**后端路由逻辑**：
```python
def _should_use_cutlass_backend(device):
    cc_major = torch.cuda.get_device_capability(device)[0]
    if cc_major in (8, 9):   # Ampere / Hopper
        return True           # 使用 ffa_fa3 (CUTLASS backend)
    return False              # sm100+ 使用 cute DSL backend
```

| GPU 架构 | 后端 | 接口 |
|----------|------|------|
| Ampere (sm80) | `flash_attn_cute.ffa_fa3` (CUTLASS) | `_flash_attn_forward_cutlass` |
| Hopper (sm90) | `flash_attn_cute.ffa_fa3` (CUTLASS) | `_flash_attn_forward_cutlass` |
| Blackwell (sm100) | `flash_attn_cute.interface` (CuTE DSL) | `_flash_attn_fwd` |

**FA4 的 AttnSlice 支持**：通过 `FA4AttnArg` 将 AttnSlice 转换为 FA4 的 block sparsity 格式：
```python
@dataclass
class FA4AttnArg:
    # 转换 AttnSlice → FA4 的 BlockSparseTensors 或 FlexAttention block_mask
    block_sparse: BlockSparseTensorsTorch | None
    mask_mod: callable | None
```

#### 6.3 SDPA / SDPA_OL（PyTorch 原生 Fallback）

- **SDPA**：使用 `torch.nn.functional.scaled_dot_product_attention`，支持 FP32/FP64
- **SDPA_OL**：Online Softmax 版本，支持 partial attention 的累加式合并

主要用于**调试、验证和全精度计算**场景。

---

### 七、CUDA Kernel 层 —— 通信 Kernel

#### 7.1 Intranode GroupCast/Reduce（NVLink）

**代码位置**：`magi_attention/csrc/comm/grpcoll/kernels/intranode*.cuh`

节点内使用 NVLink 直接 GPU-to-GPU 通信：
- `launch_group_cast<kNumDataGroups, kNumRanks, kNumWarps>`：多数据组并行发送
- 支持 1/2/3 组数据同时传输（如 K、V 或 Q、O、dO）
- 通过 `buffer_ptrs` 实现 GPU 间的共享内存直接读写
- `is_token_in_rank` + `channel_prefix_matrix` 精确控制每个 token 的路由

#### 7.2 Internode GroupCast/Reduce（RDMA/NVSHMEM）

**代码位置**：`magi_attention/csrc/comm/grpcoll/kernels/internode*.cuh`

节点间使用 NVSHMEM + RDMA（支持 IBGDA）：
- `launch_group_cast<kNumDataGroups, kNumRDMARanks>`：跨节点数据分发
- 分离 RDMA 和 NVLink 路径：`send_rdma_head` / `send_nvl_head`
- 通过 `recv_rdma_channel_prefix_matrix` 和 `gbl_channel_prefix_matrix` 管理跨节点路由
- 支持 `post_perm_idx` 进行通信后重排

#### 7.3 KernelBarrier

**代码位置**：`magi_attention/csrc/extensions/kernel_barrier.cu`

CUDA 层面的轻量级同步屏障：
- 用于 FFA persistent kernel 与通信 kernel 的精确同步
- 支持 `sm_margin` 机制：FFA kernel 预留部分 SM 给通信 kernel 使用

---

### 八、数据流全景图

以 **4 GPU、2 个序列** 的 forward pass 为例：

```
全局序列: [seq1_tokens | seq2_tokens]  (total_seqlen = 8192)
Mask:     [CAUSAL      | FULL       ]

Step 1: Key 构造（CPU）
  ├─ AttnSlice 生成: [(q=[0,4096), k=[0,4096), CAUSAL),
  │                    (q=[4096,8192), k=[4096,8192), FULL)]
  ├─ Chunking: 按 chunk_size=512 切分为 16 个 chunk
  ├─ DispatchSolver: MinHeap 将 16 chunk 分配到 4 rank
  │   (考虑 CAUSAL 面积 < FULL 面积，均衡分配)
  ├─ OverlapSolver: degree=2，分 2 个远程 stage
  └─ 生成 CommMeta + CalcMeta

Step 2: Dispatch（GPU）
  全局 x: [8192, hidden] ──select_local_chunks──→ 各 rank 的 local_x: [~2048, hidden]

Step 3: QKV 投影（GPU，各 rank 独立）
  local_x → local_q, local_k, local_v

Step 4: calc_attn（GPU，分布式 pipeline）
  ┌─ Stage 0 (Local):
  │   GroupCast(local_kv → 需要的 remote ranks)  [异步]
  │   local_out = FFA(local_q, local_kv, AttnArg_local)
  │
  ├─ Stage 1 (Remote 1):
  │   wait GroupCast stage 1
  │   remote_out_1 = FFA(local_q, remote_kv_1, AttnArg_remote_1)
  │   GroupReduce(remote_out_1 → 原始 Q 的 rank)  [异步]
  │
  └─ Stage 2 (Remote 2):
      wait GroupCast stage 2
      remote_out_2 = FFA(local_q, remote_kv_2, AttnArg_remote_2)
      GroupReduce(remote_out_2 → 原始 Q 的 rank)  [异步]

  wait all GroupReduce
  final_out = correct_attn_out_lse(local_out, reduced_remote_outs)

Step 5: Undispatch（GPU）
  各 rank 的 local_out ──all_gather + reorder──→ 全局 out: [8192, heads, dim]
```

---

### 九、与 Ring Attention 的对比

| 维度 | Ring Attention | MagiAttention |
|------|---------------|---------------|
| **通信模式** | P2P Ring，每个 rank 传全部 KV 给下一个 | GroupCast，仅发送给需要的 rank |
| **通信量** | O(N × W) 总冗余通信 | O(N) 零冗余通信 |
| **Mask 支持** | 仅 Causal / Full | 任意 AttnSlice 组合 |
| **负载均衡** | 固定均匀分片 | Solver 自适应均衡 |
| **扩展性** | 通信量随 W 线性增长 | 通信量不随 W 增长 |
| **Kernel** | 标准 FlashAttention | FFA（AttnSlice-aware persistent kernel） |

---

### 十、核心技术亮点总结

1. **AttnSlice 抽象**：将异构 Mask 统一表达为 `(q_range, k_range, mask_type)` 三元组，既紧凑又可分布式分片

2. **Solver 体系**：
   - DispatchSolver 保证计算负载均衡
   - OverlapSolver 编排通信与计算的多阶段 pipeline
   - 两者协同实现通信隐藏

3. **零冗余通信**：GroupCast/GroupReduce 原语替代 Ring P2P，通信量与 CP 并行度无关

4. **FFA Persistent Kernel**：TMA + WGMMA + Persistent Tile Scheduling，
   原生理解 AttnSlice 语义，跳过无效 tile，性能媲美 FlashAttention-3

5. **Native Group Collective Kernels**：基于 DeepEP 的 CUDA 级通信 kernel，
   支持 NVLink（节点内）和 NVSHMEM/RDMA（节点间），最大化带宽利用

6. **多后端支持**：FFA (Hopper) / FA4 (Blackwell+Ampere) / SDPA (通用 fallback)，
   运行时自动路由到最优后端
