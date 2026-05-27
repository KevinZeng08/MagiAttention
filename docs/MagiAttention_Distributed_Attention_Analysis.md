## MagiAttention 分布式 Attention 策略深度分析

---

### 一、设计动机

#### 1.1 超长上下文训练的挑战

随着大模型向超长上下文（如 128K、1M tokens）和多模态场景（如自回归视频生成 Magi-1）演进，标准 Attention 的 **O(N²) 计算复杂度**和**单卡显存瓶颈**成为核心制约。Context Parallelism（CP）成为解决这一问题的主流范式——将序列沿 token 维度切分到多卡上并行计算。

#### 1.2 现有 CP 方案的不足

现有主流 CP 方案（如 Ring Attention、Megatron-LM CP、DeepSpeed Ulysses）存在以下关键短板：

- **Mask 表达能力不足**：多数方案仅支持 causal mask 或 full mask，难以表达 varlen、bi-causal、inv-causal 等异构 mask 模式（如视频生成中文本与视频帧混合的 mask 模式）。
- **负载不均衡**：causal mask 下不同 rank 的计算量天然不同（三角形面积差异），导致严重的 GPU 空闲等待。
- **通信冗余**：Ring Attention 使用 P2P 环形传递 KV，每个 rank 都会接收到全量 KV，但 causal mask 下大量接收到的 KV 不参与计算，造成**通信浪费**。
- **Overlap 效率低**：粗粒度的 overlap 策略难以有效隐藏通信延迟。

#### 1.3 MagiAttention 的目标

MagiAttention 旨在实现：

- **线性可扩展性**：分布式 attention 的耗时随 CP 卡数近线性下降。
- **异构 Mask 支持**：原生支持 FULL / CAUSAL / INV_CAUSAL / BI_CAUSAL 四种 mask 类型的任意组合和重叠。
- **零冗余通信**：每个 token 的 KV 只被发送到真正需要它的 rank。
- **自适应 Overlap**：计算与通信的多阶段流水线调度，最大化 GPU 利用率。

---

### 二、解决的核心问题

| 问题 | 现有方案的痛点 | MagiAttention 的解决方式 |
|------|-------------|----------------------|
| **异构 Mask** | 仅支持单一 mask 类型 | `AttnSlice` 抽象 + Flex-Flash-Attention (FFA) 内核 |
| **负载不均衡** | causal mask 下各 rank 计算量差异大 | 细粒度 chunk-level dispatch solver，支持 DP/BS/MinHeap 等多种算法 |
| **通信冗余** | Ring P2P 全量传递，大量无用通信 | GroupCast / GroupReduce 原语实现按需通信 |
| **Overlap 不充分** | 粗粒度 overlap，通信无法有效隐藏 | 自适应多阶段 overlap + 动态调度 |
| **Kernel 效率** | 需要 padding 或额外 mask 计算 | FFA 内核原生支持 AttnSlice 的 `(q_range, k_range, mask_type)` 三元组 |

---

### 三、设计思路与核心抽象

#### 3.1 AttnSlice：通用 Mask 表示

MagiAttention 的核心抽象是 **`AttnSlice`**——用一个三元组 `(q_range, k_range, mask_type)` 来紧凑地表示 attention mask 的一个"切片"：

```python
@dataclass
class AttnSlice:
    q_range: AttnRange       # query 的序列范围 [start, end)
    k_range: AttnRange       # key 的序列范围 [start, end)
    mask_type: AttnMaskType   # FULL / CAUSAL / INV_CAUSAL / BI_CAUSAL
```

**设计优势**：
- **紧凑性**：任何复杂的 attention mask 都可以分解为若干 `AttnSlice` 的组合，无需存储完整的 N×N mask 矩阵。
- **可分区性**：`AttnSlice` 的 `area` 属性可以精确计算不同 mask 类型下的有效计算面积（矩形、三角形、梯形、平行四边形），为负载均衡提供量化依据。
- **Kernel 友好**：FFA 内核直接消费 `(q_ranges, k_ranges, attn_type_map)` 张量，避免了运行时 mask 计算开销。

此外，`AttnRectangle` 通过对角线索引 `d_range` 进一步精确裁剪 mask 的有效区域，使分区后的 slice 面积计算更准确。

#### 3.2 层级化的数据结构

MagiAttention 采用三层容器结构组织分布式 attention 的工作负载：

```
AttnBucket (per CP rank)
  └── AttnChunk (per chunk, dispatch 的最小调度单元)
        └── AttnSlice (per q-k range pair, kernel 的计算单元)
```

- **`AttnSlice`**：最小的 attention 计算单元，描述一个 q_range × k_range 的 mask 区域。
- **`AttnChunk`**：由若干 `AttnSlice` 组成，是 dispatch solver 进行负载均衡调度的最小单位。通过 `iou` 属性衡量 chunk 内 k_ranges 的重叠度，指导通信优化。
- **`AttnBucket`**：一个 CP rank 被分配到的所有 chunks，代表该 rank 需要完成的全部工作。

#### 3.3 整体流水线

MagiAttention 的前向传播遵循以下流程：

```
                      ┌─────────────┐
                      │  输入 Q/K/V  │
                      └──────┬──────┘
                             │
                    ┌────────▼────────┐
                    │   Dispatch       │  ← 按 dispatch_meta 将序列切分到各 rank
                    │  (load balance) │
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │   Multi-Stage Overlap Loop   │  ← 计算与通信流水线
              │  ┌─────────────────────────┐ │
              │  │ Stage 0: Local Attention │ │  ← host stage: 本地 Q×本地KV
              │  ├─────────────────────────┤ │
              │  │ Stage 1..N: Remote Attn  │ │  ← remote stages: 本地Q×远程KV
              │  │  GroupCast KV ──overlap──│ │     通信与计算重叠
              │  │  Partial Attention       │ │
              │  │  GroupReduce Out/LSE     │ │
              │  └─────────────────────────┘ │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │   Undispatch     │  ← AllGather + 恢复原始序列顺序
                    └────────┬────────┘
                             │
                      ┌──────▼──────┐
                      │  输出 O/LSE  │
                      └─────────────┘
```

---

### 四、具体实现

#### 4.1 Flex-Flash-Attention (FFA) 内核

FFA 是 MagiAttention 的核心计算内核，关键特性：

- **原生 varlen + 异构 mask 支持**：接受 `(q_ranges, k_ranges, attn_type_map)` 三个张量，在单次 kernel launch 中处理多个不同 mask 类型的 attention slice。
- **累积输出模式**（`out_acc` / `lse_acc`）：支持将多阶段的 partial attention 结果累积到同一 buffer，避免显式的 `correct_attn_out_lse` 操作，减少中间结果存储。
- **原子归约控制**：当 q_ranges 不重叠时禁用原子归约（`disable_fwd_atomic_reduction`），提升性能。
- **多后端**：支持 FFA（Hopper sm_90）、FA4（Blackwell sm_100，基于 Flash-Attention 4 fork）、SDPA（纯 PyTorch，用于调试/兼容）。

```python
# 后端与精度兼容矩阵
_BACKEND_SUPPORTED_PRECISIONS = {
    FFA:     {BF16, FP16},          # 硬件 kernel
    FA4:     {BF16, FP16},          # 硬件 kernel
    SDPA:    {BF16, FP16, FP32, FP64},  # 纯 torch
    SDPA_OL: {BF16, FP16, FP32, FP64},  # 纯 torch，online softmax
}
```

#### 4.2 Dispatch Solver：负载均衡调度

Dispatch solver 将全局序列切分为 chunks，并将 chunks 分配到各 CP rank 以实现负载均衡。

**输入**：
- `q_ranges`、`k_ranges`、`attn_mask_type`：全局的 attention 定义
- `chunk_size`、`pad_size`：控制分块粒度

**调度算法**（`DispatchAlg` 家族）：

| 算法 | 类型 | 最优性 | 特点 |
|------|------|--------|------|
| `LBDispatchAlg` | Lower Bound | ✗ | 仅计算理论下界，不实际分配 |
| `DPDispatchAlg` | Dynamic Programming | ✓ | 最优解，O(n²) 时间复杂度 |
| `BSDispatchAlg` | Binary Search | ✓ | 最优解，返回具体分区 |
| `MinHeapDispatchAlg` | Min Heap | ✗ | 贪心近似，每次将 chunk 分配给当前负载最小的 rank |
| `ToppHeapDispatchAlg` | Topp Heap | ✗ | 考虑 KV 亲和性的贪心算法 |
| `SequentialDispatchAlg` | Sequential | ✗ | 顺序分配，最简单 |
| `SortedSequentialSelectAlg` | Sorted Sequential | ✗ | 按 area 排序后顺序分配 |

**核心逻辑**：每个 chunk 的 `area`（有效计算面积）作为负载度量。对于 causal mask，三角区域的面积计算精确到梯形/三角形公式。

#### 4.3 GroupCast / GroupReduce：零冗余通信原语

MagiAttention 用两个自定义集合通信原语替代了 Ring P2P：

##### GroupCast（前向 KV 分发）

```
rank_i 持有 local KV →  只发送给需要这些 KV 的 rank 集合
```

- **输入**：`input_split_sizes`（本地 KV 的分段大小）、`dst_indices`（每段的目标 rank 列表）、`src_index`（从哪些 rank 接收）
- **语义**：每个 rank 精确地将自己的 KV 片段发送到需要它们的 rank，不多不少。
- **实现**：
  - **a2av 实现**（`_a2av_grpcoll_impl.py`）：基于 `all-to-all-v`，适用于通用场景。
  - **native 实现**（`_native_grpcoll_impl.py`）：基于 DeepEP 的 GPU kernel 级别集合通信，支持节点内和节点间，性能更优。
  - **层级实现**（`_group_collective_hier.py`）：节点内 + 节点间分层通信，优化跨节点场景。

##### GroupReduce（前向 Out/LSE 归约，反向梯度归约）

```
各 rank 的 partial out/lse → 归约到对应的 rank
```

- **支持的归约操作**：`sum`（梯度）、`avg`、`lse`（log-sum-exp 校正）
- **与 GroupCast 对称**：`src_indices` ↔ `dst_index` 关系反转

**核心优势**：通信量等于实际计算需要的数据量，与 Ring Attention 的全量 KV 循环相比，在稀疏 mask（如 causal）下通信量大幅减少。

#### 4.4 自适应多阶段 Overlap

Overlap 策略负责将通信和计算组织成流水线，最大化 GPU 利用率。

##### 流水线结构

```
Time →
Stage 0 (Host):   [Calc: local Q × local KV]
                       ↕ (overlap)
Stage 1 (Remote): [Comm: GroupCast KV_1] → [Calc: local Q × remote KV_1] → [Comm: GroupReduce partial_out_1]
Stage 2 (Remote): [Comm: GroupCast KV_2] → [Calc: local Q × remote KV_2] → [Comm: GroupReduce partial_out_2]
...
```

##### Overlap 配置（`OverlapConfig`）

```python
@dataclass
class OverlapConfig:
    degree: int | None = 1
    # degree=0: 无 overlap（blocking 通信 + 合并 attn_arg）
    # degree=1: 本地 + 1 个远程阶段，无多阶段 chunking
    # degree=N (N>=2): 本地 + N 个远程阶段（静态多阶段 overlap）
    # degree=None: 动态模式，overlap solver 自动确定最优 degree

    mode: AttnOverlapMode = STATIC  # STATIC / DYNAMIC
    min_chunk_size: int = 512
    max_num_chunks: int = 64
    calc_cost_factor: float = 1.0   # 计算代价系数
    comm_cost_factor: float = 1.0   # 通信代价系数
```

##### Overlap Solver

Overlap solver 将远程 KV 的通信/计算组织为 `OverlapStageCost` 序列，然后求解最优的阶段划分：

- **Uniform 算法**：将远程 KV 均匀切分为 N 个阶段。
- **Greedy 算法**：贪心地将阶段分配到最早完成的流水线槽位。
- **动态模式**：自动遍历不同 degree，选择总体耗时最小的方案。

##### Prefetch 策略

```python
# 两种 prefetch 模式
if self.prefetch_stage_by_stage:
    # 逐阶段 prefetch：当前计算阶段同时发起下一阶段的通信
else:
    # 一次性 prefetch：在 host stage 计算前一次性发起所有远程通信
    # 依赖 FFA 的 persistent kernel 设计的 sm_margin 支持
```

#### 4.5 DistAttnSolver：端到端求解

`DistAttnSolver` 是连接所有组件的核心调度器，其 `solve()` 方法执行以下步骤：

1. **构建 Bucket**：根据 dispatch_meta 为当前 rank 构建 `AttnBucket`，确定本 rank 需要计算的所有 `AttnSlice`。
2. **划分 Host/Remote 范围**：
   - `host_q_ranges_global`：本 rank 持有的 Q 的全局范围。
   - `host_k_ranges_global`：本 rank 持有的 KV 的全局范围。
   - `remote_k_ranges_global`：本 rank 需要但不持有的 KV 范围（通过 `find_hole_ranges` 计算"空洞"得到）。
3. **构建 Host/Remote RankEntry**：确定每个阶段每个 rank 需要发送/接收的数据。
4. **构建 TransferTable**：生成 GroupCast/GroupReduce 的通信参数（split_sizes、dst_indices、src_index）。
5. **生成 CalcMeta 和 CommMeta**：最终的计算和通信元数据，供 `DistAttnRuntime` 消费。

#### 4.6 DynamicAttnSolver：动态 Mask 场景

对于运行时 mask 动态变化的场景（如 MoE 中不同 token 路由不同），`DynamicAttnSolver` 提供了更灵活的算法族：

| 算法 | 特点 |
|------|------|
| `NON_COMMUNICATION_QO` | 无通信的 QO 路由 |
| `GREEDY_RANDOM_GRID` | 贪心随机网格搜索 |
| `SIMPLEX_NETWORK_FLOW` | 单纯形网络流 |
| `FAST_SIMPLEX_NETWORK_FLOW` | 快速单纯形网络流 |
| `BINARY_GREEDY` | 二分贪心 |
| `BINARY_GREEDY_PARALLEL` | 并行二分贪心 |

#### 4.7 Dispatch / Undispatch 数据搬运

```python
# dispatch: 将全局序列按 chunk 分配到各 rank
def dispatch_func(x, dispatch_meta, cp_group):
    # 1. torch.split 按 chunk_actual_sizes 切分（零拷贝 view）
    # 2. torch.cat 只拼接属于当前 rank 的 chunks
    return local_x

# undispatch: 反向聚合恢复原始序列
def undispatch_func(x_local, dispatch_meta, cp_group):
    # 1. all_gather_v 收集所有 rank 的 local 数据
    # 2. 按 unperm_idxs 恢复原始顺序
    return global_x
```

**精巧之处**：`torch.split` 返回 view（零拷贝），只在最终 `torch.cat` 时发生一次 O(shard_seqlen) 的数据拷贝。

#### 4.8 Attention Sink 支持

MagiAttention 原生支持 **learnable attention sink**——一种固定的全局注意力锚点，解决长序列中注意力分散的问题：

- 前向：在 FFA kernel 中注入 sink 张量，与正常 attention 融合计算。
- 反向：通过 `sink_bwd_compiled` 计算 sink 的梯度。
- 分布式：通过 `partial_dsink_reduce_work` 跨 rank 归约 sink 梯度。

---

### 五、性能表现

#### 5.1 线性可扩展性

根据官方 benchmark（H100 和 B200），MagiAttention 在 **varlen causal mask** 场景下展现了近线性的扩展性：

- **前向**：在 8 卡 H100 上，相比单卡 FlashAttention-3，加速比接近理论上限（8x）。
- **反向**：同样保持高效的线性扩展。
- **对比**：显著优于 Ring Attention 和 Megatron-LM CP 等方案，尤其在高 CP 卡数（16/32/64 卡）下差距更大。

#### 5.2 通信效率

| 方案 | varlen causal mask 通信量 | 说明 |
|------|-------------------------|------|
| Ring Attention | O(N × d × cp_size) | 全量 KV 循环传递 |
| MagiAttention | O(N × d × actual_needed) | 只传输实际需要的 KV |

在 causal mask 下，约 50% 的 KV 在 Ring Attention 中被冗余传输，MagiAttention 通过 GroupCast 实现精确按需通信，通信量减少约 50%。

#### 5.3 负载均衡效果

通过 dispatch solver 的细粒度 chunk-level 调度，各 rank 的计算面积（`bucket.area`）差异被控制在最小范围内。特别是 `DPDispatchAlg`（动态规划）和 `BSDispatchAlg`（二分搜索）可以给出最优解，确保理论最优的负载均衡。

#### 5.4 硬件适配

| 硬件架构 | Kernel Backend | 状态 |
|----------|---------------|------|
| Hopper (sm_90, H100/H200) | FFA (Flex-Flash-Attention) | ✅ 完整支持 |
| Ampere (sm_80, A100) | FFA_FA4 | ✅ v1.1.1 扩展支持 |
| Blackwell (sm_100, B200) | FA4 (Flash-Attention 4 fork) | ✅ 早期支持 |
| 通用 GPU | SDPA / SDPA_OL | ✅ 纯 PyTorch 回退 |

#### 5.5 框架集成

已支持与以下主流框架的集成：
- **Megatron-LM**：替换原生 CP 实现，提供示例和训练收敛实验。
- **PyTorch FSDP**：与数据并行无缝组合。
- **HuggingFace Transformers**：drop-in 集成。

---

### 六、架构总结

```
magi_attention/
├── common/               # 基础抽象
│   ├── enum.py           # AttnMaskType, KernelBackend 等枚举
│   ├── mask.py           # AttnMask 2D mask 矩阵表示
│   ├── range.py          # AttnRange [start, end) 区间
│   ├── ranges.py         # AttnRanges 区间集合
│   └── rectangle.py      # AttnRectangle 对角线裁剪的精确面积计算
│
├── meta/                 # 元数据与调度
│   ├── container/        # 层级数据结构
│   │   ├── slice.py      # AttnSlice: (q_range, k_range, mask_type)
│   │   ├── chunk.py      # AttnChunk: dispatch 最小单元
│   │   └── bucket.py     # AttnBucket: per-rank 工作负载
│   ├── solver/           # 求解器
│   │   ├── dispatch_solver.py   # 负载均衡调度 (DP/BS/MinHeap/...)
│   │   ├── overlap_solver.py    # 多阶段 overlap 调度
│   │   ├── dist_attn_solver.py  # 端到端分布式 attention 求解
│   │   └── dynamic_attn_solver.py # 动态 mask 场景求解
│   └── collection/       # 元数据集合
│       ├── calc_meta.py  # AttnArg → FFA kernel 参数
│       └── comm_meta.py  # GroupCollectiveArg → 通信参数
│
├── comm/                 # 通信层
│   └── primitive/grpcoll/  # GroupCast / GroupReduce
│       ├── _a2av_grpcoll_impl.py    # all-to-all-v 实现
│       ├── _native_grpcoll_impl.py  # DeepEP GPU kernel 实现
│       └── _group_collective_hier.py # 层级通信实现
│
├── functional/           # 计算层
│   ├── dist_attn.py      # DistAttnRuntime: 核心前向/反向逻辑
│   ├── flex_flash_attn.py # FFA kernel 封装
│   ├── fa4.py            # Flash-Attention 4 封装
│   ├── dispatch.py       # dispatch/undispatch 数据搬运
│   └── roll.py           # MTP 支持的 roll 操作
│
├── config.py             # DistAttnConfig 总配置
├── dist_attn_runtime_mgr.py # 运行时管理器（缓存 + 复用）
└── api/                  # 用户接口
    └── magi_attn_interface.py
```

---

### 七、关键创新点总结

1. **AttnSlice 抽象**：将任意复杂的 attention mask 分解为紧凑的三元组表示，使分布式 mask 分区在数学上可行，同时 kernel 可直接消费。

2. **计算感知的负载均衡**：基于精确面积计算（考虑 mask 类型的几何面积）的 dispatch solver，在 chunk 级别实现各 rank 工作量的最优均衡。

3. **零冗余集合通信**：GroupCast/GroupReduce 替代 Ring P2P，通信量严格等于计算所需，消除了 causal mask 等稀疏场景下的冗余通信。

4. **多阶段自适应 Overlap**：通过 overlap solver 自动或手动调优流水线阶段数，结合 FFA 的 persistent kernel 和 sm_margin 机制，实现通信延迟的有效隐藏。

5. **端到端工程化**：从 kernel（FFA/FA4）到通信原语（GroupCast/GroupReduce）到调度器（DistAttnSolver）到运行时（DistAttnRuntime）的完整系统设计，支持多硬件架构和主流训练框架。
