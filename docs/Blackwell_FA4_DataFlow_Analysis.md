## Blackwell 场景下 MagiAttention 端到端数据流分析

### 场景设定

以一个典型的 Blackwell (B200, sm_100) 上的 **varlen causal attention** 训练场景为例：

- **硬件**：4× B200 GPU（CP=4）
- **序列**：2 个样本 packing，总 seqlen=4096，cu_seqlens=[0, 2048, 4096]
- **模型**：num_heads_q=32, num_heads_kv=8 (GQA), head_dim=128
- **Kernel Backend**：FA4（Flash-Attention 4，Blackwell 原生）
- **配置**：chunk_size=512, overlap_degree=2, MinHeapDispatchAlg

---

### 一、全局调用链总览

```
用户代码
  │
  ├─ ① magi_attn_varlen_key(cu_seqlens, ...)     # API 层：生成 runtime key
  │     └─ init_dist_attn_runtime_mgr(...)         # 初始化/缓存 runtime
  │           ├─ make_dispatch_meta_from_qk_ranges  # Dispatch Solver：chunk→rank
  │           ├─ DistAttnSolver.solve()             # Attn Solver：comm/calc meta
  │           └─ DistAttnRuntime(comm_meta, calc_meta)  # 构建 runtime
  │
  ├─ ② dispatch(x, key)                           # 数据搬运：全局序列 → local shard
  │
  ├─ ③ local_q, local_k, local_v = proj(local_x)  # 用户自己做 QKV 投影
  │
  ├─ ④ calc_attn(local_q, local_k, local_v, key)  # 分布式 Attention 计算
  │     └─ DistAttnFunc.forward(...)
  │           ├─ Host Stage: local Q × local KV     → FA4 kernel
  │           ├─ Remote Stage 0: GroupCast KV → FA4  → GroupReduce Out/LSE
  │           └─ Remote Stage 1: GroupCast KV → FA4  → GroupReduce Out/LSE
  │
  └─ ⑤ undispatch(local_out, key)                  # 数据搬运：local shard → 全局序列
```

---

### 二、阶段 ①：API 入口与 Runtime 初始化

#### 2.1 用户入口

```python
# magi_attn_interface.py
key = magi_attn_varlen_key(
    cu_seqlens_q=torch.tensor([0, 2048, 4096]),
    cu_seqlens_k=torch.tensor([0, 2048, 4096]),
    num_heads_q=32, num_heads_kv=8, head_dim=128,
    pad_size=0,
    cp_group_or_mesh=cp_group,   # 4-GPU process group
    causal=True,
    dist_attn_config=DistAttnConfig(
        dispatch_config=DispatchConfig(chunk_size=512, alg=MinHeapDispatchAlg()),
        overlap_config=OverlapConfig(degree=2),
    ),
)
```

#### 2.2 从 cu_seqlens 推导 mask 定义

`infer_attn_mask_from_cu_seqlens` 将 cu_seqlens + causal 转换为 MagiAttention 的标准三元组：

```
cu_seqlens = [0, 2048, 4096], causal=True
    ↓
q_ranges = AttnRanges([ [0, 2048), [2048, 4096) ])
k_ranges = AttnRanges([ [0, 2048), [2048, 4096) ])
attn_mask_type = [CAUSAL, CAUSAL]
total_seqlen_q = total_seqlen_k = 4096
```

#### 2.3 Dispatch Solver 计算

```python
# _make_dispatch_meta.py
make_dispatch_meta_from_qk_ranges(
    q_ranges, k_ranges, attn_mask_type,
    total_seqlen_q=4096, chunk_size=512, cp_size=4, ...
)
```

**过程**：
1. 将 4096 tokens 切分为 `4096 / 512 = 8` 个 chunks
2. 为每个 chunk 计算 attention 面积（causal 下是三角/梯形面积）
3. MinHeapDispatchAlg 贪心分配：每次将面积最大的 chunk 分给当前负载最小的 rank

**输出 DispatchMeta**：
```
partitions = [
    [7, 0],    # rank 0: chunk 7（面积大）+ chunk 0（面积小）
    [6, 1],    # rank 1
    [5, 2],    # rank 2
    [4, 3],    # rank 3
]
# causal 下面积: chunk0 最小(三角尖)，chunk7 最大(梯形底)
# MinHeap 会把大小互补的 chunk 配对到同一 rank
```

#### 2.4 DistAttnSolver 求解

```python
# dist_attn_solver.py: DistAttnSolver.solve()
```

**过程**：
1. **构建 AttnBucket**：根据 partitions，为当前 rank 建立包含其所有 AttnSlice 的 bucket
2. **划分 Host/Remote 范围**：
   - `host_k_ranges_global`：本 rank 持有的 KV 范围（如 rank 0 持有 chunk 7 和 chunk 0 的 KV）
   - `remote_k_ranges_global`：本 rank 需要但不持有的 KV（通过 `find_hole_ranges` 计算"空洞"）
3. **Overlap 分阶段**：将 remote_k_ranges 按 overlap_degree=2 切分为 2 个阶段
4. **构建 TransferTable**：确定每个阶段的 GroupCast/GroupReduce 通信参数

**输出**：
- `CalcMeta`：包含 `local_attn_arg` + `remote_attn_args_list[0..1]`（每个都是 `FA4AttnArg`）
- `CommMeta`：包含每个阶段的 `GroupCollectiveArg`（split_sizes, dst_indices, src_index）

#### 2.5 FA4AttnArg 的特殊初始化

当 kernel_backend 为 FA4 时，`CalcMeta` 中的 `AttnArg` 被替换为 `FA4AttnArg`，触发关键的 mask 表示转换：

```
AttnSlice (q_ranges, k_ranges, mask_type)
    │
    ↓  magi_to_hstu_cuda.magi_to_hstu()    ← CUDA kernel 转换
    │
HSTU Functions: hstu_func [n_func, seqlen_q]  (分段函数表示)
    │
    ↓  create_block_mask_cuda.create_q2k_csr_sparse_from_func()
    │
LinearBlockSparseTensorsTorch              (CSR 稀疏 block mask)
    ├─ Q2K (前向): mask_block_cnt/offset/idx, full_block_cnt/offset/idx
    └─ K2Q (反向): 同结构，转置视角
```

**Blackwell 特殊处理**：
- SM100 上 tile_size 固定为 `(128, 128)`
- `sparse_tile_m = 2 × tile_m = 256`（kernel 内部 M 维度翻倍）
- 这与 SM80/SM90 的动态 tile 大小查询不同

---

### 三、阶段 ②：Dispatch 数据搬运

```python
# 用户调用
local_x = dispatch(x_global, key)
```

**内部逻辑**（`dispatch.py: _select_local_chunks`）：
```
x_global: [4096, hidden_size]
    │
    ├─ torch.split(x, chunk_actual_sizes=[512]*8)  ← 零拷贝 view
    │     → 8 个 [512, hidden_size] 的 view
    │
    └─ torch.cat([chunks[7], chunks[0]])            ← 只拷贝本 rank 的 chunks
          → local_x: [1024, hidden_size]
```

---

### 四、阶段 ④：分布式 Attention 前向（核心）

```python
local_out, lse = calc_attn(local_q, local_k, local_v, key)
```

最终进入 `DistAttnFunc.forward()`。以 **overlap_degree=2** 为例，前向流水线如下：

#### 4.1 整体时序（以 rank 0 为例）

```
Time ────────────────────────────────────────────────────────────────→

┌─────────────────────────────────────────────────────────────────────┐
│ Host Stage                                                          │
│ ┌──────────────────────────────────┐                                │
│ │ GroupCast KV (stage 0 & 1)       │ ← 一次性发起所有远程通信       │
│ │ (async, non-blocking)            │                                │
│ └──────────────────────────────────┘                                │
│ ┌──────────────────────────────────┐                                │
│ │ FA4 fwd: local_q × local_kv     │ ← host attention               │
│ │ (overlap with comm via sm_margin)│                                │
│ └──────────────────────────────────┘                                │
├─────────────────────────────────────────────────────────────────────┤
│ Remote Stage 0                                                      │
│ ┌─────────────┐ ┌────────────────────────────┐ ┌──────────────────┐│
│ │ wait KV_0   │ │ FA4 fwd: local_q × remote  │ │ GroupReduce      ││
│ │ (comm done) │ │ KV_0 → partial_out_0       │ │ partial_out_0    ││
│ └─────────────┘ └────────────────────────────┘ └──────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│ Remote Stage 1                                                      │
│ ┌─────────────┐ ┌────────────────────────────┐ ┌──────────────────┐│
│ │ wait KV_1   │ │ FA4 fwd: local_q × remote  │ │ GroupReduce      ││
│ │ (comm done) │ │ KV_1 → partial_out_1       │ │ partial_out_1    ││
│ └─────────────┘ └────────────────────────────┘ └──────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│ Finalize                                                            │
│ ┌──────────────────────────────────────────────────────────────────┐│
│ │ wait all GroupReduce → correct_attn_out_lse → local_out, lse    ││
│ └──────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

#### 4.2 Step-by-Step 数据流

##### Step 1: 初始化 KernelBarrier

```python
kernel_barrier_fetch = KernelBarrier(fwd_kernel_barrier_fetch_target)
kernel_barrier_reduce = KernelBarrier(fwd_kernel_barrier_reduce_target)
```

KernelBarrier 是 GPU 端的同步原语，确保 native grpcoll 的通信 kernel 在计算 kernel 之后启动。

##### Step 2: Host Stage — 获取本地 QKV 并预取远程 KV

```python
local_q, local_kv = dist_attn_runtime.get_curr_q_kv_and_fetch_next(
    local_q, local_kv=(local_k, local_v),
    overlap_stage=None,  # None = host stage
    kernel_barrier=kernel_barrier_fetch,
)
```

**内部**：
1. `_maybe_flatten_local_qkv_head_groups`：GQA 场景下可选地展平 head groups
2. `_maybe_concat(local_k, local_v)` → `local_kv: [2×kv_seqlen, num_heads_kv, head_dim]`
3. **一次性预取所有远程 KV**（因为 `prefetch_stage_by_stage=False`）：
   ```python
   for ith_stage in range(overlap_degree):  # 0, 1
       self._fetch_remote_kv(local_kv, overlap_stage=ith_stage)
       # → 发起 GroupCast：将本地 KV 的对应片段发送到需要它的 rank
   ```

##### Step 3: Host Stage — FA4 Attention Forward

```python
partial_local_out, partial_local_meta = dist_attn_runtime.apply_fwd_partial_attn(
    q=local_q, kv=local_kv,
    overlap_stage=None,  # host stage
    softmax_scale=softmax_scale,
    sink=global_sink,
)
```

**内部调用链**：
```
apply_fwd_partial_attn()
  ├─ 获取 attn_arg = calc_meta.local_attn_arg  (FA4AttnArg 类型)
  │
  └─ _launch_attn_fwd_kernel()
       │
       ├─ 检测 backend == FA4
       │
       └─ fa4_fwd(q, k, v, sink, attn_arg, ...)
            │
            ├─ attn_arg.to_fa4_args(is_bwd=False)
            │   → 返回 dict:
            │     ├─ linear_k_block_sparse_mask  (Q2K CSR 稀疏 block mask)
            │     ├─ linear_q_block_sparse_mask  (K2Q CSR, 存给反向)
            │     └─ aux_tensors = [hstu_func]   (HSTU 分段函数)
            │
            ├─ q, k, v: [s, h, d] → [1, s, h, d]  (unsqueeze batch dim)
            │
            ├─ _should_use_cutlass_backend(q.device)
            │   → SM100: False (用 cute backend)
            │   → SM80/SM90: True (用 cutlass/ffa_fa3 backend)
            │
            └─ _flash_attn_fwd(             ← Flash-Attention 4 CUDA kernel
                 q, k, v,
                 softmax_scale=softmax_scale,
                 causal=False,               # mask 由 block_sparse 控制，不用 causal flag
                 arbitrary=True,             # 启用 arbitrary mask
                 block_sparse_tensors=linear_k_block_sparse_mask,
                 aux_tensors=[hstu_func],    # HSTU 函数用于 tile 内精确 mask
                 learnable_sink=sink,
               )
               → out: [1, s, h, d] → squeeze → [s, h, d]
               → lse: [1, h, s]    → squeeze+transpose → [s, h]
```

**FA4 kernel 内部 mask 机制**（两级稀疏）：

```
Level 1: Block Sparsity (tile 级)
  LinearBlockSparseTensorsTorch 以 CSR 格式存储哪些 (q_tile, kv_tile) 对需要计算
  ├─ full_block: 整个 tile 都是 "attend" → 跳过 mask 检查
  └─ mask_block: 部分 attend → 需要 Level 2 精确 mask

Level 2: Arbitrary Mask (元素级)
  hstu_func: [1, 1, n_func, seqlen_q] 分段函数
  对 mask_block 内的每个 (q_idx, kv_idx)，用 hstu_func 判断是否 attend
  编码规则: interval 0 = [0, F0), interval 1 = [F1, F2), ...
```

##### Step 4: Remote Stage Loop

```python
for ith_overlap_stage in range(overlap_degree):  # 0, 1
    # 4a. 等待当前阶段的远程 KV 到达
    curr_remote_q, curr_remote_kv = dist_attn_runtime.get_curr_q_kv_and_fetch_next(
        local_q, local_kv, overlap_stage=ith_overlap_stage,
    )
    # 内部: wait(remote_kv_work) → 从 GroupCast buffer 取出远程 KV

    # 4b. FA4 attention with remote KV
    partial_remote_out, partial_remote_meta = dist_attn_runtime.apply_fwd_partial_attn(
        q=curr_remote_q, kv=curr_remote_kv,
        overlap_stage=ith_overlap_stage,
        # FA4 不支持 out_acc/lse_acc 累积模式
    )
    # → 同样走 fa4_fwd()，但 attn_arg 是 calc_meta.remote_attn_args_list[stage]

    # 4c. GroupReduce: 将 partial_out/lse 归约回原始 rank
    dist_attn_runtime.reduce_partial_out_lse(
        partial_remote_out, partial_remote_lse,
        partial_local_out, partial_local_lse,
        ref_remote_out=curr_remote_q,
        overlap_stage=ith_overlap_stage,
    )
    # → 发起异步 GroupReduce，与下一阶段的 FA4 计算 overlap
```

##### Step 5: Finalize

```python
local_out, local_lse = dist_attn_runtime.prepare_reduced_local_out_lse(
    partial_local_out, partial_local_lse, ref_local_out=local_q,
)
```

**内部**：
1. 等待所有 `partial_out_lse_reduce_work` 完成
2. 每次 wait 后执行 `correct_attn_out_lse`：用 LSE 对 partial out 进行 log-sum-exp 校正合并
3. `local_out.to(q.dtype)` 从 FP32 高精度转回 BF16
4. `_maybe_unflatten_local_out_lse_head_groups`：恢复 GQA head 维度

---

### 五、FA4 vs FFA 路径差异对照

| 维度 | FFA (Hopper sm_90) | FA4 (Blackwell sm_100) |
|------|-------------------|----------------------|
| **Mask 表示** | 直接消费 `(q_ranges, k_ranges, attn_type_map)` 张量 | 先转 HSTU func → 再转 CSR BlockSparseTensors |
| **Kernel 入口** | `_flex_flash_attn_forward()` (自研 kernel) | `_flash_attn_fwd()` (Flash-Attention 4 fork) |
| **累积输出** | ✅ 支持 `out_acc`/`lse_acc`，避免显式 LSE 校正 | ❌ 不支持，每个 stage 产出独立 partial_out |
| **Causal 参数** | 通过 `attn_type_map` 控制 | `causal=False` + `arbitrary=True` + block sparse |
| **Tile 大小** | 动态查询 `get_tile_sizes_by_backend()` | 固定 `(128, 128)`，kernel 内 `sparse_tile_m = 256` |
| **SM Margin** | 支持 `sm_margin` 预留 SM 给通信 | 暂不支持（`sm_margin=0`） |
| **Atomic Reduction** | 支持 `disable_fwd_atomic_reduction` | 不适用（FA4 自身处理） |
| **Sink 支持** | ✅ 前向+反向 | ✅ 仅前向（反向尚不支持） |
| **Cutlass 分支** | — | SM80/SM90 走 `ffa_fa3` cutlass 后端 |

---

### 六、Mask 转换详细流程

这是 Blackwell 路径中最独特的部分，需要将 MagiAttention 的 `AttnSlice` 语义转换为 FA4 的 block sparse 格式：

```
原始定义
  q_ranges = [[0, 512), [512, 1024)]     ← 本 rank 的 q chunk 范围
  k_ranges = [[1024, 2048), [512, 1024)]  ← 对应的 k 范围
  mask_type = [CAUSAL, CAUSAL]

        ↓ FA4AttnArg.__post_init__()

Step 1: magi_to_hstu (CUDA kernel)
  magi_to_hstu_cuda.magi_to_hstu(
      q_ranges, k_ranges, mask_types,
      seqlen_q=1024, seqlen_k=2048,
      n_max_func=5,  # 2*len(k_ranges)+1
  )
  → hstu_func: [n_func, 1024]
  每行是一个分段函数，编码 "q_idx 可以 attend 到 [0, F0) ∪ [F1, F2) ∪ ..."

        ↓

Step 2: create_block_mask (CUDA kernel)
  create_block_mask_cuda.create_q2k_csr_sparse_from_func(
      hstu_func,
      seqlen_q=1024, seqlen_k=2048,
      Q_BLOCK_SIZE=256,   # sparse_tile_m = 2×128 on SM100
      KV_BLOCK_SIZE=128,
  )
  → LinearBlockSparseTensorsTorch:
    mask_block_cnt:    每个 q_block 有多少个 mask_block
    mask_block_offset: CSR offset 数组
    mask_block_idx:    哪些 kv_block 是 mask_block
    full_block_cnt:    每个 q_block 有多少个 full_block
    full_block_offset: CSR offset 数组
    full_block_idx:    哪些 kv_block 是 full_block

        ↓

Step 3: FA4 kernel 消费
  _flash_attn_fwd(
      ...,
      block_sparse_tensors=linear_k_block_sparse_mask,  # Q→K 方向
      aux_tensors=[hstu_func],                           # tile 内精确 mask
  )
```

---

### 七、反向传播数据流

反向与前向对称，关键差异：

```
Forward:  GroupCast(KV) → FA4_fwd → GroupReduce(Out, LSE)
Backward: GroupCast(KV) + GroupCast(Q,O,dO,LSE) → FA4_bwd → GroupReduce(dQ) + GroupReduce(dKV)
```

FA4 反向调用：
```python
fa4_bwd(do, q, k, v, sink=None, o, lse, attn_arg, ...)
# 注意: FA4 反向目前不支持 learnable sink
# 反向使用 K2Q 方向的 block sparse mask（与前向的 Q2K 转置）
```

**SM100 cutlass 路径不可用时** → 走 `_flash_attn_bwd` cute backend。

---

### 八、性能关键路径分析

在 Blackwell 上 FA4 路径的 **性能关键差异**：

1. **无 out_acc 累积**：FA4 不支持 FFA 的累积输出模式，每个 remote stage 产生独立的 partial_out，需要额外的 `correct_attn_out_lse` 操作。这引入了少量额外计算和精度损失。

2. **sm_margin=0**：FA4 kernel 当前不支持预留 SM 给通信 kernel（FFA 支持），这意味着 overlap 需要依赖 GPU 硬件的天然并发能力而非显式 SM 分区。

3. **Mask 转换开销**：`magi_to_hstu` + `create_block_mask` 两步 CUDA kernel 调用是 FA4 独有的初始化开销，但只在 solver 阶段执行一次（初始化缓存），不在每个 training step 的 critical path 上。

4. **两级稀疏优势**：BlockSparseTensors 的 CSR 格式让 FA4 kernel 可以完全跳过不参与计算的 (q_tile, kv_tile) 对，在稀疏 mask（如 causal 的三角区域）下减少大量无效计算。
