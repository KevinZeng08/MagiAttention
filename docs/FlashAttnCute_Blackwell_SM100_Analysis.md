# Flash Attention CuTE (Blackwell SM100) 实现深度分析

## 一、仓库与技术栈概览

### 1.1 仓库信息

| 属性 | 值 |
|------|------|
| 仓库 | `https://github.com/demonatic/flash-attention` |
| 分支 | `magi_attn_blackwell_support` |
| 包名 | `flash-attn-cute` v0.1.0 |
| 作者 | Tri Dao（原始）、demonatic（MagiAttention 适配） |
| 核心依赖 | `nvidia-cutlass-dsl >= 4.4.2`、`torch`、`cuda.bindings` |

### 1.2 技术栈选型

FA4 Blackwell 实现**没有使用传统的 CUDA C++ kernel**，而是采用 **NVIDIA CuTE DSL (CUDA Template Engine Domain-Specific Language)**——一种基于 Python 的 GPU kernel 编程框架：

- **编译方式**：JIT 编译，运行时根据参数（dtype、head_dim、mask 模式等）编译生成 SASS
- **硬件抽象**：通过 `cutlass.cute` 库提供 Tensor Memory (TMEM)、TMA、tcgen05 MMA 等 SM100 专属原语
- **编译缓存**：以 `(dtype, head_dim, causal, arbitrary, func_num, block_sparsity, ...)` 为 key 缓存已编译 kernel

---

## 二、文件组织结构

```
flash_attn/cute/
├── __init__.py              # 包入口，v0.1.0，patch cute.compile
├── interface.py        (69KB)  # Python 调用入口 _flash_attn_fwd / _flash_attn_bwd
├── flash_fwd_sm100.py (129KB)  # ★ SM100 前向 kernel（本文核心）
├── flash_bwd_sm100.py (123KB)  # ★ SM100 反向 kernel
├── flash_fwd.py       (105KB)  # SM80/SM90 前向 kernel
├── flash_bwd.py        (65KB)  # SM80 反向 kernel
├── flash_bwd_sm90.py   (59KB)  # SM90 反向 kernel
├── flash_fwd_combine.py (32KB) # SplitKV combine kernel
├── flash_bwd_preprocess.py     # 反向预处理
├── flash_bwd_postprocess.py    # 反向后处理
├── softmax.py          (19KB)  # Online softmax 实现
├── mask.py             (38KB)  # Attention mask（causal/local/arbitrary/R2P 指令）
├── mask_definitions.py (12KB)  # Flex Attention mask 函数定义
├── block_sparsity.py   (29KB)  # Block sparse 数据结构
├── block_sparse_utils.py(51KB) # Block sparse 运行时工具
├── pipeline.py         (15KB)  # Pipeline 异步 producer/consumer
├── tile_scheduler.py   (31KB)  # Tile 调度器（Single/Persistent/LPT/Varlen）
├── tile_size.py        (21KB)  # Tile size 启发式选择
├── blackwell_helpers.py (33KB) # SM100 GEMM PTX 辅助
├── mma_sm100_desc.py   (11KB)  # MMA 指令描述符编码
├── copy_utils.py       (12KB)  # TMA/TMEM copy 工具
├── seqlen_info.py               # 序列长度信息
├── block_info.py                # Block 边界计算
├── pack_gqa.py                  # GQA packing
├── paged_kv.py                  # Paged KV cache
├── cute_dsl_utils.py            # CuTE DSL 补丁
├── utils.py            (33KB)  # 通用工具
└── pyproject.toml               # 包配置
```

---

## 三、SM100 前向 Kernel 架构

### 3.1 核心类：`FlashAttentionForwardSm100`

```python
class FlashAttentionForwardSm100:
    arch = 100
    def __init__(self,
        head_dim: int,           # 64, 96, 128, 192
        head_dim_v: int,         # 可与 head_dim 不同（DeepSeek: 192,128）
        qhead_per_kvhead: int,   # GQA ratio
        is_causal: bool,
        is_arbitrary: bool,      # ★ HSTU 函数 mask
        func_num: int,           # HSTU 函数数量（奇数）
        m_block_size: int = 128, # Q tile 大小
        n_block_size: int = 128, # KV tile 大小
        is_persistent: bool = True,
        pack_gqa: bool = False,
        ...
    )
```

### 3.2 Warp 专业化架构

SM100 kernel 使用 **16 个 warp（512 线程）**，按功能划分为 6 类角色：

| Warp ID | 角色 | 数量 | 职责 |
|---------|------|------|------|
| 0-3 | **Softmax0** | 4 warps | S 矩阵第一个 stage 的 softmax |
| 4-7 | **Softmax1** | 4 warps | S 矩阵第二个 stage 的 softmax |
| 8-11 | **Correction** | 4 warps | Online softmax 修正 + rescale O |
| 12 | **MMA** | 1 warp | 驱动 tcgen05 MMA 指令（Q×K^T 和 P×V） |
| 13 | **Epilogue** | 1 warp | O 从 SMEM 写回 GMEM |
| 14 | **Load** | 1 warp | TMA 加载 Q/K/V 到 SMEM |
| 15 | **Empty** | 1 warp | 占位，释放寄存器给其他 warp |

**寄存器预算分配**：
```python
num_regs_softmax = 200      # softmax warps 需要最多寄存器
num_regs_correction = 64    # correction warps
num_regs_other = 48         # load/mma/epilogue warps
num_regs_empty = 24         # empty warp 释放寄存器
```

### 3.3 Tensor Memory (TMEM) 布局

SM100 引入了 512 列 TMEM（Tensor Memory），kernel 中精心规划了 TMEM 的列分配：

```
TMEM 列分配（512 columns total）:
┌──────────────────────────────────────────────────────────────────┐
│ S0          │ S1          │ O0              │ O1              │
│ [0, 128)    │ [128, 256)  │ [256, 256+hdim_v) │ [256+hdim_v, ...) │
│ QK^T stage0 │ QK^T stage1 │ PV acc stage0   │ PV acc stage1   │
└──────────────────────────────────────────────────────────────────┘

P (softmax 输出) 与 S 共享 TMEM，偏移 n_block_size/2:
  tmem_p_offset[i] = tmem_s_offset[i] + 64  (n_block_size=128 时)
```

### 3.4 2-Stage Q Tiling

前向 kernel 采用 **q_stage=2** 策略——每个 CTA 同时处理 2 个 Q tile：

```python
self.q_stage = 2
self.cta_tiler = (2 * m_block_size, n_block_size, head_dim_padded)
# 即 (256, 128, head_dim) — 每个 CTA 处理 256 行 Q
```

两个 softmax warp group（0-3 和 4-7）分别负责各自 stage 的 softmax 计算，实现 **softmax 计算与 MMA 计算的 pipeline overlap**。

---

## 四、SMEM Pipeline 设计

### 4.1 Pipeline 阶段配置

```python
kv_stage = 3     # K/V 在 SMEM 中的 buffer 数量（3 级流水线）
q_stage = 2      # Q 的 buffer 数量
acc_stage = 1    # 累加器 stage
epi_stage = 2    # Epilogue stage
```

### 4.2 mbarrier 同步矩阵

kernel 使用大量 mbarrier 进行细粒度的 warp 间同步：

```
mbar 布局:
  load_q_full      [0..1]   — Q 加载完成信号
  load_q_empty     [2..3]   — Q 可覆盖信号
  load_kv_full     [4..6]   — KV 加载完成信号（3-stage）
  load_kv_empty    [7..9]   — KV 可覆盖信号（3-stage）
  P_full_O_rescaled [10..11] — P 就绪 + O 已 rescale
  S_full           [12..13] — S 矩阵就绪
  O_full           [14..15] — O 累加就绪
  softmax_corr     [16..19] — softmax ↔ correction 同步
  corr_epi         [20..23] — correction ↔ epilogue 同步
  s0_s1_sequence   [24..31] — stage0 ↔ stage1 顺序控制
  tmem_dealloc     [32]     — TMEM 释放同步
  P_full_2         [33..34] — P 第二信号
```

### 4.3 KV Pipeline 流程

Load warp 通过 TMA 异步加载 K/V 到 SMEM 的 3 级 circular buffer：

```
Load warp:                    MMA warp:
  producer_acquire(stage)       consumer_wait(stage)
  TMA_load K[n_block]           GEMM: S = Q @ K^T (TMEM)
  producer_acquire(stage)       consumer_wait(stage)
  TMA_load V[n_block]           GEMM: O += P @ V (TMEM)
  advance(stage)                consumer_release(stage)
```

---

## 五、MMA 指令配置

### 5.1 tcgen05 MMA

SM100 使用 `tcgen05` (Tensor Core Generation 05) MMA 指令：

```python
# QK^T: S = Q @ K^T
tiled_mma_qk = make_trivial_tiled_mma(
    dtype=BF16/FP16,
    a_major=OperandMajorMode.K,    # Q: K-major
    b_major=OperandMajorMode.K,    # K: K-major
    acc_dtype=Float32,             # 累加器 FP32
    cta_group=CtaGroup.ONE,
    tile_mn=(128, 128),            # m_block × n_block
)

# PV: O = P @ V
tiled_mma_pv = make_trivial_tiled_mma(
    dtype=BF16/FP16,
    a_major=OperandMajorMode.K,    # P: K-major (from TMEM)
    b_major=OperandMajorMode.MN,   # V: MN-major (transposed)
    acc_dtype=Float32,
    cta_group=CtaGroup.ONE,
    tile_mn=(128, head_dim_v),
    a_source=OperandSource.TMEM,   # ★ P 直接从 TMEM 读
)
```

**关键特性**：P 矩阵（softmax 输出）**直接从 TMEM 读取**作为 MMA 的 A 操作数，避免了 TMEM→SMEM→MMA 的数据搬运。

### 5.2 反向 MMA 配置

反向 kernel 需要 5 个不同的 MMA 配置：

```python
# S  = K  @ Q^T    → 重算 attention scores
# dP = V  @ dO^T   → 计算 dP
# dV = P^T @ dO    → P 从 TMEM，dO 从 SMEM
# dK = dS^T @ Q    → dS 从 TMEM/SMEM（deterministic 模式用 SMEM）
# dQ = dS  @ K     → dQ 需要 TMA reduce-add 累加
```

---

## 六、Arbitrary Mask（HSTU 函数）支持

### 6.1 Mask 模式分类

interface.py 中根据参数选择 mask 模式：

```python
if arbitrary:
    # HSTU 函数 mask
    func_num = aux_tensors[0].shape[2]  # 必须为奇数
    causal, local = False, False
elif causal:
    # 标准 causal mask
elif mask_mod is not None:
    # Flex Attention 自定义 mask
```

### 6.2 R2P 指令优化的 Arbitrary Mask

`mask.py` 中实现了一个关键优化——`mask_r2p_intervals`：利用 SM100 的 **R2P（Register to Predicate）指令**批量设置 mask predicate：

```python
@cute.jit
def mask_r2p_intervals(X, col_limits, num_intervals):
    """
    interval: [0, col_max[0]) ∪ [col_min[0], col_max[1]) ∪ [col_min[1], col_max[2]) ...
    col_limits: [col_max[0], col_min[0], col_max[1], col_min[1], ...]
    """
    for s in range_constexpr(ceil_div(ncol, 24)):
        # 第一区间 [0, F0)
        combined_mask = (1 << col_max_0_s) - 1
        # 后续区间 [F_{2j+1}, F_{2j+2})，用 XOR 生成区间 mask
        for j in range_constexpr(num_intervals):
            interval_mask = ((1 << col_max_s) - 1) ^ ((1 << col_min_s) - 1)
            combined_mask |= interval_mask
        # R2P 批量设置 predicate
        for i in range_constexpr(min(24, ncol - s*24)):
            in_bound = Boolean(combined_mask & (1 << i))
            X[c] = X[c] if in_bound else -inf
```

**性能关键**：
- 每 24 列只需一次 mask 计算（R2P 指令宽度为 24 bit）
- 多区间通过位运算 OR 合并，无需逐元素分支判断
- `range_constexpr` 确保循环完全展开，零分支开销

### 6.3 Flex Attention 的 CuTE 版本

`mask_definitions.py` 定义了 `flex_arbitrary_mask` 的 CuTE JIT 版本，直接编译进 kernel：

```python
def flex_arbitrary_mask(b, h, q_idx, kv_idx, arbitrary_func):
    value_valid = kv_idx < arbitrary_func[b, 0, 0, q_idx]      # [0, F0)
    for i in range(n_func // 2):
        in_range = (kv_idx >= arbitrary_func[..., 2*i+1, q_idx]) & \
                   (kv_idx < arbitrary_func[..., 2*i+2, q_idx])  # [F_{2i+1}, F_{2i+2})
        value_valid |= in_range
    return value_valid
```

---

## 七、Block Sparsity 机制

### 7.1 数据结构

```python
class LinearBlockSparseTensors:
    mask_block_cnt: Tensor     # [B, H, num_m_blocks] 每个 Q-block 的 mask block 数
    mask_block_offset: Tensor  # [B*H*num_m_blocks+1] CSR offset
    mask_block_idx: Tensor     # [total_mask_blocks] 具体的 K-block 索引
    full_block_cnt: Tensor     # 同上，full block 版本
    full_block_offset: Tensor
    full_block_idx: Tensor
```

### 7.2 Load warp 中的稀疏加载

`block_sparse_utils.py` 中的 `load_block_list` 实现了按 CSR 索引加载 KV：

```python
@cute.jit
def load_block_list(block_indices, block_offset, block_count, ...):
    # 反向遍历 block 索引
    for offset in range(block_count):
        n_block = block_indices[block_offset + block_count - 1 - offset]
        pipeline_k.producer_acquire(state)
        load_K(src_idx=n_block, producer_state=state)
        pipeline_v.producer_acquire(state)
        load_V(src_idx=n_block, producer_state=state)
        state.advance()
```

**关键优化**：`intra_wg_overlap` 模式下，K 和 V 的加载交错进行——加载第 i+1 个 block 的 K 同时加载第 i 个 block 的 V，最大化 TMA 带宽利用。

### 7.3 Softmax 中的 Block Sparsity 处理

`softmax_block_sparse_sm100` 需要处理三种 block 类型的转换：
1. **Full block**：跳过 mask 判断，直接 softmax
2. **Mask block**：应用 HSTU 函数做 per-element mask 后 softmax
3. **Empty block**：tile 中无有效注意力，需要特殊修正

---

## 八、Online Softmax 实现

### 8.1 核心算法

`softmax.py` 实现了标准的 online softmax（2-pass → 1-pass）：

```python
@cute.jit
def online_softmax(self, acc_S, is_first, check_inf):
    for r in range(num_rows):
        acc_S_row = acc_S_mn[r, :].load()

        # 1. 计算当前行 max
        row_max_cur = warp_reduce(fmax_reduce(acc_S_row, ...), fmax, width=4)

        if is_first:
            # 首次：直接 exp2f
            acc_S_row_exp = exp2f(acc_S_row * scale_log2 - row_max_cur * scale_log2)
            row_scale[r] = 1.0
        else:
            # 后续：rescale 之前的累积
            row_scale[r] = exp2f((row_max_prev - row_max_cur) * scale_log2)
            acc_S_row_exp = exp2f(acc_S_row * scale_log2 - row_max_cur * scale_log2)

        # 2. 累加 row sum
        row_sum = fadd_reduce(acc_S_row_exp, ...)

        # 3. 更新状态
        row_max[r] = row_max_cur
        row_sum[r] = row_sum[r] * row_scale[r] + row_sum_new
```

### 8.2 使用 exp2f 而非 expf

注意 FA4 统一使用 `exp2f`（以 2 为底的指数函数）而非 `expf`（以 e 为底）：
- `softmax_scale_log2 = softmax_scale * log2(e)`
- `exp2f` 在 GPU 上比 `expf` 快（直接对应硬件指令）
- 数值上等价：`exp(x * scale) = exp2(x * scale * log2(e))`

### 8.3 Correction Warp 的职责

Correction warp 在每个 KV block 迭代后：
1. 读取 softmax warp 写入 SMEM 的 `row_scale`
2. 使用 `row_scale` rescale TMEM 中的 O 累加器：`O *= row_scale`
3. 在最后一个 block 后，除以 `row_sum` 得到最终归一化的 O
4. 将 O 从 TMEM copy 到 SMEM，触发 epilogue warp 写回

---

## 九、Tile Scheduler

### 9.1 调度器类型

```python
# 根据场景自动选择
if varlen:
    TileScheduler = SingleTileVarlenScheduler
elif causal or local or arbitrary:
    TileScheduler = SingleTileLPTScheduler     # ★ LPT (Longest Processing Time)
else:
    TileScheduler = StaticPersistentTileScheduler  # 持久化调度
```

### 9.2 LPT 调度器

对于 causal/arbitrary mask，不同 Q block 的有效 KV 数量不同（三角形特征）。LPT 调度器按照工作量降序排列 tile，让计算量大的 tile 先调度，减少尾部不均衡：

```python
class SingleTileLPTScheduler:
    # 按 (num_kv_blocks_per_q_block) 降序排列
    # 使得 GPU SM 利用率最大化
```

### 9.3 SplitKV

当 `seqlen_k` 很长时，启用 SplitKV 将 K 维度拆分给多个 CTA 并行：

```python
num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits=128)
```

每个 split 独立计算 partial_out 和 partial_lse，最后由 `FlashAttentionForwardCombine` kernel 合并。

---

## 十、TMA (Tensor Memory Access) 使用

### 10.1 TMA 加载

Q/K/V 的全局内存加载统一使用 TMA bulk copy：

```python
tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)

# TMA atom 创建
tma_atom_Q, mQ = make_tiled_tma_atom_A(tma_load_op, mQ, sQ_layout, ...)
tma_atom_K, mK = make_tiled_tma_atom_B(tma_load_op, mK, sK_layout, ...)
tma_atom_V, mV = make_tiled_tma_atom_B(tma_load_op, mV, sV_layout, ...)
```

### 10.2 TMA Descriptor Prefetch

kernel 入口处，warp 0 预取所有 TMA descriptor：

```python
if warp_idx == 0:
    cpasync.prefetch_descriptor(tma_atom_Q)
    cpasync.prefetch_descriptor(tma_atom_K)
    cpasync.prefetch_descriptor(tma_atom_V)
    cpasync.prefetch_descriptor(tma_atom_O)
```

### 10.3 TMA Store（O 写回）

O 写回支持两种模式：
- **TMA Store**（非 varlen 时）：通过 `CopyBulkTensorTileS2GOp` 直接 S2G
- **Universal Copy**（varlen 时）：通过 128-bit copy atom 逐元素写回

---

## 十一、反向 Kernel 架构

### 11.1 `FlashAttentionBackwardSm100`

反向 kernel 同样使用 16 warps，但角色分配不同：

| Warp ID | 角色 | 职责 |
|---------|------|------|
| 0-3 | **Reduce** | dQ 的 TMA reduce-add |
| 4-11 | **Compute** | S/dP 计算 + dK/dV 累加 |
| 12 | **MMA** | 驱动所有 MMA 指令 |
| 13 | **Load** | TMA 加载 Q/dO/K/V |
| 14 | **Epilogue** | dK/dV 写回 |
| 15 | **Empty** | 占位 |

### 11.2 TMEM 布局（反向）

```
TMEM 列分配:
  S/P     [0, tile_n)               — S 和 P 共享
  dV      [tile_n, tile_n+hdim)     — dV 累加器
  dP/dQ/dS [tile_n+hdim, ...]       — dP、dQ、dS 相互复用
  dK      [...]                     — dK 累加器
```

### 11.3 dQ 的 TMA Reduce-Add

dQ 的累加使用 **TMA reduce-add** 指令——让 TMA 单元直接在全局内存上做原子加法：

```python
self.dQ_reduce_ncol = 32        # 每次 reduce 32 列
self.sdQaccum_stage = 64 // 32  # 2 个 pipeline stage
self.dQaccum_reduce_stage = tile_hdim // 32  # 总 reduce 次数
```

这避免了传统的 "计算 partial_dQ → 通信归约" 两步走，直接在 GPU 内存层面完成累加。

### 11.4 Deterministic 模式

`deterministic=True` 时：
- dK 使用 SMEM 版本的 dS 做 MMA（`use_smem_dS_for_mma_dK=True`），确保归约顺序固定
- 增加寄存器预算（`num_regs_reduce=152, num_regs_compute=136`）

---

## 十二、SMEM 布局的特殊处理

### 12.1 DeepSeek 异形 head_dim (192, 128)

当 `head_dim=192, head_dim_v=128` 时，3 stage KV 需要 `128×192×2×3 = 144KB` SMEM，超出预算。解决方案是 **不均匀 SMEM 分配**：

```python
self.uneven_kv_smem = (head_dim_padded == 192 and head_dim_v_padded == 128 and kv_stage == 3)
# 布局: [smem_large, smem_small, smem_large]
# smem_large = 128 × 192, smem_small = 128 × 128
# stride = 128 × 160（(192+128)/2）
```

第 0 和第 2 stage 存储完整的 192 列，第 1 stage 只存储 128 列，通过 stride 偏移实现正确寻址。

### 12.2 sO 与 sQ 的 SMEM 复用

当 `overlap_sO_sQ=True`（hdim=192 或 SplitKV 时），O 和 Q 共享同一块 SMEM：

```python
if self.overlap_sO_sQ:
    sO = make_tensor(recast_ptr(sQ.iterator, sO_layout.inner, o_dtype), sO_layout.outer)
```

---

## 十三、与 MagiAttention 的集成接口

### 13.1 调用链

```
MagiAttention                              flash_attn_cute
─────────────                              ───────────────
fa4_fwd()
  ├─ attn_arg.to_fa4_args()
  │   ├─ aux_tensors = [hstu_func]         → aux_tensors
  │   ├─ linear_k_block_sparse_mask        → block_sparse_tensors
  │   └─ linear_q_block_sparse_mask        → （反向用）
  │
  ├─ q.unsqueeze(0)                        → (1, s, h, d) 格式
  │
  └─ _flash_attn_fwd(
        q, k, v,
        arbitrary=True,                    ← 启用 HSTU mask
        block_sparse_tensors=...,          ← CSR 稀疏 mask
        aux_tensors=[hstu_func],           ← HSTU 函数张量
      )
        ↓
      interface.py
        → FlashAttentionForwardSm100.__call__()
          → JIT 编译 + launch kernel
```

### 13.2 编译参数影响

MagiAttention 的 mask 配置直接影响 kernel 编译：

```python
compile_key = (
    dtype,              # BF16/FP16
    head_dim,           # 128
    head_dim_v,         # 128
    qhead_per_kvhead,   # GQA ratio
    causal=False,       # ★ arbitrary 时为 False
    arbitrary=True,     # ★ 启用 HSTU
    func_num=N,         # ★ HSTU 函数数量（如 3, 5, 7）
    use_block_sparsity=True,  # ★ 启用 tile 跳过
    ...
)
```

**不同的 func_num 会编译出不同的 kernel**——这是因为 `mask_r2p_intervals` 中的循环需要 `range_constexpr` 展开，func_num 必须是编译时常量。

### 13.3 性能特征

| 配置 | SM100 kernel 特性 |
|------|-------------------|
| 标准 causal | 内置 causal mask，最高性能 |
| arbitrary + block_sparse | HSTU 函数 mask + tile 跳过，接近 causal 性能 |
| arbitrary 无 block_sparse | 每个 tile 都需要 HSTU 判断，性能下降 |
| func_num 增大 | mask_r2p_intervals 循环展开增加，寄存器压力增大 |

---

## 十四、总结

| 维度 | 设计选择 | 原因 |
|------|---------|------|
| **编程模型** | CuTE DSL (Python JIT) | 快速迭代 SM100 新特性，避免 C++ 编译周期 |
| **Warp 专业化** | 16 warps 6 种角色 | 最大化 MMA/TMA/Softmax 三路 overlap |
| **数据通路** | P 直接从 TMEM 读 | 省去 TMEM→SMEM 的额外 copy |
| **Mask 实现** | R2P 指令 + 位运算 | 多区间 mask 零分支开销 |
| **稀疏调度** | CSR Block Sparse + LPT | 跳过空 tile + 负载均衡 |
| **SMEM 管理** | 不均匀分配 + 复用 | 支持 DeepSeek (192,128) 异形 head_dim |
| **反向 dQ** | TMA reduce-add | 直接全局内存原子累加，避免二次通信 |
