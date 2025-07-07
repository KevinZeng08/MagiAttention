# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
import torch.distributed as dist

import magi_attention

# from magi_attention.comm.primitive.magi_nccl_interface import (  # type: ignore[attr-defined]
#     MagiNCCLBackend,
# )
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.config import (
    DispatchConfig,
    DistAttnConfig,
    MinHeapDispatchAlg,
    OverlapConfig,
)
from magi_attention.dist_attn_runtime_mgr import (
    DistAttnRuntimeMgr,
    init_dist_attn_runtime_mgr,
)
from magi_attention.testing.precision import (
    EPSILON,
    calc_inf_norm,
    extract_mismatch_threshold,
    torch_attn_ref,
)
from magi_attention.utils import (
    clearup_dist_env,
    get_attn_mask_from_ranges,
    setup_dist_env,
)

rank, world_size, group, device, manual_seed = setup_dist_env(
    backend="cpu:gloo,cuda:magi_nccl"
    if magi_attention.is_magi_nccl_backend_enable()
    else "cpu:gloo,cuda:nccl",
    base_seed=42,
)
device = torch.cuda.current_device()


def assert_close_to_torch_ref(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    is_causal_mapping: list[bool],
    total_seqlen_q: int,
    total_seqlen_k: int,
    total_q: torch.Tensor,
    total_k: torch.Tensor,
    total_v: torch.Tensor,
    total_out: torch.Tensor,
    grad_total_q: torch.Tensor | None,
    grad_total_k: torch.Tensor | None,
    grad_total_v: torch.Tensor | None,
    grad_total_out: torch.Tensor | None,
    dtype: torch.dtype,
    run_bwd: bool,
    test_case: str = "",
) -> None:
    # -----   customize tolerance threshold  ---- #

    o_atol = EPSILON
    o_rtol = {torch.bfloat16: 0.05, torch.float16: 0.05}.get(dtype, 0.05)

    dq_atol = EPSILON
    dq_rtol = {torch.bfloat16: 0.3, torch.float16: 0.2}.get(dtype, 0.2)

    dk_atol = EPSILON
    dk_rtol = {torch.bfloat16: 0.15, torch.float16: 0.08}.get(dtype, 0.08)

    dv_atol = EPSILON
    dv_rtol = {torch.bfloat16: 0.05, torch.float16: 0.05}.get(dtype, 0.05)

    # NOTE: an experimental value from magi_attention testing
    mismatch_thres_ratio: float = 2.0
    # NOTE: an experimental value from fa testing
    norm_rtol_ratio: float = 2.0

    # -----   build attn mask   ---- #

    mask = get_attn_mask_from_ranges(
        q_ranges=q_ranges.to_naive_ranges(),
        k_ranges=k_ranges.to_naive_ranges(),
        is_causal_mapping=is_causal_mapping,
        total_seqlen_q=total_seqlen_q,
        total_seqlen_k=total_seqlen_k,
    )

    # -----   ref1. torch ref with high precision (fp32)   ---- #

    total_q.grad, total_k.grad, total_v.grad = None, None, None

    total_out_ref_high_precision = torch_attn_ref(
        q=total_q,
        k=total_k,
        v=total_v,
        mask=mask,
        layout="thd",
        high_precision=True,
    )

    if run_bwd:
        total_out_ref_high_precision.backward(grad_total_out)
        (
            grad_total_q_ref_high_precision,
            grad_total_k_ref_high_precision,
            grad_total_v_ref_high_precision,
        ) = (
            total_q.grad,
            total_k.grad,
            total_v.grad,
        )

    # -----   ref2. torch ref with low precision (fp16/bf16)   ---- #

    total_q.grad, total_k.grad, total_v.grad = None, None, None

    total_out_ref_low_precision = torch_attn_ref(
        q=total_q,
        k=total_k,
        v=total_v,
        mask=mask,
        layout="thd",
        high_precision=False,
    )

    if run_bwd:
        total_out_ref_low_precision.backward(grad_total_out)
        (
            grad_total_q_ref_low_precision,
            grad_total_k_ref_low_precision,
            grad_total_v_ref_low_precision,
        ) = (
            total_q.grad,
            total_k.grad,
            total_v.grad,
        )

    # -----   init error message list   ---- #

    err_msg_list: list[str] = []

    # -----   assert close for fwd out   ---- #

    # fa style with Linf norm
    out_norm = calc_inf_norm(total_out, total_out_ref_high_precision)
    out_ref_norm = calc_inf_norm(
        total_out_ref_low_precision, total_out_ref_high_precision
    )
    try:
        assert (
            out_norm <= norm_rtol_ratio * out_ref_norm
        ), f"For {test_case=}: {out_norm=} should be no greater than {norm_rtol_ratio}x of {out_ref_norm=}"
    except Exception as e:
        err_msg_list.append(str(e))

    # torch style with atol + rtol + mismatch threshold
    o_thres = extract_mismatch_threshold(
        actual=total_out_ref_low_precision,
        expected=total_out_ref_high_precision,
        atol=o_atol,
        rtol=o_rtol,
        mismatch_thres_ratio=mismatch_thres_ratio,
    )
    try:
        magi_attention.testing.assert_close(  # type: ignore[attr-defined]
            total_out,
            total_out_ref_high_precision,
            atol=o_atol,
            rtol=o_rtol,
            mismatch_threshold=o_thres,
            test_case=f"{test_case} => o",
        )
    except Exception as e:
        err_msg_list.append(str(e))

    if run_bwd:
        # -----   assert close for bwd dq   ---- #

        # fa style with Linf norm
        dq_norm = calc_inf_norm(grad_total_q, grad_total_q_ref_high_precision)
        dq_ref_norm = calc_inf_norm(
            grad_total_q_ref_low_precision, grad_total_q_ref_high_precision
        )
        try:
            assert (
                dq_norm <= norm_rtol_ratio * dq_ref_norm
            ), f"For {test_case=}: {dq_norm=} should be no greater than {norm_rtol_ratio}x of {dq_ref_norm=}"
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        dq_thres = extract_mismatch_threshold(
            actual=grad_total_q_ref_low_precision,
            expected=grad_total_q_ref_high_precision,
            atol=dq_atol,
            rtol=dq_rtol,
            mismatch_thres_ratio=mismatch_thres_ratio,
        )
        try:
            magi_attention.testing.assert_close(  # type: ignore[attr-defined]
                grad_total_q,
                grad_total_q_ref_high_precision,
                atol=dq_atol,
                rtol=dq_rtol,
                mismatch_threshold=dq_thres,
                test_case=f"{test_case} => dq",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # -----   assert close for bwd dk   ---- #

        # fa style with Linf norm
        dk_norm = calc_inf_norm(grad_total_k, grad_total_k_ref_high_precision)
        dk_ref_norm = calc_inf_norm(
            grad_total_k_ref_low_precision, grad_total_k_ref_high_precision
        )
        try:
            assert (
                dk_norm <= norm_rtol_ratio * dk_ref_norm
            ), f"For {test_case=}: {dk_norm=} should be no greater than {norm_rtol_ratio}x of {dk_ref_norm=}"
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        dk_thres = extract_mismatch_threshold(
            actual=grad_total_k_ref_low_precision,
            expected=grad_total_k_ref_high_precision,
            atol=dk_atol,
            rtol=dk_rtol,
            mismatch_thres_ratio=mismatch_thres_ratio,
        )
        try:
            magi_attention.testing.assert_close(  # type: ignore[attr-defined]
                grad_total_k,
                grad_total_k_ref_high_precision,
                atol=dk_atol,
                rtol=dk_rtol,
                mismatch_threshold=dk_thres,
                test_case=f"{test_case} => dk",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # -----   assert close for bwd dv   ---- #

        # fa style with Linf norm
        dv_norm = calc_inf_norm(grad_total_v, grad_total_v_ref_high_precision)
        dv_ref_norm = calc_inf_norm(
            grad_total_v_ref_low_precision, grad_total_v_ref_high_precision
        )
        try:
            assert (
                dv_norm <= norm_rtol_ratio * dv_ref_norm
            ), f"For {test_case=}: {dv_norm=} should be no greater than {norm_rtol_ratio}x of {dv_ref_norm=}"
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        dv_thres = extract_mismatch_threshold(
            actual=grad_total_v_ref_low_precision,
            expected=grad_total_v_ref_high_precision,
            atol=dv_atol,
            rtol=dv_rtol,
            mismatch_thres_ratio=mismatch_thres_ratio,
        )
        try:
            magi_attention.testing.assert_close(  # type: ignore[attr-defined]
                grad_total_v,
                grad_total_v_ref_high_precision,
                atol=dv_atol,
                rtol=dv_rtol,
                mismatch_threshold=dv_thres,
                test_case=f"{test_case} => dv",
            )
        except Exception as e:
            err_msg_list.append(str(e))

    # -----   raise error if any error occurs   ---- #

    if err_msg_list:
        if rank == 0:
            print("\n\n".join(err_msg_list))


# backend = get_pg_backend(group)
# if magi_attention.is_magi_nccl_backend_enable():
#     assert isinstance(backend, MagiNCCLBackend)

profile_mode = os.environ.get("DEBUG_PROFILE_MODE", "0") == "1"

if profile_mode:  # [start_iter, end_iter)
    prof_iters, prof_start_iter, prof_end_iter = 10, 2, 9
else:
    prof_iters, prof_start_iter, prof_end_iter = 50, -1, -1


is_run_bwd = True
num_heads, head_dim = 1, 128
dtype = torch.float16


times = 1
name = f"varlen_block_causal_{12 * times}k_with_q_overlap"
q_ranges = AttnRanges.from_ranges(
    [
        [0 * times, 8192 * times],
        [2048 * times, 8192 * times],
        [4096 * times, 8192 * times],
        [6144 * times, 8192 * times],
        [8192 * times, 12288 * times],
        [10240 * times, 12288 * times],
    ]
)
k_ranges = AttnRanges.from_ranges(
    [
        [0 * times, 2048 * times],
        [2048 * times, 4096 * times],
        [4096 * times, 6144 * times],
        [6144 * times, 8192 * times],
        [8192 * times, 10240 * times],
        [10240 * times, 12288 * times],
    ]
)
is_causal_mapping = [False] * 6
total_seqlen_q = 12288 * times
total_seqlen_k = 12288 * times
chunk_size = 512 * times


# name = "varlen_block_causal_6k_with_q_overlap"
# q_ranges = AttnRanges.from_ranges(
#     [
#         [0, 8192 // 2],
#         [2048 // 2, 8192 // 2],
#         [4096 // 2, 8192 // 2],
#         [6144 // 2, 8192 // 2],
#         [8192 // 2, 12288 // 2],
#         [10240 // 2, 12288 // 2],
#     ]
# )
# k_ranges = AttnRanges.from_ranges(
#     [
#         [0, 2048 // 2],
#         [2048 // 2, 4096 // 2],
#         [4096 // 2, 6144 // 2],
#         [6144 // 2, 8192 // 2],
#         [8192 // 2, 10240 // 2],
#         [10240 // 2, 12288 // 2],
#     ]
# )
# is_causal_mapping = [False] * 6
# total_seqlen_q = 12288 // 2
# total_seqlen_k = 12288 // 2
# chunk_size = 512 // 2


# name = "full_attn_32k"
# s = 32 * 1024
# q_ranges = AttnRanges.from_ranges([[0, s]])
# k_ranges = AttnRanges.from_ranges([[0, s]])
# is_causal_mapping = [False]
# total_seqlen_q = s
# total_seqlen_k = s
# chunk_size = 512

# name = "full_attn_14k"
# q_ranges = AttnRanges.from_ranges([[0, 14336]])
# k_ranges = AttnRanges.from_ranges([[0, 14336]])
# is_causal_mapping = [False]
# total_seqlen_q = 14336
# total_seqlen_k = 14336
# chunk_size = 512

# name = "full_attn_7k"
# q_ranges = AttnRanges.from_ranges([[0, 14336//2]])
# k_ranges = AttnRanges.from_ranges([[0, 14336//2]])
# is_causal_mapping = [False]
# total_seqlen_q = 14336 // 2
# total_seqlen_k = 14336 // 2
# chunk_size = 512 // 2


dist_attn_config = DistAttnConfig(
    dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
    overlap_config=OverlapConfig(enable=False),
    high_bandwith_domain_size=1,
    deterministic=False,
)


# -----    run pipeline test   ---- #

for iter in range(prof_iters):
    torch.manual_seed(42)
    # -----    profile control if using profile mode   ---- #

    if profile_mode:
        if rank == 0 and iter == prof_start_iter:
            torch.cuda.profiler.start()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
        if rank == 0 and iter == prof_end_iter:
            torch.cuda.profiler.stop()

    # -----    barrier at the beginning of each iteration   ---- #

    dist.barrier()
    torch.cuda.synchronize()

    # -----    init dist attn runtime mgr   ---- #

    dist_attn_runtime_mgr: DistAttnRuntimeMgr = init_dist_attn_runtime_mgr(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=[
            AttnMaskType.CAUSAL if is_causal else AttnMaskType.FULL
            for is_causal in is_causal_mapping
        ],
        total_seqlen_q=total_seqlen_q,
        total_seqlen_k=total_seqlen_k,
        chunk_size=chunk_size,
        cp_group=group,
        is_same_source=True,
        is_q_permutable=True,
        is_k_permutable=True,
        dist_attn_config=dist_attn_config,
        cp_mesh=None,
    )

    # -----   init global qkv   ---- #

    total_q = torch.randn(
        total_seqlen_q,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=is_run_bwd,
    )
    total_k = torch.randn(
        total_seqlen_k,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=is_run_bwd,
    )
    total_v = torch.randn(
        total_seqlen_k,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=is_run_bwd,
    )
    dist.all_reduce(total_q.data, group=group)
    dist.all_reduce(total_k.data, group=group)
    dist.all_reduce(total_v.data, group=group)

    # -----   dispatch global qkv to local qkv   ---- #

    local_q = dist_attn_runtime_mgr.dispatch_qo(total_q)
    local_k = dist_attn_runtime_mgr.dispatch_kv(total_k)
    local_v = dist_attn_runtime_mgr.dispatch_kv(total_v)

    # -----   run dist attn forward on local qkv for local o   ---- #

    local_out, _ = dist_attn_runtime_mgr.calc_attn(local_q, local_k, local_v)

    # -----   undispatch local o to global o   ---- #

    total_out = dist_attn_runtime_mgr.undispatch_qo(local_out)

    if is_run_bwd:
        grad_total_out = torch.randn_like(total_out).detach()
        dist.all_reduce(grad_total_out.data, group=group)
        total_out.backward(grad_total_out)
        grad_total_q, grad_total_k, grad_total_v = (
            total_q.grad,
            total_k.grad,
            total_v.grad,
        )

    # -----   assert close if not using profile mode   ---- #

    # if profile_mode:
    #     continue

    # -----   assert close to torch ref   ---- #

    assert_close_to_torch_ref(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        is_causal_mapping=is_causal_mapping,
        total_seqlen_q=total_seqlen_q,
        total_seqlen_k=total_seqlen_k,
        total_q=total_q,
        total_k=total_k,
        total_v=total_v,
        total_out=total_out,
        grad_total_q=grad_total_q,
        grad_total_k=grad_total_k,
        grad_total_v=grad_total_v,
        grad_total_out=grad_total_out,
        dtype=dtype,
        run_bwd=is_run_bwd,
        test_case=name,
    )

clearup_dist_env()
