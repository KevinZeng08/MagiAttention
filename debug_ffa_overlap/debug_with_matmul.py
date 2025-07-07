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

import magi_attention
from magi_attention.common import AttnRanges
from magi_attention.functional import flex_flash_attn_func
from magi_attention.testing.precision import (
    EPSILON,
    calc_inf_norm,
    extract_mismatch_threshold,
    torch_attn_ref,
)
from magi_attention.utils import get_attn_mask_from_ranges

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
        print("\n\n".join(err_msg_list))


profile_mode = os.environ.get("DEBUG_PROFILE_MODE", "0") == "1"
run_side_matmul = os.environ.get("DEBUG_RUN_SIDE_MATMUL", "0") == "1"

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


q_range_tensor = q_ranges.to_tensor(device=device)
k_range_tensor = k_ranges.to_tensor(device=device)
max_seqlen_q = q_ranges.max_seqlen
max_seqlen_k = k_ranges.max_seqlen
attn_type_map = torch.tensor(
    [1 if is_causal else 0 for is_causal in is_causal_mapping],
    dtype=torch.int32,
    device=device,
)


torch.manual_seed(42)


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

grad_total_out = torch.randn_like(total_q).detach()


# prepare side matmul
if run_side_matmul:
    side_stream = torch.cuda.Stream()

    # m, n, k = 16 * 1024, 16 * 1024, 8 * 1024
    # a = torch.randn(m, k, device=device, dtype=dtype)
    # b = torch.randn(k, n, device=device, dtype=dtype)

    total_q_fork = torch.randn(
        total_seqlen_q,
        num_heads * 10,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=False,
    )
    total_k_fork = torch.randn(
        total_seqlen_k,
        num_heads * 10,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=False,
    )
    total_v_fork = torch.randn(
        total_seqlen_k,
        num_heads * 10,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=False,
    )

    grad_total_out_fork = torch.randn_like(total_q_fork).detach()


# -----    run pipeline test   ---- #

for iter in range(prof_iters):
    # -----    profile control if using profile mode   ---- #

    if profile_mode:
        if iter == prof_start_iter:
            torch.cuda.profiler.start()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
        if iter == prof_end_iter:
            torch.cuda.profiler.stop()

    # -----    barrier at the beginning of each iteration   ---- #

    torch.cuda.synchronize()

    if run_side_matmul:
        with torch.cuda.stream(side_stream):
            # c = a @ b
            flex_flash_attn_func(
                q=total_q_fork,
                k=total_k_fork,
                v=total_v_fork,
                q_ranges=q_range_tensor,
                k_ranges=k_range_tensor,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                attn_type_map=attn_type_map,
                deterministic=True,
                sm_margin=90,
            )

    # -----   run ffa forward   ---- #

    total_out, _ = flex_flash_attn_func(
        q=total_q,
        k=total_k,
        v=total_v,
        q_ranges=q_range_tensor,
        k_ranges=k_range_tensor,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        attn_type_map=attn_type_map,
        deterministic=True,
        sm_margin=32,
    )

    # -----   run ffa backward   ---- #

    if is_run_bwd:
        total_q.grad = None
        total_k.grad = None
        total_v.grad = None
        total_out.backward(grad_total_out)

        # -----   run side matmul   ---- #

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
