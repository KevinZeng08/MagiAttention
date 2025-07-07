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


def setup_dist_env(
    backend: str = "nccl",
    base_seed: int | None = None,
) -> tuple[int, int, dist.ProcessGroup, str, int | None]:
    """set up distributed environment with the specified process group backend,
    NOTE: the test script using this func to set up should be executed through torchrun
    """
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    manual_seed = None
    if base_seed is not None:
        torch.manual_seed(manual_seed)

    return rank, world_size, dist.group.WORLD, f"cuda:{rank}", manual_seed  # noqa: E231


def clearup_dist_env() -> None:
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


def _group_cast_impl_with_batch_p2p(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    group: dist.Backend = None,
    async_op: bool = False,
    **kwargs,
):
    input_list = input.split(input_split_size_list, dim=0)
    output_list = output.split(output_split_size_list, dim=0)

    p2p_op_list = []

    # send
    for input_split_idx in range(len(input_split_size_list)):
        for dst_rank in dst_indices_list[input_split_idx]:
            p2p_op_list.append(
                dist.P2POp(
                    op=dist.isend,
                    tensor=input_list[input_split_idx],
                    peer=dst_rank,
                    group=group,
                )
            )
    # recv
    for output_split_idx in range(len(output_split_size_list)):
        src_rank = src_index_list[output_split_idx]
        p2p_op_list.append(
            dist.P2POp(
                op=dist.irecv,
                tensor=output_list[output_split_idx],
                peer=src_rank,
                group=group,
            )
        )

    work_list = dist.batch_isend_irecv(p2p_op_list)

    return work_list


prof_iters, prof_start_iter, prof_end_iter = 10, 5, 8


rank, world_size, group, device, manual_seed = setup_dist_env()
dtype = torch.float16
device = torch.cuda.current_device()

input = torch.randn(12288, 1, 128, dtype=dtype, device=device)
output = torch.randn(12288, 1, 128, dtype=dtype, device=device)

input_split_size_list_per_rank = [
    # rank0
    [
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
    ],
    # rank1
    [
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
    ],
]

output_split_size_list_per_rank = [
    # rank0
    [
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
    ],
    # rank1
    [
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
    ],
]

dst_indices_list_per_rank = [
    # rank0
    [
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
    ],
    # rank1
    [
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
    ],
]

src_index_list_per_rank = [
    # rank0
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    # rank1
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

input_split_size_list = input_split_size_list_per_rank[rank]
output_split_size_list = output_split_size_list_per_rank[rank]
dst_index_list = dst_indices_list_per_rank[rank]
src_index_list = src_index_list_per_rank[rank]


for iter in range(prof_iters):
    # -----    profile control if using profile mode   ---- #

    if rank == 0 and iter == prof_start_iter:
        torch.cuda.profiler.start()
        torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
    if rank == 0 and iter == prof_end_iter:
        torch.cuda.profiler.stop()

    dist.barrier()
    torch.cuda.synchronize()

    work_list = _group_cast_impl_with_batch_p2p(
        input=input,
        output=output,
        input_split_size_list=input_split_size_list,
        output_split_size_list=output_split_size_list,
        dst_indices_list=dst_index_list,
        src_index_list=src_index_list,
        group=group,
        async_op=True,
    )

    for work in work_list:
        work.wait()

clearup_dist_env()
