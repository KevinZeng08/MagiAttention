#!/bin/bash

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

export CUDA_VISIBLE_DEVICES="0,1"
export WORLD_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

export MASTER_ADDRESS="localhost"
export MASTER_PORT=23457

export PYTHONPATH=$PYTHONPATH:$(pwd)


# --------  for cuda -------- #


# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# --------  for torch -------- #

export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_USE_CUDA_DSA=1

# --------  for torch-dist -------- #

export OMP_NUM_THREADS=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_NCCL_USE_COMM_NONBLOCKING=1 # always segment fault
export TORCH_NCCL_TRACE_CPP_STACK=true
# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_NCCL_ENABLE_MONITORIN=1


# --------  for nccl -------- #

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=1
# export NCCL_CHECK_POINTERS=1
# export NCCL_LAUNCH_RACE_FATAL=1 # since 2.26
# export NCCL_LAUNCH_ORDER_IMPLICIT=1 # since 2.26
# export NCCL_RAS_ENABLE=1
# export NCCL_CUMEM_ENABLE=0
# export NCCL_PXN_DISABLE=1
# export NCCL_COMM_BLOCKING=1
# export NCCL_NCHANNELS_PER_NET_PEER=1
# export NCCL_WORK_FIFO_BYTES=1073741824

# --------  for magi-attn -------- #

# export MAGI_ATTENTION_UNITEST_PROFILE_MODE=1
# export MAGI_ATTENTION_HIERARCHICAL_COMM=1
export MAGI_ATTENTION_SANITY_CHECK=1

# export MAGI_NCCL_BACKEND=1 # whether to use magi_nccl backend
export MAGI_ATTENTION_USE_BATCH_P2P_FOR_GROUP_COLLECTIVE=1 # whether to use batch p2p instead of a2av with nccl backend


# --------  for magi-attn debug -------- #

export MAGI_ATTENTION_DEBUG_FWD_USE_FA3=0
export MAGI_ATTENTION_DEBUG_BWD_USE_FA3=0
export MAGI_ATTENTION_DEBUG_FWD_USE_REF=none # whether to use ref tensors for fwd, choose from get, set, none
export MAGI_ATTENTION_DEBUG_BWD_USE_REF=none # whether to use ref tensors for bwd, choose from get, set, none
export MAGI_ATTENTION_DEBUG_DUMMY_ALL2ALLV_TIMES=0 # set it larger to let all2all-v slower


# --------  for script -------- #

export DEBUG_PROFILE_MODE=0 # whether to profile, default with nsys
export DEBUG_USE_NCU_FOR_PROFILE=0 # whether to profile with ncu
export DEBUG_SANITIZER_MODE=0 # whether to use compute-sanitizer, but seem useless for multi-proc


# --------  run cmds -------- #

CMD="torchrun \
    --standalone \
    --nnode 1 \
    --nproc_per_node=$WORLD_SIZE \
    --master_addr=$MASTER_ADDRESS \
    --master_port=$MASTER_PORT \
    debug.py
"

if [[ $DEBUG_PROFILE_MODE == "1" ]]; then
    if [[ $DEBUG_USE_NCU_FOR_PROFILE == "1" ]]; then
        ncu \
            --target-processes all \
            --set full \
            --kernel-name device_kernel \
            -f -o debug.ncu-rep \
            $CMD > debug.log 2>&1
    else
        nsys profile \
            --force-overwrite true \
            -o debug.nsys-rep \
            --capture-range=cudaProfilerApi \
            $CMD > debug.log 2>&1
    fi
elif [[ $DEBUG_SANITIZER_MODE == "1" ]]; then
    compute-sanitizer --tool memcheck $CMD > debug.log 2>&1
else
    $CMD > debug.log 2>&1
fi
