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

export PYTHONPATH=$PYTHONPATH:$(pwd)

# --------  for cuda -------- #


# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_DEVICE_MAX_CONNECTIONS=1

# --------  for torch -------- #

export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_USE_CUDA_DSA=1


# --------  for script -------- #

export DEBUG_RUN_SIDE_MATMUL=1

export DEBUG_PROFILE_MODE=0 # whether to profile, default with nsys
export DEBUG_USE_NCU_FOR_PROFILE=0 # whether to profile with ncu
export DEBUG_SANITIZER_MODE=0 # whether to use compute-sanitizer, but seem useless for multi-proc


# --------  run cmds -------- #

CMD="python debug_with_matmul.py"

if [[ $DEBUG_PROFILE_MODE == "1" ]]; then
    if [[ $DEBUG_USE_NCU_FOR_PROFILE == "1" ]]; then
        ncu \
            --target-processes all \
            --set full \
            --kernel-name device_kernel \
            -f -o debug_with_matmul.ncu-rep \
            $CMD > debug_with_matmul.log 2>&1
    else
        nsys profile \
            --force-overwrite true \
            -o debug_with_matmul.nsys-rep \
            --capture-range=cudaProfilerApi \
            $CMD > debug_with_matmul.log 2>&1
    fi
elif [[ $DEBUG_SANITIZER_MODE == "1" ]]; then
    compute-sanitizer --tool racecheck $CMD > debug_with_matmul.log 2>&1
else
    $CMD > debug_with_matmul.log 2>&1
fi
