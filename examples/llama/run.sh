#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# This script is used to run GPT2 training with 3D parallelism enabled.
# It is tested on a single AWS p3d instance with 8*V100 GPUs.
deepspeed deepspeed_hf.py --batch_size 4 --micro_batch_size 2 --model_name decapoda-research/llama-7b-hf --iter_nums 20 --hidden-size 4096 --nlayers 32 --num-attn-heads 32 --dropout 0.0 --activation_function silu --seq_len 1024 --pmp 2 --tmp 2 --checkpoint 0.0