#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# PP=1, MP=8
deepspeed deepspeed_hf.py --batch_size 8 --micro_batch_size 8 --model_name decapoda-research/llama-7b-hf --iter_nums 20 --hidden-size 4096 --nlayers 32 --num-attn-heads 32 --dropout 0.0 --activation_function silu --seq_len 1024 --pmp 1 --tmp 8 --fp16 --checkpoint 1.0

# ZeRO-3
deepspeed deepspeed_hf.py --batch_size 8 --micro_batch_size 1 --model_name decapoda-research/llama-7b-hf --iter_nums 20 --hidden-size 4096 --nlayers 32 --num-attn-heads 32 --dropout 0.0 --activation_function silu --seq_len 1024 --pmp 1 --tmp 1 --fp16 --checkpoint 1.0
