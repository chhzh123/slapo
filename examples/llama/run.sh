#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# deepspeed deepspeed_hf.py --batch_size 8 --micro_batch_size 8 --model_name decapoda-research/llama-7b-hf --iter_nums 20 --hidden-size 4096 --nlayers 32 --num-attn-heads 32 --dropout 0.0 --activation_function silu --seq_len 1024 --pmp 1 --tmp 8 --fp16 --checkpoint 0.9

deepspeed --num_gpus 8 --num_nodes 2 --master_addr 172.31.7.13 --hostfile 2nodes_hostfile deepspeed_hf.py --batch_size 256 --micro_batch_size 8 --model_name decapoda-research/llama-7b-hf --iter_nums 20 --hidden-size 4096 --nlayers 32 --num-attn-heads 32 --dropout 0.0 --activation_function silu --seq_len 1024 --pmp 2 --tmp 8 --fp16 --checkpoint 1.0
