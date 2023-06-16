# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser

import torch
import torch.distributed as dist
import deepspeed

import slapo
from slapo.model_schedule.base import (
    shard_attention,
    shard_mlp,
    trace_attention,
    replace_sdp,
)
from utils import perf_model, get_model

parser = ArgumentParser()
parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--local_rank", required=True, type=int, help="local_rank")
parser.add_argument("--bs", default=1, type=int, help="batch size")
args = parser.parse_args()

bs = args.bs


def optimize(mod, config):
    sch = slapo.create_schedule(mod)
    if dist.get_world_size() == 1:
        mod, _ = slapo.build(sch, init_weights=mod._init_weights)
        return mod
    for i in range(config.num_hidden_layers):
        shard_attention(
            sch[f"encoder.layer.{i}.attention"],
            names=["self.query", "self.key", "self.value", "output.dense"],
            attrs=["self.num_attention_heads", "self.all_head_size"],
        )
        shard_mlp(
            sch[f"encoder.layer.{i}"], names=["intermediate.dense", "output.dense"]
        )
        trace_attention(sch[f"encoder.layer.{i}.attention"], config)
        replace_sdp(sch[f"encoder.layer.{i}.attention"], config)
    mod, _ = slapo.build(sch, init_weights=mod._init_weights)
    return mod


if __name__ == "__main__":
    dist.init_process_group("nccl", world_size=int(os.environ["WORLD_SIZE"]))
    torch.cuda.set_device(args.local_rank)

    for kernel_opt in [False]:
        mod, config, seq_len = get_model(args.name)
        mod = optimize(mod, config)
        # Initialize the DeepSpeed-Inference engine
        # https://www.deepspeed.ai/tutorials/inference-tutorial/
        ds_engine = deepspeed.init_inference(
            mod,
            mp_size=1,
            dtype=torch.float16,
            max_out_tokens=seq_len,
            checkpoint=None,
            replace_with_kernel_inject=kernel_opt,
        )
        mod = ds_engine.module
        input_ids = torch.ones((bs, seq_len), dtype=torch.long, device="cuda")
        if dist.get_rank() == 0:
            print(mod)
        perf_model(mod, input_ids)
        del mod
