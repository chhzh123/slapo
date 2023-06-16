# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser

import torch
import torch.distributed as dist

import slapo
from slapo.model_schedule.base import (
    shard_attention,
    shard_mlp,
    trace_attention,
    replace_sdp,
    fuse_bias_gelu,
    fuse_ln_residual,
)
from utils import perf_model, get_model

parser = ArgumentParser()
parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--bs", default=1, type=int, help="batch size")
args = parser.parse_args()

bs = args.bs


def optimize(mod, config):
    sch = slapo.create_schedule(mod)
    for i in range(config.num_hidden_layers):
        # May degrade performance for BERT
        # fuse_bias_gelu(sch[f"encoder.layer.{i}.intermediate"], "dense")
        # fuse_ln_residual(
        #     sch[f"encoder.layer.{i}.attention.output"], names=["dense", "LayerNorm"]
        # )
        # fuse_ln_residual(sch[f"encoder.layer.{i}.output"], names=["dense", "LayerNorm"])
        if sch.world_size > 1:
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
    if sch.world_size == 1:
        mod = torch.compile(mod, fullgraph=True, backend="inductor")
    return mod


if __name__ == "__main__":
    dist.init_process_group("nccl", world_size=int(os.environ["WORLD_SIZE"]))
    torch.cuda.set_device(dist.get_rank())

    for kernel_opt in [False]:
        mod, config, seq_len = get_model(args.name)
        mod = optimize(mod, config)
        input_ids = torch.ones((bs, seq_len), dtype=torch.long, device="cuda")
        if dist.get_rank() == 0:
            print(mod)
        perf_model(mod, input_ids, use_cuda_graph=True)
        del mod
