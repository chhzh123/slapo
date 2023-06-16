# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser

import torch
import torch.distributed as dist
import deepspeed

from utils import perf_model, get_model

parser = ArgumentParser()
parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--local_rank", required=True, type=int, help="local_rank")
parser.add_argument("--bs", default=1, type=int, help="batch size")
args = parser.parse_args()

bs = args.bs


if __name__ == "__main__":
    dist.init_process_group("nccl", world_size=int(os.environ["WORLD_SIZE"]))

    for kernel_opt in [False, True]:
        mod, _, seq_len = get_model(args.name)
        # Initialize the DeepSpeed-Inference engine
        # https://www.deepspeed.ai/tutorials/inference-tutorial/
        ds_engine = deepspeed.init_inference(
            mod,
            mp_size=dist.get_world_size(),
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
