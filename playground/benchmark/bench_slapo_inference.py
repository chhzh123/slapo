# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser

import torch
import torch.distributed as dist
import slapo
from utils import perf_model, get_model
from bert_schedule import schedule_bert
from llama_schedule import schedule_llama

parser = ArgumentParser()
parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--bs", default=1, type=int, help="batch size")
parser.add_argument("--max_seq_len", default=1024, type=int, help="sequence length")
parser.add_argument("--nsys", default=False, type=bool, help="Use nsys to profile")
# import deepspeed
# parser.add_argument("--local_rank", default=0, type=int, help="local rank")
args = parser.parse_args()

bs = args.bs

SCHEDULE_MAP = {
    "bert": schedule_bert,
    "llama": schedule_llama,
}


def create_optimized_schedule(name, mod, config):
    if name not in SCHEDULE_MAP:
        raise ValueError(f"Unknown model {name}")
    return SCHEDULE_MAP[name](mod, config)


if __name__ == "__main__":
    # https://pytorch.org/docs/stable/notes/cuda.html#id5
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    dist.init_process_group("nccl", world_size=int(os.environ["WORLD_SIZE"]))
    # deepspeed.init_distributed()
    torch.cuda.set_device(dist.get_rank())

    mod, config, seq_len = get_model(
        args.name, meta=False if args.name == "bert" else True
    )
    if seq_len > args.max_seq_len:
        seq_len = args.max_seq_len
    sch = create_optimized_schedule(args.name, mod, config)
    mod, _ = slapo.build(sch, init_weights=mod._init_weights)
    input_ids = torch.ones(
        (bs, seq_len), dtype=torch.long, device="cuda", requires_grad=False
    )
    if dist.get_rank() == 0:
        print(mod)
        if args.nsys:
            print("Use nsys to profile...")
    perf_model(
        mod,
        input_ids,
        use_cuda_graph=False if args.name == "bert" else False,
        nsys=args.nsys,
    )
    del mod
