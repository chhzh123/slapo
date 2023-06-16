# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser

import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoModel, AutoConfig

parser = ArgumentParser()
parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--local_rank", required=True, type=int, help="local_rank")
parser.add_argument("--bs", default=1, type=int, help="batch size")
args = parser.parse_args()

bs = args.bs

MODEL_SETTINGS = {
    "bert": ("bert-large-uncased", 512),
    "gpt": ("EleutherAI/gpt-neo-1.3B", 1024),
    "llama": ("decapoda-research/llama-7b-hf", 2048),
    "vicuna": ("lmsys/vicuna-13b-delta-v1.1", 2048),
    "opt": ("facebook/opt-13b", 2048),
}


def perf_model(mod, input_tensor):
    """Measure the performance of a mod with certain resharding schemes"""
    # warmup
    mod.eval()
    torch.cuda.empty_cache()
    # mod.to(torch.float16)
    for _ in range(10):
        mod(input_tensor)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    iters = 40
    for _ in range(iters):
        mod(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    if dist.get_rank() == 0:
        print(f"{start_event.elapsed_time(end_event) / iters:.3f} ms")


def get_model(name):
    model_name, seq_len = MODEL_SETTINGS[name]
    if dist.get_rank() == 0:
        print(f"Loading {model_name} with seq_len {seq_len}")
    config = AutoConfig.from_pretrained(model_name)
    if model_name != "bert":
        config.use_cache = False
    mod = AutoModel.from_pretrained(model_name)
    mod.eval()
    mod.to(torch.float16)
    return mod, seq_len


if __name__ == "__main__":
    dist.init_process_group("nccl", world_size=int(os.environ["WORLD_SIZE"]))

    for kernel_opt in [False, True]:
        for mod, seq_len in [get_model(args.name)]:
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
