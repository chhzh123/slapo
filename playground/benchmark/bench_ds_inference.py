# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.distributed as dist
import deepspeed

# Config for verification
bs = 4
seq_len = 1024


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


def get_bert_model():
    from transformers import BertLMHeadModel, AutoConfig

    config = AutoConfig.from_pretrained("bert-large-uncased")
    mod = BertLMHeadModel(config)
    bs, seq_len = 8, 512
    input_ids = torch.ones(
        bs, seq_len, dtype=torch.long, device=f"cuda:{dist.get_rank()}"
    )
    return mod, input_ids


def get_gpt_model():
    from transformers import GPTNeoModel, AutoConfig

    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-1.3B")
    config.use_cache = False
    mod = GPTNeoModel(config)
    bs, seq_len = 4, 1024
    input_ids = torch.ones(
        bs, seq_len, dtype=torch.long, device=f"cuda:{dist.get_rank()}"
    )
    return mod, input_ids


if __name__ == "__main__":
    dist.init_process_group("nccl", world_size=int(os.environ["WORLD_SIZE"]))

    for kernel_opt in [False, True]:
        for mod, input_ids in [get_gpt_model()]:
            # Initialize the DeepSpeed-Inference engine
            # https://www.deepspeed.ai/tutorials/inference-tutorial/
            ds_engine = deepspeed.init_inference(
                mod,
                mp_size=dist.get_world_size(),
                dtype=torch.float32,
                checkpoint=None,
                replace_with_kernel_inject=kernel_opt,
            )
            mod = ds_engine.module
            perf_model(mod, input_ids)
            del mod
