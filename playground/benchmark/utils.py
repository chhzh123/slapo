# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.distributed as dist
from transformers import AutoModel, AutoConfig
import slapo

MODEL_SETTINGS = {
    "bert": ("bert-large-uncased", 512),
    "gpt": ("EleutherAI/gpt-neo-1.3B", 1024),
    "llama": ("decapoda-research/llama-7b-hf", 2048),
    "vicuna": ("lmsys/vicuna-13b-delta-v1.1", 2048),
    "opt": ("facebook/opt-13b", 2048),
}


def get_model(name, meta=False):
    model_name, seq_len = MODEL_SETTINGS[name]
    if dist.get_rank() == 0:
        print(f"Loading {model_name} with seq_len {seq_len}")
    config = AutoConfig.from_pretrained(model_name)
    if model_name != "bert":
        config.use_cache = False
    with slapo.init_empty_weights(enable=meta):
        mod = AutoModel.from_pretrained(model_name)
    mod.eval()
    mod.to(torch.float16)
    return mod, config, seq_len


def perf_model(mod, input_tensor):
    """Measure the performance of a mod with certain resharding schemes"""
    # warmup
    torch.cuda.empty_cache()
    mod.to(torch.float16)
    mod.eval()
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
