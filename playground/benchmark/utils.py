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
    if name != "bert":
        config.use_cache = False
    with slapo.init_empty_weights(enable=meta):
        mod = AutoModel.from_pretrained(model_name, config=config)
    mod.eval()
    mod.to(torch.float16)
    return mod, config, seq_len


def perf_model(mod, input_tensor, use_cuda_graph=False, iters=100):
    """Measure the performance of a mod with certain resharding schemes"""
    # warmup
    torch.cuda.empty_cache()
    mod.to(torch.float16)
    mod.eval()
    mod.to("cuda")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    if use_cuda_graph:
        s = torch.cuda.Stream()
        fake_inputs = torch.ones_like(
            input_tensor, dtype=input_tensor.dtype, device="cuda"
        )
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(15):
                mod(fake_inputs)
        torch.cuda.current_stream().wait_stream(s)

        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            mod(fake_inputs)

        input_tensor.copy_(fake_inputs)
        start_event.record()
        for _ in range(iters):
            g.replay()
        end_event.record()
        torch.cuda.synchronize()
    else:
        for _ in range(10):
            mod(input_tensor)

        start_event.record()
        for _ in range(iters):
            mod(input_tensor)
        end_event.record()
        torch.cuda.synchronize()

    if dist.get_rank() == 0:
        print(
            f"# GPUs: {dist.get_world_size()} Time: {start_event.elapsed_time(end_event) / iters:.3f} ms"
        )
