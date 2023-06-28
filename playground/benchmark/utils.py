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
        print(f"Loading {model_name} with seq_len {seq_len} (meta={meta})")
    config = AutoConfig.from_pretrained(model_name)
    if name != "bert":
        config.use_cache = False
    if name == "llama":
        config.pad_token_id = 0
    with slapo.init_empty_weights(enable=meta):
        mod = AutoModel.from_pretrained(model_name, config=config)
    mod.eval()
    mod.to(torch.float16)
    return mod, config, seq_len


def perf_model(mod, input_tensor, use_cuda_graph=False, iters=100, nsys=False):
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
            input_tensor, dtype=input_tensor.dtype, device="cuda", requires_grad=False
        )
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(15):
                    mod(fake_inputs)
        torch.cuda.current_stream().wait_stream(s)

        # capture
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            with torch.no_grad():
                mod(fake_inputs)

        input_tensor.copy_(fake_inputs)
        if not nsys:
            start_event.record()
            for i in range(iters):
                g.replay()
            end_event.record()
            torch.cuda.synchronize()
        else:
            torch.cuda.cudart().cudaProfilerStart()
            start_event.record()
            for i in range(iters):
                torch.cuda.nvtx.range_push(f"perf_model_{i}")
                g.replay()
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            end_event.record()
            torch.cuda.synchronize()
    else:
        for _ in range(15):
            mod(input_tensor)

        torch.cuda.synchronize()
        # https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964/10
        # https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59?u=ptrblck
        # https://gist.github.com/mcarilli/376821aa1a7182dfcf59928a7cde3223
        if not nsys:
            start_event.record()
            for i in range(iters):
                mod(input_tensor)
            end_event.record()
            torch.cuda.synchronize()
        else:
            torch.cuda.cudart().cudaProfilerStart()
            start_event.record()
            for i in range(iters):
                torch.cuda.nvtx.range_push(f"perf_model_{i}")
                mod(input_tensor)
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
            end_event.record()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()

    if dist.get_rank() == 0:
        print(
            f"# GPUs: {dist.get_world_size()} Time: {start_event.elapsed_time(end_event) / iters:.3f} ms"
        )
