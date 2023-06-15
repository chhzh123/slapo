# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import torch
import torch.nn.functional as F
from torch import nn

import slapo


def perf_model(mod, input_tensor):
    """Measure the performance of a mod with certain resharding schemes"""
    # warmup
    mod.eval()
    mod.to(torch.float16)
    for _ in range(10):
        mod(input_tensor)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    iters = 100
    for _ in range(iters):
        mod(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    print(f"{start_event.elapsed_time(end_event) / iters:.3f} ms")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4096, 11008)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(11008, 4096)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


mod = Model().cuda()
sch = slapo.create_schedule(copy.deepcopy(mod))

sch["linear1"].decompose()
sch.trace(flatten=True)


def pattern(x, bias):
    x = F.gelu(bias + x)
    return x


subgraph = sch.find(pattern)
assert len(subgraph[0]) == 2
sch.fuse(subgraph, compiler="TorchScript", name="BiasGeLU")
slapo_mod, _ = slapo.build(sch, init_weights=False)

inp = torch.randn(2, 2048, 4096, dtype=torch.float16, device="cuda")
perf_model(mod, inp)
perf_model(slapo_mod, inp)
opt_mod = torch.compile(mod, mode="reduce-overhead", backend="inductor")
perf_model(opt_mod, inp)
torch._dynamo.reset()
opt_mod_2 = torch.compile(mod, mode="max-autotune", backend="inductor")
perf_model(opt_mod_2, inp)
