# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import BertLMHeadModel, AutoConfig
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

config = AutoConfig.from_pretrained("bert-large-uncased")
model = BertLMHeadModel(config)
print(config)

import slapo

sch = slapo.create_schedule(model)
input_names = ["hidden_states"]
subsch = sch["bert.encoder.layer.0.attention"]
sig = inspect.signature(subsch.mod.forward)
concrete_args = {
    p.name: p.default for p in sig.parameters.values() if p.name not in input_names
}
print(subsch.mod)
print(concrete_args)
subsch.trace(
    recursive=False, flatten=True, tracer="pytorch", concrete_args=concrete_args
)
print(subsch.mod.graph)

from slapo.pattern import call_module


def pattern(x):
    x = call_module(r"self\.(query|key|value)", x)
    new_shape = x.size()[:-1] + (16, -1)  # FIXME: hard-coded
    x = x.view(new_shape)
    return x.permute(0, 2, 1, 3)


qkv_subgraphs = subsch.find(pattern)
print(qkv_subgraphs)


class FusedQKV(nn.Linear):
    def __init__(self, hidden_size, num_heads) -> None:
        super().__init__(hidden_size, hidden_size * 3)
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.world_size = 1

    def reshape_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_heads // self.world_size,
            self.head_size,
            3,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3, 4)

    def forward(self, hidden_states):
        qkv = F.linear(hidden_states, self.weight, self.bias)
        reshaped_qkv = self.reshape_for_scores(qkv)
        q, k, v = torch.split(reshaped_qkv, 1, dim=-1)
        q = torch.squeeze(q, -1).contiguous()
        k = torch.squeeze(k, -1).contiguous()
        v = torch.squeeze(v, -1).contiguous()
        return [q, k, v]


fused_qkv = FusedQKV(
    hidden_size=config.hidden_size, num_heads=config.num_attention_heads
)
subsch.replace(fused_qkv, qkv_subgraphs)
print(subsch.mod.graph)

from solver import Solver

sol = Solver(subsch.mod)
sol.solve([torch.randn(8, 512, 1024)])
