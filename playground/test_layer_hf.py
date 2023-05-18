# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import BertLMHeadModel, AutoConfig
import inspect
import torch
import slapo
import time
from solver import Solver

config = AutoConfig.from_pretrained("bert-large-uncased")
model = BertLMHeadModel(config)
print(config)

sch = slapo.create_schedule(model)
input_names = ["hidden_states"]
tot_time = 0
i = 0
subsch = sch[f"bert.encoder.layer.{i}"]
sig = inspect.signature(subsch.mod.forward)
concrete_args = {
    p.name: p.default for p in sig.parameters.values() if p.name not in input_names
}
subsch.trace(
    recursive=False,
    flatten=True,
    tracer="pytorch",
    concrete_args=concrete_args,
)
print(subsch.mod.graph)

st = time.time()
sol = Solver(subsch.mod, p=8)
sol.solve([torch.randn(8, 512, 1024)])
tot_time += time.time() - st
print("Time: ", tot_time)


# from transformers import AutoConfig, GPTNeoModel
# config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-1.3B")
# model = GPTNeoModel(config)
# print(config)

# sch = slapo.create_schedule(model)
# input_names = ["hidden_states"]
# tot_time = 0
# for i in range(config.num_layers):
#     subsch = sch[f"h.{i}.mlp"]
#     sig = inspect.signature(subsch.mod.forward)
#     concrete_args = {
#         p.name: p.default for p in sig.parameters.values() if p.name not in input_names
#     }
#     subsch.trace(
#         recursive=False,
#         flatten=True,
#         tracer="pytorch",
#         concrete_args=concrete_args,
#     )
#     print(subsch.mod.graph)

#     st = time.time()
#     sol = Solver(subsch.mod, p=8)
#     sol.solve([torch.randn(8, 1024, 2048)])
#     tot_time += time.time() - st
# print("Time: ", tot_time)