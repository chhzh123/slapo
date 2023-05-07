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
subsch = sch["bert.encoder.layer.0"]
sig = inspect.signature(subsch.mod.forward)
concrete_args = {
    p.name: p.default for p in sig.parameters.values() if p.name not in input_names
}
subsch.trace(
    recursive=False,
    flatten=True,
    leaf_modules=["BertAttention"],
    tracer="pytorch",
    concrete_args=concrete_args,
)
print(subsch.mod.graph)

from solver import Solver

sol = Solver(subsch.mod, p=8)
sol.solve([torch.randn(8, 512, 1024)])
