import torch
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp
from typing import NamedTuple, Tuple

class ShardSpec(NamedTuple):
    inputs : Tuple
    outputs : Tuple


class Solver():

    def __init__(self, gm) -> None:
        self.gm = gm
        self.named_modules = dict(self.gm.named_modules())
        self.shard_spec = {} # {node_name: shard_spec}

    def inference_shape(self, inputs):
        sp = ShapeProp(self.gm)
        sp.propagate(*inputs)
        for node in self.gm.graph.nodes:
            if "tensor_meta" in node.meta:
                if isinstance(node.meta["tensor_meta"], list):
                    lst = node.meta["tensor_meta"]
                else:
                    lst = [node.meta["tensor_meta"]]
                # for data in lst:
                #     print(node.name, data)

    def solve(self, inputs):
        self.inference_shape(inputs)
        for node in self.gm.graph.nodes:
            self.shard_spec[node.name] = None
            node.meta["shard_spec"] = ShardSpec()
            if node.op == "placeholder": # input
                print("placeholder", node.name, node)
            elif node.op == "call_module":
                mod = self.named_modules[node.target]
                if isinstance(mod, nn.Linear):
                    print("cm", node.name, node)
                    print(node.args[0].meta["tensor_meta"].shape)
                    print(mod.weight.shape)
            elif node.op == "call_function":
                if node.target == torch.matmul:
                    print("cf", node.name, node)
                    print(node.args[0].meta["tensor_meta"].shape)
                    print(node.args[1].meta["tensor_meta"].shape)
