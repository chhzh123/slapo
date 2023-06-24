import torch
from torch import fx
import torch.nn.functional as F
from slapo.pattern import call_module


def fuse_gemm_bias_gelu(sch, name="dense"):
    sch.trace(recursive=False, flatten=True, leaf_modules=["Linear"])

    subgraph = sch.find(lambda x: F.gelu(call_module(name, x)))
    assert len(subgraph[0]) == 2
    torch.ops.load_library("/home/ubuntu/ByteTransformer/torch/build/libbt.so")

    class GEMMBiasGeLU(torch.nn.Module):
        def __init__(self, weight, bias):
            super().__init__()
            self.weight = (
                weight.transpose(1, 0).contiguous().to(torch.float16).to("cuda")
            )
            self.bias = bias.contiguous().to(torch.float16).to("cuda")

        def forward(self, x):
            return torch.ops.bt.gemm_bias_gelu(x, self.weight, self.bias)

    sch.replace(GEMMBiasGeLU(sch[name].mod.weight, sch[name].mod.bias), subgraph)


def fuse_ln_residual(sch, names=["dense", "LayerNorm"], lib="FasterTransformer"):
    dense, ln = names
    assert not isinstance(sch.mod, fx.GraphModule)
    sch[dense].decompose()
    sch.trace(recursive=False, flatten=True)

    def pattern(x, bias, residual):
        x = F.dropout(x + bias)
        x = call_module(ln, x + residual)
        return x

    class BiasLayerNorm(torch.nn.Module):
        def __init__(self, weight, bias, eps=1e-5):
            super().__init__()
            self.weight = weight
            self.bias = bias
            self.eps = eps
            if lib == "FasterTransformer":
                torch.ops.load_library(
                    "/home/ubuntu/FasterTransformer/torch/build/libft.so"
                )
                self.fn = torch.ops.ft.add_bias_residual_layernorm
            else:
                torch.ops.load_library(
                    "/home/ubuntu/ByteTransformer/torch/build/libbt.so"
                )
                self.fn = torch.ops.bt.add_bias_residual_layernorm

        def forward(self, hidden_states, dense_bias, residual):
            return self.fn(hidden_states, residual, dense_bias, self.weight, self.bias)

    subgraph = sch.find(pattern)
    assert len(subgraph[0]) == 4
    sch.replace(
        BiasLayerNorm(sch[ln].mod.weight, sch[ln].mod.bias, sch[ln].mod.eps), subgraph
    )
