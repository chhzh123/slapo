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


def fuse_ln_residual(sch, names=["dense", "LayerNorm"]):
    dense, ln = names
    if not isinstance(sch.mod, fx.GraphModule):
        sch.trace(recursive=False, flatten=True, leaf_modules=["Linear"])
        weight = sch[ln].mod.weight
        bias = sch[ln].mod.bias
    else:
        weight = getattr(sch["output"].mod, "LayerNorm").weight
        bias = getattr(sch["output"].mod, "LayerNorm").bias
    torch.ops.load_library("/home/ubuntu/ByteTransformer/torch/build/libbt.so")

    def pattern(hidden_states, residual):
        x = F.dropout(hidden_states)  # bias has been added in the previous dense layer
        x = call_module(ln, x + residual)
        return x

    class BiasLayerNorm(torch.nn.Module):
        def __init__(self, weight, bias):
            super().__init__()
            self.weight = weight
            self.bias = bias

        def forward(self, hidden_states, residual):
            return torch.ops.bt.add_bias_layernorm(
                hidden_states, residual, self.weight, self.bias
            )

    subgraph = sch.find(pattern)
    assert len(subgraph[0]) == 3
    sch.replace(BiasLayerNorm(weight, bias), subgraph)
