import torch
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
            out = torch.zeros(
                x.shape[:-1] + (self.weight.shape[1],),
                device=x.device,
                dtype=torch.float16,
            )
            torch.ops.bt.gemm_bias_gelu(x, self.weight, self.bias, out)
            return out

    sch.replace(GEMMBiasGeLU(sch[name].mod.weight, sch[name].mod.bias), subgraph)
