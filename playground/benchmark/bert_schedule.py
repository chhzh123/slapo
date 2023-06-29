import torch
from torch import fx
import torch.nn.functional as F
import slapo
from slapo.model_schedule.base import (
    shard_attention,
    shard_mlp,
    trace_attention,
    replace_sdp,
    fuse_ln_residual,
    shard_word_embedding,
)
from slapo.pattern import call_module


def fuse_gemm_gelu(sch, name="dense"):
    sch.trace(recursive=False, flatten=True, leaf_modules=["Linear"])

    subgraph = sch.find(lambda x: F.gelu(call_module(name, x)))
    assert len(subgraph[0]) == 2
    torch.ops.load_library("/home/ubuntu/ByteTransformer/torch/build/libbt.so")

    class GEMMBiasGeLU(torch.nn.Module):
        def __init__(self, weight, bias):
            super().__init__()
            device = f"cuda:{sch.rank}"
            self.weight = (
                weight.transpose(1, 0).to(torch.float16).contiguous().to(device)
            )
            self.bias = bias.to(torch.float16).contiguous().to(device)

        def forward(self, x):
            return torch.ops.bt.gemm_bias_gelu(x, self.weight, self.bias)

    sch.replace(GEMMBiasGeLU(sch[name].mod.weight, sch[name].mod.bias), subgraph)


def fuse_ln_residual(sch, names=["dense", "LayerNorm"], lib="FasterTransformer"):
    dense, ln = names
    assert not isinstance(sch.mod, fx.GraphModule)
    if sch.world_size == 1:
        sch[dense].decompose()
    sch.trace(flatten=True)

    def pattern(x, bias, residual):
        x = F.dropout(x + bias)
        x = call_module(ln, x + residual)
        return x

    class BiasLayerNorm(torch.nn.Module):
        def __init__(self, weight, bias, eps=1e-5):
            super().__init__()
            device = f"cuda:{sch.rank}"
            self.weight = weight.to(torch.float16).contiguous().to(device)
            self.bias = bias.to(torch.float16).contiguous().to(device)
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
            return self.fn(
                hidden_states, residual, dense_bias, self.weight, self.bias, self.eps
            )

    subgraph = sch.find(pattern)
    assert len(subgraph[0]) == 4
    sch.replace(
        BiasLayerNorm(sch[ln].mod.weight, sch[ln].mod.bias, sch[ln].mod.eps), subgraph
    )


def cudagraphify(sch):
    from torch._inductor.compile_fx import cudagraphify_impl

    class CudaGraphModule(torch.nn.Module):
        def __init__(self, mod, inputs):
            super().__init__()
            mod.to(torch.float16).cuda(sch.rank).eval()
            # Need to preserve the inputs
            self.inputs = inputs
            self.mod = cudagraphify_impl(
                model=lambda args: mod(*args), inputs=self.inputs
            )

        def forward(self, *args):
            # should input a list
            return self.mod([args[0]])

    sch.replace(
        CudaGraphModule(
            sch.mod, [torch.randn((1, 512, 1024), device="cuda", dtype=torch.float16)]
        )
    )


def schedule_bert(mod, config):
    sch = slapo.create_schedule(mod)
    # if sch.world_size > 1:
    #     shard_word_embedding(sch, config.vocab_size, "embeddings.word_embeddings")
    for i in range(config.num_hidden_layers):
        if sch.world_size > 1:
            shard_attention(
                sch[f"encoder.layer.{i}.attention"],
                names=["self.query", "self.key", "self.value", "output.dense"],
                attrs=["self.num_attention_heads", "self.all_head_size"],
            )
            shard_mlp(
                sch[f"encoder.layer.{i}"], names=["intermediate.dense", "output.dense"]
            )
        # Use this for ByteTransformer
        fuse_gemm_gelu(sch[f"encoder.layer.{i}.intermediate"], "dense")
        trace_attention(sch[f"encoder.layer.{i}.attention.self"], config)
        # fuse_qkv(sch[f"encoder.layer.{i}.attention"], config=config)
        replace_sdp(sch[f"encoder.layer.{i}.attention.self"], config)
        fuse_ln_residual(
            sch[f"encoder.layer.{i}.attention.output"],
            names=["dense", "LayerNorm"],
        )
        fuse_ln_residual(sch[f"encoder.layer.{i}.output"], names=["dense", "LayerNorm"])
        cudagraphify(sch[f"encoder.layer.{i}"])
    # if sch.world_size == 1:
    #     mod = torch.compile(mod, backend="inductor")
    return sch
