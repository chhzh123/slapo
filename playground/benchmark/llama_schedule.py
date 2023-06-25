import math
import torch
from torch import fx
from torch import nn
import torch.nn.functional as F
import slapo
from slapo.model_schedule.base import (
    shard_attention,
    trace_attention,
    shard_word_embedding,
)
from slapo.pattern import call_module

# from transformers.models.llama.modeling_llama import LlamaPreTrainedModel

# Related issue in FasterTransformer:
# https://github.com/NVIDIA/FasterTransformer/issues/506
# https://github.com/void-main/FasterTransformer/blob/f3bd8e681f8bf70141012c5fe621416a36edca46/src/fastertransformer/models/llama/LlamaDecoder.cc

# Pre layernorm
# layernrom(hidden_states + mha_bias + residual)


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
            return self.fn(
                hidden_states, residual, dense_bias, self.weight, self.bias, self.eps
            )

    subgraph = sch.find(pattern)
    assert len(subgraph[0]) == 4
    sch.replace(
        BiasLayerNorm(sch[ln].mod.weight, sch[ln].mod.bias, sch[ln].mod.eps), subgraph
    )


def shard_mlp(sch, names):
    l0, l1, l2 = names
    sch[l0].shard("weight", axis=0)
    sch[l1].shard("weight", axis=0)
    if sch[l0].mod.bias is not None:
        sch[l0].shard("bias", axis=0)
    if sch[l1].mod.bias is not None:
        sch[l1].shard("bias", axis=0)
    sch[l2].shard("weight", axis=1)
    sch[l2].sync("fwd_post", sync_op_or_fn="all_reduce")


def replace_sdp(sch, config):
    # Replace efficient kernels
    def pattern(query_states, key_states, value_states):
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(config.hidden_size // config.num_attention_heads)
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output

    subgraphs = sch.find(pattern)
    assert len(subgraphs) > 0
    sch.replace(F.scaled_dot_product_attention, subgraphs)


def fix_reshape(sch):
    reshape_op = sch.find_node(
        lambda node: node.op == "call_method" and node.target == "reshape"
    )
    assert len(reshape_op) == 1

    def new_reshape(tensor, *args):
        return tensor.reshape(args[0], args[1], args[2] // sch.world_size)

    sch.replace(new_reshape, reshape_op[0])


def replace_layernorm(sch, names=["input_layernorm", "post_attention_layernorm"]):
    torch.ops.load_library("/home/ubuntu/FasterTransformer/torch/build/libft.so")

    class FTRmsNorm(torch.nn.Module):
        def __init__(self, weight, eps=1e-5):
            super().__init__()
            self.weight = weight
            self.eps = eps

        def forward(self, hidden_states):
            return torch.ops.ft.rms_norm(
                hidden_states,
                self.weight,
                self.eps,
            )

    for name in names:
        sch[name].replace(
            FTRmsNorm(sch[name].mod.weight, sch[name].mod.variance_epsilon)
        )


def replace_silu(sch, name="act_fn"):
    # Since there are no bias for MLP in LLaMA, no need to fuse bias
    class FTSiLU(torch.nn.Module):
        def forward(self, hidden_states):
            return torch.ops.ft.silu(hidden_states)

    sch[name].replace(FTSiLU())


def schedule_llama(mod, config):
    sch = slapo.create_schedule(mod)
    # if sch.world_size > 1:
    #     shard_word_embedding(sch, config.vocab_size, "embed_tokens")
    replace_layernorm(sch, names=["norm"])
    for i in range(config.num_hidden_layers):
        replace_layernorm(sch[f"layers.{i}"])
        if sch.world_size > 1:
            shard_attention(
                sch[f"layers.{i}.self_attn"],
                names=["q_proj", "k_proj", "v_proj", "o_proj"],
                attrs=["num_heads"],
            )
            shard_mlp(
                sch[f"layers.{i}.mlp"], names=["gate_proj", "up_proj", "down_proj"]
            )
        trace_attention(
            sch[f"layers.{i}.self_attn"],
            config,
            input_names=["hidden_states", "position_ids"],
            leaf_modules=["LlamaRotaryEmbedding"],
        )
        fix_reshape(sch[f"layers.{i}.self_attn"])
        replace_sdp(sch[f"layers.{i}.self_attn"], config)
        replace_silu(sch[f"layers.{i}.mlp"])
        # fuse_ln_residual(
        #     sch[f"encoder.layer.{i}.attention.output"],
        #     names=["dense", "LayerNorm"],
        # )
        # fuse_ln_residual(sch[f"encoder.layer.{i}.output"], names=["dense", "LayerNorm"])
    # if sch.world_size == 1:
    #     mod = torch.compile(mod, backend="inductor")
    return sch
