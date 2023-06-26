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
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# Related issue in FasterTransformer:
# https://github.com/NVIDIA/FasterTransformer/issues/506
# https://github.com/void-main/FasterTransformer/blob/f3bd8e681f8bf70141012c5fe621416a36edca46/src/fastertransformer/models/llama/LlamaDecoder.cc


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

    class EfficientAttention(torch.nn.Module):
        # Be careful of the order of the arguments
        def forward(self, key_layer, query_layer, value_layer):
            return F.scaled_dot_product_attention(query_layer, key_layer, value_layer)

    subgraphs = sch.find(pattern)
    assert len(subgraphs) > 0
    sch.replace(EfficientAttention(), subgraphs)


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


def replace_rotary_pos_emb(sch, name="apply_rotary_pos_emb"):
    subgraph = sch.find_node(
        lambda node: node.op == "call_function" and name in node.target.__name__
    )

    class JitApplyRotary(nn.Module):
        def __init__(self):
            super().__init__()
            # self.apply_rotary_emb = torch.jit.script(apply_rotary_pos_emb)
            self.apply_rotary_emb = torch.compile(apply_rotary_pos_emb)

        def forward(self, query_states, key_states, cos, sin, position_ids):
            return self.apply_rotary_emb(
                query_states, key_states, cos, sin, position_ids
            )

    assert len(subgraph) == 1
    sch.replace(JitApplyRotary(), target_ops=[subgraph])


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
            leaf_functions=[apply_rotary_pos_emb],
        )
        fix_reshape(sch[f"layers.{i}.self_attn"])
        replace_sdp(sch[f"layers.{i}.self_attn"], config)
        replace_rotary_pos_emb(sch[f"layers.{i}.self_attn"])
        print(sch[f"layers.{i}.self_attn"].mod)
        replace_silu(sch[f"layers.{i}.mlp"])
    return sch
