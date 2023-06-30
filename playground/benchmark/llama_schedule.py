import math
import torch
from torch import nn
import torch.nn.functional as F
import slapo
from slapo.pattern import call_module, call_function
from slapo.model_schedule.base import (
    shard_attention,
    trace_attention,
    shard_word_embedding,
)
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# Related issue in FasterTransformer:
# https://github.com/NVIDIA/FasterTransformer/issues/506
# https://github.com/void-main/FasterTransformer/blob/f3bd8e681f8bf70141012c5fe621416a36edca46/src/fastertransformer/models/llama/LlamaDecoder.cc


class FusedQKV(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        world_size: int,
        bias: bool = True,
        reshape: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.world_size = world_size
        self.num_heads = num_heads // self.world_size
        self.head_dim = embed_dim // num_heads
        self.fused_linear = nn.Linear(
            embed_dim, embed_dim * 3 // self.world_size, bias=bias
        )
        self.reshape = reshape

    def forward(self, hidden_states: torch.Tensor):
        bsz, tgt_len, _ = hidden_states.size()
        qkv = self.fused_linear(hidden_states)
        if not self.reshape:
            reshaped_qkv = qkv.view(bsz, tgt_len, 3, -1)
            q, k, v = reshaped_qkv.unbind(dim=2)
            v = v.view(bsz, tgt_len, self.num_heads, -1).transpose(1, 2)
        else:
            reshaped_qkv = (
                qkv.view(bsz, tgt_len, 3 * self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )
            reshaped_qkv = reshaped_qkv.view(
                bsz, 3, self.num_heads, tgt_len, self.head_dim
            )
            q, k, v = reshaped_qkv.unbind(dim=1)

        query_states = q.contiguous()
        key_states = k.contiguous()
        value_states = v.contiguous()

        return query_states, key_states, value_states


def fuse_qkv(sch, config):
    def qkv_pattern(hidden_states):
        new_states = call_module(r"[qkv]_proj", hidden_states)
        new_states.view(
            1,  # bs (fake number)
            1024,  # seq_len (fake number)
            config.num_attention_heads // sch.world_size,
            config.hidden_size // config.num_attention_heads,
        ).transpose(1, 2)
        return new_states

    subgraphs = sch.find(qkv_pattern)
    assert len(subgraphs) == 3
    fused_qkv = FusedQKV(
        config.hidden_size,
        config.num_attention_heads,
        sch.world_size,
        bias=False,
        reshape=True,
    )
    fused_qkv = torch.compile(fused_qkv, backend="inductor")
    # out = fused_qkv(torch.randn(bs, seq_len, config.hidden_size).cuda())
    sch.replace(fused_qkv, subgraphs)


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

    # from slapo.op import FlashAttentionOp

    # class EfficientAttention(torch.nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.op = FlashAttentionOp(attn_op_name="auto", apply_causal_mask=False)

    #     # Be careful of the order of the arguments
    #     def forward(self, key_layer, query_layer, value_layer):
    #         return self.op(query_layer, key_layer, value_layer, None, 0)

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


def replace_ds_rotary_pos_emb(sch, config):
    def rotary_pattern(query_states, key_states, value_states):
        kv_seq_len = key_states.size()[-2]
        cos, sin = call_module("rotary_emb", value_states, seq_len=kv_seq_len)
        return call_function(
            "apply_rotary_pos_emb", query_states, key_states, cos, sin, None
        )

    subgraphs = sch.find(rotary_pattern)
    assert len(subgraphs) > 0

    torch.ops.load_library("deepspeed/build/libds.so")

    class DSApplyRotary(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_num = config.num_attention_heads // sch.world_size
            self.max_position_embeddings = config.max_position_embeddings

        def forward(self, key_states, value_states, query_states, position_ids):
            # getitem_9, getitem_10, getitem_8
            bsz, q_len, hs = query_states.size()
            query_states, key_states = torch.ops.ds.apply_rotary_pos_emb(
                query_states,
                key_states,
                hs,  # rotary_dim
                0,  # offset
                self.head_num,  # num_heads
                True,  # rotate_half
                self.max_position_embeddings,  # max_token_length
            )
            query_states = query_states.view(bsz, q_len, self.head_num, -1).transpose(
                1, 2
            )
            key_states = key_states.view(bsz, q_len, self.head_num, -1).transpose(1, 2)
            return query_states, key_states

    sch.replace(DSApplyRotary(), target_ops=subgraphs)


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


def cudagraphify(sch):
    from torch._inductor.compile_fx import cudagraphify_impl

    class CudaGraphModule(torch.nn.Module):
        def __init__(self, mod, inputs):
            super().__init__()
            mod.to(torch.float16).cuda(sch.rank).eval()
            # Need to preserve the inputs
            self.inputs = inputs
            self.repr = str(mod)
            self.mod = cudagraphify_impl(
                model=lambda args: mod(*args), inputs=self.inputs
            )

        def forward(self, *args, **kwargs):
            # should input a list
            new_args = list(args + tuple(kwargs.values()))
            return self.mod(new_args[:3])

        def extra_repr(self) -> str:
            return self.repr

    device = f"cuda:{sch.rank}"
    hidden_states = torch.randn((1, 1024, 4096), device=device, dtype=torch.float16)
    attention_mask = torch.ones(
        1, 1, 1024, 1024, dtype=torch.float16, requires_grad=False, device=device
    )
    position_ids = torch.ones(
        1, 1024, dtype=torch.long, device=device, requires_grad=False
    )
    sch.replace(CudaGraphModule(sch.mod, [hidden_states, attention_mask, position_ids]))


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
        fuse_qkv(sch[f"layers.{i}.self_attn"], config)
        fix_reshape(sch[f"layers.{i}.self_attn"])
        replace_sdp(sch[f"layers.{i}.self_attn"], config)
        # replace_ds_rotary_pos_emb(sch[f"layers.{i}.self_attn"], config)
        replace_rotary_pos_emb(sch[f"layers.{i}.self_attn"])
        replace_silu(sch[f"layers.{i}.mlp"])
    mod, _ = slapo.build(sch, init_weights=mod._init_weights)
    mod.to(torch.float16).cuda(sch.rank).eval()
    for i in range(config.num_hidden_layers):
        cudagraphify(sch[f"layers.{i}"])
        if sch.rank == 0:
            print("Done layer", i)
    return mod
