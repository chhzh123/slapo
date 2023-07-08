# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from slapo.pattern import call_module


def shard_word_embedding(sch, vocab_size, word_embed_name="embeddings.word_embeddings"):
    if sch.world_size == 1:
        return

    # Embedding
    sch[word_embed_name].shard("weight", axis=0)
    # Build the mask
    vocab_start_index = sch.rank * vocab_size // sch.world_size
    vocab_end_index = (sch.rank + 1) * vocab_size // sch.world_size

    def fwd_pre_hook(_module, _input):
        # Mask the input
        input_mask = (_input[0] < vocab_start_index) | (_input[0] >= vocab_end_index)
        masked_input = _input[0].clone() - vocab_start_index
        masked_input[input_mask] = 0
        return masked_input

    sch[word_embed_name].sync(mode="fwd_pre", sync_op_or_fn=fwd_pre_hook)

    def fwd_post_hook(_module, _input, output):
        # Mask the output embedding. Note that the input is already masked.
        output[_input[0] == 0, :] = 0.0
        # Reduce across all the model parallel GPUs
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=sch.group)
        return output

    sch[word_embed_name].sync(mode="fwd_post", sync_op_or_fn=fwd_post_hook)


def trace_attention(
    sch, config, input_names=["hidden_states"], leaf_modules=[], leaf_functions=[]
):
    sig = inspect.signature(sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    sch.trace(
        recursive=False,
        flatten=True,
        tracer="huggingface",
        leaf_modules=leaf_modules,
        leaf_functions=leaf_functions,
        concrete_args=concrete_args,
        config=config,
    )


class FusedQKV(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        world_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.world_size = world_size
        self.head_dim = embed_dim // num_heads
        self.fused_linear = nn.Linear(
            embed_dim, embed_dim * 3 // self.world_size, bias=bias
        )

    def forward(self, hidden_states: torch.Tensor):
        bsz, tgt_len, _ = hidden_states.size()
        qkv = self.fused_linear(hidden_states)
        reshaped_qkv = (
            qkv.view(bsz, tgt_len, 3 * self.num_heads // self.world_size, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        reshaped_qkv = reshaped_qkv.view(
            bsz, 3, self.num_heads // self.world_size, tgt_len, self.head_dim
        )
        q, k, v = reshaped_qkv.unbind(dim=1)

        query_states = q.contiguous()
        key_states = k.contiguous()
        value_states = v.contiguous()

        return query_states, key_states, value_states


def fuse_qkv(sch, config, name_pattern=r"self.(query|key|value)"):
    def qkv_pattern(x):
        x = call_module(name_pattern, x)
        new_x_shape = x.size()[:-1] + (
            config.num_attention_heads,
            int(config.hidden_size / config.num_attention_heads),
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    subgraphs = sch.find(qkv_pattern)
    assert len(subgraphs) == 3
    fused_qkv = FusedQKV(
        config.hidden_size, config.num_attention_heads, sch.world_size, bias=True
    )
    # fused_qkv = torch.compile(fused_qkv.cuda(), mode="reduce-overhead", backend="inductor")
    # for _ in range(10):
    #     fused_qkv(torch.randn(1, 512, config.hidden_size, device="cuda"))
    sch.replace(fused_qkv, subgraphs)


def shard_attention(
    sch,
    names=["q_proj", "k_proj", "v_proj", "out_proj"],
    attrs=["num_heads", "all_head_size"],
    backward=False,
):
    if len(names) == 4:
        q, k, v, out = names
        sch[q].shard("weight", axis=0)
        sch[k].shard("weight", axis=0)
        sch[v].shard("weight", axis=0)
        if sch[q].mod.bias is not None:
            sch[q].shard("bias", axis=0)
            sch[k].shard("bias", axis=0)
            sch[v].shard("bias", axis=0)
        if backward:
            sch.sync(mode="bwd_post", sync_op_or_fn="all_reduce")
    elif len(names) == 2:  # q,k,v have been fused
        qkv, out = names
        sch[qkv].shard("weight", axis=0)
        if sch[qkv].mod.bias is not None:
            sch[qkv].shard("bias", axis=0)
        if backward:
            sch[qkv].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
    else:
        raise ValueError(f"Invalid names {names}")
    sch[out].shard("weight", axis=1)
    sch[out].sync("fwd_post", sync_op_or_fn="all_reduce")
    # Update the number of heads
    for attr in attrs:
        if "." not in attr:
            path, attr = "", attr
            subsch = sch
        else:
            path, attr = attr.rsplit(".", 1)
            subsch = sch[path]
        if hasattr(subsch.mod, attr):
            setattr(subsch.mod, attr, getattr(subsch.mod, attr) // sch.world_size)
        else:
            raise ValueError(f"Invalid attribute {attr}")


def shard_mlp(sch, names=["c_fc", "c_proj"], backward=False):
    l1, l2 = names
    sch[l1].shard("weight", axis=0)
    if sch[l1].mod.bias is not None:
        sch[l1].shard("bias", axis=0)
    sch[l2].shard("weight", axis=1)
    sch[l2].sync("fwd_post", sync_op_or_fn="all_reduce")
    if backward:
        sch[l1].sync("bwd_post", sync_op_or_fn="all_reduce")


def find_gpt_attention(sch):
    subgraph = []
    flag = False
    for node in sch.mod.graph.nodes:
        # Start capturing subgraph
        if node.op == "call_method" and node.target == "to":
            flag = True
        if flag:
            subgraph.append(("", node))
        # Stop capturing subgraph
        if (
            node.op == "call_function"
            and node.target == torch.matmul
            and node.args[0].op == "call_module"
        ):
            flag = False
    return [subgraph]


def replace_sdp(sch, config, mask=False):
    if not mask:

        def scaled_dot_product(query_layer, key_layer, value_layer):
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(
                config.hidden_size // config.num_attention_heads
            )
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = F.dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, value_layer)
            return context_layer

        subgraphs = sch.find(scaled_dot_product)
        assert len(subgraphs[0]) == 6

        class EfficientAttention(torch.nn.Module):
            # Be careful of the order of the arguments
            def forward(self, key_layer, query_layer, value_layer):
                return F.scaled_dot_product_attention(
                    query_layer, key_layer, value_layer
                )

    else:
        subgraphs = find_gpt_attention(sch)
        assert len(subgraphs) > 0

        class EfficientAttention(torch.nn.Module):
            # TODO: Add attention mask
            def forward(self, query_layer, key_layer, value_layer):
                return F.scaled_dot_product_attention(
                    query_layer, key_layer, value_layer
                )

    sch.replace(EfficientAttention(), subgraphs)


def fuse_bias_gelu(sch, name="dense", act="act"):
    sch[name].decompose()
    sch.trace(flatten=True, leaf_modules=[act])

    def pattern(x, bias):
        x = call_module(act, bias + x)
        return x

    subgraph = sch.find(pattern)
    assert len(subgraph[0]) == 2
    sch.fuse(subgraph, compiler="TorchInductor", name="FusedGeLU")


def fuse_ln_residual(sch, names=["dense", "LayerNorm"]):
    dense, ln = names
    sch[dense].decompose()
    sch.trace(flatten=True)

    def pattern(bias, x, residual):
        x = F.dropout(bias + x)
        x = call_module(ln, x + residual)
        return x

    subgraph = sch.find(pattern)
    assert len(subgraph[0]) == 4
    sch.fuse(subgraph, compiler="TorchInductor", name="LNResidual")


def generate_pipeline_stages(
    sch,
    pipeline_cuts,
    prefix="",
    input_names=["input_ids", "attention_mask", "token_type_ids"],
):
    sig = inspect.signature(sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    _prefix = f"{prefix}." if prefix else ""
    sch.trace_until(
        f"{_prefix}encoder", tracer="huggingface", concrete_args=concrete_args
    )
    for cut in pipeline_cuts:
        sch[f"{_prefix}encoder.layer.{cut}"].cut_pipeline_stage()

    return sch


def uniform_checkpoint(sch, num_layers, path="encoder.layer.N", ckpt_ratio=1.0):
    if ckpt_ratio == 0.0:
        return 0

    n_ckpt = int(num_layers * ckpt_ratio)
    for idx in range(n_ckpt):
        sch[path.replace("N", str(idx))].checkpoint()
    return n_ckpt


def broadcast_input(sch):
    def broadcast(inputs):
        for inp in inputs:
            dist.broadcast(inp, src=0, group=sch.group)
        return inputs

    sch.sync(mode="fwd_pre", sync_op_or_fn=broadcast)
