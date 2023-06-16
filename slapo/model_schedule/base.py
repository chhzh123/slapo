# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import inspect

import torch
import torch.nn.functional as F


def trace_attention(sch, config, input_names=["hidden_states"]):
    sig = inspect.signature(sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    sch.trace(
        recursive=False,
        flatten=True,
        tracer="huggingface",
        concrete_args=concrete_args,
        config=config,
    )


def shard_attention(
    sch,
    names=["q_proj", "k_proj", "v_proj", "out_proj"],
    attrs=["num_heads", "all_head_size"],
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
    elif len(names) == 2:  # q,k,v have been fused
        qkv, out = names
        sch[qkv].shard("weight", axis=0)
        if sch[qkv].mod.bias is not None:
            sch[qkv].shard("bias", axis=0)
    else:
        raise ValueError(f"Invalid names {names}")
    sch[out].shard("weight", axis=1)
    sch[out].sync("fwd_post", sync_op_or_fn="all_reduce")
    # Update the number of heads
    for attr in attrs:
        path, attr = attr.rsplit(".", 1)
        subsch = sch[path]
        if hasattr(subsch.mod, attr):
            setattr(subsch.mod, attr, getattr(subsch.mod, attr) // sch.world_size)
        else:
            raise ValueError(f"Invalid attribute {attr}")


def shard_mlp(sch, names=["c_fc", "c_proj"]):
    l1, l2 = names
    sch[l1].shard("weight", axis=0)
    if sch[l1].mod.bias is not None:
        sch[l1].shard("bias", axis=0)
    sch[l2].shard("weight", axis=1)
    sch[l2].sync("fwd_post", sync_op_or_fn="all_reduce")


def replace_sdp(sch, config):
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
            return F.scaled_dot_product_attention(query_layer, key_layer, value_layer)

    sch.replace(EfficientAttention(), subgraphs)


def fuse_bias_gelu(sch, name="dense"):
    sch[name].decompose()
    sch.trace(flatten=True)

    def pattern(x, bias):
        x = F.gelu(bias + x)
        return x

    subgraph = sch.find(pattern)
    assert len(subgraph[0]) == 2
    sch.fuse(subgraph, compiler="TorchScript", name="BiasGeLU")
