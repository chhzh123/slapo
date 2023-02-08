# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import torch
import torch.nn as nn
import torch.distributed as dist

from slapo import init_empty_weights
from slapo.pattern import call_module
from slapo.op.linear import FusedQKV


def trace_attention(sch, config, attn_path="h.N.attn.attention"):
    cnt = 0
    for idx in range(config.num_hidden_layers):
        sub_sch = sch[attn_path.replace("N", str(idx))]
        input_names = ["hidden_states", "attention_mask"]
        sig = inspect.signature(sub_sch.mod.forward)
        concrete_args = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.name not in input_names
        }
        if sub_sch.trace(tracer="pytorch", concrete_args=concrete_args):
            cnt += 1
    return cnt


def fix_attention_mask_shape(sch):
    # Attention mask may needed to be expanded from (B, 1, 1, S)
    # to (B, H, S, S), where H is sharded.
    ops = sch.find_node(
        lambda node: node.op == "call_method" and node.target == "expand"
    )

    def new_expand(tensor, *args):
        # (B, 1, 1, S) -> (B, H, S, S)
        assert len(args) == 4
        out = tensor.expand(args[0], args[1] // sch.world_size, *args[2:])
        return out.contiguous()

    for op in ops:
        sch.replace(new_expand, op[1])


def replace_and_shard_attention(
    sch,
    config,
    attn_path="decoder.layers.N.self_attn",
    delay_init=True,
    disable_flash_attn=False,
):
    from epoi.inject.policy.gpt import InjectHFGPTAttentionPolicy
    from epoi.ops.xformers_attn import GenericSelfAttention, MemoryEfficientAttentionOp

    try:
        # Backward compatibility
        from epoi.ops.flash_attention import FlashSelfAttention, FlashAttentionTritonOp
    except ImportError:
        FlashSelfAttention = None
        FlashAttentionTritonOp = None

    cuda_sm = torch.cuda.get_device_capability("cuda")
    if not disable_flash_attn and FlashSelfAttention is not None and cuda_sm == (8, 0):
        SelfAttentionModule = FlashSelfAttention
        AttentionOp = FlashAttentionTritonOp
        attn_op_name = "triton"
    else:
        SelfAttentionModule = GenericSelfAttention
        AttentionOp = MemoryEfficientAttentionOp
        attn_op_name = "native" if disable_flash_attn else "cutlass"

    class SelfAttention(nn.Module):
        """A wrapper to align the original GPTNeoAttention forward signature."""

        def __init__(self, **kwargs):
            super().__init__()
            self.module = SelfAttentionModule(**kwargs)

        def forward(
            self,
            hidden_states,
            past_key_value=None,
            attention_mask=None,
            layer_head_mask=None,
            output_attentions=False,
            use_cache=False,
        ):
            outputs = self.module(
                hidden_states, attention_mask, past_key_value, use_cache
            )
            # FIXME: The original output is (hidden_states, None) where the None
            # is present_key_value and only used by in inference.
            # OPT output is (hidden_states, self_attn_weights, present_key_value)
            return outputs[0], None, None

    num_layers, num_heads, hidden_size = (
        config.num_hidden_layers,
        config.num_attention_heads,
        config.hidden_size,
    )

    cnt = 0
    for idx in range(num_layers):
        sub_sch = sch[attn_path.replace("N", str(idx))]
        init_config = InjectHFGPTAttentionPolicy.gen_init_config_from_object(
            sub_sch.mod, attn_op_name=attn_op_name
        )
        with init_empty_weights(enable=delay_init):
            new_mod = SelfAttention(**init_config)
        sub_sch.replace(new_mod)
        sub_sch.trace(
            tracer="pytorch",
            leaf_modules=["MemoryEfficientAttentionOp"],
            concrete_args={
                "layer_past": None,
                "use_cache": False,
            },
        )

        def pattern(x: torch.Tensor) -> torch.Tensor:
            x = call_module("query|key|value", x)
            new_x_shape = x.size()[:-1] + (num_heads, hidden_size)
            x = x.view(new_x_shape)
            return x

        subgraphs = sub_sch["module"].find(pattern)
        assert len(subgraphs) == 3
        with init_empty_weights(enable=delay_init):
            new_fused_qkv = FusedQKV(hidden_size, num_heads, sch.world_size)
        sub_sch["module"].replace(new_fused_qkv, subgraphs)
        if sch.world_size > 1:
            sub_sch["module.FusedQKV_0.fused_linear"].shard("weight", axis=0)
            sub_sch["module.FusedQKV_0.fused_linear"].shard("bias", axis=0)
            fix_attention_mask_shape(sub_sch["module"])
            sub_sch["module.FusedQKV_0.fused_linear"].sync(
                mode="bwd_post", sync_op_or_fn="all_reduce"
            )
            sub_sch["module.out_proj"].shard("weight", axis=1)
            sub_sch["module.out_proj"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")
        cnt += 1

    return cnt


def remove_cast(sch, config, attn_path="h.N.attn.attention"):
    """[Untested] Remove .to(torch.float32) in GPT-Neo attention to align
    HF and Megatron GPT-2 behavior.
    """
    cnt = 0
    for idx in range(config.num_hidden_layers):
        sub_sch = sch[attn_path.replace("N", str(idx))]
        ops = sub_sch.find_node(
            lambda node: node.op == "call_method"
            and node.target == "to"
            and len(node.args) == 2
            and node.args[1] == torch.float32
        )

        for op in ops:
            sub_sch.replace(lambda x, *args: x, op[1])
            cnt += 1
    return cnt


def shard_word_embedding(
    sch, head_sch, vocab_size, word_embed_name="decoder.embed_tokens"
):
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
        # Mask the output embedding
        input_mask = (_input[0] < vocab_start_index) | (_input[0] >= vocab_end_index)
        output[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=sch.group)
        return output

    sch[word_embed_name].sync(mode="fwd_post", sync_op_or_fn=fwd_post_hook)

    # Shard output embedding.
    if head_sch is not None:
        head_sch.shard("weight", axis=0)


def replace_and_shard_mlp(
    sch,
    config,
    path="decoder.layers.N",
    fc_names=["fc1", "fc2"],
    delay_init=True,
    disable_fuse_bias_gelu=True,
):
    from epoi.inject.policy.gpt import InjectHFGPTMLPPolicy

    for idx in range(config.num_hidden_layers):
        prefix = path.replace("N", str(idx))
        replaced_new_mlp = False
        if config.activation_function in ["gelu", "gelu_new"]:
            if disable_fuse_bias_gelu:
                sub_sch = sch[prefix]
                with init_empty_weights(enable=delay_init):
                    new_mod = InjectHFGPTMLPPolicy.init_from_object(sub_sch.mod)
                sub_sch.replace(new_mod)
                sub_sch.trace(leaf_modules=["FusedBiasGELU", "FusedBiasNewGELU"])
                replaced_new_mlp = True
            else:

                def bias_gelu_pattern(x, bias):
                    x = x + bias
                    x = call_module("activation_fn", x)
                    return x

                subsch = sch[prefix]
                subsch["fc1"].decompose()
                subsch.trace(flatten=True)

                subgraphs = subsch.find(bias_gelu_pattern)
                assert len(subgraphs) == 1
                assert len(subgraphs[0]) == 2
                subsch.fuse(subgraphs, compiler="TorchScript", name="FusedBiasGeLU")
        if sch.world_size > 1:
            if replaced_new_mlp:
                sub_sch["fc_in"].shard("weight", axis=0)
                sub_sch["act"].shard("bias", axis=0)
                sub_sch["fc_in"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
                sub_sch["fc_out"].shard("weight", axis=1)
                sub_sch["fc_out"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")
            else:
                sch[f"{prefix}.{fc_names[0]}"].shard("weight", axis=0)
                sch[f"{prefix}.{fc_names[0]}"].shard("bias", axis=0)
                sch[f"{prefix}.{fc_names[0]}"].sync(
                    mode="bwd_post", sync_op_or_fn="all_reduce"
                )
                sch[f"{prefix}.{fc_names[1]}"].shard("weight", axis=1)
                sch[f"{prefix}.{fc_names[1]}"].sync(
                    mode="fwd_post", sync_op_or_fn="all_reduce"
                )


def checkpoint(sch, config, path="decoder.layers.N", ckpt_ratio=1.0):
    if ckpt_ratio == 0.0:
        return

    n_ckpt = int(config.num_hidden_layers * ckpt_ratio)
    for idx in range(n_ckpt):
        sch[path.replace("N", str(idx))].checkpoint()
    return n_ckpt


def broadcast_input(sch):
    def broadcast_input(inputs):
        for inp in inputs:
            dist.broadcast(inp, src=0, group=sch.group)
        return inputs

    sch.sync(mode="fwd_pre", sync_op_or_fn=broadcast_input)
