# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace T5 with model schedule."""
# pylint: disable=unused-argument, import-error

import torch
from torch import nn
from torch.nn import functional as F

from ..schedule import create_schedule
from ..logger import get_logger
from .registry import register_schedule
from .base import (
    shard_attention,
    shard_mlp,
    shard_word_embedding,
    broadcast_input,
    uniform_checkpoint,
    trace_attention,
    generate_pipeline_stages,
)


@register_schedule()
def _apply_schedule(
    model,
    **sch_config,
):
    model_config = sch_config.get("model_config", None)
    if model_config is None:
        raise ValueError(
            "Model config is not specified in sch_config. Please provide `model_config` in the kwarg."
        )
    try:
        model_name = model_config._name_or_path
    except Exception:
        model_name = model_config.get("_name_or_path", None)
    logger = get_logger(f"{model_name}")

    # Change data type.
    fp16 = sch_config.get("fp16", False)
    bf16 = sch_config.get("bf16", False)
    if fp16 and bf16:
        raise ValueError("Cannot use both fp16 and bf16")
    if fp16:
        logger.info("Change model dtype to fp16", ranks=0)
        model.half()
    elif bf16:
        logger.info("Change model dtype to bf16", ranks=0)
        model.bfloat16()
    else:
        logger.info("Use fp32 as default model dtype", ranks=0)

    group = sch_config.get("group", None)
    orig_sch = create_schedule(model, group=group)
    logger.info(
        "Scheduling %s with TP=%d, config: %s",
        model_name,
        orig_sch.world_size,
        sch_config,
        ranks=0,
    )
    prefix = sch_config.get("prefix", "")
    sch = orig_sch[prefix]

    # Shard parameters if MP group > 1.
    if sch.world_size > 1:
        shard_word_embedding(
            sch,
            model_config.vocab_size,
            word_embed_name="shared",
        )
    for idx in range(model_config.num_layers):
        # Encoder
        optimize_attention(
            sch[f"encoder.block.{idx}.layer.0.SelfAttention"], model_config
        )
        if sch.world_size > 1:
            shard_mlp(
                sch[f"encoder.block.{idx}.layer.1.DenseReluDense"],
                names=["wi", "wo"],
                backward=True,
            )
        # Decoder
        optimize_attention(
            sch[f"decoder.block.{idx}.layer.0.SelfAttention"], model_config
        )
        optimize_attention(
            sch[f"decoder.block.{idx}.layer.1.EncDecAttention"],
            model_config,
            has_key_value=True,
        )
        if sch.world_size > 1:
            shard_mlp(
                sch[f"decoder.block.{idx}.layer.2.DenseReluDense"],
                names=["wi", "wo"],
                backward=True,
            )
    logger.info("Shard %d attention layers", model_config.num_layers, ranks=0)

    if sch.world_size > 1 and sch_config.get("bcast_input", False):
        # Broadcast input to all devices within the MP group.
        # This is not required when running on Megatron.
        logger.info("Broadcast input to all devices", ranks=0)
        broadcast_input(sch)

    # Insert activation checkpoints.
    ckpt_ratio = sch_config.get("ckpt_ratio", 0.0)
    if ckpt_ratio > 0.0:
        checkpoint_method = sch_config.get("checkpoint_method", "uniform")
        logger.info("Checkpoint ratio: %.2f", ckpt_ratio, ranks=0)
        n_ckpt = checkpoint(
            sch,
            model_config,
            ckpt_ratio=ckpt_ratio,
            checkpoint_method=checkpoint_method,
        )
        logger.info("Checkpointed %d layers", n_ckpt, ranks=0)

    # Pipeline parallelism.
    if sch_config.get("pipeline_cuts", None):
        logger.info("Generate pipeline schedule", ranks=0)
        generate_pipeline_stages(sch, sch_config)

    return orig_sch


def optimize_attention(sch, model_config, has_key_value=False):
    if sch.world_size > 1:
        shard_attention(
            sch,
            names=["q", "k", "v", "o"],
            # Split along the num_heads dimension
            attrs=["n_heads", "inner_dim"],
            backward=True,
        )
    fix_position_bias_shape(sch)
    trace_attention(
        sch,
        model_config,
        input_names=["hidden_states", "mask"]
        + (["key_value_states"] if has_key_value else []),
    )
    replace_sdp(sch, model_config)


def replace_sdp(sch, config):
    # Replace efficient kernels
    def pattern(query_states, key_states, position_bias, value_states):
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=config.dropout_rate, training=True
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output

    class EfficientAttention(torch.nn.Module):
        # Be careful of the order of the arguments
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        def forward(self, key_layer, query_layer, attention_mask, value_layer):
            return F.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=attention_mask,
                dropout_p=config.dropout_rate,
            )

    subgraphs = sch.find(pattern)
    assert len(subgraphs) > 0
    sch.replace(EfficientAttention(), subgraphs)


def fix_position_bias_shape(sch):
    if "relative_attention_bias" in sch:
        sch["relative_attention_bias"].shard("weight", axis=1)
        print("Sharded relative_attention_bias.weight along the num_heads dimension")


def _checkpoint(sch, model_config, path, ckpt_ratio=1.0):
    if ckpt_ratio == 0.0:
        return 0

    n_ckpt = int(model_config.num_hidden_layers * ckpt_ratio)
    for idx in range(n_ckpt):
        sch[path.replace("N", str(idx))].checkpoint()
    return n_ckpt


def checkpoint(
    sch,
    model_config,
    path="",
    ckpt_ratio=1.0,
    checkpoint_method="uniform",
):
    if checkpoint_method != "uniform":
        raise NotImplementedError(
            f"Checkpoint method {checkpoint_method} is not supported yet."
        )
    n_ckpt = 0
    if ckpt_ratio > 0.0:
        n_ckpt += _checkpoint(
            sch, model_config, "encoder.block.N", ckpt_ratio=ckpt_ratio
        )
        n_ckpt += _checkpoint(
            sch, model_config, "decoder.block.N", ckpt_ratio=ckpt_ratio
        )
    return n_ckpt


def generate_pipeline_schedule(sch, sch_config):
    pipeline_cuts = sch_config.get("pipeline_cuts", None)
    prefix = sch_config.get("prefix", "")
    # Cut pipeline stages. For example, [[11], [11]] means to cut
    # encoder.block.11, decoder.block.11. And we always cut between encoder/decoder,
    # so there will be 4 stages in total.
    if pipeline_cuts:
        assert len(pipeline_cuts) == 2
        input_names = [
            "decoder_input_ids",
            "input_ids",
            "decoder_attention_mask",
            "attention_mask",
        ]
        sig = inspect.signature(sch.mod.forward)
        concrete_args = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.name not in input_names
        }
        _prefix = f"{prefix}." if prefix else ""
        sch.trace_until(
            [f"{_prefix}encoder", f"{_prefix}decoder"],
            tracer="huggingface",
            concrete_args=concrete_args,
        )
        for cut in pipeline_cuts[0]:
            sch[f"{_prefix}encoder.block.{cut}"].cut_pipeline_stage()
        sch[f"{_prefix}encoder"].cut_pipeline_stage()
        for cut in pipeline_cuts[1]:
            sch[f"{_prefix}decoder.block.{cut}"].cut_pipeline_stage()

    return sch
