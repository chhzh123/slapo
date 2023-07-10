# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace OPT with model schedule."""
# pylint: disable=unused-argument

import torch
from torch import nn

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
    replace_sdp,
    fuse_bias_gelu,
    fuse_ln_residual,
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
    # TODO: sequence parallel

    # Shard parameters if MP group > 1.
    if sch.world_size > 1:
        shard_word_embedding(
            sch["decoder"],
            model_config.vocab_size,
            word_embed_name="embed_tokens",
        )
        for idx in range(model_config.num_hidden_layers):
            shard_attention(
                sch[f"decoder.layers.{idx}.self_attn"],
                names=["q_proj", "k_proj", "v_proj", "out_proj"],
                # Split along the num_heads dimension
                attrs=["num_heads", "embed_dim"],
                backward=True,
                use_kwargs=True,
            )
            shard_mlp(
                sch[f"decoder.layers.{idx}"],
                names=["fc1", "fc2"],
                backward=True,
            )
        logger.info(
            "Shard %d attention layers", model_config.num_hidden_layers, ranks=0
        )

    # Replace efficient kernels.
    for idx in range(model_config.num_hidden_layers):
        trace_attention(sch[f"decoder.layers.{idx}.self_attn"], model_config)

        def attn_pattern(query_states, key_states, value_states):
            attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_probs = nn.functional.dropout(
                attn_weights, p=model_config.attention_dropout
            )
            attn_output = torch.bmm(attn_probs, value_states)
            return attn_output

        replace_sdp(
            sch[f"decoder.layers.{idx}.self_attn"],
            model_config,
            pattern=attn_pattern,
            mask=False,
        )
        if not sch_config.get("disable_fusion", False):
            fuse_bias_gelu(
                sch[f"decoder.layers.{idx}"],
                name="fc1",
                act="activation_fn",
                tracer="huggingface",
                leaf_modules=["self_attn"],
                config=model_config,
            )
        # fuse_ln_residual(
        #     sch[f"encoder.layer.{idx}.attention.output"],
        #     names=["dense", "LayerNorm"],
        # )
        # fuse_ln_residual(
        #     sch[f"encoder.layer.{idx}.output"], names=["dense", "LayerNorm"]
        # )

    if sch.world_size > 1 and sch_config.get("bcast_input", False):
        # Broadcast input to all devices within the MP group.
        # This is not required when running on Megatron.
        logger.info("Broadcast input to all devices", ranks=0)
        broadcast_input(sch)

    # Insert activation checkpoints.
    ckpt_ratio = sch_config.get("ckpt_ratio", 0.0)
    if ckpt_ratio > 0.0:
        prefix = sch_config.get("prefix", "")
        logger.info("Checkpoint ratio: %.2f", ckpt_ratio, ranks=0)
        n_ckpt = uniform_checkpoint(
            sch, model_config.num_hidden_layers, ckpt_ratio=ckpt_ratio
        )
        logger.info("Checkpointed %d layers", n_ckpt, ranks=0)

    # Pipeline parallelism.
    if sch_config.get("pipeline_cuts", None):
        logger.info("Generate pipeline schedule", ranks=0)
        generate_pipeline_stages(sch, sch_config)

    return orig_sch
