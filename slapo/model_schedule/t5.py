# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace T5 with model schedule."""
# pylint: disable=unused-argument, import-error

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

    # Shard parameters if MP group > 1.
    if sch.world_size > 1:
        shard_word_embedding(
            sch,
            model_config.vocab_size,
            word_embed_name="shared",
        )
        for idx in range(model_config.num_layers):
            # Encoder
            shard_attention(
                sch[f"encoder.block.{idx}.layer.0.SelfAttention"],
                names=["q", "k", "v", "o"],
                # Split along the num_heads dimension
                attrs=["n_heads", "inner_dim"],
                backward=True,
            )
            fix_position_bias_shape(sch[f"encoder.block.{idx}.layer.0.SelfAttention"])
            shard_mlp(
                sch[f"encoder.block.{idx}.layer.1.DenseReluDense"],
                names=["wi", "wo"],
                backward=True,
            )
            # Decoder
            shard_attention(
                sch[f"decoder.block.{idx}.layer.0.SelfAttention"],
                names=["q", "k", "v", "o"],
                # Split along the num_heads dimension
                attrs=["n_heads", "inner_dim"],
                backward=True,
            )
            fix_position_bias_shape(sch[f"decoder.block.{idx}.layer.0.SelfAttention"])
            shard_attention(
                sch[f"decoder.block.{idx}.layer.1.EncDecAttention"],
                names=["q", "k", "v", "o"],
                # Split along the num_heads dimension
                attrs=["n_heads", "inner_dim"],
                backward=True,
            )
            fix_position_bias_shape(sch[f"decoder.block.{idx}.layer.1.EncDecAttention"])
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
        prefix = sch_config.get("prefix", "")
        logger.info("Checkpoint ratio: %.2f", ckpt_ratio, ranks=0)
        n_ckpt = uniform_checkpoint(sch, model_config.num_layers, ckpt_ratio=ckpt_ratio)
        logger.info("Checkpointed %d layers", n_ckpt, ranks=0)

    # Pipeline parallelism.
    if sch_config.get("pipeline_cuts", None):
        logger.info("Generate pipeline schedule", ranks=0)
        generate_pipeline_stages(sch, sch_config)

    return orig_sch


def fix_position_bias_shape(sch):
    if "relative_attention_bias" in sch:
        sch["relative_attention_bias"].shard("weight", axis=1)
        print("Sharded relative_attention_bias.weight along the num_heads dimension")
