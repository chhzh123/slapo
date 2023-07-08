# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace Bert with model schedule."""

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

logger = get_logger()


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
    # logger = get_logger(f"{model_name}")

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
            word_embed_name="embeddings.word_embeddings",
        )
        for idx in range(model_config.num_hidden_layers):
            shard_attention(
                sch[f"encoder.layer.{idx}.attention"],
                names=["self.query", "self.key", "self.value", "output.dense"],
                # Split along the n_heads dimension,
                # so self.attention_head_size needs not be changed
                attrs=["self.num_attention_heads", "self.all_head_size"],
                backward=True,
            )
            shard_mlp(
                sch[f"encoder.layer.{idx}"],
                names=["intermediate.dense", "output.dense"],
                backward=True,
            )
        logger.info(
            "Shard %d attention layers", model_config.num_hidden_layers, ranks=0
        )

    # Replace efficient kernels.
    for idx in range(model_config.num_hidden_layers):
        trace_attention(sch[f"encoder.layer.{idx}.attention.self"], model_config)
        replace_sdp(sch[f"encoder.layer.{idx}.attention.self"], model_config)
        if not sch_config.get("disable_fusion", False):
            fuse_bias_gelu(sch[f"encoder.layer.{idx}.intermediate"], name="dense", act="intermediate_act_fn")
            fuse_ln_residual(
                sch[f"encoder.layer.{idx}.attention.output"],
                names=["dense", "LayerNorm"],
            )
            fuse_ln_residual(
                sch[f"encoder.layer.{idx}.output"], names=["dense", "LayerNorm"]
            )

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
