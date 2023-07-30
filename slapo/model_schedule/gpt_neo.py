# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace GPT-Neo with model schedule."""

from torch import nn

from ..schedule import create_schedule
from ..op import LinearWithSyncFunc
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
            sch,
            model_config.vocab_size,
            word_embed_name="wte",
        )
        for idx in range(model_config.num_layers):
            shard_attention(
                sch[f"h.{idx}.attn.attention"],
                names=["q_proj", "k_proj", "v_proj", "out_proj"],
                # Split along the n_heads dimension
                attrs=["num_heads"],
                backward=True,
            )
            shard_mlp(
                sch[f"h.{idx}.mlp"],
                names=["c_fc", "c_proj"],
                backward=True,
            )
        logger.info(
            "Shard %d attention layers", model_config.num_hidden_layers, ranks=0
        )

    # Replace efficient kernels.
    for idx in range(model_config.num_layers):
        trace_attention(
            sch[f"h.{idx}.attn.attention"],
            model_config,
            input_names=["hidden_states", "attention_mask"],
        )
        replace_sdp(
            sch[f"h.{idx}.attn.attention"],
            model_config,
            mask=True,
            dropout=model_config.attention_dropout,
        )
        if not sch_config.get("disable_fusion", False):
            fuse_bias_gelu(sch[f"h.{idx}.mlp"], name="c_fc", act="act")
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

    sequence_parallel = sch_config.get("sequence_parallel", False)
    if sequence_parallel:
        annotate_layernorm_and_bias(sch)

    # Pipeline parallelism.
    if sch_config.get("pipeline_cuts", None):
        logger.info("Generate pipeline schedule", ranks=0)
        generate_pipeline_stages(sch, sch_config)

    return orig_sch


def annotate_layernorm_and_bias(sch):
    """Annotate parameters that require additional allreduce on tensor parallel group
    when sequence parallelism is turned on. This is specific for DeepSpeed pipeline
    runtime.

    Parameters
    ----------
    sch : slapo.Schedule
        The schedule of the model.
    """
    for sub_sch in sch.child.values():
        if isinstance(sub_sch.mod, nn.LayerNorm):
            for name, _ in sub_sch.mod.named_parameters(recurse=False):
                sub_sch.annotate(name, "replicated_param", True)
        if issubclass(sub_sch.mod.__class__, LinearWithSyncFunc):
            sub_sch.annotate("bias", "replicated_param", True)
        annotate_layernorm_and_bias(sub_sch)


def checkpoint(
    sch, model_config, path="h.N", ckpt_ratio=1.0, checkpoint_method="uniform"
):
    """Add activation checkpointing to the model. The ckpt_ratio specifies
    the ratio of the attention layers to be checkpointed. For example, if
    ckpt_ratio is 0.5, then half of the attention layers will be checkpointed.

    Parameters
    ----------
    sch : slapo.Schedule
        The schedule of the model.
    model_config : GPT2Config
        The configuration of the model.
    path : str
        The path to the attention layer. Default: "h.N.attn".
    ckpt_ratio : float
        The ratio of the attention layers to be checkpointed. Default: 1.0.
    checkpoint_method : str
        The checkpointing method. Default: "uniform".
    """
    if ckpt_ratio == 0.0:
        return 0

    def order_args_fn(*args, **kwargs):
        assert len(args) == 1
        attention_mask = kwargs.get("attention_mask", None)
        head_mask = kwargs.get("head_mask", None)
        output_attentions = kwargs.get("output_attentions", False)
        # Forward: (
        #   hidden_states,
        #   layer_past,
        #   attention_mask,
        #   head_mask,
        #   use_cache,
        #   output_attentions
        # )
        return (args[0], None, attention_mask, head_mask, False, output_attentions)

    n_ckpt = int(model_config.num_layers * ckpt_ratio)
    if checkpoint_method == "head":
        for idx in range(n_ckpt):
            sch[path.replace("N", str(idx))].checkpoint(order_args_fn=order_args_fn)
    elif checkpoint_method == "uniform" and ckpt_ratio > 0:
        for idx in range(0, model_config.num_layers, max(1, int(1 / ckpt_ratio))):
            sch[path.replace("N", str(idx))].checkpoint(order_args_fn=order_args_fn)
    return n_ckpt
