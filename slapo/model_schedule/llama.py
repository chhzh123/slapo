# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace LLaMA with model schedule."""

import math
import torch
from torch import nn
import torch.nn.functional as F

from ..schedule import create_schedule
from ..op import LinearWithSyncFunc
from ..logger import get_logger
from .registry import register_schedule
from .base import (
    shard_attention,
    shard_word_embedding,
    broadcast_input,
    uniform_checkpoint,
    trace_attention,
    generate_pipeline_stages,
)
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRMSNorm


def shard_mlp(sch, names):
    sch.sync("bwd_post", sync_op_or_fn="all_reduce")
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
    def pattern(query_states, key_states, attention_mask, value_states):
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(config.hidden_size // config.num_attention_heads)
        attn_weights = attn_weights + attention_mask
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output

    class EfficientAttention(torch.nn.Module):
        # Be careful of the order of the arguments
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        def forward(self, key_layer, query_layer, attention_mask, value_layer):
            return F.scaled_dot_product_attention(
                query_layer, key_layer, value_layer, attn_mask=attention_mask
            )

    subgraphs = sch.find(pattern)
    assert len(subgraphs) > 0
    sch.replace(EfficientAttention(), subgraphs)


def replace_layernorm(sch, config):
    class JITRMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            """
            LlamaRMSNorm is equivalent to T5LayerNorm
            """
            super().__init__()
            self.ln = torch.jit.script(LlamaRMSNorm(hidden_size, eps).cuda())

        def forward(self, hidden_states):
            return self.ln(hidden_states)

    sch.replace(JITRMSNorm(config.hidden_size))


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
            word_embed_name="embed_tokens",
        )
        for idx in range(model_config.num_hidden_layers):
            shard_attention(
                sch[f"layers.{idx}.self_attn"],
                names=["q_proj", "k_proj", "v_proj", "o_proj"],
                attrs=["num_heads", "hidden_size"],
            )
            shard_mlp(
                sch[f"layers.{idx}.mlp"], names=["gate_proj", "up_proj", "down_proj"]
            )
        logger.info(
            "Shard %d attention layers", model_config.num_hidden_layers, ranks=0
        )

    # Replace efficient kernels.
    if not sch_config.get("disable_fusion", False):
        replace_layernorm(sch["norm"], model_config)
    for idx in range(model_config.num_hidden_layers):
        if not sch_config.get("disable_fusion", False):
            replace_layernorm(sch[f"layers.{idx}.input_layernorm"], model_config)
            replace_layernorm(
                sch[f"layers.{idx}.post_attention_layernorm"], model_config
            )
        trace_attention(
            sch[f"layers.{idx}.self_attn"],
            model_config,
            input_names=["hidden_states", "attention_mask", "position_ids"],
            leaf_modules=["LlamaRotaryEmbedding"],
            leaf_functions=[apply_rotary_pos_emb],
        )
        replace_sdp(sch[f"layers.{idx}.self_attn"], model_config)
        replace_rotary_pos_emb(sch[f"layers.{idx}.self_attn"])

    if sch.world_size > 1 and sch_config.get("bcast_input", False):
        # Broadcast input to all devices within the MP group.
        # This is not required when running on Megatron.
        logger.info("Broadcast input to all devices", ranks=0)
        broadcast_input(sch)

    # Insert activation checkpoints.
    ckpt_ratio = sch_config.get("ckpt_ratio", 0.0)
    if ckpt_ratio > 0.0:
        logger.info("Checkpoint ratio: %.2f", ckpt_ratio, ranks=0)
        n_ckpt = uniform_checkpoint(
            sch, model_config.num_hidden_layers, path="layers.N", ckpt_ratio=ckpt_ratio
        )
        logger.info("Checkpointed %d layers", n_ckpt, ranks=0)

    sequence_parallel = sch_config.get("sequence_parallel", False)
    if sequence_parallel:
        annotate_layernorm_and_bias(sch)

    # Pipeline parallelism.
    pipeline_cuts = sch_config.get("pipeline_cuts", None)
    if pipeline_cuts:
        logger.info("Generate pipeline schedule", ranks=0)
        generate_pipeline_stages(
            sch,
            pipeline_cuts,
            prefix="",
            input_names=["hidden_states", "attention_mask", "position_ids"],
        )

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
