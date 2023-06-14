# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import math
import copy
import inspect

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed
from transformers import LlamaModel, AutoConfig

import slapo
from slapo.logger import get_logger

logger = get_logger(__name__)

# Config for verification
bs = 2
seq_len = 2048


def perf_model(mod, input_tensor):
    """Measure the performance of a mod with certain resharding schemes"""
    # warmup
    mod.eval()
    # mod.to(torch.float16)
    for _ in range(10):
        mod(input_tensor)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    iters = 40
    for _ in range(iters):
        mod(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    if dist.get_rank() == 0:
        print(f"{start_event.elapsed_time(end_event) / iters:.3f} ms")


def trace_and_find_view(sch, config):
    input_names = ["hidden_states", "position_ids"]
    sig = inspect.signature(sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    sch.trace(
        recursive=False,
        flatten=True,
        tracer="huggingface",
        concrete_args=concrete_args,
        leaf_modules=["LlamaRotaryEmbedding"],
        config=config,
    )
    ops = sch.find_node(
        lambda node: node.op == "call_method"
        and node.target == "view"
        and node.args[0].op == "call_module"
        and "proj" in node.args[0].target
    )
    sch.mod.embed_dim = config.hidden_size
    sch.mod.num_attention_heads = config.num_attention_heads
    assert len(ops) == 3  # q,k,v
    reshape_op = sch.find_node(
        lambda node: node.op == "call_method" and node.target == "reshape"
    )
    assert len(reshape_op) == 1
    return ops, reshape_op


def fix_attention_mask_shape_megatron(sch, config):
    ops, reshape_op = trace_and_find_view(sch, config)

    def new_view(tensor, *args):
        return tensor.view(args[0], args[1], args[2] // sch.world_size, args[3])

    for op in ops:
        sch.replace(new_view, op)

    def new_reshape(tensor, *args):
        return tensor.reshape(args[0], args[1], args[2] // sch.world_size)

    sch.replace(new_reshape, reshape_op[0])


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


def scheme_megatron(model, input_ids, config):
    sch = slapo.create_schedule(model)
    # Shard embedding
    # shard_word_embedding(sch, config.vocab_size, "embed_tokens")

    # Shard attention
    for i in range(config.num_hidden_layers):
        # shard self attention
        subsch = sch[f"layers.{i}.self_attn"]
        # no bias for GPTNeo
        subsch["q_proj"].shard("weight", axis=0)
        subsch["k_proj"].shard("weight", axis=0)
        subsch["v_proj"].shard("weight", axis=0)
        subsch["o_proj"].shard("weight", axis=1)
        subsch["o_proj"].sync("fwd_post", sync_op_or_fn="all_reduce")
        fix_attention_mask_shape_megatron(subsch, config)
        # shard MLP
        # * is element-wise multiplication
        # self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        subsch = sch[f"layers.{i}.mlp"]
        subsch["gate_proj"].shard("weight", axis=0)
        subsch["up_proj"].shard("weight", axis=0)
        subsch["down_proj"].shard("weight", axis=1)
        subsch["down_proj"].sync("fwd_post", sync_op_or_fn="all_reduce")

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

        subsch = sch[f"layers.{i}.self_attn"]
        subgraphs = subsch.find(pattern)
        assert len(subgraphs[0]) > 0
        subsch.replace(F.scaled_dot_product_attention, subgraphs)

    return sch


def test_schemes(init_dist):
    torch.cuda.set_device(dist.get_rank())
    device = torch.cuda.current_device()

    config = AutoConfig.from_pretrained("decapoda-research/llama-7b-hf")
    config.use_cache = False
    config.pad_token_id = 0
    with slapo.init_empty_weights():
        model = LlamaModel(config)

    schs = []
    input_ids = torch.ones(bs, seq_len, dtype=torch.long, device=device)
    # 1. Slapo-Megatron
    # RR x RS = RS, RS x SR = RR
    schs.append(scheme_megatron(copy.deepcopy(model), input_ids, config))
    # 2. Sequence-Parallel
    # RR->RS x RR = RS, RS x RR = RS->RR
    # schs.append(scheme_sequence_parallel(copy.deepcopy(model), input_ids, config))
    return schs


if __name__ == "__main__":
    dist.init_process_group("nccl", world_size=int(os.environ["WORLD_SIZE"]))

    input_ids = torch.ones(
        bs, seq_len, dtype=torch.long, device=f"cuda:{dist.get_rank()}"
    )
    schs = test_schemes(None)
    mod, _ = slapo.build(schs[0], init_weights=schs[0].mod._init_weights)
    mod.eval()
    mod.to(f"cuda:{dist.get_rank()}")
    mod.to(torch.float16)

    ds_engine = deepspeed.init_inference(
        mod,
        mp_size=1,
        dtype=torch.float16,
        checkpoint=None,
        replace_with_kernel_inject=False,
    )
    mod = ds_engine.module
    perf_model(mod, input_ids)
    del mod
