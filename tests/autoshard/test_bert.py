# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import copy
import inspect
import operator
import argparse

import torch
import torch.distributed as dist
from transformers import BertLMHeadModel, AutoConfig

import slapo
from slapo.logger import get_logger

logger = get_logger(__name__)

# Config for verification
bs = 4
seq_len = 512


def perf_model(mod, input_tensor):
    """Measure the performance of a mod with certain resharding schemes"""
    # warmup
    mod.eval()
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


def trace_and_find_view(sch):
    input_names = ["hidden_states"]
    sig = inspect.signature(sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    sch.trace(
        recursive=False, flatten=True, tracer="pytorch", concrete_args=concrete_args
    )
    ops = sch.find_node(lambda node: node.op == "call_method" and node.target == "view")
    assert len(ops) == 4  # q,k,v,context_layer
    return ops


def fix_attention_mask_shape_megatron(sch):
    ops = trace_and_find_view(sch)

    def new_view(tensor, args):
        if len(args) == 4:  # q,k,v
            out = tensor.view(args[0], args[1], args[2] // sch.world_size, args[3])
        else:  # context_layer
            out = tensor.view(args[0], args[1], args[2] // sch.world_size)
        return out

    for op in ops:
        sch.replace(new_view, op)


def scheme_megatron(model, input_ids, config):
    sch = slapo.create_schedule(model)

    enable = True if input_ids.shape[0] <= 4 else False
    with slapo.Verify(sch, [input_ids], enable=enable):
        for i in range(config.num_hidden_layers):
            # shard attention
            subsch = sch[f"bert.encoder.layer.{i}.attention.self"]
            subsch["query"].shard("weight", axis=0)
            subsch["query"].shard("bias", axis=0)
            subsch["key"].shard("weight", axis=0)
            subsch["key"].shard("bias", axis=0)
            subsch["value"].shard("weight", axis=0)
            subsch["value"].shard("bias", axis=0)
            fix_attention_mask_shape_megatron(subsch)
            subsch = sch[f"bert.encoder.layer.{i}.attention.output"]
            subsch["dense"].shard("weight", axis=1)  # replace
            subsch["dense"].sync("fwd_post", sync_op_or_fn="all_reduce")  # replace
            # shard MLP
            subsch = sch[f"bert.encoder.layer.{i}"]
            subsch["intermediate.dense"].shard("weight", axis=0)
            subsch["intermediate.dense"].shard("bias", axis=0)
            subsch["output.dense"].shard("weight", axis=1)
            subsch["output.dense"].sync("fwd_post", sync_op_or_fn="all_reduce")

    return sch


def scheme_sequence_parallel(model, input_ids, config):
    sch = slapo.create_schedule(model)

    from slapo.sharding.reshard_ops import (
        reshard_SR_to_RR,
        reshard_RS_to_RR,
    )

    def new_matmul(lhs, rhs):
        return torch.matmul(lhs, reshard_RS_to_RR(rhs, sch.group))

    def new_matmul_1(lhs, rhs):
        return torch.matmul(lhs, reshard_SR_to_RR(rhs, sch.group))

    enable = True if input_ids.shape[0] <= 4 else False
    with slapo.Verify(sch, [input_ids], enable=enable):
        sch["bert.embeddings.LayerNorm"].sync(mode="fwd_post", sync_op_or_fn="RR->SR")
        for i in range(config.num_hidden_layers):
            subsch = sch[f"bert.encoder.layer.{i}.attention.self"]
            trace_and_find_view(subsch)
            ops = subsch.find_node(
                lambda node: node.op == "call_function" and node.target == torch.matmul
            )
            assert len(ops) == 2
            subsch.replace(new_matmul, ops[0])
            subsch.replace(new_matmul_1, ops[1])
        sch[f"bert.encoder.layer.{config.num_hidden_layers - 1}.output.LayerNorm"].sync(
            mode="fwd_post", sync_op_or_fn="SR->RR"
        )

    return sch


def scheme_sequence_parallel_attn(model, input_ids, config):
    sch = slapo.create_schedule(model)

    from slapo.sharding.reshard_ops import (
        reshard_SR_to_RR,
        reshard_RR_to_SR,
    )

    def scaled_dot_product(query_layer, key_layer, value_layer):
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / 8  # math.sqrt(64)
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = torch.nn.functional.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        return context_layer

    class EfficientAttention(torch.nn.Module):
        def forward(self, key_layer, query_layer, value_layer):
            query_layer = reshard_SR_to_RR(query_layer, sch.group)
            # transpose is done inside the scaled_dot_product
            key_layer = reshard_SR_to_RR(key_layer, sch.group)
            value_layer = reshard_SR_to_RR(value_layer, sch.group)
            score = torch.nn.functional.scaled_dot_product_attention(
                query_layer, key_layer, value_layer
            )
            return reshard_RR_to_SR(score, sch.group)

    with slapo.Verify(sch, [input_ids], enable=True):
        sch["bert.embeddings.LayerNorm"].sync(mode="fwd_post", sync_op_or_fn="RR->SR")
        for i in range(config.num_hidden_layers):
            subsch = sch[f"bert.encoder.layer.{i}.attention.self"]
            trace_and_find_view(subsch)
            ops = subsch.find_node(
                lambda node: node.op == "call_function" and node.target == torch.matmul
            )
            assert len(ops) == 2
            subgraphs = subsch.find(scaled_dot_product)
            subsch.replace(EfficientAttention(), subgraphs)
        sch[f"bert.encoder.layer.{config.num_hidden_layers - 1}.output.LayerNorm"].sync(
            mode="fwd_post", sync_op_or_fn="SR->RR"
        )

    return sch


def scheme_activation_stationary(model, input_ids, config):
    sch = slapo.create_schedule(model)
    enable = True if input_ids.shape[0] <= 4 else False
    with slapo.Verify(sch, [input_ids], enable=enable):
        for i in range(config.num_hidden_layers):
            # shard attention
            subsch = sch[f"bert.encoder.layer.{i}.attention.self"]
            subsch["query"].shard("weight", axis=0)
            subsch["query"].shard("bias", axis=0)
            subsch["key"].shard("weight", axis=0)
            subsch["key"].shard("bias", axis=0)
            subsch["value"].shard("weight", axis=0)
            subsch["value"].shard("bias", axis=0)
            fix_attention_mask_shape_megatron(subsch)
            subsch = sch[f"bert.encoder.layer.{i}.attention.output"]
            # shape here: [4096, 256](RS). Need to matmul with [1024, 1024] (without shard)
            subsch["dense"].sync("fwd_pre", sync_op_or_fn="RS->RR")
            subsch["dense"].shard("weight", axis=0)
            subsch["dense"].shard("bias", axis=0)
            subsch["dense"].sync("fwd_post", sync_op_or_fn="RS->RR")
            # shard MLP
            subsch = sch[f"bert.encoder.layer.{i}"]
            subsch["intermediate.dense"].shard("weight", axis=0)
            subsch["intermediate.dense"].shard("bias", axis=0)
            subsch["intermediate.dense"].sync("fwd_post", sync_op_or_fn="RS->RR")
            subsch["output.dense"].shard("weight", axis=0)
            subsch["output.dense"].shard("bias", axis=0)
            subsch["output.dense"].sync("fwd_post", sync_op_or_fn="RS->RR")

    return sch


def scheme_activation_sharding(model, input_ids, config):
    sch = slapo.create_schedule(model)

    from slapo.sharding.reshard_ops import reshard_RR_to_SR

    def reshard_and_add(dropout, hidden_states):
        """Replace the add operator with reshard_and_add"""
        reshard_hidden_states = reshard_RR_to_SR(hidden_states, sch.group)
        return dropout + reshard_hidden_states

    enable = True if input_ids.shape[0] <= 4 else False
    with slapo.Verify(sch, [input_ids], enable=enable):
        for i in range(config.num_hidden_layers):
            # shard attention
            subsch = sch[f"bert.encoder.layer.{i}.attention.self"]
            subsch["query"].shard("weight", axis=0)
            subsch["query"].shard("bias", axis=0)
            subsch["key"].shard("weight", axis=0)
            subsch["key"].shard("bias", axis=0)
            subsch["value"].shard("weight", axis=0)
            subsch["value"].shard("bias", axis=0)
            fix_attention_mask_shape_megatron(subsch)
            subsch = sch[f"bert.encoder.layer.{i}.attention.output"]

            subsch.trace(recursive=False, flatten=False, tracer="pytorch")
            ops = subsch.find_node(
                lambda node: node.op == "call_function" and node.target == operator.add
            )
            subsch.replace(reshard_and_add, ops[0])

            # shape here: RS
            subsch["dense"].sync(
                "fwd_pre", sync_op_or_fn="RS->SR"
            )  # LayerNorm will crash for SR x RR = SR
            # shard MLP
            subsch = sch[f"bert.encoder.layer.{i}"]
            subsch["output.LayerNorm"].sync("fwd_post", sync_op_or_fn="SR->RR")

    return sch


def replace_attn(model, input_ids, config):
    sch = slapo.create_schedule(model)

    # https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html
    def scaled_dot_product(query_layer, key_layer, value_layer):
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / 8  # math.sqrt(64)
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = torch.nn.functional.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        return context_layer

    class EfficientAttention(torch.nn.Module):
        def forward(self, key_layer, query_layer, value_layer):
            return torch.nn.functional.scaled_dot_product_attention(
                query_layer, key_layer, value_layer
            )

    with slapo.Verify(sch, [input_ids], eval_mode=True, enable=True):
        for i in range(config.num_hidden_layers):
            subsch = sch[f"bert.encoder.layer.{i}.attention.self"]
            trace_and_find_view(subsch)
            subgraphs = subsch.find(scaled_dot_product)
            subsch.replace(EfficientAttention(), subgraphs)


def test_schemes(init_dist):
    torch.cuda.set_device(dist.get_rank())
    device = torch.cuda.current_device()

    config = AutoConfig.from_pretrained("bert-large-uncased")
    with slapo.init_empty_weights():
        model = BertLMHeadModel(config)

    schs = []
    input_ids = torch.ones(bs, seq_len, dtype=torch.long, device=device)
    # 1. Slapo-Megatron
    # RR x RS = RS, RS x SR = RR
    schs.append(scheme_megatron(copy.deepcopy(model), input_ids, config))
    # 2. Sequence-Parallel
    # RR->RS x RR = RS, RS x RR = RS->RR
    schs.append(scheme_sequence_parallel(copy.deepcopy(model), input_ids, config))
    schs.append(scheme_sequence_parallel_attn(copy.deepcopy(model), input_ids, config))
    # replace_attn(copy.deepcopy(model), input_ids, config)
    # 3. Activation-Stationary
    # RR x RS = RS
    schs.append(scheme_activation_stationary(copy.deepcopy(model), input_ids, config))
    # 4. Activation Sharding. SR x RR = SR
    schs.append(scheme_activation_sharding(copy.deepcopy(model), input_ids, config))
    return schs


if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser(description="Resharding schemes on BERT")
    # Add arguments
    parser.add_argument("--bs", type=int, help="Batch size", default=8)
    parser.add_argument("--seq", type=int, help="Sequence length", default=512)
    # Parse the arguments
    args = parser.parse_args()

    bs = args.bs
    seq_len = args.seq

    dist.init_process_group("nccl", world_size=int(os.environ["WORLD_SIZE"]))

    logger.info(
        "Number of GPUs: %d, bs=%d, seq_len=%d; Model: BERT-large",
        dist.get_world_size(),
        bs,
        seq_len,
        ranks=0,
    )

    schs = test_schemes(None)

    input_ids = torch.ones(
        bs, seq_len, dtype=torch.long, device=f"cuda:{dist.get_rank()}"
    )
    for i, sch in enumerate(schs):
        mod, _ = slapo.build(sch, init_weights=sch.mod._init_weights)
        mod.to(f"cuda:{dist.get_rank()}")
        torch.cuda.empty_cache()
        perf_model(mod, input_ids)
        del mod
