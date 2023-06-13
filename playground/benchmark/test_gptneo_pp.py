# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import copy
import inspect
import operator

import torch
from torch import nn
from torch import fx
import torch.distributed as dist
from transformers import GPTNeoModel, AutoConfig

import deepspeed

import slapo
from slapo.framework_dialect.deepspeed.pipeline import (
    get_ds_config,
    create_dist_group_for_pipeline,
)
from slapo.logger import get_logger

logger = get_logger(__name__)

# Config for verification
bs = 4
seq_len = 1024


def perf_model(mod, dataloader, is_deepspeed=False):
    """Measure the performance of a mod with certain resharding schemes"""
    # warmup
    mod.eval()
    # mod.to(torch.float16)
    if is_deepspeed:
        for _ in range(10):
            mod.eval_batch(dataloader, compute_loss=False)
    else:
        for _ in range(10):
            mod(dataloader)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    iters = 40
    if is_deepspeed:
        start_event.record()
        for _ in range(iters):
            mod.eval_batch(dataloader, compute_loss=False)
        end_event.record()
    else:
        start_event.record()
        for _ in range(iters):
            mod(dataloader)
        end_event.record()
    torch.cuda.synchronize()
    if dist.get_rank() == 0:
        print(f"{start_event.elapsed_time(end_event) / iters:.3f} ms")


def trace_and_find_view(sch, config):
    input_names = ["hidden_states"]
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
    ops = sch.find_node(
        lambda node: node.op == "call_method"
        and node.target == "view"
        and (
            (node.args[0].op == "call_module" and "proj" in node.args[0].target)
            or (
                len(node.args) > 1
                and isinstance(node.args[1], fx.Node)
                and node.args[1].op == "call_function"
                and node.args[1].target == operator.add
            )
        )
    )
    sch.mod.embed_dim = config.hidden_size
    sch.mod.num_heads = config.num_heads
    assert len(ops) == 4  # q,k,v,context_layer
    return ops


def fix_attention_mask_shape_megatron(sch, config):
    ops = trace_and_find_view(sch, config)

    def new_view(tensor, args):
        if len(args) == 4:  # q,k,v
            out = tensor.view(args[0], args[1], args[2] // sch.world_size, args[3])
        else:  # context_layer
            out = tensor.view(args[0], args[1], args[2] // sch.world_size)
        return out

    for op in ops:
        sch.replace(new_view, op)


def scheme_megatron(model, input_ids, config):
    num_pp = dist.get_world_size()
    num_mp = 1
    num_dp = dist.get_world_size() // (num_pp * num_mp)
    topology, group = create_dist_group_for_pipeline(num_pp=num_pp, num_mp=num_mp)
    sch = slapo.create_schedule(model, group=group)

    with slapo.Verify(sch, [input_ids], enable=False):
        # Tensor parallel
        # for i in range(config.num_hidden_layers):
        #     # shard attention
        #     subsch = sch[f"h.{i}.attn.attention"]
        #     # no bias for GPTNeo
        #     subsch["q_proj"].shard("weight", axis=0)
        #     subsch["k_proj"].shard("weight", axis=0)
        #     subsch["v_proj"].shard("weight", axis=0)
        #     subsch["out_proj"].shard("weight", axis=1)
        #     subsch["out_proj"].sync("fwd_post", sync_op_or_fn="all_reduce")
        #     fix_attention_mask_shape_megatron(subsch, config)
        #     # shard MLP
        #     subsch = sch[f"h.{i}.mlp"]
        #     subsch["c_fc"].shard("weight", axis=0)
        #     subsch["c_fc"].shard("bias", axis=0)
        #     subsch["c_proj"].shard("weight", axis=1)
        #     subsch["c_proj"].sync("fwd_post", sync_op_or_fn="all_reduce")
        # Pipeline parallel
        input_names = ["input_ids"]
        sig = inspect.signature(sch.mod.forward)
        concrete_args = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.name not in input_names
        }
        sch.trace_until("", tracer="huggingface", concrete_args=concrete_args)
        sch["h.2"].cut_pipeline_stage()
        sch["h.5"].cut_pipeline_stage()
        sch["h.8"].cut_pipeline_stage()
        sch["h.11"].cut_pipeline_stage()
        sch["h.14"].cut_pipeline_stage()
        sch["h.17"].cut_pipeline_stage()
        sch["h.20"].cut_pipeline_stage()

    micro_bs = bs // num_dp
    ds_config_dict = get_ds_config(
        batch_size=bs,
        micro_batch_size_per_gpu=micro_bs,
        fp16=False,
    )
    mod, _ = slapo.build(
        sch,
        init_weights=model._init_weights,
        target="deepspeed",
        topology=topology,
        config=ds_config_dict,
        loss_fn=None,
    )

    return mod


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

    class NewMask(nn.Module):
        def forward(self, query, key, bias):
            query_length, key_length = (
                query.size(-2) * sch.world_size,
                key.size(-2) * sch.world_size,
            )
            size_per_chunk = query_length // sch.world_size
            start_idx = key_length - query_length + size_per_chunk * sch.rank
            end_idx = start_idx + size_per_chunk
            causal_mask = bias[:, :, start_idx:end_idx, :key_length]
            return causal_mask

    enable = True if input_ids.shape[0] == 1 else False
    with slapo.Verify(sch, [input_ids], eval_mode=True, enable=enable):
        sch["drop"].sync(mode="fwd_post", sync_op_or_fn="RR->SR")
        for i in range(config.num_hidden_layers):
            subsch = sch[f"h.{i}.attn.attention"]
            trace_and_find_view(subsch, config)
            ops = subsch.find_node(
                lambda node: node.op == "call_function" and node.target == torch.matmul
            )
            assert len(ops) == 2
            subsch.replace(new_matmul, ops[0])
            subsch.replace(new_matmul_1, ops[1])

            # Need to shard the tril matrix (causal mask)
            def pattern(query, key, bias):
                query_length, key_length = query.size(-2), key.size(-2)
                causal_mask = bias[
                    :, :, key_length - query_length : key_length, :key_length
                ]
                return causal_mask

            ops = subsch.find(pattern)
            subsch.replace(NewMask(), target_ops=[ops[-1]])
        sch[f"ln_f"].sync(mode="fwd_post", sync_op_or_fn="SR->RR")

    return sch


def test_schemes(init_dist):
    device = torch.cuda.current_device()

    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-1.3B")
    config.use_cache = False
    with slapo.init_empty_weights():
        model = GPTNeoModel(config)

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
    deepspeed.init_distributed(dist_backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    num_dp = 1
    micro_bs = bs // num_dp
    input_ids = torch.ones(
        bs, seq_len, dtype=torch.long, device=f"cuda:{dist.get_rank()}"
    )
    labels = torch.randint(
        0, 10, (micro_bs, seq_len), dtype=torch.long, device=dist.get_rank()
    )
    mods = test_schemes(None)
    mod = mods[0]

    # ds_engine = deepspeed.init_inference(
    #     mods[0],
    #     mp_size=1,
    #     dtype=torch.float32,
    #     checkpoint=None,
    #     replace_with_kernel_inject=False,
    # )
    # mod = ds_engine.module
    from deepspeed.utils import RepeatingLoader

    data_iter = RepeatingLoader(
        [
            # First batch: (inputs, labels)
            (
                tuple((input_ids,)),  # inputs
                labels,  # labels
            ),
            # Rest of the batches
            # ...
        ]
    )
    perf_model(mod, data_iter, is_deepspeed=True)
