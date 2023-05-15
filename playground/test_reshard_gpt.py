# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoConfig, GPTNeoModel
import inspect
import torch
import torch.fx as fx
import slapo
import torch.distributed as dist
import copy
from util_reshard_mlp \
    import reshard_RS_to_SR_post, reshard_SR_to_RS_post, reshard_RS_to_SR_pre, \
    reshard_RS_to_RR_post, \
    reshard_RR_to_RS_pre, \
    reshard_RR_to_SR_pre, \
    reshard_RS_to_RR_pre

# profile time breakdown
PROFILE = False

NUM_PROC = 4

# Performance Testing
TIMES = 10
BS = 2
SEQ = 1024

dist.init_process_group("nccl", world_size=NUM_PROC)
torch.cuda.set_device(dist.get_rank())
device = torch.cuda.current_device()

config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-1.3B")
with slapo.init_empty_weights(enable=True):
    model = GPTNeoModel(config)
if dist.get_rank() == 0:
    # print(config)
    print(model)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)


def perf_model(mod, input_tensor, times=TIMES):
    """Measure the performance of a mod with certain resharding schemes"""
    for _ in range(5):
        mod(input_tensor)
    start_event.record()
    for _ in range(times):
        mod(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    if dist.get_rank() == 0:
        print(f"{start_event.elapsed_time(end_event) / 1000}")


input_ids = torch.ones(BS, SEQ, dtype=torch.long, device=device)

# 1. Naive
sch = slapo.create_schedule(model)
mod_1, _ = slapo.build(sch, init_weights=model._init_weights)
mod_1.to(device)
perf_model(mod_1, input_ids)
del mod_1
torch.cuda.empty_cache()

# 2. Slapo-Megatron
sch = slapo.create_schedule(copy.deepcopy(model))


def fix_attention_mask_shape(sch):
    import operator

    input_names = ["input_ids"]
    sig = inspect.signature(sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    sch.trace(
        recursive=True, flatten=True, tracer="huggingface", concrete_args=concrete_args
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
    assert len(ops) == 4 * config.num_layers + 1  # q,k,v,context_layer
    # last one is the output layer view

    def new_view(tensor, args):
        if torch.prod(torch.tensor(tensor.shape)) == torch.prod(
            torch.tensor(args)
        ):  # TODO: figure out why this is needed
            return tensor.view(*args)
        if len(args) == 4:  # q,k,v
            out = tensor.view(args[0], args[1], args[2] // sch.world_size, args[3])
        else:  # context_layer
            out = tensor.view(args[0], args[1], args[2] // sch.world_size)
        return out

    for op in ops[: config.num_layers * 4]:
        sch.replace(new_view, op)


for i in range(12):
    # shard attention
    subsch = sch[f"h.{i}.attn.attention"]
    subsch["q_proj"].shard("weight", axis=0)
    subsch["k_proj"].shard("weight", axis=0)
    subsch["v_proj"].shard("weight", axis=0)
    subsch["out_proj"].shard("weight", axis=1) # replace
    subsch["out_proj"].sync("fwd_post", sync_op_or_fn="all_reduce") # replace
    # shard MLP
    subsch = sch[f"h.{i}.mlp"]
    subsch["c_fc"].shard("weight", axis=0)
    subsch["c_fc"].shard("bias", axis=0)
    subsch["c_proj"].shard("weight", axis=1)
    subsch["c_proj"].sync("fwd_post", sync_op_or_fn="all_reduce")
fix_attention_mask_shape(sch)

mod_2, _ = slapo.build(sch, init_weights=False)
mod_2.to(device)
perf_model(mod_2, input_ids)
del mod_2
torch.cuda.empty_cache()


# 3. Weight Sharding. RR x RS = RS
sch = slapo.create_schedule(copy.deepcopy(model))

for i in range(12):
    # shard attention
    subsch = sch[f"h.{i}.attn.attention"]
    subsch["q_proj"].shard("weight", axis=0)
    subsch["k_proj"].shard("weight", axis=0)
    subsch["v_proj"].shard("weight", axis=0)
    # RS here
    subsch["out_proj"].sync("fwd_pre", sync_op_or_fn=reshard_RS_to_RR_pre)
    subsch["out_proj"].shard("weight", axis=0) # RR x RS = RS
    subsch["out_proj"].shard("bias", axis=0)
    subsch["out_proj"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_RR_post)
    # shard MLP
    subsch = sch[f"h.{i}.mlp"]
    subsch["c_fc"].shard("weight", axis=0)
    subsch["c_fc"].shard("bias", axis=0)
    subsch["c_fc"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_RR_post)
    subsch["c_proj"].shard("weight", axis=0)
    subsch["c_proj"].shard("bias", axis=0)
    subsch["c_proj"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_RR_post)
fix_attention_mask_shape(sch)

mod_3, _ = slapo.build(sch, init_weights=model._init_weights)
mod_3.to(device)
perf_model(mod_3, input_ids)
del mod_3
torch.cuda.empty_cache()


# 4. Activation Sharding. SR x RR = SR
# sch = slapo.create_schedule(copy.deepcopy(model))

# for i in range(12):
#     # shard attention
#     subsch = sch[f"bert.encoder.layer.{i}.attention.self"]
#     subsch["query"].shard("weight", axis=0)
#     subsch["query"].shard("bias", axis=0)
#     subsch["key"].shard("weight", axis=0)
#     subsch["key"].shard("bias", axis=0)
#     subsch["value"].shard("weight", axis=0)
#     subsch["value"].shard("bias", axis=0)
#     fix_attention_mask_shape(subsch)
#     subsch = sch[f"bert.encoder.layer.{i}.attention.output"]
#     # shape here: [4096, 256](RS). Need to matmul with [1024, 1024] (without shard)
#     subsch["dense"].sync("fwd_pre", sync_op_or_fn=reshard_RS_to_SR_pre) # LayerNorm will crash for SR x RR = SR
#     # shard MLP
#     subsch = sch[f"bert.encoder.layer.{i}"]
#     subsch["output.dense"].sync("fwd_post", sync_op_or_fn=reshard_SR_to_RR_post)

# mod_4, _ = slapo.build(sch, init_weights=model._init_weights)
# mod_4.to(device)
# perf_model(mod_4, input_ids)
# del mod_4
# torch.cuda.empty_cache()