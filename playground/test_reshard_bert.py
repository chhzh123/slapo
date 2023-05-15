# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import BertLMHeadModel, AutoConfig
import inspect
import torch
import slapo
import torch.distributed as dist
import copy
import operator
import sys
from util_reshard_mlp \
    import reshard_RS_to_SR_post, reshard_SR_to_RS_post, reshard_RS_to_SR_pre, \
    reshard_SR_to_RR_post, reshard_RS_to_RR_post, \
    reshard_RR_to_RS_post, reshard_RR_to_RS_pre, \
    reshard_RR_to_SR_post, reshard_RR_to_SR_pre, \
    reshard_RS_to_RR_pre
# profile time breakdown
PROFILE = False

NUM_PROC = 4

# Performance Testing
TIMES = 10
BS = 8
SEQ = 512

dist.init_process_group("nccl", world_size=NUM_PROC)
torch.cuda.set_device(dist.get_rank())
device = torch.cuda.current_device()

config = AutoConfig.from_pretrained("bert-large-uncased")
with slapo.init_empty_weights():
    model = BertLMHeadModel(config)
# print(config)
# print(model)

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
sch = slapo.create_schedule(copy.deepcopy(model))
# print("==== Schedule ====")
# print(sch)
mod_1, _ = slapo.build(sch, init_weights=model._init_weights)
mod_1.to(device)
perf_model(mod_1, input_ids)
del mod_1
torch.cuda.empty_cache()

# 2. Slapo-Megatron
sch = slapo.create_schedule(copy.deepcopy(model))


def fix_attention_mask_shape(sch):
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

    def new_view(tensor, args):
        if len(args) == 4:  # q,k,v
            out = tensor.view(args[0], args[1], args[2] // sch.world_size, args[3])
        else:  # context_layer
            out = tensor.view(args[0], args[1], args[2] // sch.world_size)
        return out

    for op in ops:
        sch.replace(new_view, op)


for i in range(12):
    # shard attention
    subsch = sch[f"bert.encoder.layer.{i}.attention.self"]
    subsch["query"].shard("weight", axis=0)
    subsch["query"].shard("bias", axis=0)
    subsch["key"].shard("weight", axis=0)
    subsch["key"].shard("bias", axis=0)
    subsch["value"].shard("weight", axis=0)
    subsch["value"].shard("bias", axis=0)
    fix_attention_mask_shape(subsch)
    subsch = sch[f"bert.encoder.layer.{i}.attention.output"]
    subsch["dense"].shard("weight", axis=1) # replace
    subsch["dense"].sync("fwd_post", sync_op_or_fn="all_reduce") # replace
    # shard MLP
    subsch = sch[f"bert.encoder.layer.{i}"]
    subsch["intermediate.dense"].shard("weight", axis=0)
    subsch["intermediate.dense"].shard("bias", axis=0)
    subsch["output.dense"].shard("weight", axis=1)
    subsch["output.dense"].sync("fwd_post", sync_op_or_fn="all_reduce")

mod_2, _ = slapo.build(sch, init_weights=model._init_weights)
mod_2.to(device)
perf_model(mod_2, input_ids)
del mod_2
torch.cuda.empty_cache()


# 3. Weight Sharding. RR x RS = RS
sch = slapo.create_schedule(copy.deepcopy(model))

for i in range(12):
    # shard attention
    subsch = sch[f"bert.encoder.layer.{i}.attention.self"]
    subsch["query"].shard("weight", axis=0)
    subsch["query"].shard("bias", axis=0)
    subsch["key"].shard("weight", axis=0)
    subsch["key"].shard("bias", axis=0)
    subsch["value"].shard("weight", axis=0)
    subsch["value"].shard("bias", axis=0)
    fix_attention_mask_shape(subsch)
    subsch = sch[f"bert.encoder.layer.{i}.attention.output"]
    # shape here: [4096, 256](RS). Need to matmul with [1024, 1024] (without shard)
    subsch["dense"].sync("fwd_pre", sync_op_or_fn=reshard_RS_to_RR_pre)
    subsch["dense"].shard("weight", axis=0)
    subsch["dense"].shard("bias", axis=0)
    subsch["dense"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_RR_post)
    # shard MLP
    subsch = sch[f"bert.encoder.layer.{i}"]
    subsch["intermediate.dense"].shard("weight", axis=0)
    subsch["intermediate.dense"].shard("bias", axis=0)
    subsch["intermediate.dense"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_RR_post)
    subsch["output.dense"].shard("weight", axis=0)
    subsch["output.dense"].shard("bias", axis=0)
    subsch["output.dense"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_RR_post)

mod_3, _ = slapo.build(sch, init_weights=model._init_weights)
mod_3.to(device)
perf_model(mod_3, input_ids)
del mod_3
torch.cuda.empty_cache()


# 4. Activation Sharding. SR x RR = SR
sch = slapo.create_schedule(copy.deepcopy(model))

for i in range(12):
    # shard attention
    subsch = sch[f"bert.encoder.layer.{i}.attention.self"]
    subsch["query"].shard("weight", axis=0)
    subsch["query"].shard("bias", axis=0)
    subsch["key"].shard("weight", axis=0)
    subsch["key"].shard("bias", axis=0)
    subsch["value"].shard("weight", axis=0)
    subsch["value"].shard("bias", axis=0)
    fix_attention_mask_shape(subsch)
    subsch = sch[f"bert.encoder.layer.{i}.attention.output"]

    if dist.get_rank == 0:
        subsch.trace(recursive=False, flatten=False, tracer="pytorch")
        print(subsch.mod.graph)   
        ops = subsch.find_node(lambda node: node.op == "call_function" and node.target == operator.add)
        print(ops)
    # synchronize all ranks
    sys.exit(0)


    # shape here: [4096, 256](RS). Need to matmul with [1024, 1024] (without shard)
    subsch["dense"].sync("fwd_pre", sync_op_or_fn=reshard_RS_to_SR_pre) # LayerNorm will crash for SR x RR = SR
    # shard MLP
    subsch = sch[f"bert.encoder.layer.{i}"]
    subsch["output.dense"].sync("fwd_post", sync_op_or_fn=reshard_SR_to_RR_post)

mod_4, _ = slapo.build(sch, init_weights=model._init_weights)
mod_4.to(device)
perf_model(mod_4, input_ids)
del mod_4
torch.cuda.empty_cache()



# sequence parallelism
    # subsch.trace(recursive=False, flatten=False, tracer="pytorch")
    # print(subsch.mod.graph)
    # ops = subsch.find_node(lambda node: node.op == "call_function" and node.target == operator.add)
    # print(ops)
    # def reshard_and_add(dropout, hidden_states):
    #     return dropout + reshard(hidden_states)
    # subsch.replace(reshard_and_add, ops[0])
    # print(subsch.mod.graph)
    # sys.exit()