# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import BertLMHeadModel, AutoConfig
import inspect
import operator
import torch
import slapo
import torch.distributed as dist
import copy

# profile time breakdown
PROFILE = False

NUM_PROC = 1

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
print(config)

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
    # sequence parallelism
    subsch.trace(recursive=False, flatten=False, tracer="pytorch")
    print(subsch.mod.graph)
    ops = subsch.find_node(lambda node: node.op == "call_function" and node.target == operator.add)
    print(ops)
    def reshard_and_add(dropout, hidden_states):
        return dropout + reshard(hidden_states)
    subsch.replace(reshard_and_add, ops[0])
    print(subsch.mod.graph)
    sys.exit()
    subsch["dense"].shard("weight", axis=1)
    subsch["dense"].sync("fwd_post", sync_op_or_fn="all_reduce")
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
