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
import argparse
from util_reshard_mlp \
    import reshard_RS_to_SR_post, reshard_SR_to_RS_post, reshard_RS_to_SR_pre, \
    reshard_SR_to_RR_post, reshard_RS_to_RR_post, \
    reshard_RR_to_RS_post, reshard_RR_to_RS_pre, \
    reshard_RR_to_SR_post, reshard_RR_to_SR_pre, \
    reshard_RS_to_RR_pre

def perf_model(mod, input_tensor, times):
    """Measure the performance of a mod with certain resharding schemes
    """
    # warmup
    for _ in range(5):
        mod(input_tensor)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(times):
        mod(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    if dist.get_rank() == 0:
        print(f"{start_event.elapsed_time(end_event) / times:.3f} ms")

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


if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser(description="Resharding schemes on BERT")
    # Add arguments
    parser.add_argument('--bs', type=int, help='Batch size', default=8)
    parser.add_argument('--times', type=int, help='Number of times to run the model', default=10)
    parser.add_argument('--p', type=int, help='Number of processes', default=4)
    # Parse the arguments
    args = parser.parse_args()    


    # profile time breakdown
    PROFILE = False

    NUM_PROC = args.p
    # Performance Testing
    TIMES = args.times
    BS = args.bs
    SEQ = 512

    dist.init_process_group("nccl", world_size=NUM_PROC)

    if dist.get_rank() == 0:
        print(f"===== Setting =====")
        print(f"NUM_PROC: {NUM_PROC}; BS: {BS}; SEQ: {SEQ}; Model: Bert-Large")


    torch.cuda.set_device(dist.get_rank())
    device = torch.cuda.current_device()

    config = AutoConfig.from_pretrained("bert-large-uncased")
    with slapo.init_empty_weights():
        model = BertLMHeadModel(config)
    # print(config)
    # print(model)


    input_ids = torch.ones(BS, SEQ, dtype=torch.long, device=device)

    # 1. Naive
    sch = slapo.create_schedule(copy.deepcopy(model))
    mod_1, _ = slapo.build(sch, init_weights=model._init_weights)
    mod_1.to(device)
    perf_model(mod_1, input_ids, TIMES)
    del mod_1
    torch.cuda.empty_cache()

    # 2. Slapo-Megatron
    sch = slapo.create_schedule(copy.deepcopy(model))

    for i in range(24):
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
    perf_model(mod_2, input_ids, TIMES)
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
    perf_model(mod_3, input_ids, TIMES)
    del mod_3
    torch.cuda.empty_cache()


    # 4. Activation Sharding. SR x RR = SR
    sch = slapo.create_schedule(copy.deepcopy(model))

    def reshard_RR_to_SR(input):
        """Reshard from RR to SR
        input: torch.Tensor
        """
        in_tensor = input
        # get the current rank's tensor. Slice across the 2nd last dimension
        shard_dim_size = in_tensor.shape[-2] // dist.get_world_size()
        start_idx = (int)(dist.get_rank() * shard_dim_size)
        end_idx = (int)((dist.get_rank() + 1) * shard_dim_size)

        slices = [slice(None)] * len(in_tensor.shape)
        slices[-2] = slice(start_idx, end_idx)

        # Slice the tensor
        ret = in_tensor[slices]
        return ret

    def reshard_and_add(dropout, hidden_states):
        """Replace the add operator with reshard_and_add
        """
        reshard_hidden_states = reshard_RR_to_SR(hidden_states)
        return dropout + reshard_hidden_states


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

        subsch.trace(recursive=False, flatten=False, tracer="pytorch")
        ops = subsch.find_node(lambda node: node.op == "call_function" and node.target == operator.add)
        subsch.replace(reshard_and_add, ops[0]) 

        # shape here: RS
        subsch["dense"].sync("fwd_pre", sync_op_or_fn=reshard_RS_to_SR_pre) # LayerNorm will crash for SR x RR = SR
        # shard MLP
        subsch = sch[f"bert.encoder.layer.{i}"]
        subsch["output.LayerNorm"].sync("fwd_post", sync_op_or_fn=reshard_SR_to_RR_post)

    mod_4, _ = slapo.build(sch, init_weights=model._init_weights)
    mod_4.to(device)
    perf_model(mod_4, input_ids, TIMES)
    del mod_4
    torch.cuda.empty_cache()
