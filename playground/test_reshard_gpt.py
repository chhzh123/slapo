# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoConfig, GPTNeoModel
import inspect
import torch
import torch.fx as fx
import slapo
import torch.distributed as dist
import copy
import argparse
import operator
from torch import fx
from util_reshard_mlp \
    import reshard_RS_to_SR_post, reshard_SR_to_RS_post, reshard_RS_to_SR_pre, \
    reshard_RS_to_RR_post, \
    reshard_RR_to_RS_pre, \
    reshard_RR_to_SR_pre, \
    reshard_RS_to_RR_pre, \
    reshard_SR_to_RR_post, reshard_SR_to_RR_pre

def perf_model(mod, input_tensor, times):
    """Measure the performance of a mod with certain resharding schemes
    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # warmup
    for _ in range(5):
        mod(input_tensor)

    start_event.record()
    for _ in range(times):
        mod(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    if dist.get_rank() == 0:
        print(f"{start_event.elapsed_time(end_event) / times:.3f} ms")

def fix_attention_mask_shape(sch):
    input_names = ["input_ids"]
    sig = inspect.signature(sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    sch.trace(
        recursive=True, flatten=True, tracer="huggingface", concrete_args=concrete_args
    )
    # print(sch.mod.graph)
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
    

def fix_attention_mask_shape_replace_add(sch):
    input_names = ["input_ids"]
    sig = inspect.signature(sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    sch.trace(
        recursive=True, flatten=True, tracer="huggingface", concrete_args=concrete_args
    )
    # print(sch.mod.graph)
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
    

    ops = sch.find_node(lambda node: node.op == "call_function" 
                        and node.target == operator.add 
                        and isinstance(node.args[0], fx.Node)
                        and node.args[0].op == "call_module"
                        and "dropout" in node.args[0].target)
    print(ops)
    for op in ops:
        sch.replace(reshard_and_add, op)
    
    ops = sch.find_node(lambda node: node.op == "call_function"
                        and node.target == operator.add
                        and isinstance(node.args[1], fx.Node)
                        and "dropout" in node.args[1].target)
    print(ops)
    for op in ops:
        sch.replace(add_and_gather, op)


def reshard_and_add(dropout, hidden_states):
    """Replace the add operator with reshard_and_add
    """
    reshard_hidden_states = reshard_RR_to_SR(hidden_states)
    return dropout + reshard_hidden_states

def add_and_gather(residual, hidden_states):
    """Replace the add operator with add_and_gather
    """
    ret = reshard_SR_to_RR(residual + hidden_states)
    return ret

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

def reshard_SR_to_RR(input):
    """Reshard from SR to RR
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


if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser(description="Resharding schemes on BERT")
    # Add arguments
    parser.add_argument('--bs', type=int, help='Batch size', default=4)
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
    SEQ = 1024

    dist.init_process_group("nccl", world_size=NUM_PROC)
    torch.cuda.set_device(dist.get_rank())
    device = torch.cuda.current_device()

    if dist.get_rank() == 0:
        print(f"===== Setting =====")
        print(f"NUM_PROC: {NUM_PROC}; BS: {BS}; SEQ: {SEQ}; Model: GPT-Neo")

    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-1.3B")
    with slapo.init_empty_weights(enable=False):
        model = GPTNeoModel(config)


    input_ids = torch.ones(BS, SEQ, dtype=torch.long, device=device)

    # 1. Naive
    sch = slapo.create_schedule(model)
    mod_1, _ = slapo.build(sch, init_weights=model._init_weights)
    mod_1.to(device)
    perf_model(mod_1, input_ids, TIMES)
    del mod_1
    torch.cuda.empty_cache()

    # 2. Slapo-Megatron
    sch = slapo.create_schedule(copy.deepcopy(model))

    for i in range(24):
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
    perf_model(mod_2, input_ids, TIMES)
    del mod_2
    torch.cuda.empty_cache()


    # 3. Weight Sharding. RR x RS = RS
    sch = slapo.create_schedule(copy.deepcopy(model))

    for i in range(24):
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
    perf_model(mod_3, input_ids, TIMES)
    del mod_3
    torch.cuda.empty_cache()


    # # 4. Activation Sharding. SR x RR = SR
    # sch = slapo.create_schedule(copy.deepcopy(model))

 
    # for i in range(24):
    #     # shard attention
    #     subsch = sch[f"h.{i}.attn.attention"]
    #     subsch["q_proj"].shard("weight", axis=0)
    #     subsch["k_proj"].shard("weight", axis=0)
    #     subsch["v_proj"].shard("weight", axis=0)
    #     # RS here

    #     # shape here: RS
    #     subsch["out_proj"].sync("fwd_pre", sync_op_or_fn=reshard_RS_to_SR_pre) # LayerNorm will crash for SR x RR = SR
    #     # shard MLP
    #     subsch = sch[f"h.{i}.mlp"]
    #     # subsch["c_proj"].sync("fwd_post", sync_op_or_fn=reshard_SR_to_RR_post)
        
    #     # sch[f"h.{i+1}.ln_1"].sync("fwd_pre", sync_op_or_fn=reshard_SR_to_RR_pre)

    # fix_attention_mask_shape(sch)

    # mod_4, _ = slapo.build(sch, init_weights=model._init_weights)
    # mod_4.to(device)
    # perf_model(mod_4, input_ids, TIMES)
    # del mod_4
    # torch.cuda.empty_cache()