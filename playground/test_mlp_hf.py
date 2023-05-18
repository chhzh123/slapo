# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import BertLMHeadModel, AutoConfig
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import slapo

config = AutoConfig.from_pretrained("bert-large-uncased")
with slapo.init_empty_weights():
    model = BertLMHeadModel(config)
print(config)



# sig = inspect.signature(subsch.mod.forward)
# concrete_args = {
#     p.name: p.default for p in sig.parameters.values() if p.name not in input_names
# }
# subsch.trace(
#     recursive=False,
#     flatten=True,
#     leaf_modules=["BertAttention"],
#     tracer="pytorch",
#     concrete_args=concrete_args,
# )
# print(subsch.mod.graph)

# from solver import Solver

# sol = Solver(subsch.mod, p=8)
# sol.solve([torch.randn(8, 512, 1024)])

import slapo
import torch.distributed as dist
import sys
print(sys.argv)
p = 4
dist.init_process_group("nccl", world_size=p)
torch.cuda.set_device(dist.get_rank())
device = f"cuda:{dist.get_rank()}"
# model.to(device)

sch = slapo.create_schedule(model)
input_names = ["hidden_states"]

# def reshard_RS_to_SR(_module, _input, output):
#     in_tensor = output
#     in_shape = in_tensor.shape
#     chunk_shape = list(in_shape[:-2]) + [in_shape[-2] // sch.world_size, in_shape[-1]]
#     splitted_tensor = torch.split(in_tensor, in_shape[-2] // sch.world_size, dim=-2)
#     for i in range(dist.get_world_size()):
#         send_tensor = splitted_tensor[i].contiguous()
#         if dist.get_rank() != i:
#             dist.gather(send_tensor, dst=i, async_op=True)
#         else:
#             gather_list = [
#                 torch.empty(chunk_shape, dtype=in_tensor.dtype, device=in_tensor.device)
#                 for _ in range(dist.get_world_size())
#             ]
#             dist.gather(send_tensor, gather_list, dst=i)
#             ret = torch.cat(gather_list, dim=-1)
#     return ret

def reshard_RS_to_SR(_module, _input, output):
    # beg = time.time()
    # in_tensor = output
    # in_shape = in_tensor.shape
    # chunk_shape = list(in_shape[:-2]) + [in_shape[-2] // sch.world_size, in_shape[-1]]
    # splitted_tensor = torch.split(in_tensor, in_shape[-2] // sch.world_size, dim=-2)
    # comm_start = time.time()
    # for i in range(dist.get_world_size()):
    #     in_time = time.time()
    #     send_tensor = splitted_tensor[i].contiguous()
    #     if dist.get_rank() != i:
    #         dist.gather(send_tensor, dst=i, async_op=True)  # send
    #     else:
    #         gather_list = [
    #             torch.empty(chunk_shape, dtype=in_tensor.dtype, device=in_tensor.device)
    #             for _ in range(dist.get_world_size())
    #         ]
    #         handle = dist.gather(send_tensor, gather_list, dst=i, async_op=True)  # recv
    #     if dist.get_rank() == 0:
    #         print(f"in_comm{i}: {time.time() - in_time}")
    # if dist.get_rank() == 0:
    #     print(f"comm0: {time.time() - comm_start}")
    # handle.wait()
    # if dist.get_rank() == 0:
    #     print(f"comm1: {time.time() - comm_start}")
    # ret = torch.cat(gather_list, dim=-1)
    # if dist.get_rank() == 0:
    #     print(f"shard: {time.time() - beg}")
    in_tensor = output
    temp = in_tensor
    # temp = temp.contiguous()
    gather_shape = list(temp.shape)
    gather_shape[-1] = dist.get_world_size() * gather_shape[-1]
    ret = torch.empty(gather_shape, dtype=temp.dtype, device=in_tensor.device)
    dist.all_gather_into_tensor(ret, temp)
    # ret = ret.contiguous()
    ret = ret.split(ret.shape[-2] // dist.get_world_size(), dim=-2)[dist.get_rank()]
    return ret

def reshard_SR_to_RR(_module, _input, output):
    in_tensor = output
    temp = in_tensor.transpose(0, -2)
    temp = temp.contiguous()
    gather_shape = list(temp.shape)
    gather_shape[0] = dist.get_world_size() * gather_shape[0]
    ret = torch.empty(gather_shape, dtype=temp.dtype, device=in_tensor.device)
    dist.all_gather_into_tensor(ret, temp)
    ret = ret.transpose(0, -2).contiguous()
    return ret


print(sch.mod)
# for i in range(12):
#     subsch = sch[f"bert.encoder.layer.{i}"]
#     subsch["intermediate.dense"].shard("weight", axis=0)
#     subsch["intermediate.dense"].shard("bias", axis=0)
#     subsch["output.dense"].shard("weight", axis=1)
#     subsch["output.dense"].sync("fwd_post", sync_op_or_fn="all_reduce")
    # subsch["intermediate.dense"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_SR)
    # subsch["output.dense"].sync("fwd_post", sync_op_or_fn=reshard_SR_to_RR)

bs = 8 * p
input_ids = torch.ones(bs, 512, dtype=torch.long, device=device)
attention_mask = torch.ones(bs, 512, dtype=torch.float16, device=device)
token_type_ids = torch.ones(bs, 512, dtype=torch.long, device=device)
model, _ = slapo.build(sch, init_weights=model._init_weights)
model.to(device)

import time
start_time = time.time()
iteration = 10
for i in range(iteration):
    model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
end_time = time.time()
exec_time = (end_time - start_time) / iteration
print(f"Time: {exec_time}s")
print(f"Thoughput: {1 / (exec_time / iteration) * bs} samples/sec")