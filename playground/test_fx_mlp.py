# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import time
import torch
from torch import fx
import torch.nn as nn
import torch.nn.functional as F
import slapo
import torch.distributed as dist


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 1024)

    def forward(self, x):
        beg = time.time()
        x = self.fc1(x)
        if dist.get_rank() == 0:
            print(f"fc1: {time.time() - beg}")
        x = F.relu(x)
        beg = time.time()
        x = self.fc2(x)
        if dist.get_rank() == 0:
            print(f"fc2: {time.time() - beg}")
        return x


# from solver import Solver

# sol = Solver(fx.symbolic_trace(MLP()), p=2)
# sol.solve([torch.randn(512, 1024)])

dist.init_process_group("nccl", world_size=8)
torch.cuda.set_device(dist.get_rank())
device = f"cuda:{dist.get_rank()}"


def reshard_RS_to_SR(_module, _input, output):
    beg = time.time()
    in_tensor = output
    in_shape = in_tensor.shape
    chunk_shape = list(in_shape[:-2]) + [in_shape[-2] // sch.world_size, in_shape[-1]]
    splitted_tensor = torch.split(in_tensor, in_shape[-2] // sch.world_size, dim=-2)
    comm_start = time.time()
    for i in range(dist.get_world_size()):
        in_time = time.time()
        send_tensor = splitted_tensor[i].contiguous()
        if dist.get_rank() != i:
            dist.gather(send_tensor, dst=i, async_op=True)  # send
        else:
            gather_list = [
                torch.empty(chunk_shape, dtype=in_tensor.dtype, device=in_tensor.device)
                for _ in range(dist.get_world_size())
            ]
            handle = dist.gather(send_tensor, gather_list, dst=i, async_op=True)  # recv
        if dist.get_rank() == 0:
            print(f"in_comm{i}: {time.time() - in_time}")
    if dist.get_rank() == 0:
        print(f"comm0: {time.time() - comm_start}")
    handle.wait()
    if dist.get_rank() == 0:
        print(f"comm1: {time.time() - comm_start}")
    ret = torch.cat(gather_list, dim=-1)
    if dist.get_rank() == 0:
        print(f"shard: {time.time() - beg}")
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


mlp = MLP()
mlp_reshard = copy.deepcopy(mlp).to(device=device)
sch = slapo.create_schedule(mlp_reshard)
print(sch.mod)
sch["fc1"].shard("weight", axis=0)
sch["fc1"].shard("bias", axis=0)
# sch["fc2"].shard("weight", axis=1)
# sch["fc2"].sync("fwd_post", sync_op_or_fn="all_reduce")
sch["fc1"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_SR)
sch["fc2"].sync("fwd_post", sync_op_or_fn=reshard_SR_to_RR)
mod_reshard, _ = slapo.build(sch)

mlp_mega = copy.deepcopy(mlp).to(device=device)
sch = slapo.create_schedule(mlp_mega)
print(sch.mod)
sch["fc1"].shard("weight", axis=0)
sch["fc1"].shard("bias", axis=0)
sch["fc2"].shard("weight", axis=1)
sch["fc2"].sync("fwd_post", sync_op_or_fn="all_reduce")
mod_mega, _ = slapo.build(sch)

start = time.time()
for _ in range(100):
    mod_reshard(torch.randn(8, 512, 1024).to(device=device))
end = time.time()
if dist.get_rank() == 0:
    print("reshard", end - start)

start = time.time()
for _ in range(100):
    mod_mega(torch.randn(8, 512, 1024).to(device=device))
end = time.time()
if dist.get_rank() == 0:
    print("mega", end - start)
