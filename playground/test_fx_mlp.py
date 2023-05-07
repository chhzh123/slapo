# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import fx
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 1024)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# from solver import Solver

# sol = Solver(fx.symbolic_trace(MLP()), p=2)
# sol.solve([torch.randn(512, 1024)])

import slapo
import torch.distributed as dist

dist.init_process_group("nccl", world_size=2)
torch.cuda.set_device(dist.get_rank())
device = f"cuda:{dist.get_rank()}"


def reshard_RS_to_SR(_module, _input, output):
    in_tensor = output
    in_shape = in_tensor.shape
    chunk_shape = list(in_shape[:-2]) + [in_shape[-2] // sch.world_size, in_shape[-1]]
    splitted_tensor = torch.split(in_tensor, in_shape[-2] // sch.world_size, dim=-2)
    for i in range(dist.get_world_size()):
        send_tensor = splitted_tensor[i].contiguous()
        if dist.get_rank() != i:
            dist.gather(send_tensor, dst=i, async_op=True)
        else:
            gather_list = [
                torch.empty(chunk_shape, dtype=in_tensor.dtype, device=in_tensor.device)
                for _ in range(dist.get_world_size())
            ]
            dist.gather(send_tensor, gather_list, dst=i)
            ret = torch.cat(gather_list, dim=-1)
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


import copy

mlp = MLP().to(device=device)
sch = slapo.create_schedule(copy.deepcopy(mlp))
print(sch.mod)
with slapo.Verify(sch, [torch.randn(8, 512, 1024).to(device=device)]):
    sch["fc1"].shard("weight", axis=0)
    sch["fc1"].shard("bias", axis=0)
    # sch["fc2"].shard("weight", axis=1)
    # sch["fc2"].sync("fwd_post", sync_op_or_fn="all_reduce")
    sch["fc1"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_SR)
    sch["fc2"].sync("fwd_post", sync_op_or_fn=reshard_SR_to_RR)
