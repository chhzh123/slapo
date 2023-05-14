# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

'''
Implementation of different resharding schemes on MLP. 
Verified by different combinations of resharding schemes. 
'''

import copy
import time
import torch
from torch import fx
import torch.nn as nn
import torch.nn.functional as F
import slapo
import torch.distributed as dist

NUM_PROC = 4

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

dist.init_process_group("nccl", world_size=NUM_PROC)

# check how many GPUs are available
if dist.get_rank() == 0:
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

torch.cuda.set_device(dist.get_rank())
device = f"cuda:{dist.get_rank()}"




def reshard_RS_to_SR(_module, _input, output):
    """Reshard from RS to SR
    """

    in_tensor = output
    in_shape = in_tensor.shape
    chunk_shape = list(in_shape[:-2]) + [in_shape[-2] // dist.get_world_size(), in_shape[-1]] # [8, 256, 2048]

    splitted_tensor = torch.split(in_tensor, in_shape[-2] // dist.get_world_size(), dim=-2)

    for i in range(dist.get_world_size()):
        send_tensor = splitted_tensor[i].contiguous()

        if dist.get_rank() != i:
            dist.gather(send_tensor, dst=i, async_op=True)  # send
        else:
            gather_list = [
                torch.empty(chunk_shape, dtype=in_tensor.dtype, device=in_tensor.device)
                for _ in range(dist.get_world_size())
            ]
            handle = dist.gather(send_tensor, gather_list, dst=i, async_op=True)  # recv
    handle.wait()

    ret = torch.cat(gather_list, dim=-1)
    return ret

def reshard_SR_to_RS(_module, _input, output):
    """Reshard from SR to RS
    """
    in_tensor = output
    in_shape = in_tensor.shape # [8, 256, 4096]
    # chunk shape = [8, 256, 4096 // p]
    chunk_shape = list(in_shape[:-1]) + [in_shape[-1] // dist.get_world_size()] # [8, 256, 2048]
    
    splitted_tensor = torch.split(in_tensor, in_shape[-1] // dist.get_world_size(), dim=-1) # [8, 256, 2048]
    
    for i in range(dist.get_world_size()):
        send_tensor = splitted_tensor[i].contiguous()

        if dist.get_rank() != i:
            dist.gather(send_tensor, dst=i, async_op=True)  # send
        else:
            gather_list = [
                torch.empty(chunk_shape, dtype=in_tensor.dtype, device=in_tensor.device)
                for _ in range(dist.get_world_size())
            ]
            handle = dist.gather(send_tensor, gather_list, dst=i, async_op=True)
    handle.wait()

    ret = torch.cat(gather_list, dim=-2) # [8, 512, 2048]
    return ret

def reshard_RS_to_SR_to_RS(_module, _input, output):
    in_tensor = output
    sr_tensor = reshard_RS_to_SR(_, _, in_tensor)
    rs_tensor = reshard_SR_to_RS(_, _, sr_tensor)
    return rs_tensor


def reshard_SR_to_RR(_module, _input, output):
    """Reshard from SR to RR
    """
    in_tensor = output # [8, 256, 1024]

    temp = in_tensor.transpose(0, -2) # [256, 8, 1024]
    temp = temp.contiguous()
    gather_shape = list(temp.shape) # [256, 8, 1024]

    gather_shape[0] = dist.get_world_size() * gather_shape[0] # [512, 8, 1024]

    ret = torch.empty(gather_shape, dtype=temp.dtype, device=in_tensor.device)
    dist.all_gather_into_tensor(ret, temp) # [512, 8, 1024]

    ret = ret.transpose(0, -2).contiguous() # [8, 512 1024]
    return ret

def reshard_RS_to_RR(_module, _input, output):
    """Reshard from RS to RR
    """
    # [8, 512, 1024] -> [8, 512, 2048]
    in_tensor = output
    temp = in_tensor.transpose(0, -1).contiguous() # [1024, 512, 8]
    gather_shape = list(temp.shape)
    
    gather_shape[0] = dist.get_world_size() * gather_shape[0] # [2048, 512, 8]

    ret = torch.empty(gather_shape, dtype=temp.dtype, device=in_tensor.device)
    dist.all_gather_into_tensor(ret, temp) # [2048, 512, 8]
    ret = ret.transpose(0, -1).contiguous() # [8, 512, 2048]
    return ret

def reshard_RR_to_RS(_module, _input, output):
    """Reshard from RR to RS
    """
    # [8, 512, 2048] -> [8, 512, 1024]
    in_tensor = output
    # get the current rank's tensor. Slice across the last dimension
    shard_dim_size = in_tensor.shape[-1] // dist.get_world_size()
    start_idx = (int)(dist.get_rank() * shard_dim_size)
    end_idx = (int)((dist.get_rank() + 1) * shard_dim_size)

    slices = [slice(None)] * len(in_tensor.shape)
    slices[-1] = slice(start_idx, end_idx)

    # Slice the tensor
    ret = in_tensor[slices]
    return ret

def reshard_RR_to_RS_pre(_module, input):
    """Reshard from RR to RS
    """
    # [8, 512, 2048] -> [8, 512, 1024]
    in_tensor = input[0]
    # get the current rank's tensor. Slice across the last dimension
    shard_dim_size = in_tensor.shape[-1] // dist.get_world_size()
    start_idx = (int)(dist.get_rank() * shard_dim_size)
    end_idx = (int)((dist.get_rank() + 1) * shard_dim_size)

    slices = [slice(None)] * len(in_tensor.shape)
    slices[-1] = slice(start_idx, end_idx)

    # Slice the tensor
    ret = in_tensor[slices]
    return ret


def reshard_RR_to_SR(_module, _input, output):
    """Reshard from RR to SR
    """
    # [8, 512, 2048] -> [8, 256, 2048]
    in_tensor = output
    # get the current rank's tensor. Slice across the 2nd last dimension
    shard_dim_size = in_tensor.shape[-2] // dist.get_world_size()
    start_idx = (int)(dist.get_rank() * shard_dim_size)
    end_idx = (int)((dist.get_rank() + 1) * shard_dim_size)

    slices = [slice(None)] * len(in_tensor.shape)
    slices[-2] = slice(start_idx, end_idx)

    # Slice the tensor
    ret = in_tensor[slices]
    return ret

def reshard_RR_to_SR_pre(_module, input):
    """Reshard from RR to SR
    """
    # [8, 512, 2048] -> [8, 256, 2048]
    in_tensor = input[0]
    # get the current rank's tensor. Slice across the 2nd last dimension
    shard_dim_size = in_tensor.shape[-2] // dist.get_world_size()
    start_idx = (int)(dist.get_rank() * shard_dim_size)
    end_idx = (int)((dist.get_rank() + 1) * shard_dim_size)

    slices = [slice(None)] * len(in_tensor.shape)
    slices[-2] = slice(start_idx, end_idx)

    # Slice the tensor
    ret = in_tensor[slices]
    return ret

# ========== Verification =========

mlp = MLP()

# 1. Naive. RR * RR -> RR; RR * RR -> RR
if dist.get_rank() == 0:
    print("===== 1. Naive RR =====")
mlp_naive = copy.deepcopy(mlp).to(device=device)
sch = slapo.create_schedule(mlp_naive)
with slapo.Verify(sch, [torch.randn(8, 512, 1024).to(device=device)]):
    # do nothing
    pass
mod_1, _ = slapo.build(sch)

# 2. RR * RS -> RS; RS -> SR (reshard); SR * RR -> SR; SR -> RR (reshard)
if dist.get_rank() == 0:
    print("===== 2. RS -> SR =====")
mlp_2 = copy.deepcopy(mlp).to(device=device)
sch = slapo.create_schedule(mlp_2)
with slapo.Verify(sch, [torch.randn(8, 512, 1024).to(device=device)]):
    sch["fc1"].shard("weight", axis=0)
    sch["fc1"].shard("bias", axis=0)
    sch["fc1"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_SR)
    sch["fc2"].sync("fwd_post", sync_op_or_fn=reshard_SR_to_RR)
mod_2, _ = slapo.build(sch)

# 3. RR * RS -> RS; RS -> SR (reshard); SR -> RS (reshard); RS * SR -> RR (with all reduce)
if dist.get_rank() == 0:
    print("===== 3. SR -> RS =====")
mlp_3 = copy.deepcopy(mlp).to(device=device)
sch = slapo.create_schedule(mlp_3)
with slapo.Verify(sch, [torch.randn(8, 512, 1024).to(device=device)]):
    sch["fc1"].shard("weight", axis=0)
    sch["fc1"].shard("bias", axis=0)
    sch["fc1"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_SR_to_RS)
    sch["fc2"].shard("weight", axis=1)
    sch["fc2"].sync("fwd_post", sync_op_or_fn="all_reduce")
mod_3, _ = slapo.build(sch)

# 4. Megatron. RR * RS -> RS; RS * SR -> RR (with all reduce)
if dist.get_rank() == 0:
    print("===== 4. Megatron =====")
mlp_4 = copy.deepcopy(mlp).to(device=device)
sch = slapo.create_schedule(mlp_4)
with slapo.Verify(sch, [torch.randn(8, 512, 1024).to(device=device)]):
    sch["fc1"].shard("weight", axis=0)
    sch["fc1"].shard("bias", axis=0)
    sch["fc2"].shard("weight", axis=1)
    sch["fc2"].sync("fwd_post", sync_op_or_fn="all_reduce")
mod_4, _ = slapo.build(sch)

# 5. RR * RS -> RS; RS -> RR (reshard); RR * RS -> RS; RS -> RR (reshard)
if dist.get_rank() == 0:
    print("===== 5. RR * RS -> RS =====")
mlp_5 = copy.deepcopy(mlp).to(device=device)
sch = slapo.create_schedule(mlp_5)
with slapo.Verify(sch, [torch.randn(8, 512, 1024).to(device=device)]):
    sch["fc1"].shard("weight", axis=0)
    sch["fc1"].shard("bias", axis=0)
    sch["fc1"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_RR)
    sch["fc2"].shard("weight", axis=0)
    sch["fc2"].shard("bias", axis=0)
    sch["fc2"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_RR)
mod_5, _ = slapo.build(sch)

# 6. RR * RR -> RR; RR -> RS (reshard); RS * SR -> RR (with all reduce)
if dist.get_rank() == 0:
    print("===== 6. RR -> RS =====")
mlp_6 = copy.deepcopy(mlp).to(device=device)
sch = slapo.create_schedule(mlp_6)
with slapo.Verify(sch, [torch.randn(8, 512, 1024).to(device=device)]):
    sch["fc1"].sync("fwd_post", sync_op_or_fn=reshard_RR_to_RS)
    sch["fc2"].shard("weight", axis=1)
    sch["fc2"].sync("fwd_post", sync_op_or_fn="all_reduce")
mod_6, _ = slapo.build(sch)

# 7. RR * RR -> RR; RR -> SR (reshard); SR * RR -> SR; SR -> RR (reshard)
if dist.get_rank() == 0:
    print("===== 7. RR -> SR =====")
mlp_7 = copy.deepcopy(mlp).to(device=device)
sch = slapo.create_schedule(mlp_7)
with slapo.Verify(sch, [torch.randn(8, 512, 1024).to(device=device)]):
    sch["fc1"].sync("fwd_post", sync_op_or_fn=reshard_RR_to_SR)
    sch["fc2"].sync("fwd_post", sync_op_or_fn=reshard_SR_to_RR)
mod_7, _ = slapo.build(sch)

# 8. RR -> SR (reshard); SR * RR -> SR; SR * RR -> SR; SR -> RR (reshard)
if dist.get_rank() == 0:
    print("===== 8. RR -> RS =====")
mlp_8 = copy.deepcopy(mlp).to(device=device)
sch = slapo.create_schedule(mlp_8)
with slapo.Verify(sch, [torch.randn(8, 512, 1024).to(device=device)]):
    sch["fc1"].sync("fwd_pre", sync_op_or_fn=reshard_RR_to_SR_pre)
    sch["fc2"].sync("fwd_post", sync_op_or_fn=reshard_SR_to_RR)
mod_8, _ = slapo.build(sch)



# =============== Performance ==============
TIMES = 10
BS = 1024

# Mod 1: Naive
start = time.time()
for _ in range(TIMES):
    mod_1(torch.randn(BS, 512, 1024).to(device=device))
end = time.time()
if dist.get_rank() == 0:
    print("Mod 1: ", end - start)

# Mod 2: Ours
start = time.time()
for _ in range(TIMES):
    mod_2(torch.randn(BS, 512, 1024).to(device=device))
end = time.time()
if dist.get_rank() == 0:
    print("Mod 2: ", end - start)

# Mod 3
start = time.time()
for _ in range(TIMES):
    mod_3(torch.randn(BS, 512, 1024).to(device=device))
end = time.time()
if dist.get_rank() == 0:
    print("Mod 3: ", end - start)

# Mod 4: Slapo-Megatron
start = time.time()
for _ in range(TIMES):
    mod_4(torch.randn(BS, 512, 1024).to(device=device))
end = time.time()
if dist.get_rank() == 0:
    print("Mod 4: ", end - start)
# [BS, 512, 1024] (RR) * [1024, 2048] (RS) -> [BS, 512, 2048] (RS);
# [BS, 512, 2048] (RS) * [2048, 1024] (SR) -> [BS, 512, 1024] (RR) (All reduce)

# Mod 5: Weight Parallelism
start = time.time()
for _ in range(TIMES):
    mod_5(torch.randn(BS, 512, 1024).to(device=device))
end = time.time()
if dist.get_rank() == 0:
    print("Mod 5: ", end - start)

# Mod 6
start = time.time()
for _ in range(TIMES):
    mod_6(torch.randn(BS, 512, 1024).to(device=device))
end = time.time()
if dist.get_rank() == 0:
    print("Mod 6: ", end - start)

# Mod 7
start = time.time()
for _ in range(TIMES):
    mod_7(torch.randn(BS, 512, 1024).to(device=device))
end = time.time()
if dist.get_rank() == 0:
    print("Mod 7: ", end - start)

# Mod 8: Data Parallelism
start = time.time()
for _ in range(TIMES):
    mod_8(torch.randn(BS, 512, 1024).to(device=device))
end = time.time()
if dist.get_rank() == 0:
    print("Mod 8: ", end - start)