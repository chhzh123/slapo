# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import torch
import torch.distributed as dist

dist.init_process_group("nccl", world_size=3)
torch.cuda.set_device(dist.get_rank())
device = f"cuda:{dist.get_rank()}"

p = dist.get_world_size()

# if dist.get_rank() == 0:
#     x = torch.tensor([[1, 2]]).to(device)
# else:
#     x = torch.tensor([[3, 4]]).to(device)

if dist.get_rank() == 0:
    x = torch.tensor([[1, 2, 3]]).to(device)
elif dist.get_rank() == 1:
    x = torch.tensor([[4, 5, 6]]).to(device)
else:
    x = torch.tensor([[7, 8, 9]]).to(device)

in_shape = x.shape
print(x, in_shape)

# 1st implementation
st = time.time()
for i in range(dist.get_world_size()):
    send_tensor = torch.split(x, in_shape[-1] // p, dim=-1)[i].contiguous()
    ret = torch.empty((p, 1), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(ret, send_tensor)
    if i == dist.get_rank():
        print(ret)
if dist.get_rank() == 0:
    print("1st", time.time() - st)

# 2nd implementation
st = time.time()
for i in range(dist.get_world_size()):
    send_tensor = torch.split(x, in_shape[-1] // p, dim=-1)[i].contiguous()
    if dist.get_rank() != i:
        dist.gather(send_tensor, dst=i, async_op=False)
    else:
        gather_list = [
            torch.empty((1, 1), dtype=x.dtype, device=x.device)
            for _ in range(dist.get_world_size())
        ]
        dist.gather(send_tensor, gather_list, dst=i)
        ret = torch.cat(gather_list, dim=0)
        print(ret)
if dist.get_rank() == 0:
    print("2nd", time.time() - st)

# 3rd implementation (async)
st = time.time()
splitted_tensor = torch.split(x, in_shape[-1] // p, dim=-1)
for i in range(dist.get_world_size()):
    send_tensor = splitted_tensor[i].contiguous()
    if dist.get_rank() != i:
        dist.gather(send_tensor, dst=i, async_op=True)
gather_list = [
    torch.empty((1, 1), dtype=x.dtype, device=x.device)
    for _ in range(dist.get_world_size())
]
dist.gather(splitted_tensor[dist.get_rank()], gather_list, dst=dist.get_rank())
ret = torch.cat(gather_list, dim=0)
print(ret)
if dist.get_rank() == 0:
    print("3rd", time.time() - st)

# test SR->RR
outshape = (3, 3)
ret = torch.empty(outshape, dtype=x.dtype, device=x.device)
dist.all_gather_into_tensor(ret, x)
print(ret)
