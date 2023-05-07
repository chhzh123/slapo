# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import torch
import torch.distributed as dist

dist.init_process_group("nccl", world_size=3)
torch.cuda.set_device(dist.get_rank())
device = f"cuda:{dist.get_rank()}"

p = dist.get_world_size()

if dist.get_rank() == 0:
    x = torch.tensor([[1], [4], [7]]).to(device)
elif dist.get_rank() == 1:
    x = torch.tensor([[2], [5], [8]]).to(device)
else:
    x = torch.tensor([[3], [6], [9]]).to(device)
in_shape = x.shape
print(x, in_shape)

def reshard_RS_to_SR(output):
    in_tensor = output
    in_shape = in_tensor.shape
    chunk_shape = list(in_shape[:-2]) + [in_shape[-2] // p, in_shape[-1]]
    splitted_tensor = torch.split(in_tensor, in_shape[-2] // p, dim=-2)
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

print(reshard_RS_to_SR(x))