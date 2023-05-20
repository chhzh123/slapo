# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os, sys
import time
import torch
import torch.distributed as dist

dist.init_process_group("nccl", world_size=int(os.environ['WORLD_SIZE']))
torch.cuda.set_device(dist.get_rank())
device = f"cuda:{dist.get_rank()}"

p = dist.get_world_size()

if dist.get_rank() == 0:
    x = torch.tensor([[[0],[1],[2],[3]],
                      [[4],[5],[6],[7]]]).to(device)
elif dist.get_rank() == 1:
    x = torch.tensor([[[8],[9],[10],[11]],
                      [[12],[13],[14],[15]]]).to(device)
in_shape = x.shape #(2,4)
# print(in_shape, x.transpose(-1, -2).chunk(2))


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

def reshard_RS_to_SR_2(in_tensor, group=None):
    # TODO: Need a more efficient implementation
    in_shape = in_tensor.shape
    chunk_shape = list(in_shape[:-2]) + [
        in_shape[-2] // dist.get_world_size(group),
        in_shape[-1],
    ]

    splitted_tensor = torch.split(
        in_tensor, in_shape[-2] // dist.get_world_size(group), dim=-2
    )

    for i in range(dist.get_world_size(group)):
        send_tensor = splitted_tensor[i].contiguous()

        if dist.get_rank() != i:
            dist.gather(send_tensor, dst=i, async_op=True, group=group)  # send
        else:
            gather_list = [
                torch.empty(chunk_shape, dtype=in_tensor.dtype, device=in_tensor.device)
                for _ in range(dist.get_world_size(group))
            ]
            handle = dist.gather(
                send_tensor, gather_list, dst=i, async_op=True, group=group
            )  # recv
    handle.wait()

    ret = torch.cat(gather_list, dim=-1)
    return ret


def reshard_RS_to_SR_3(in_tensor, group=None):
    # (bs, seq, hs/p) => (bs, seq/p, hs)
    world_size = dist.get_world_size(group)
    # Since all_to_all can only chunk the 0th dimension, we need to permute the tensor
    # to make the 0th dimension the one we want to send data.
    dims = list(range(len(in_tensor.shape)))
    dims = [-2] + dims[:-2] + [-1]
    # (seq, bs, hs/p)
    in_tensor = in_tensor.permute(dims).contiguous()
    in_shape = in_tensor.shape
    in_tensor_lst = list(in_tensor.chunk(world_size))
    out_tensor_lst = list(torch.empty(in_shape, dtype=in_tensor.dtype, device=in_tensor.device).chunk(world_size))
    # (p, seq/p, bs, hs/p)
    dist.all_to_all(out_tensor_lst, in_tensor_lst, group=group)
    # (seq/p, bs, hs)
    output = torch.cat(out_tensor_lst, dim=-1)
    dims = list(range(1, len(in_tensor.shape) - 1)) + [0, -1]
    # Permute back to the original layout
    # (bs, seq/p, hs)
    output = output.permute(dims)
    return output

def reshard_RS_to_SR_4(in_tensor, group=None):
    # (bs, seq, hs/p) => (bs, seq/p, hs)
    # Since all_to_all can only chunk the 0th dimension, we need to permute the tensor
    # to make the 0th dimension the one we want to send data.
    dims = list(range(len(in_tensor.shape)))
    dims = [-2] + dims[:-2] + [-1]
    # (seq, bs, hs/p)
    in_tensor = in_tensor.permute(dims).contiguous()
    in_shape = in_tensor.shape
    output = torch.empty(in_shape, dtype=in_tensor.dtype, device=in_tensor.device)
    # (p*seq/p, bs, hs/p)
    dist.all_to_all_single(output, in_tensor, group=group)
    dims = list(range(1, len(in_tensor.shape) - 1)) + [0, -1]
    # Permute back to the original layout
    # (bs, p*seq/p, hs/p)
    output = output.permute(dims).contiguous()
    # (bs, p, seq/p, hs/p)
    out_shape = list(output.shape)
    world_size = dist.get_world_size(group)
    output = output.view(out_shape[:-2] + [world_size, out_shape[-2] // world_size, out_shape[-1]])
    dims = list(range(len(output.shape)))[:-3] + [-2, -3, -1]
    # (bs, seq/p, p, hs/p)
    output = output.permute(dims)
    out_shape = list(output.shape)
    out_shape = list(out_shape[:-2] + [-1])
    # (bs, seq/p, hs)
    output = output.view(out_shape)
    return output

start = time.time()
out = reshard_RS_to_SR(x)
if dist.get_rank() == 0:
    print(f"Time 1:{time.time() - start:.8f}s")
print(out, out.shape)

start = time.time()
out = reshard_RS_to_SR_2(x)
if dist.get_rank() == 0:
    print(f"Time 2:{time.time() - start:.8f}s")
print(out, out.shape)

start = time.time()
out = reshard_RS_to_SR_3(x)
if dist.get_rank() == 0:
    print(f"Time 3:{time.time() - start:.8f}s")
print(out, out.shape)

start = time.time()
out = reshard_RS_to_SR_4(x)
if dist.get_rank() == 0:
    print(f"Time 4:{time.time() - start:.8f}s")
print(out, out.shape)