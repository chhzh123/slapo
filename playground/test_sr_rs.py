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
    x = torch.tensor([[[1, 2, 3]],
                      [[1, 2, 3]]]).to(device)
elif dist.get_rank() == 1:
    x = torch.tensor([[[4, 5, 6]],
                      [[4, 5, 6]]]).to(device)
else:
    x = torch.tensor([[[7, 8, 9]],
                      [[7, 8, 9]]]).to(device)

in_shape = x.shape # 3*(1,3) -> (3,1)*3
# (2,1,3) -> (2,3,1)
# print(in_shape, x.chunk(3, dim=0))

# SR to RS
def reshard_SR_to_RS(in_tensor, group=None):
    in_shape = in_tensor.shape
    chunk_shape = list(in_shape[:-1]) + [in_shape[-1] // dist.get_world_size(group)]

    splitted_tensor = torch.split(
        in_tensor, in_shape[-1] // dist.get_world_size(group), dim=-1
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

    ret = torch.cat(gather_list, dim=-2)
    return ret


def reshard_SR_to_RS_all_to_all(in_tensor, group=None):
    # (bs, seq/p, hs) => (bs, seq, hs/p)
    world_size = dist.get_world_size(group)
    # Since all_to_all can only chunk the 0th dimension, we need to permute the tensor
    # to make the 0th dimension the one we want to send data.
    dims = list(range(len(in_tensor.shape)))
    dims = [-1] + dims[:-1]
    # (hs, bs, seq/p)
    in_tensor = in_tensor.permute(dims).contiguous()
    in_shape = in_tensor.shape
    in_tensor_lst = list(in_tensor.chunk(world_size))
    out_tensor_lst = list(
        torch.empty(in_shape, dtype=in_tensor.dtype, device=in_tensor.device).chunk(
            world_size
        )
    )
    # (p, hs/p, bs, seq/p)
    dist.all_to_all(out_tensor_lst, in_tensor_lst, group=group)
    # (hs/p, bs, seq)
    output = torch.cat(out_tensor_lst, dim=-1)
    dims = list(range(1, len(in_tensor.shape))) + [0]
    # Permute back to the original layout
    # (bs, seq, hs/p)
    output = output.permute(dims)
    return output

# start = time.time()
# out = reshard_RS_to_SR(x)
# if dist.get_rank() == 0:
#     print(f"Time 1:{time.time() - start:.3f}s")
# print(out)

# start = time.time()
# out = reshard_RS_to_SR_2(x)
# if dist.get_rank() == 0:
#     print(f"Time 2:{time.time() - start:.3f}s")
# print(out)

start = time.time()
out = reshard_SR_to_RS(x)
if dist.get_rank() == 0:
    print(f"Time 1:{time.time() - start:.8f}s")
print(out)

start = time.time()
out = reshard_SR_to_RS_all_to_all(x)
if dist.get_rank() == 0:
    print(f"Time 2:{time.time() - start:.8f}s")
print(out)
