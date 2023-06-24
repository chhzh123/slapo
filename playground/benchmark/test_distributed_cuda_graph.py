# https://github.com/bottler/pytorch/blob/1bacd4c58bf06765498cfc497f8b9ab76f57c4ad/test/distributed/test_c10d_nccl.py

import os
import torch
import torch.distributed as dist
from utils import perf_model


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1024, 512, bias=False)
        self.linear2 = torch.nn.Linear(512, 1024, bias=False)
        self.linear3 = torch.nn.Linear(1024, 512, bias=False)
        self.linear4 = torch.nn.Linear(512, 1024, bias=False)

    def forward(self, data):
        out = self.linear1(data)
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.linear4(out)
        # pg.allreduce(out)
        # dist.all_reduce(out, group=pg)
        return out

dist.init_process_group("nccl", world_size=int(os.environ["WORLD_SIZE"]))
pg = dist.distributed_c10d._get_default_group()
torch.cuda.set_device(dist.get_rank())
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
model = Model()
import slapo
sch = slapo.create_schedule(model)
sch["linear2"].sync("fwd_post", "all_reduce")
sch["linear4"].sync("fwd_post", "all_reduce")
# def hook_fn(_module, _input, output):
#     dist.all_reduce(output, op=dist.ReduceOp.SUM)
#     return output
# model.linear2.register_forward_hook(hook_fn)
model, _ = slapo.build(sch, init_weights=False)
model.cuda().to(torch.float16)
x = torch.randn(1024, 1024, dtype=torch.float16).cuda()
out = model(x)
perf_model(model, x, use_cuda_graph=True)
