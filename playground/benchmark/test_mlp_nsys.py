import os
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput
from transformers import AutoConfig
import torch.distributed as dist
from utils import perf_model

os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
dist.init_process_group("nccl", world_size=int(os.environ["WORLD_SIZE"]))
torch.cuda.set_device(dist.get_rank())
config = AutoConfig.from_pretrained("bert-large-uncased")
world_size = dist.get_world_size()


# class BertMLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.intermediate = nn.Linear(
#             config.hidden_size, config.intermediate_size // world_size
#         )
#         self.output = nn.Linear(
#             config.intermediate_size // world_size, config.hidden_size
#         )

#     def forward(self, x):
#         x = self.intermediate(x)
#         x = self.output(x)
#         dist.all_reduce(x)
#         return x

class BertMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, inp):
        x = self.intermediate(inp)
        x = self.output(x, inp)
        return x

mlp = BertMLP().eval()

import slapo
sch = slapo.create_schedule(mlp)
sch["intermediate.dense"].shard("weight", axis=0)
sch["intermediate.dense"].shard("bias", axis=0)
sch["output.dense"].shard("weight", axis=1)
sch["output.dense"].sync("fwd_post", "all_reduce")

from bert_schedule import fuse_gemm_gelu, fuse_ln_residual
fuse_gemm_gelu(sch["intermediate"], "dense")
fuse_ln_residual(
    sch["output"],
    names=["dense", "LayerNorm"],
)

mlp, _ = slapo.build(sch, init_weights=False)

inp = torch.randn(1, 512, 1024, dtype=torch.float16, device=f"cuda:{sch.rank}")
# perf_model(mlp, inp, True, nsys=False)
# mlp = torch.cuda.make_graphed_callables(mlp.to(torch.float16).cuda(sch.rank).eval(), (torch.randn((1, 512, 1024), device=f"cuda:{sch.rank}", dtype=torch.float16, requires_grad=False),), allow_unused_input=True)
from torch._inductor.compile_fx import cudagraphify_impl
inputs = [inp]
mlp.to(torch.float16).cuda(sch.rank).eval()
mlp(*inputs)
mlp = cudagraphify_impl(model=lambda args: mlp(*args), inputs=inputs)
new_inputs = [torch.randn(1, 512, 1024, dtype=torch.float16, device=f"cuda:{sch.rank}")]
perf_model(mlp, new_inputs, False, nsys=True)
