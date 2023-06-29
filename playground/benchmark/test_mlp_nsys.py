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
        config.intermediate_size = config.intermediate_size // world_size
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, inp):
        x = self.intermediate(inp)
        x = self.output(x, inp)
        dist.all_reduce(x)
        return x

mlp = BertMLP().eval()
inp = torch.randn(1, 512, 1024, dtype=torch.float16, device="cuda")
perf_model(mlp, inp, False, nsys=False)
