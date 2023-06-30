import os
import torch
import torch.distributed as dist
from kernl.model_optimization import optimize_model

from utils import get_model

dist.init_process_group("nccl", world_size=int(os.environ["WORLD_SIZE"]))
torch.cuda.set_device(dist.get_rank())
model, config, seq_len = get_model("bert")
optimize_model(model)

bs = 1
input_ids = torch.ones((bs, seq_len), dtype=torch.long, device="cuda")

with torch.inference_mode():
    outputs = model(input_ids)