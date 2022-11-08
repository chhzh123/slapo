import time
import inspect
from transformers import BertLMHeadModel, AutoConfig
import numpy as np
import torch
import ms
from bert_schedule import replace_layernorm, replace_gelu, replace_xformer_attention, replace_qkv

# https://huggingface.co/bert-large-uncased/blob/main/config.json
bert_config = AutoConfig.from_pretrained("bert-large-uncased")
bert = BertLMHeadModel(bert_config)
optimizer = torch.optim.AdamW(bert.parameters(), lr=0.001)
bert.half()

input_names = list(bert.dummy_inputs.keys())
input_names += ["attention_mask", "labels"]
sig = inspect.signature(bert.forward)
concrete_args = {
    p.name: p.default for p in sig.parameters.values() if p.name not in input_names
}

sch = ms.create_schedule(
    bert,
    optimizer,
    tracer="huggingface",
    leaf_modules=["BertSelfAttention"],
    concrete_args=concrete_args,
)
replace_xformer_attention(sch, bert_config)

# sch = ms.create_schedule(
#     bert, optimizer, tracer="huggingface", concrete_args=concrete_args
# )
# # replace_layernorm(sch)
# # replace_gelu(sch)
# replace_qkv(sch, bert_config)
# print(gm.graph)

device = "cuda:0"
model, optimizer = ms.build(sch)
model.half()
model.to(device)

bs = 8
seq_length = 512
bert_input_dict = {
    "input_ids": torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(
        bert.config.vocab_size
    ),
    "labels": torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(
        bert.config.vocab_size
    ),
    "attention_mask": torch.ones(bs, seq_length, device=device),
}

fw_time = []
bw_time = []
total_time = []
for i in range(10):
    start_time = time.time()
    output = model(
        bert_input_dict["input_ids"],
        bert_input_dict["attention_mask"],
        bert_input_dict["labels"],
    )
    mid_time = time.time()
    output["logits"].mean().backward()
    final_time = time.time()
    optimizer.step()
    fw_time.append(mid_time - start_time)
    bw_time.append(final_time - mid_time)
    total_time.append(final_time - start_time)
    print(
        f"Finish step {i}, fw: {fw_time[-1]:.10f}s, bw: {bw_time[-1]:.10f}s, total: {total_time[-1]:.10f}s"
    )
fw_avg = np.array(fw_time[1:-1]).mean()
bw_avg = np.array(bw_time[1:-1]).mean()
total_avg = np.array(total_time[1:-1]).mean()
print(
    f"Average fw: {fw_avg*1000:.10f}ms, bw: {bw_avg*1000:.10f}ms, total: {total_avg*1000:.10f}ms"
)

# ms.profile(model, [bert_input_dict["input_ids"], bert_input_dict["attention_mask"], bert_input_dict["labels"]])
