from transformers import BertLMHeadModel, AutoConfig
config = AutoConfig.from_pretrained("bert-large-uncased")
model = BertLMHeadModel(config)

import slapo
from slapo.pattern import call_module
import torch.nn.functional as F

sch = slapo.create_schedule(model)

# Shard embeddings
sch["embeddings.word_embeddings"].sync(mode="fwd_pre", sync_op_or_fn=slapo.op.embed_fwd_hook)
sch["embeddings.word_embeddings"].sync(mode="fwd_post", sync_op_or_fn=slapo.op.embed_bwd_hook)
for idx in range(config.num_hidden_layers):
    # Shard self attention module
    subsch = sch[f"encoder.layer.{idx}.attention"]
    subsch["self.query"].shard(["weight", "bias"], axis=0)
    subsch["self.key"].shard(["weight", "bias"], axis=0)
    subsch["self.value"].shard(["weight", "bias"], axis=0)
    subsch.sync(mode="bwd_post", sync_op_or_fn="all_reduce")
    subsch["output.dense"].shard("weight", axis=1)
    subsch["output.dense"].sync("fwd_post", sync_op_or_fn="all_reduce")
    # Shard MLP module
    subsch = sch[f"encoder.layer.{idx}"]
    subsch["intermediate.dense"].shard(["weight", "bias"], axis=0)
    subsch["intermediate.dense"].sync("bwd_post", sync_op_or_fn="all_reduce")
    subsch["output.dense"].shard("weight", axis=1)
    subsch["output.dense"].sync("fwd_post", sync_op_or_fn="all_reduce")
    # Decompose linear bias and trace module
    subsch["attention.output.dense"].decompose()
    subsch["output.dense"].decompose()
    subsch.trace(tracer="huggingface", flatten=True)
    # Replace scaled dot product attention
    subgraphs = subsch.find(slapo.pattern.scaled_dot_product)
    subsch.replace(F.scaled_dot_product_attention, subgraphs)
    # Fuse linear bias and gelu
    subgraph = subsch.find(lambda x, bias: F.gelu(bias + x))
    subsch.fuse(subgraph, compiler="TorchInductor", name="BiasGeLU")
    # Fuse bias add, layer norm, and residual
    for ln in ["attention.output.LayerNorm", "output.LayerNorm"]:
        subgraph = subsch.find(lambda x, bias, residual: call_module(ln, F.dropout(bias + x) + residual))
        subsch.fuse(subgraph, compiler="TorchInductor", name="LNResidual")
