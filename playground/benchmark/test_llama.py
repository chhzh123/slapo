# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import math
import copy
import inspect

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import LlamaModel, AutoConfig

import slapo
from slapo.pattern import call_module
from slapo.logger import get_logger
from utils import perf_model

logger = get_logger(__name__)

# Config for verification
bs = 1
seq_len = 2048


def trace_module(sch, config):
    input_names = ["hidden_states", "position_ids"]
    sig = inspect.signature(sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    sch.trace(
        recursive=False,
        flatten=True,
        tracer="huggingface",
        concrete_args=concrete_args,
        leaf_modules=["LlamaRotaryEmbedding"],
        config=config,
    )


def fix_attention_mask_shape_megatron(sch, config):
    # ops = sch.find_node(
    #     lambda node: node.op == "call_method"
    #     and node.target == "view"
    #     and node.args[0].op == "call_module"
    #     and "proj" in node.args[0].target
    # )
    # sch.mod.embed_dim = config.hidden_size
    # sch.mod.num_attention_heads = config.num_attention_heads
    # assert len(ops) == 3  # q,k,v

    # def new_view(tensor, *args):
    #     return tensor.view(args[0], args[1], args[2] // sch.world_size, args[3])

    # for op in ops:
    #     sch.replace(new_view, op)
    reshape_op = sch.find_node(
        lambda node: node.op == "call_method" and node.target == "reshape"
    )
    assert len(reshape_op) == 1

    def new_reshape(tensor, *args):
        return tensor.reshape(args[0], args[1], args[2] // sch.world_size)

    sch.replace(new_reshape, reshape_op[0])


def shard_word_embedding(sch, vocab_size, word_embed_name="embeddings.word_embeddings"):
    if sch.world_size == 1:
        return

    # Embedding
    sch[word_embed_name].shard("weight", axis=0)
    # Build the mask
    vocab_start_index = sch.rank * vocab_size // sch.world_size
    vocab_end_index = (sch.rank + 1) * vocab_size // sch.world_size

    def fwd_pre_hook(_module, _input):
        # Mask the input
        input_mask = (_input[0] < vocab_start_index) | (_input[0] >= vocab_end_index)
        masked_input = _input[0].clone() - vocab_start_index
        masked_input[input_mask] = 0
        return masked_input

    sch[word_embed_name].sync(mode="fwd_pre", sync_op_or_fn=fwd_pre_hook)

    def fwd_post_hook(_module, _input, output):
        # Mask the output embedding. Note that the input is already masked.
        output[_input[0] == 0, :] = 0.0
        # Reduce across all the model parallel GPUs
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=sch.group)
        return output

    sch[word_embed_name].sync(mode="fwd_post", sync_op_or_fn=fwd_post_hook)


class FusedQKV(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        world_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.world_size = world_size
        self.head_dim = embed_dim // num_heads
        self.fused_linear = nn.Linear(
            embed_dim, embed_dim * 3 // self.world_size, bias=bias
        )

    def forward(self, hidden_states: torch.Tensor):
        bsz, tgt_len, _ = hidden_states.size()
        qkv = self.fused_linear(hidden_states)
        reshaped_qkv = (
            qkv.view(bsz, tgt_len, 3 * self.num_heads // self.world_size, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        reshaped_qkv = reshaped_qkv.view(
            bsz, 3, self.num_heads // self.world_size, tgt_len, self.head_dim
        )
        q, k, v = reshaped_qkv.unbind(dim=1)

        query_states = q.contiguous()
        key_states = k.contiguous()
        value_states = v.contiguous()

        return query_states, key_states, value_states


def scheme_megatron(model, input_ids, config):
    sch = slapo.create_schedule(model)
    # Shard embedding
    # shard_word_embedding(sch, config.vocab_size, "embed_tokens")

    # Shard attention
    for i in range(config.num_hidden_layers):
        # shard self attention
        subsch = sch[f"layers.{i}.self_attn"]
        # Replace QKV kernels
        trace_module(subsch, config)

        def qkv_pattern(hidden_states):
            new_states = call_module(r"[qkv]_proj", hidden_states)
            new_states.view(
                bs,
                seq_len,
                config.num_attention_heads,
                config.hidden_size // config.num_attention_heads,
            ).transpose(1, 2)
            return new_states

        subgraphs = subsch.find(qkv_pattern)
        assert len(subgraphs) == 3
        fused_qkv = FusedQKV(
            config.hidden_size, config.num_attention_heads, sch.world_size, bias=False
        )
        # fused_qkv = torch.compile(fused_qkv.cuda(), mode="reduce-overhead", backend="inductor")
        # out = fused_qkv(torch.randn(bs, seq_len, config.hidden_size).cuda())
        # logger.info(f"Replace QKV with fused QKV", ranks=0)
        subsch.replace(fused_qkv, subgraphs)

        # no bias for GPTNeo
        # subsch["FusedQKV_0.fused_linear"].shard("weight", axis=0)
        subsch["o_proj"].shard("weight", axis=1)
        subsch["o_proj"].sync("fwd_post", sync_op_or_fn="all_reduce")
        fix_attention_mask_shape_megatron(subsch, config)
        # shard MLP
        # * is element-wise multiplication
        # self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        subsch = sch[f"layers.{i}.mlp"]
        subsch["gate_proj"].shard("weight", axis=0)
        subsch["up_proj"].shard("weight", axis=0)
        subsch["down_proj"].shard("weight", axis=1)
        subsch["down_proj"].sync("fwd_post", sync_op_or_fn="all_reduce")

        subsch = sch[f"layers.{i}.self_attn"]

        # Replace efficient kernels
        def pattern(query_states, key_states, value_states):
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(config.hidden_size // config.num_attention_heads)
            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            return attn_output

        subgraphs = subsch.find(pattern)
        assert len(subgraphs) > 0
        subsch.replace(F.scaled_dot_product_attention, subgraphs)

        # Fuse add and layernorm
        # subsch = sch[f"layers.{i}"]
        # input_names = ["hidden_states"]
        # sig = inspect.signature(sch.mod.forward)
        # concrete_args = {
        #     p.name: p.default for p in sig.parameters.values() if p.name not in input_names
        # }
        # subsch.trace(recursive=False, concrete_args=concrete_args, leaf_modules=["LlamaRMSNorm", "LlamaAttention", "LlamaMLP"])
        # def add_ln_pattern(residual, hidden_states):
        #     hidden_states = residual + hidden_states
        #     hidden_states = call_module("post_attention_layernorm", hidden_states)
        #     return hidden_states
        # subgraphs = subsch.find(add_ln_pattern)
        # print(subgraphs)
        # assert len(subgraphs) > 0
        # subsch.fuse(subgraphs, compiler="TorchScript", name="fused_add_ln")
        # print(subsch.mod.graph)

    return sch


def test_schemes(init_dist):
    torch.cuda.set_device(dist.get_rank())
    device = torch.cuda.current_device()

    config = AutoConfig.from_pretrained("decapoda-research/llama-7b-hf")
    # config = AutoConfig.from_pretrained("lmsys/vicuna-13b-delta-v1.1")
    config.use_cache = False
    config.pad_token_id = 0
    with slapo.init_empty_weights():
        model = LlamaModel(config)

    schs = []
    input_ids = torch.ones(bs, seq_len, dtype=torch.long, device=device)
    # 1. Slapo-Megatron
    # RR x RS = RS, RS x SR = RR
    schs.append(scheme_megatron(copy.deepcopy(model), input_ids, config))
    # 2. Sequence-Parallel
    # RR->RS x RR = RS, RS x RR = RS->RR
    # schs.append(scheme_sequence_parallel(copy.deepcopy(model), input_ids, config))
    return schs


if __name__ == "__main__":
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    dist.init_process_group("nccl", world_size=int(os.environ["WORLD_SIZE"]))

    input_ids = torch.ones(
        bs, seq_len, dtype=torch.long, device=f"cuda:{dist.get_rank()}"
    )
    schs = test_schemes(None)
    mod, _ = slapo.build(schs[0], init_weights=schs[0].mod._init_weights)
    mod.eval()
    mod.to(f"cuda:{dist.get_rank()}")
    mod.to(torch.float16)

    perf_model(mod, input_ids, use_cuda_graph=True)
    del mod
