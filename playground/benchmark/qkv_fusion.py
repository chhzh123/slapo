from argparse import ArgumentParser
import time

import torch
from torch import nn


class UnfusedQKV(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self,
                hidden_states: torch.Tensor
    ):
        #import pdb; pdb.set_trace()
        bsz, tgt_len, _ = hidden_states.size()
        #query_states = self.q_proj(hidden_states) * self.scaling
        query_states = self.q_proj(hidden_states)
        #print(hidden_states)
        #print(query_states)
        #print(self.q_proj.weight.data)
        #print(self.q_proj.bias.data)
        #query_states = query_states * self.scaling

        #proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        return query_states, key_states, value_states


class FusedQKV(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.fused_linear = nn.Linear(embed_dim, embed_dim * 3, bias=bias)

    def forward(self,
                hidden_states: torch.Tensor
    ):
        #import pdb; pdb.set_trace()
        bsz, tgt_len, _ = hidden_states.size()
        qkv = self.fused_linear(hidden_states)
        #import pdb; pdb.set_trace()
        reshaped_qkv = qkv.view(bsz, tgt_len, 3 * self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        reshaped_qkv = reshaped_qkv.view(bsz, 3, self.num_heads, tgt_len, self.head_dim)
        q, k, v = reshaped_qkv.unbind(dim=1)
        #q = q * self.scaling
        #proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        #q = q.view(*proj_shape)

        query_states = q.contiguous()
        key_states = k.contiguous()
        value_states = v.contiguous()

        return query_states, key_states, value_states


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dtype', type=str, default="float16")
    args = parser.parse_args()

    data_type = getattr(torch, args.dtype)

    #import pdb; pdb.set_trace()

    model = UnfusedQKV(args.embed_dim, args.num_heads)
    model.eval()
    model = model.to(dtype=data_type).to('cuda:0')

    model_opt = FusedQKV(args.embed_dim, args.num_heads)
    model_opt.eval()
    model_opt = model_opt.to(dtype=data_type).to('cuda:0')

    hidden_states = torch.randn(args.batch_size, args.seq_len, args.embed_dim).to('cuda:0').to(dtype=data_type)

    # correctness check
    #import pdb; pdb.set_trace()
    q, k, v = model(hidden_states)
    q_weight = model.q_proj.weight.data
    k_weight = model.k_proj.weight.data
    v_weight = model.v_proj.weight.data
    # The weights are already transposed in the linear layer.
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
    q_bias = model.q_proj.bias.data
    k_bias = model.k_proj.bias.data
    v_bias = model.v_proj.bias.data
    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

    #import pdb; pdb.set_trace()
    model_opt.fused_linear.weight.data = qkv_weight
    model_opt.fused_linear.bias.data = qkv_bias
    q_opt, k_opt, v_opt = model_opt(hidden_states)

    rtol = 5e-2
    atol = 5e-2
    #assert torch.allclose(q, q_opt)
    assert torch.allclose(q, q_opt, rtol=rtol, atol=atol)
    assert torch.allclose(k, k_opt, rtol=rtol, atol=atol)
    assert torch.allclose(v, v_opt, rtol=rtol, atol=atol)

    n_benchmark_iters = 10
    latencies = []

    for _ in range(n_benchmark_iters):
        torch.cuda.synchronize()
        t0 = time.time()
        q, k, v = model(hidden_states)

        torch.cuda.synchronize()
        latency = time.time() - t0

        latencies.append(latency)

    latencies = latencies[3:]
    print(f"Latency(baseline): {sum(latencies)/len(latencies)*1e3:.3f} ms")

    latencies = []
    for _ in range(n_benchmark_iters):
        torch.cuda.synchronize()
        t0 = time.time()
        q, k, v = model_opt(hidden_states)

        torch.cuda.synchronize()
        latency = time.time() - t0

        latencies.append(latency)

    latencies = latencies[3:]
    print(f"Latency(opt): {sum(latencies)/len(latencies)*1e3:.3f} ms")

    # Use torch.compile
    model_compile = torch.compile(model, backend="inductor")

    latencies = []
    for _ in range(n_benchmark_iters):
        torch.cuda.synchronize()
        t0 = time.time()
        q, k, v = model_compile(hidden_states)

        torch.cuda.synchronize()
        latency = time.time() - t0

        latencies.append(latency)

    latencies = latencies[3:]
    print(f"Latency(compile): {sum(latencies)/len(latencies)*1e3:.3f} ms")


if __name__ == "__main__":
    main()