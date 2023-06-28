# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom linear modules."""
# pylint: disable=arguments-renamed
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

try:
    from functorch.compile import memory_efficient_fusion
except ImportError:
    memory_efficient_fusion = None

from .fused_bias import bias_dropout, bias_new_gelu, BiasGeLUFunction


class FusedQKV(nn.Module):
    """A linear module with fused QKV weights.

    Parameters
    ----------
    hidden_size: int
        The hidden size of the input.
    num_heads: int
        The number of heads.
    world_size: int
        The size of tensor parallelism group.
    """

    def __init__(self, hidden_size, num_heads, world_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.fused_linear = nn.Linear(hidden_size, self.num_heads * self.head_size * 3)
        self.world_size = world_size

    def reshape_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_heads // self.world_size,
            self.head_size,
            3,
        )
        x = x.view(new_x_shape)
        return x.contiguous()

    def forward(self, hidden_states):
        qkv = self.fused_linear(hidden_states)
        reshaped_qkv = self.reshape_for_scores(qkv)
        q, k, v = torch.split(reshaped_qkv, 1, dim=-1)
        q = torch.squeeze(q, -1).contiguous()
        k = torch.squeeze(k, -1).contiguous()
        v = torch.squeeze(v, -1).contiguous()
        return [q, k, v]


class SyncOpWrapper(nn.Module):
    def __init__(self, fn):
        super().__init__()
        if fn is None:
            # Identity function.
            self.fn = lambda x: x
            self.traceable = True
        else:
            self.fn = fn
            self.traceable = False

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def extra_repr(self):
        fn_name = "None"
        if self.fn is not None:
            if hasattr(self.fn, "__name__"):
                # Simply get the function name.
                fn_name = self.fn.__name__
            elif isinstance(self.fn, partial):
                # If the sync function is a partial function, extract the original
                # function name and the partial arguments.
                fixed_args = ", ".join([str(arg) for arg in self.fn.args])
                fixed_args += (
                    ", ".join([f"{k}={v}" for k, v in self.fn.keywords.items()])
                    if self.fn.keywords
                    else ""
                )
                fn_name = f"partial({self.fn.func.__name__}, {fixed_args})"
            else:
                fn_name = str(self.fn)
        return f"sync_fn={fn_name}"


class LinearWithSyncFunc(nn.Linear):
    """Derived from `nn.Linear` but with a sync function that will be invoked
    before the bias addition. Notice that the bias is disabled in this module.
    We need to call LinearWithSyncNBias to include the bias calculation.
    The reason of doing this is to avoid collective op to be traced.

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    out_features: int
        Size of each output sample.
    device: torch.device
        The device of the module.
    dtype: torch.dtype
        The data type of the module.
    sync_fn: Callable
        The sync function to be invoked before the bias addition.
    """

    def __init__(
        self,
        in_features,
        out_features,
        weight,
        device=None,
        dtype=None,
        sync_fn=None,
    ):
        super().__init__(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )
        self.weight = weight
        # Untraceable module.
        self.sync_fn = SyncOpWrapper(sync_fn)

    def forward(self, x):
        x = F.linear(x, self.weight, None)
        x = self.sync_fn(x)
        return x


class LinearWithSyncNBias(LinearWithSyncFunc):
    def __init__(
        self,
        in_features,
        out_features,
        weight,
        bias,
        device=None,
        dtype=None,
        sync_fn=None,
    ):
        super().__init__(
            in_features,
            out_features,
            weight=weight,
            device=device,
            dtype=dtype,
            sync_fn=sync_fn,
        )
        assert bias is not None, "bias must be provided"
        self.bias = bias

    def forward(self, x):
        x = F.linear(x, self.weight, None)
        x = self.sync_fn(x)
        x = x + self.bias
        return x


class LinearWithAct(LinearWithSyncFunc):
    """Derived from `LinearWithSyncFunc` but fusing the activation function.

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    out_features: int
        Size of each output sample.
    bias: bool
        This is to align the interface with `nn.Linear`. However, this module
        requires bias to be True.
    act_fn: str
        The activation function to be fused. Currently supports "gelu" and "gelu_new".
    device: torch.device
        The device of the module.
    dtype: torch.dtype
        The data type of the module.
    sync_fn: Callable
        The sync function to be invoked before the bias addition.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_fn="gelu",
        device=None,
        dtype=None,
        sync_fn=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype, sync_fn)
        self.act_fn = act_fn
        if not bias:
            raise ValueError(
                "LinearWithAct requires bias. Please set bias=True or "
                "simply use nn.Linear"
            )

        if self.act_fn == "gelu":
            self.act = BiasGeLUFunction.apply
        elif self.act_fn == "gelu_new":
            if memory_efficient_fusion is not None:
                self.act = memory_efficient_fusion(bias_new_gelu)
            else:
                self.act = torch.jit.script(bias_new_gelu)
        else:
            raise NotImplementedError(f"Unsupported activation: {self.act_fn}")

    def forward(self, x: Tensor) -> Tensor:
        x = F.linear(x, self.weight, None)
        if self.sync_fn is not None:
            x = self.sync_fn(x)
        return self.act(x, self.bias)

    def extra_repr(self):
        return f"{super().extra_repr()}, act_fn={self.act_fn}"


class LinearWithDropout(LinearWithSyncFunc):
    """Derived from `LinearWithSyncFunc` but fusing the dropout.

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    out_features: int
        Size of each output sample.
    bias: bool
        This is to align the interface with `nn.Linear`. However, this module
        requires bias to be True.
    p: float
        The probability of an element to be zeroed.
    inplace: bool
        If set to True, will do dropout in-place.
    device: torch.device
        The device of the module.
    dtype: torch.dtype
        The data type of the module.
    sync_fn: Callable
        The sync function to be invoked before the bias addition.
    use_torchscript: bool
        Whether to use torchscript or memory_efficient_fusion to fuse dropout.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        p=0.5,
        inplace=False,
        device=None,
        dtype=None,
        sync_fn=None,
        use_torchscript=False,
    ):
        super().__init__(in_features, out_features, bias, device, dtype, sync_fn)
        if not bias:
            raise ValueError(
                "LinearWithDropout requires bias. Please set bias=True or "
                "simply use nn.Linear"
            )
        self.p = p
        self.inplace = inplace
        self.use_torchscript = use_torchscript or memory_efficient_fusion is None

        # Somehow memory_efficient_fusion generates a different dropout mask
        # against the original dropout function even the random seed is the same.
        if self.use_torchscript:
            self.dropout = torch.jit.script(bias_dropout)
        else:
            bias_dropout_func = partial(
                bias_dropout, p=p, training=self.training, inplace=inplace
            )
            self.dropout = memory_efficient_fusion(bias_dropout_func)

    def forward(self, x: Tensor) -> Tensor:
        x = F.linear(x, self.weight, None)
        if self.sync_fn is not None:
            x = self.sync_fn(x)
        if self.use_torchscript:
            return self.dropout(x, self.bias, self.p, self.training, self.inplace)
        return self.dropout(x, self.bias)

    def extra_repr(self):
        return (
            f"{super().extra_repr()}, p={self.p}, inplace={self.inplace}, "
            f"use_torchscript={self.use_torchscript}"
        )
