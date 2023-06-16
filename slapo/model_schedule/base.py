# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


def shard_attention(
    sch,
    names=["q_proj", "k_proj", "v_proj", "out_proj"],
    attrs=["num_heads", "all_head_size"],
):
    if len(names) == 4:
        q, k, v, out = names
        sch[q].shard("weight", axis=0)
        sch[k].shard("weight", axis=0)
        sch[v].shard("weight", axis=0)
        if sch[q].mod.bias is not None:
            sch[q].shard("bias", axis=0)
            sch[k].shard("bias", axis=0)
            sch[v].shard("bias", axis=0)
    elif len(names) == 2:  # q,k,v have been fused
        qkv, out = names
        sch[qkv].shard("weight", axis=0)
        if sch[qkv].mod.bias is not None:
            sch[qkv].shard("bias", axis=0)
    else:
        raise ValueError(f"Invalid names {names}")
    sch[out].shard("weight", axis=1)
    sch[out].sync("fwd_post", sync_op_or_fn="all_reduce")
    # Update the number of heads
    for attr in attrs:
        path, attr = attr.rsplit(".", 1)
        subsch = sch[path]
        if hasattr(subsch.mod, attr):
            setattr(subsch.mod, attr, getattr(subsch.mod, attr) // sch.world_size)
        else:
            raise ValueError(f"Invalid attribute {attr}")


def shard_mlp(sch, names=["c_fc", "c_proj"]):
    l1, l2 = names
    sch[l1].shard("weight", axis=0)
    if sch[l1].mod.bias is not None:
        sch[l1].shard("bias", axis=0)
    sch[l2].shard("weight", axis=1)
    sch[l2].sync("fwd_post", sync_op_or_fn="all_reduce")
