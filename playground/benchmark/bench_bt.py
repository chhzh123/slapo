# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.distributed as dist
import deepspeed
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--name", required=True, type=str, help="model_name")


def perf_model(mod, input_tensor):
    """Measure the performance of a mod with certain resharding schemes"""
    # warmup
    mod.eval()
    torch.cuda.empty_cache()
    mod.to(torch.float16)
    for _ in range(10):
        mod(input_tensor)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    iters = 40
    for _ in range(iters):
        mod(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    if dist.get_rank() == 0:
        bs, seq_len, _ = input_tensor.shape
        print(f"{bs}\t{seq_len}\t{start_event.elapsed_time(end_event) / iters:.3f} ms")


def get_bert_model():
    from transformers import BertLMHeadModel, AutoConfig

    config = AutoConfig.from_pretrained("bert-base-uncased")
    print(config)
    mod = BertLMHeadModel(config)
    bs, seq_len = 1, 512
    input_ids = torch.ones(
        bs, seq_len, dtype=torch.long, device=f"cuda:{dist.get_rank()}"
    )
    return mod, input_ids


def get_gpt_model():
    from transformers import GPTNeoModel, AutoConfig

    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-1.3B")
    config.use_cache = False
    mod = GPTNeoModel(config)
    bs, seq_len = 4, 1024
    input_ids = torch.ones(
        bs, seq_len, dtype=torch.long, device=f"cuda:{dist.get_rank()}"
    )
    return mod, input_ids


def get_llama_model():
    from transformers import LlamaModel, AutoConfig

    # config = AutoConfig.from_pretrained("decapoda-research/llama-7b-hf")
    # elinas/llama-7b-hf-transformers-4.29
    config = AutoConfig.from_pretrained("lmsys/vicuna-13b-delta-v1.1")
    config.use_cache = False
    mod = LlamaModel(config)
    mod.eval()
    mod.to(torch.float16)
    bs, seq_len = 1, 2048
    input_ids = torch.ones(
        bs, seq_len, dtype=torch.long, device=f"cuda:{dist.get_rank()}"
    )
    return mod, input_ids


def get_opt_model():
    from transformers import OPTModel, AutoConfig

    # config = AutoConfig.from_pretrained("decapoda-research/llama-7b-hf")
    # elinas/llama-7b-hf-transformers-4.29
    config = AutoConfig.from_pretrained("facebook/opt-13b")
    config.use_cache = False
    with deepspeed.OnDevice(dtype=torch.float16, device="meta", enabled=False):
        mod = OPTModel(config)
    mod.eval()
    mod.to(torch.float16)
    bs, seq_len = 1, 2048
    input_ids = torch.ones(
        bs, seq_len, dtype=torch.long, device=f"cuda:{dist.get_rank()}"
    )
    return mod, input_ids


def generate_json(model_name, checkpoint_path=None):
    import io
    from pathlib import Path
    from huggingface_hub import snapshot_download
    import json

    if checkpoint_path is None:
        repo_root = snapshot_download(
            model_name,
            allow_patterns=["*"],
            cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
            ignore_patterns=["*.safetensors"],
            local_files_only=False,
            revision=None,
        )
    else:
        assert os.path.exists(
            checkpoint_path
        ), f"Checkpoint path {checkpoint_path} does not exist"
        repo_root = checkpoint_path

    if os.path.exists(os.path.join(repo_root, "ds_inference_config.json")):
        checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
    else:
        checkpoints_json = "checkpoints.json"

        with io.open(checkpoints_json, "w", encoding="utf-8") as f:
            file_list = [
                str(entry).split("/")[-1]
                for entry in Path(repo_root).rglob("*.[bp][it][n]")
                if entry.is_file()
            ]
            data = {"type": "BLOOM", "checkpoints": file_list, "version": 1.0}
            json.dump(data, f)

    return repo_root, checkpoints_json


if __name__ == "__main__":
    dist.init_process_group("nccl", world_size=int(os.environ["WORLD_SIZE"]))

    for kernel_opt in [True]:
        for mod, input_ids in [get_bert_model()]:
            # Initialize the DeepSpeed-Inference engine
            # https://www.deepspeed.ai/tutorials/inference-tutorial/
            # repo_root, checkpoints_json = generate_json("facebook/opt-13b", "ds_inference_config.json")
            ds_engine = deepspeed.init_inference(
                mod,
                mp_size=dist.get_world_size(),
                dtype=torch.float16,
                checkpoint=None,  # "checkpoints.json",
                replace_with_kernel_inject=kernel_opt,
            )
            mod = ds_engine.module
            if dist.get_rank() == 0:
                print(mod)
            for bs in [16]:
                for seq_len in [32, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]:
                    # input_ids = torch.ones(
                    #     bs, seq_len, dtype=torch.long, device=f"cuda:{dist.get_rank()}"
                    # )
                    # out = mod(input_ids)
                    hidden_states = torch.randn((bs, seq_len, 1024), device="cuda")
                    perf_model(mod.bert.encoder, hidden_states)
            del mod
