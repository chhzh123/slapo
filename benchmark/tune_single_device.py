# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

models = ["bert", "roberta", "gpt", "opt", "t5", "wideresnet"]
impls = ["slapo-megatron"]

model_name_mapping = {
    "bert": "bert-xlarge",
    "roberta": "roberta-xlarge",
    "albert": "albert-large-v2",
    "gpt_neo": "EleutherAI/gpt-neo-2.7B",
    "opt": "facebook/opt-2.7b",
    "t5": "t5-3b",
    "wideresnet": "wideresnet-2.4B",
}

for model in models:
    for impl in impls:
        n_gpu = 1
        if any(m in model for m in ["opt", "t5", "gpt"]):
            seq_len = 1024
        else:
            seq_len = 512
        print(f"Running {impl} on {model} with {n_gpu} GPU", flush=True)
        batch_size = "batch_size"
        if impl != "slapo-megatron":
            ckpt_ratio = "full"
        else:
            ckpt_ratio = "ckpt_ratio"
        cmd = "python3 -m slapo.autotune.tune"
        cmd += f" --config {os.getcwd()}/../examples/{model}/tune_cfg.py"
        cmd += f" --db results/{model}-gpu{n_gpu}-{impl}.json"
        cmd += f" --error-stop symbol"
        cmd += f"  bench_single_node.py {impl}"
        cmd += f" --model {model_name_mapping[model]} --gpus {n_gpu} --seq-len {seq_len}"
        if model == "t5":
            cmd += " --seq-len-dec 512"
        cmd += f" --batch-size {batch_size}"
        cmd += f" --gradient-checkpoint {ckpt_ratio}"
        print(cmd, flush=True)
        os.system(cmd)
        print("\n", flush=True)
