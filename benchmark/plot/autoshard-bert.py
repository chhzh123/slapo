# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import seaborn as sns
# sns.set_theme(context="paper", style="whitegrid", palette=sns.color_palette("Set3", 10))

bs = 8

def draw_bar(
    data,
    ax,
    idx,
    x_ticklabels,
    x_label=None,
    y_label=None,
    legends=None,
    title=None,
):
    x = np.arange(0, len(x_ticklabels) * 4, 4)
    width = 0.25
    interval = np.arange(-len(data) + 1, len(data), 2)
    bars = []
    plt.rcParams["hatch.linewidth"] = 0.6
    hatches = ["//", "\\\\", "..", "x", "|", "-", "o", "O", ".", "*"]
    for i, key in enumerate(data):
        label = legends.get(key, key) if legends is not None else key
        hatch = None
        kwargs = {}
        kwargs["alpha"] = 0.95
        kwargs["label"] = label
        kwargs["hatch"] = hatch
        # if i >= 4:
        #     kwargs["color"] = bars[i % 4].patches[0].get_facecolor()
        data[key] = 1000 / data[key] * bs * 10
        bars.append(ax.bar(x + interval[i] * width, data[key], width * 2, **kwargs))
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticklabels)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    for bar in bars:
        for patch in bar.patches:
            height = patch.get_height()
            if height == 0:
                ax.text(
                    patch.get_x() + patch.get_width() / 2.0,
                    height,
                    "X",
                    ha="center",
                    va="bottom",
                    color=patch.get_facecolor(),
                    fontweight="bold",
                )
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    # ax.legend(loc="upper left")
    if title is not None:
        ax.set_title(title)
    return bars


def plot(file_name):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    all_data = []
    data_v100 = {
        "Naive": np.array([278,278,278]),
        "Column-row (Megatron-LM)": np.array([197,117,86]),
        "Weight-sharding": np.array([289,175,141]),
        "Sequence parallelism": np.array([174,388,861]),
    }
    data_a100 = {
        "Naive": np.array([207.155, 207.132, 207.125]),
        "Column-row (Megatron-LM)": np.array([199.020, 128.270, 80.891 ]),
        "Weight-sharding": np.array([232.710, 169.689, 110.934]),
        "Sequence parallelism": np.array([204.908, 125.794, 290.711]),
    }
    for i in range(2):
        data = data_v100 if i == 0 else data_a100
        draw_bar(
            data,
            axs[i % 2],
            idx=i,
            x_ticklabels=list(["2", "4", "8"]) if i == 0 else list(["1", "2", "4"]),
            x_label="# of GPUs",
            y_label="Throughput (samples/sec)",
            legends={k: k for k in data.keys()},
            title="V100" if i == 0 else "A100",
        )
        for key in data:
            data[key] = np.array(data[key])
        all_data.append(data)
    # legend as a separate figure
    label_params = axs[0].get_legend_handles_labels()
    axs[2].axis(False)
    axs[2].legend(*label_params, ncol=1, loc="center", frameon=False)
    # axs[1][3].text(
    #     0.5,
    #     0,
    #     "X: Unsupported or OOM",
    #     ha="center",
    #     va="bottom",
    #     color="black",
    #     fontweight="bold",
    # )
    plt.tight_layout()
    plt.savefig(
        "autoshard-bert.pdf",
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
    # speedup_bert = all_data[0]["slapo-megatron"] / all_data[0]["megatron"]
    # speedup_gpt = all_data[3]["slapo-megatron"] / all_data[3]["megatron"]
    # speedup_t5 = all_data[5]["slapo-megatron"] / all_data[5]["megatron"]
    # print("BERT speedup vs Megatron: ", speedup_bert)
    # print("GPT speedup vs Megatron: ", speedup_gpt)
    # print("T5 speedup vs Megatron:", speedup_t5)
    # for i, (model, long_name) in enumerate(model_name_mapping.items()):
    #     speedup_ds = all_data[i]["slapo-deepspeed"] / all_data[i]["deepspeed"]
    #     print(f"{model} speedup vs DS: ", speedup_ds)


if __name__ == "__main__":
    # assert len(sys.argv) > 1
    file_name = ""#sys.argv[1]
    plot(file_name)
