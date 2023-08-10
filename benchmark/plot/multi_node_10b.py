# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import seaborn as sns
# sns.set_theme(context="paper", style="whitegrid", palette=sns.color_palette("Set3", 10))


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
    x = np.arange(0, len(x_ticklabels) * 3, 3)
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
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="upper left", prop={'size': 8}, ncol=1)
    return bars


def plot(file_name):
    with open(file_name, "r") as csv_file:
        for line in csv_file:
            if "Impl" in line:
                break
        headers = line.strip().split(",")
        lines = []
        for line in csv_file.readlines():
            lines.append(line.strip().split(","))
        results = pd.DataFrame(lines, columns=headers)
    model_name_mapping = {
        "GPT-10B": "gpt-10b",
        "LLaMA-7B": "llama-7b",
    }
    legend_name_mapping = {
        "megatron": "Megatron-LM",
        "deepspeed": "DeepSpeed",
        "slapo": "Epos",
    }
    fig, axs = plt.subplots(2, 1, figsize=(3, 4.5))
    all_data = []
    for i, (model, long_name) in enumerate(model_name_mapping.items()):
        data = {key: [] for key in legend_name_mapping}
        for impl in data:
            if "ckpt" in impl:
                new_impl_name = impl.split("|")[0]
            else:
                new_impl_name = impl
            res = results[results["Impl"] == new_impl_name]
            res = res[res["Model"] == long_name]
            for n_gpu in [16, 32, 64]:
                selected_model_res = res[res["nGPU"] == str(n_gpu)]
                # including ckpt and non-ckpt
                thrpt = selected_model_res["Thrpt"].values.tolist()
                thrpt = [0.0 if val in ["fail", "oom"] else float(val) for val in thrpt]
                if "slapo" in impl:
                    print(impl, thrpt[1] / thrpt[0] if len(thrpt) > 1 and min(thrpt) > 0 else "na")
                thrpt = max(thrpt) if len(thrpt) > 0 else 0.0
                data[impl].append(thrpt)
        print(model, data)
        draw_bar(
            data,
            axs[i // 1],
            idx=i,
            x_ticklabels=list(["16", "32", "64"]),
            x_label="# of GPUs",
            y_label="Throughput (samples/sec)",
            legends=legend_name_mapping,
            title=model,
        )
        for key in data:
            data[key] = np.array(data[key])
        all_data.append(data)
    # legend as a separate figure
    # label_params = axs[1][2].get_legend_handles_labels()
    # axs[1][3].axis(False)
    # axs[1][3].legend(*label_params, ncol=1, loc="center", frameon=False)
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
        "multi_node_v100_10b.pdf",
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    assert len(sys.argv) > 1
    file_name = sys.argv[1]
    plot(file_name)
