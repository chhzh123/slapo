# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    legend_name_mapping = {
        "vanilla": "Vanilla",
        "slapo-kernel": "+ Kernel Opt",
        "slapo-mp-attn": "+ Attn/FFN TP",
        "slapo-mp": "+ Embedding TP",
    }
    data = {
        "vanilla": None,
        "slapo-kernel": None,
        "slapo-mp-attn": None,
        "slapo-mp": None,
    }
    for impl in data:
        res = results[results["Impl"] == impl]
        selected_model_res = res
        thrpt = selected_model_res["Thrpt"].values.tolist()
        data[impl] = float(thrpt[0])
    fig, ax = plt.subplots(1, 1, figsize=(6, 2.2))
    print(data)
    speedup = []
    for val in data.values():
        speedup.append(val / data["vanilla"])

    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barh.html
    y_pos = np.arange(len(data))
    ax.barh(
        y_pos,
        list(data.values()),
        align="center",
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(data)],
    )
    ax.xaxis.set_tick_params(labelsize=12)
    ax.set_yticks(y_pos, labels=legend_name_mapping.values(), fontsize=12)
    for i in range(len(data)):
        ax.text(
            12,
            y_pos[i],
            f"{speedup[i]:.2f}x",
            ha="center",
            va="center",
            color="white",
        )
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    ax.set_xlabel("Throughput (samples/sec)", fontsize=12)
    plt.tight_layout()
    plt.savefig("ablation-1b.pdf", format="pdf", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    assert len(sys.argv) > 1
    file_name = sys.argv[1]
    plot(file_name)
