# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import matplotlib.pyplot as plt

thrpt = [
    [
        13.759020582371607,
        13.759310349669004,
        13.774879118407737,
        13.759503534647974,
        0,
        0,
        0,
    ],
    [
        13.745808283737652,
        13.74601592120965,
        13.723523778644084,
        13.73077126194841,
        0,
        0,
        0,
    ],
    [
        13.51644903574618,
        13.504497479964309,
        13.499397348332662,
        13.488017341736581,
        0,
        0,
        0,
    ],
    [
        13.2637417532255,
        13.274260003103073,
        13.279754410775572,
        13.263741753225503,
        15.217541675312996,
        15.268689272258573,
        0,
    ],
    [
        14.010507880910684,
        13.994262352435502,
        13.999160050396977,
        13.99230423267203,
        16.21533971136695,
        16.21271076523995,
        0,
    ],
    [
        13.744504925114265,
        13.737162294760253,
        13.719064054963358,
        13.66594360086768,
        15.891433760468168,
        15.87761634134355,
        17.24940448484517,
    ],
    [
        13.338255784872988,
        13.350020859407593,
        13.372495641999189,
        13.345248733988681,
        15.406624848684935,
        15.412985440233399,
        16.728652297949246,
    ],
    [
        13.036769009737668,
        13.036769009737668,
        13.039544414284954,
        13.010288748755393,
        15.01846658391185,
        15.007656967840735,
        16.181229773462782,
    ],
    [
        12.846786773927143,
        12.831087892952068,
        12.853470437017995,
        12.889366272824923,
        14.733998702004875,
        14.787951340598209,
        15.962905248755273,
    ],
    [
        12.315270935960593,
        12.274672090902715,
        12.26155651701729,
        12.257906349595492,
        14.061589763162653,
        14.051146172066323,
        15.188664916353853,
    ],
    [
        11.532362692806688,
        11.514341523594119,
        11.529750875025735,
        11.514104778353484,
        13.120592301023875,
        13.160677774905407,
        14.146058049359638,
    ],
]

X = [28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8]
Y = [1.0, 0.92, 0.84, 0.67, 0.5, 0.34, 0.25]
thrpt = np.array(thrpt).reshape((len(X), len(Y)))
print(X, Y, thrpt)

fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.5))
CS = ax.contourf(
    Y,
    X,
    thrpt,
    levels=[0, 11, 12, 13, 14, 15, 16, 17, 18],
    colors=[
        "#9e0142",
        "#d53e4f",
        "#f46d43",
        "#fdae61",
        "#fee08b",
        "#ffffbf",
        "#e6f598",
        "#abdda4",
        "#66c2a5",
        "#3288bd",
        "#5e4fa2",
    ][:-1][::-1],
)  # , cmap="viridis")
# ax.clabel(
# CS,
# inline=True,
# fontsize=5,
# colors="black"
# manual=[
#     (0.6, 175),
#     (0.3, 120),
#     (0.66, 159),
#     (0.46, 135),
#     (0.67, 110),
#     (0.84, 100),
#     (0.82, 184),
# ],
# )
# best result
# explored_points = [
#     [1.0, 176],
#     [0.92, 176],
#     [0.84, 176],
#     [0.67, 176],
#     [0.84, 168],
#     [0.67, 168],
#     [0.5, 168],
#     [0.67, 160],
#     [0.5, 160],
#     [0.67, 152],
#     [0.5, 152],
#     [0.5, 144],
#     [0.34, 144],
#     [0.5, 136],
#     [0.34, 136],
#     [0.5, 128],
#     [0.34, 128],
# ]
# x = [point[0] for point in explored_points]
# y = [point[1] for point in explored_points]
# ax.plot(x, y, "*", color="purple", markersize=5, linewidth=1)
# # ax.plot([0.5], [144], "x", color='red', markersize=10)
# ax.text(0.5, 144, "$\\times$", color="red", va="center", ha="center", fontsize=20)
# ax.set_xlim(0.25, 1.0)
# ax.set_ylim(96, 192)
# ax.set_xlabel("Activation Checkpointing Ratio")
# ax.set_ylabel("Batch Size")
# ax.text(
#     0.4,
#     168,
#     "OOM",
#     weight="bold",
#     horizontalalignment="center",
#     color="black",
#     fontsize=10,
# )
# ax.text(
#     0.55,
#     135,
#     "125",
#     horizontalalignment="center",
#     weight="bold",
#     color="black",
#     fontsize=10,
# )
# ax.text(
#     0.67,
#     128,
#     "120",
#     horizontalalignment="center",
#     weight="bold",
#     color="black",
#     fontsize=10,
# )
# ax.text(
#     0.75,
#     115,
#     "115",
#     horizontalalignment="center",
#     weight="bold",
#     color="black",
#     fontsize=10,
# )
# ax.text(
#     0.84,
#     107,
#     "110",
#     horizontalalignment="center",
#     weight="bold",
#     color="black",
#     fontsize=10,
# )
# ax.text(
#     0.92,
#     100,
#     "105",
#     horizontalalignment="center",
#     weight="bold",
#     color="black",
#     fontsize=10,
# )
# ax.text(
#     1.16,
#     100,
#     "Throughput (samples / sec)",
#     horizontalalignment="center",
#     color="black",
#     rotation=270,
#     fontsize=10,
# )
# ax.set_xticks([0.34, 0.5, 0.67, 0.84, 1.0])
# ax.set_yticks([184, 168, 152, 136, 120, 104])
# make a colorbar for the contour lines
CB = fig.colorbar(CS)
# draw search space
# from matplotlib.patches import Polygon
# y = np.array([[0.25, 104], [0.25, 176], [1.0, 176], [1.0, 120], [0.67, 120], [0.67, 104]])
# p_defined = Polygon(y, facecolor="#FFE7B2")
# y = np.array([[0.75, 200], [0.67, 176], [0.5, 152], [0.34, 128], [0, 100], [0, 200]])
# p_invalid = Polygon(y, facecolor=(0.1, 0.2, 0.5, 0.3))
# ax.add_patch(p_defined)
# ax.add_patch(p_invalid)
plt.show()
plt.savefig("autotune-1b.pdf", format="pdf", dpi=200, bbox_inches="tight")
