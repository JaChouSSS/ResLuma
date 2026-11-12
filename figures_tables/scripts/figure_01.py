#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ResLuma 7-axis radar chart, tightly cropped top and bottom.

Dependencies:
    pip install numpy matplotlib

Run:
    python fastluma_radar_7axis_tight.py

Outputs (in the same directory as this script):
    ResLuma_radar_7axis_tight.png
    ResLuma_radar_7axis_tight.pdf

Source: original experiment results from p2实验.zip:
- sec5/5.3/tables/table5_2.csv (LOL/BIMEF quality)
- sec5/5.4/tables/table5_3.csv (weights and parameters)
- sec5/5.4/tables/efficiency_raw.csv (4K GPU FPS)
The values are embedded below: no archive or CSV is required.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Columns: [PSNR, SSIM, NIQE, CLIP-IQA, 4K GPU FPS, Weight Size (KB), Params]
# The original evaluation used LOL for PSNR/SSIM, BIMEF for NIQE/CLIP-IQA,
# and an RTX 4090 for the 4K GPU throughput.
DATA = {
    "Zero-DCE":         [14.861065, 0.592723, 3.585352, 0.478033,  11.7924529363,  310.2,  79416],
    "RUAS":             [16.447909, 0.545959, 6.773844, 0.279189,   8.0797121997,   13.4,   3438],
    "Zero-DCE++":       [15.355358, 0.605377, 3.625719, 0.474215,  85.9276593175,   41.3,  10561],
    "SCI":              [14.853409, 0.557093, 3.886416, 0.415833, 104.7466157233,    1.0,    258],
    "URetinex-Net":     [19.860626, 0.834634, 3.904697, 0.507487,   1.5471780715, 1328.5, 340105],
    "FourLLIE":         [20.022528, 0.832022, 3.594229, 0.489969,   3.6196128048,  466.6, 119446],
    "Retinexformer":    [25.155450, 0.860180, 3.361414, 0.405519,   0.6879541100, 6272.3,1605701],
    "LiteIE":           [19.202528, 0.664905, 3.961074, 0.432102,  69.5260200754,    0.2,     58],
    "MobileIE":         [23.634311, 0.829096, 4.044331, 0.352820,  39.6147286898,   15.9,   4075],
    "ResLuma (Ours)":   [19.332213, 0.632885, 3.665704, 0.453891, 118.3353516221,    0.1,     28],
}

AXIS_LABELS = ["PSNR", "SSIM", "NIQE", "CLIP-IQA", "4K GPU FPS", "Weight Size", "Parameters"]
# PSNR, SSIM, CLIP-IQA, 4K FPS: higher is better.
# NIQE, Weight Size, Parameters: lower is better.
LOWER_IS_BETTER = {2, 5, 6}
# Log scaling prevents throughput, size and parameter counts from being dominated by outliers.
LOG_SCALED = {4, 5, 6}

PALETTE = {
    "Zero-DCE": "#4e79a7",
    "RUAS": "#a0a0a0",
    "Zero-DCE++": "#f28e2b",
    "SCI": "#59a14f",
    "URetinex-Net": "#af7aa1",
    "FourLLIE": "#9c755f",
    "Retinexformer": "#e15759",
    "LiteIE": "#7f90a1",
    "MobileIE": "#b6992d",
    "ResLuma (Ours)": "#0d99aa",
}


def normalized_matrix():
    raw = np.asarray(list(DATA.values()), dtype=float)
    result = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        v = np.log10(raw[:, j]) if j in LOG_SCALED else raw[:, j]
        span = v.max() - v.min()
        if span == 0:
            result[:, j] = 1.0
        elif j in LOWER_IS_BETTER:
            result[:, j] = (v.max() - v) / span
        else:
            result[:, j] = (v - v.min()) / span
    return result


def draw():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })

    names = list(DATA.keys())
    matrix = normalized_matrix()
    n = len(AXIS_LABELS)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    closed = np.r_[angles, angles[0]]

    # Keep the radar circle physically small while reducing canvas height to remove excess vertical whitespace.
    fig = plt.figure(figsize=(9.3, 4.65), facecolor="white")
    ax = fig.add_axes([0.080, 0.120, 0.490, 0.760], projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(AXIS_LABELS, fontsize=11.5)
    ax.tick_params(axis="x", pad=13)
    ax.set_ylim(0.0, 1.055)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="#808080")
    ax.set_rlabel_position(2)
    ax.grid(alpha=0.21, linewidth=0.75)
    ax.spines["polar"].set_color("#cbd2dc")
    ax.spines["polar"].set_linewidth(0.8)

    plotted = []
    for name, row in zip(names, matrix):
        if name == "ResLuma (Ours)":
            continue
        values = np.r_[row, row[0]]
        line, = ax.plot(closed, values, color=PALETTE[name], linewidth=1.45,
                        alpha=0.68, zorder=3, label=name)
        plotted.append(line)

    own_row = matrix[names.index("ResLuma (Ours)")]
    own_values = np.r_[own_row, own_row[0]]
    ax.fill(closed, own_values, color=PALETTE["ResLuma (Ours)"], alpha=0.11, zorder=2)
    our_line, = ax.plot(closed, own_values, color=PALETTE["ResLuma (Ours)"],
                        linewidth=3.7, marker="o", markersize=4.7,
                        solid_capstyle="round", zorder=10, label="ResLuma (Ours)")
    plotted.append(our_line)

    # Position legend relative to FIGURE, not the axes: truly centered vertically on the right.
    legend = fig.legend(handles=plotted, labels=names, loc="center left",
                        bbox_to_anchor=(0.655, 0.50), frameon=False,
                        fontsize=10.4, handlelength=2.8,
                        labelspacing=0.85, borderaxespad=0.0)
    for label in legend.get_texts():
        if label.get_text() == "ResLuma (Ours)":
            label.set_fontweight("bold")
            label.set_color("#0a8796")

    # Intentionally no title or subtitle at the top.
    outdir = Path(__file__).resolve().parent.parent / 'generated_figures' 
    for ext in ("png", "pdf"):
        outpath = outdir / f'Figure_01.{ext}'
        fig.savefig(outpath, dpi=320, bbox_inches="tight", pad_inches=0.045)
        print(f"Saved: {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    draw()
