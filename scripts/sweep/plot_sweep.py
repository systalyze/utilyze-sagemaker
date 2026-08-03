#!/usr/bin/env python3
"""Render a sweep run as the deck's performance-plane scatter.

  plot_sweep.py results-sweep/<run> [-o out.png]

Every measured point in the (tokens/s per user, tokens/s per GPU) plane:
blue = measured, grey = pruned by the stop rule, star = best config's best point.
Requires matplotlib (pip install matplotlib).
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACCENT, GHOST, GOOD, INK2 = "#2a78d6", "#dcdbd7", "#0ca30c", "#52514e"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("-o", "--out", default="")
    args = p.parse_args()

    with open(os.path.join(args.run_dir, "summary.json")) as f:
        summary = json.load(f)
    rows = list(csv.DictReader(open(os.path.join(args.run_dir, "sweep.csv"))))
    rows = [r for r in rows if r["tok_s_user"]]
    ngpu = summary["gpu_count"]

    fig = plt.figure(figsize=(9.9, 8.8), dpi=240)
    ax = fig.add_axes([0.11, 0.10, 0.86, 0.86])
    for pruned, color, z in ((True, GHOST, 2), (False, ACCENT, 3)):
        pts = [r for r in rows if (r["pruned"] == "True") == pruned]
        ax.scatter([float(r["tok_s_user"]) for r in pts],
                   [float(r["whole_tok_s"]) / ngpu for r in pts],
                   s=14, color=color, zorder=z, edgecolors="none")
    win = summary.get("winner_pruned") or summary.get("winner_full")
    if win:
        wr = max((r for r in rows if r["config"] == win and r["pruned"] != "True"),
                 key=lambda r: float(r["whole_tok_s"]), default=None)
        if wr:
            ax.scatter([float(wr["tok_s_user"])],
                       [float(wr["whole_tok_s"]) / ngpu],
                       s=340, marker="*", color=GOOD, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("tokens/s per user", fontsize=12, color=INK2)
    ax.set_ylabel("tokens/s per GPU", fontsize=12, color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.tick_params(length=0, labelsize=10, labelcolor=INK2)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", color="none", markerfacecolor=c,
                      markeredgecolor="none") for c in (ACCENT, GHOST)]
    handles.append(Line2D([], [], marker="*", color="none", markerfacecolor=GOOD,
                          markeredgecolor="none", markersize=12))
    ax.legend(handles, ["measured", "pruned", "best config"], frameon=False,
              fontsize=10, loc="lower left", labelcolor=INK2)
    out = args.out or os.path.join(args.run_dir, "sweep-plane.png")
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
