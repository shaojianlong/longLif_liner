#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot continual accuracy curves until the final global epoch.

This script reads metrics/continual_eval.csv from an fc.py output folder.
Each task curve starts from the global epoch where that task first appears
and continues until the final epoch, for example:
    MNIST: 1-300
    QMNIST: 51-300
    EMNIST-Digits: 101-300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RUN_DIR = Path(__file__).resolve().parents[3] / "results" / "continual_learning"
TASK_ORDER = [
    "MNIST",
    "QMNIST",
    "EMNIST-Digits",
    "Fashion-MNIST",
    "EMNIST-Letters",
    "EMNIST-Balanced",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot each learned task's accuracy from its start epoch to the final epoch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--epochs-per-task", type=int, default=50)
    parser.add_argument("--output-name", type=str, default="continual_accuracy_full.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    csv_path = run_dir / "metrics" / "continual_eval.csv"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find continual evaluation CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"global_epoch", "evaluated_task_name", "test_acc_percent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    final_epoch = args.epochs_per_task * len(TASK_ORDER)
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(15.5, 5.4), dpi=180)

    for task_index, task_name in enumerate(TASK_ORDER, start=1):
        part = df[df["evaluated_task_name"] == task_name].copy()
        if part.empty:
            raise ValueError(f"Task {task_name!r} was not found in {csv_path}")
        part = part.sort_values("global_epoch")

        ax.plot(
            part["global_epoch"],
            part["test_acc_percent"],
            color=colors((task_index - 1) % 10),
            linewidth=2.1,
            label=task_name,
        )

    for task_index, task_name in enumerate(TASK_ORDER, start=1):
        start = (task_index - 1) * args.epochs_per_task + 1
        end = task_index * args.epochs_per_task
        mid = (start + end) / 2
        if task_index > 1:
            ax.axvline(start - 0.5, color="#8a8a8a", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.text(
            mid,
            38,
            f"Learning\n{task_name}",
            ha="center",
            va="center",
            fontsize=8,
            color="#333333",
        )

    ax.axhspan(0, 20, color="#f2f2f2", zorder=0)
    ax.set_xlim(1, final_epoch)
    ax.set_ylim(0, 102)
    ax.set_xticks([1, 50, 100, 150, 200, 250, 300])
    ax.set_xlabel("Global epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Continual evaluation accuracy on learned tasks")
    ax.grid(alpha=0.22)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 0.86, 1])

    out_path = fig_dir / args.output_name
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
