#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot sequential task accuracy over global epochs for fc.py outputs."""

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
        description="Draw one accuracy-vs-global-epoch figure from fc.py epoch_metrics.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--epochs-per-task", type=int, default=50)
    parser.add_argument("--show-train", action="store_true", help="Also draw train accuracy as dashed lines.")
    parser.add_argument("--output-name", type=str, default="task_sequence_accuracy.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metrics_csv = run_dir / "metrics" / "epoch_metrics.csv"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_csv.exists():
        raise FileNotFoundError(f"Cannot find epoch metrics: {metrics_csv}")

    df = pd.read_csv(metrics_csv)
    required = {"task_name", "epoch", "train_acc", "val_acc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {metrics_csv}: {sorted(missing)}")

    rows = []
    for task_index, task_name in enumerate(TASK_ORDER, start=1):
        part = df[df["task_name"] == task_name].copy()
        if part.empty:
            raise ValueError(f"Task {task_name!r} was not found in {metrics_csv}")
        part = part.sort_values("epoch")
        part["plot_epoch"] = (task_index - 1) * args.epochs_per_task + part["epoch"].astype(int)
        rows.append(part)
    plot_df = pd.concat(rows, ignore_index=True)

    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(15.5, 5.4), dpi=180)

    for task_index, task_name in enumerate(TASK_ORDER, start=1):
        part = plot_df[plot_df["task_name"] == task_name]
        color = colors((task_index - 1) % 10)
        ax.plot(
            part["plot_epoch"],
            part["val_acc"] * 100.0,
            color=color,
            linewidth=2.1,
            label=f"{task_name} val",
        )
        if args.show_train:
            ax.plot(
                part["plot_epoch"],
                part["train_acc"] * 100.0,
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.45,
                label=f"{task_name} train",
            )

    total_epochs = args.epochs_per_task * len(TASK_ORDER)
    for task_index, task_name in enumerate(TASK_ORDER, start=1):
        start = (task_index - 1) * args.epochs_per_task + 1
        end = task_index * args.epochs_per_task
        mid = (start + end) / 2
        if task_index > 1:
            ax.axvline(start - 0.5, color="#8a8a8a", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.text(mid, 8, task_name, ha="center", va="center", fontsize=8, color="#333333")

    ax.set_xlim(1, total_epochs)
    ax.set_ylim(0, 102)
    ax.set_xticks([1, 50, 100, 150, 200, 250, 300])
    ax.set_xlabel("Global epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy over sequential tasks")
    ax.grid(alpha=0.22)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 0.86, 1])

    out_path = fig_dir / args.output_name
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
