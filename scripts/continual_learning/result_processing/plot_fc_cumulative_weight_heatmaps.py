#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot cumulative FC weight-usage heatmaps from saved frozen masks.

For each FC layer, this script creates one long figure with six subplots.
The i-th subplot shows the cumulative frozen/used weights after task i.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap


DEFAULT_RUN_DIR = Path(__file__).resolve().parents[3] / "results" / "continual_learning"
TASK_ORDER = [
    "MNIST",
    "QMNIST",
    "EMNIST-Digits",
    "Fashion-MNIST",
    "EMNIST-Letters",
    "EMNIST-Balanced",
]
LAYER_SHAPES = {
    "fc1": (512, 784),
    "fc2": (512, 512),
    "fc3": (256, 512),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cumulative used/frozen FC weight masks after each task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--width", type=int, default=512, help="Model hidden width used by fc.py.")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--output-prefix", type=str, default="cumulative_usage")
    return parser.parse_args()


def layer_shapes(width: int) -> Dict[str, Tuple[int, int]]:
    return {
        "fc1": (width, 784),
        "fc2": (width, width),
        "fc3": (256, width),
    }


def task_index_from_name(path: Path) -> int:
    match = re.search(r"task_(\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot parse task index from file name: {path.name}")
    return int(match.group(1))


def load_mask_snapshots(run_dir: Path) -> List[Tuple[int, str, Dict[str, torch.Tensor]]]:
    mask_dir = run_dir / "masks"
    files = sorted(mask_dir.glob("task_*_frozen.pt"), key=task_index_from_name)
    if not files:
        raise FileNotFoundError(f"No task_*_frozen.pt files found in {mask_dir}")

    snapshots = []
    for path in files:
        data = torch.load(path, map_location="cpu", weights_only=False)
        task_index = int(data.get("ti", task_index_from_name(path)))
        task_name = str(data.get("name", path.stem))
        frozen_masks = data["frozen_masks"]
        snapshots.append((task_index, task_name, frozen_masks))
    return snapshots


def plot_layer(
    layer_name: str,
    snapshots: List[Tuple[int, str, Dict[str, torch.Tensor]]],
    shape: Tuple[int, int],
    out_path: Path,
    dpi: int,
) -> None:
    cmap = ListedColormap(["#f4f4f4", "#1f77b4"])
    fig, axes = plt.subplots(1, len(snapshots), figsize=(3.2 * len(snapshots), 3.6), dpi=dpi)
    if len(snapshots) == 1:
        axes = [axes]

    for ax, (task_index, task_name, masks) in zip(axes, snapshots):
        if layer_name not in masks:
            raise KeyError(f"Layer {layer_name!r} not found in snapshot for {task_name}")
        mask = masks[layer_name].bool().view(shape).numpy().astype(float)
        used = int(mask.sum())
        total = int(mask.size)
        ratio = used / total * 100.0

        ax.imshow(mask, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_title(f"Task {task_index}\n{task_name}\nused {ratio:.1f}%", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{layer_name.upper()} cumulative used/frozen weight mask", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    snapshots = load_mask_snapshots(run_dir)
    shapes = layer_shapes(args.width)

    for layer_name, shape in shapes.items():
        out_path = fig_dir / f"{args.output_prefix}_{layer_name}.png"
        plot_layer(layer_name, snapshots, shape, out_path, args.dpi)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
