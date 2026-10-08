#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a 3x3 Fashion-MNIST image grid with visible x/y axes and save as TIFF.

Default data layout:
    <repository-root>/data/datasets/FashionMNIST/raw/
        train-images-idx3-ubyte(.gz)
        train-labels-idx1-ubyte(.gz)
        t10k-images-idx3-ubyte(.gz)
        t10k-labels-idx1-ubyte(.gz)

Example:
    python make_fashionmnist_3x3_tif_with_axes.py \
        --data-root "<repository-root>/data/datasets" \
        --output fashionmnist_3x3_axes.tif
"""

from __future__ import annotations

import argparse
import gzip
import random
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "datasets"

FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def open_binary(path: Path):
    """Open plain or gzipped IDX file."""
    return gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")


def existing_idx_file(raw_dir: Path, file_name: str) -> Path:
    """Find IDX file with or without .gz suffix."""
    candidates = [raw_dir / file_name, raw_dir / f"{file_name}.gz"]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(f"Missing IDX file: {file_name} or {file_name}.gz in {raw_dir}")


def read_idx(path: Path) -> np.ndarray:
    """Read an IDX-format array."""
    dtype_map = {
        0x08: np.uint8,
        0x09: np.int8,
        0x0B: np.int16,
        0x0C: np.int32,
        0x0D: np.float32,
        0x0E: np.float64,
    }
    with open_binary(path) as f:
        magic = int.from_bytes(f.read(4), byteorder="big")
        dtype_code = (magic >> 8) & 0xFF
        ndim = magic & 0xFF
        if dtype_code not in dtype_map:
            raise ValueError(f"Unsupported IDX dtype code {dtype_code} in {path}")
        shape = [int.from_bytes(f.read(4), byteorder="big") for _ in range(ndim)]
        data = f.read()

    dtype = np.dtype(dtype_map[dtype_code]).newbyteorder(">")
    array = np.frombuffer(data, dtype=dtype).reshape(shape)
    return array.astype(dtype_map[dtype_code], copy=False)


def load_fashion_mnist(data_root: Path, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """Load Fashion-MNIST from raw IDX files."""
    raw_dir = data_root / "FashionMNIST" / "raw"

    if split == "train":
        image_name = "train-images-idx3-ubyte"
        label_name = "train-labels-idx1-ubyte"
    elif split == "test":
        image_name = "t10k-images-idx3-ubyte"
        label_name = "t10k-labels-idx1-ubyte"
    else:
        raise ValueError("split must be 'train' or 'test'")

    images = read_idx(existing_idx_file(raw_dir, image_name))
    labels = read_idx(existing_idx_file(raw_dir, label_name)).astype(np.int64)
    return images, labels


def make_figure(
    images: np.ndarray,
    labels: np.ndarray,
    output: Path,
    seed: int = 42,
    show_titles: bool = False,
    dpi: int = 300,
) -> None:
    """Create a 3x3 grid with axes and save as TIFF."""
    rng = random.Random(seed)
    indices = rng.sample(range(len(images)), 9)

    fig, axes = plt.subplots(3, 3, figsize=(4.2, 4.2), dpi=dpi)

    for ax, idx in zip(axes.ravel(), indices):
        img = images[idx]
        label = int(labels[idx])

        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")

        # Show x/y axes like the reference image.
        ax.set_xticks([0, 10, 20])
        ax.set_yticks([0, 10, 20])
        ax.tick_params(axis="both", which="major", labelsize=6, length=2, width=0.6)

        # Keep full image range and make the image coordinate system explicit.
        ax.set_xlim(-0.5, 27.5)
        ax.set_ylim(27.5, -0.5)

        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

        if show_titles:
            ax.set_title(FASHION_CLASSES[label], fontsize=6)

    plt.tight_layout(pad=0.35, w_pad=0.35, h_pad=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="tiff", dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    label_grid = labels[indices].reshape(3, 3)
    label_csv = output.with_suffix(".labels.csv")
    np.savetxt(label_csv, label_grid, delimiter=",", fmt="%d")
    print(f"Saved TIFF figure: {output}")
    print(f"Saved label grid:  {label_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 3x3 Fashion-MNIST TIFF figure with x/y axes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("fashionmnist_3x3_axes.tif"))
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show-titles", action="store_true", help="Show Fashion-MNIST class names above each subplot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images, labels = load_fashion_mnist(args.data_root.resolve(), args.split)
    make_figure(
        images=images,
        labels=labels,
        output=args.output,
        seed=args.seed,
        show_titles=args.show_titles,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
