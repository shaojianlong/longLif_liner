#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a 6x6 random MNIST digit montage and save it as a .tif image.
"""

from __future__ import annotations

import argparse
import gzip
import random
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "datasets"


def open_binary(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")


def existing_idx_file(raw_dir: Path, file_name: str) -> Path:
    candidates = [raw_dir / file_name, raw_dir / f"{file_name}.gz"]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(f"Missing IDX file: {file_name} in {raw_dir}")


def read_idx(path: Path) -> np.ndarray:
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


def load_mnist(data_root: Path, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    raw_dir = data_root / "MNIST" / "raw"

    if split == "train":
        image_name = "train-images-idx3-ubyte"
        label_name = "train-labels-idx1-ubyte"
    else:
        image_name = "t10k-images-idx3-ubyte"
        label_name = "t10k-labels-idx1-ubyte"

    images = read_idx(existing_idx_file(raw_dir, image_name))
    labels = read_idx(existing_idx_file(raw_dir, label_name))

    return images, labels.astype(np.int64)


def make_montage(
    images: np.ndarray,
    labels: np.ndarray,
    grid_size: int = 6,
    seed: int = 42,
    gap: int = 2,
    margin: int = 2,
) -> Tuple[Image.Image, np.ndarray]:
    rng = random.Random(seed)
    total = grid_size * grid_size

    indices = rng.sample(range(len(images)), total)
    selected = images[indices].astype(np.uint8)
    selected_labels = labels[indices]

    h, w = selected.shape[1], selected.shape[2]

    canvas_h = margin * 2 + grid_size * h + (grid_size - 1) * gap
    canvas_w = margin * 2 + grid_size * w + (grid_size - 1) * gap

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    for n, img in enumerate(selected):
        r = n // grid_size
        c = n % grid_size

        y0 = margin + r * (h + gap)
        x0 = margin + c * (w + gap)

        canvas[y0:y0 + h, x0:x0 + w] = img

    return Image.fromarray(canvas, mode="L"), selected_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 6x6 random MNIST digit montage and save it as TIFF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("mnist_6x6_random.tif"))
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gap", type=int, default=2)
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--save-labels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images, labels = load_mnist(args.data_root.resolve(), args.split)

    montage, selected_labels = make_montage(
        images=images,
        labels=labels,
        grid_size=args.grid_size,
        seed=args.seed,
        gap=args.gap,
        margin=args.margin,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(args.output, format="TIFF")

    if args.save_labels:
        label_grid = selected_labels.reshape(args.grid_size, args.grid_size)
        label_csv = args.output.with_suffix(".labels.csv")
        np.savetxt(label_csv, label_grid, delimiter=",", fmt="%d")
        print(f"Saved labels to: {label_csv}")

    print(f"Saved MNIST montage to: {args.output}")
    print(f"Image size: {montage.size[0]} x {montage.size[1]} pixels")


if __name__ == "__main__":
    main()