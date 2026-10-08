#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from torchvision import datasets
from torchvision.transforms import ToTensor


def download_one_emnist_split(root: Path, split: str):
    print(f"\n==> Downloading EMNIST-{split}")

    train_set = datasets.EMNIST(
        root=str(root),
        split=split,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_set = datasets.EMNIST(
        root=str(root),
        split=split,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    print(f"    train: {len(train_set):,}")
    print(f"    test : {len(test_set):,}")


def main():
    root = Path("./data").resolve()
    root.mkdir(parents=True, exist_ok=True)

    splits = [
        "digits",
        "letters",
        "balanced",
    ]

    failed = []

    for split in splits:
        try:
            download_one_emnist_split(root, split)
        except Exception as e:
            print(f"\n[FAILED] EMNIST-{split}")
            print(f"原因: {repr(e)}")
            failed.append(split)

    print("\n========== Summary ==========")
    print(f"数据目录: {root}")

    if failed:
        print("以下 EMNIST split 下载失败：")
        for split in failed:
            print(f"  - EMNIST-{split}")
        print("\n可以稍后重新运行脚本，已下载完成的部分不会重复下载。")
    else:
        print("EMNIST-Digits / Letters / Balanced 全部下载完成。")


if __name__ == "__main__":
    main()