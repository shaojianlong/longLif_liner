from pathlib import Path
import gzip
import hashlib
import shutil
import time

import requests
from tqdm import tqdm
from torchvision import datasets


# Repository root resolved from this script location.
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "datasets"
RAW_DIR = DATA_DIR / "MNIST" / "raw"

# Google 提供的稳定 MNIST 文件镜像
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def md5sum(file_path: Path) -> str:
    """计算文件 MD5，用于判断下载是否完整。"""
    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)

    return md5.hexdigest()


def download_file(url: str, save_path: Path, expected_md5: str, retries: int = 5):
    """带重试机制地下载单个文件，并校验 MD5。"""
    if save_path.exists():
        current_md5 = md5sum(save_path)

        if current_md5 == expected_md5:
            print(f"已存在且校验通过：{save_path.name}")
            return
        else:
            print(f"发现损坏或未下载完整的文件，删除后重下：{save_path.name}")
            save_path.unlink()

    temp_path = save_path.with_suffix(save_path.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            print(f"\n下载：{url}")
            print(f"第 {attempt}/{retries} 次尝试")

            with requests.get(
                url,
                stream=True,
                timeout=(15, 180),
                headers={"User-Agent": "Mozilla/5.0"},
            ) as response:
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))

                with open(temp_path, "wb") as f, tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=save_path.name,
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))

            downloaded_md5 = md5sum(temp_path)

            if downloaded_md5 != expected_md5:
                temp_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"MD5 校验失败：期望 {expected_md5}，实际 {downloaded_md5}"
                )

            temp_path.replace(save_path)
            print(f"下载成功：{save_path.name}")
            return

        except Exception as e:
            print(f"下载失败：{e}")

            if temp_path.exists():
                temp_path.unlink()

            if attempt == retries:
                raise RuntimeError(
                    f"\n文件下载失败：{save_path.name}\n"
                    f"请检查网络、代理、VPN 或 SSL 证书设置。"
                ) from e

            wait_seconds = attempt * 3
            print(f"{wait_seconds} 秒后重试……")
            time.sleep(wait_seconds)


def extract_gz(gz_path: Path):
    """解压 .gz 文件，得到 torchvision 可直接识别的 IDX 原始文件。"""
    output_path = gz_path.with_suffix("")

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"已解压：{output_path.name}")
        return

    print(f"解压：{gz_path.name}")

    with gzip.open(gz_path, "rb") as src, open(output_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    print(f"解压完成：{output_path.name}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"数据保存位置：{DATA_DIR}")
    print("=" * 70)

    for filename, expected_md5 in FILES.items():
        file_url = BASE_URL + filename
        gz_path = RAW_DIR / filename

        download_file(
            url=file_url,
            save_path=gz_path,
            expected_md5=expected_md5,
        )

        extract_gz(gz_path)

    print("\n" + "=" * 70)
    print("MNIST 文件已下载并解压。")

    # 检查 torchvision 是否可以正常读取
    train_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=False,
    )

    test_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=False,
    )

    print(f"训练集数量：{len(train_dataset)}")
    print(f"测试集数量：{len(test_dataset)}")
    print("MNIST 已可正常使用。")


if __name__ == "__main__":
    main()