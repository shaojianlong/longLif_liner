import os
import copy
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ============================================================
# 1. 路径和参数
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data", "datasets")

CURRENT_CSV = os.path.join(BASE_DIR, "results", "current_analysis", "modeling", "alpha_sigma_analysis", "alpha_sigma_per_target.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "results", "mnist_noise", "inference_scan")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "mnist_clean_best_model.pth")
TRAIN_LOG_CSV = os.path.join(OUTPUT_DIR, "mnist_clean_training_log.csv")
CODEBOOK_CSV = os.path.join(OUTPUT_DIR, "differential_current_codebook.csv")
ALPHA_ACC_CSV = os.path.join(OUTPUT_DIR, "alpha_accuracy_results.csv")
ALPHA_ACC_FIG = os.path.join(OUTPUT_DIR, "alpha_accuracy_curve.tif")
CONFIG_JSON = os.path.join(OUTPUT_DIR, "experiment_config.json")


# 你自己实验拟合得到的 alpha
OWN_ALPHA = 0.037749195736928795

# alpha 扫描列表
ALPHA_LIST = [
    0.00,
    0.005,
    0.01,
    0.015,
    0.02,
    0.025,
    0.03,
    0.035,
    OWN_ALPHA,
    0.04,0.045,
    0.05,0.055,0.06,0.07,0.065,0.075,
    0.08,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
    0.80,
    1.00,
]

ALPHA_LIST = sorted(set(ALPHA_LIST))

EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
SEED = 42

# 如果 data 文件夹里已经有 MNIST，这里保持 False
# 如果没有，可以改成 True
DOWNLOAD_MNIST = False

# 是否重新训练 clean model
# 如果已经训练过且想直接加载 best model，可以改成 False
TRAIN_CLEAN_MODEL = True

# 是否将加入噪声后的电流限制在 [I_min, I_max]
CLIP_CURRENT = True


# ============================================================
# 2. 固定随机种子
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# 3. 定义 MNIST 模型
# ============================================================

class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ============================================================
# 4. MNIST 数据加载
# ============================================================

def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        transform=transform,
        download=DOWNLOAD_MNIST
    )

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        transform=transform,
        download=DOWNLOAD_MNIST
    )

    # 50000 train + 10000 val
    train_size = 50000
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ============================================================
# 5. 训练和评估函数
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_num += labels.size(0)

    avg_loss = total_loss / total_num
    acc = total_correct / total_num

    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        if criterion is not None:
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_num += labels.size(0)

    acc = total_correct / total_num

    if criterion is not None:
        avg_loss = total_loss / total_num
        return avg_loss, acc

    return acc


# ============================================================
# 6. 阶段1：正常训练 MNIST 50 epoch，保存 best model
# ============================================================

def train_clean_mnist_model(train_loader, val_loader):
    model = MNISTCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_epoch = 0

    logs = []

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion
        )

        elapsed = time.time() - start_time

        logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "time_sec": elapsed
        })

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"train_loss={train_loss:.6f}, "
            f"train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.6f}, "
            f"val_acc={val_acc:.4f}, "
            f"time={elapsed:.2f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    log_df = pd.DataFrame(logs)
    log_df.to_csv(TRAIN_LOG_CSV, index=False, encoding="utf-8-sig")

    print("\nClean MNIST training finished.")
    print("Best epoch:", best_epoch)
    print("Best val accuracy:", best_val_acc)
    print("Best model saved to:", BEST_MODEL_PATH)

    return best_epoch, best_val_acc


def load_best_clean_model():
    model = MNISTCNN().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()
    return model


# ============================================================
# 7. 构建归一化电流全组合差分码本
# ============================================================

def find_target_current_column(df):
    candidates = [
        "target",
        "target_current",
        "Target",
        "Target_Current",
        "abs_target_current"
    ]

    for col in candidates:
        if col in df.columns:
            return col

    # 如果列名不匹配，默认使用第二列
    return df.columns[1]


def build_differential_codebook(current_csv):
    df = pd.read_csv(current_csv)

    target_col = find_target_current_column(df)

    currents = pd.to_numeric(
        df[target_col],
        errors="coerce"
    ).dropna().to_numpy(dtype=np.float64)

    currents = np.unique(currents)
    currents = np.sort(currents)

    if len(currents) < 2:
        raise ValueError("目标电流状态数量不足，无法构建差分码本。")

    I_min = currents.min()
    I_max = currents.max()
    I_range = I_max - I_min

    if I_range <= 0:
        raise ValueError("I_max - I_min 必须大于 0。")

    # 电流归一化
    g = (currents - I_min) / I_range

    rows = []

    for i in range(len(currents)):
        for j in range(len(currents)):
            I_pos = currents[i]
            I_neg = currents[j]

            g_pos = g[i]
            g_neg = g[j]

            code_value = g_pos - g_neg

            rows.append({
                "pos_index": i,
                "neg_index": j,
                "I_pos": I_pos,
                "I_neg": I_neg,
                "g_pos": g_pos,
                "g_neg": g_neg,
                "code_value": code_value
            })

    codebook_df = pd.DataFrame(rows)
    codebook_df = codebook_df.sort_values("code_value").reset_index(drop=True)

    codebook_df.to_csv(CODEBOOK_CSV, index=False, encoding="utf-8-sig")

    print("\nDifferential codebook built.")
    print("Target current column:", target_col)
    print("Current states:", len(currents))
    print("Differential code count:", len(codebook_df))
    print("I_min:", I_min)
    print("I_max:", I_max)
    print("Codebook saved to:", CODEBOOK_CSV)

    return codebook_df, I_min, I_max, I_range


# ============================================================
# 8. 权重映射：最近差分码本 + alpha 电流噪声
# ============================================================

def find_nearest_indices(w_norm, code_values):
    """
    对每个归一化权重，寻找最近的差分码本值。
    """
    order = np.argsort(code_values)
    sorted_codes = code_values[order]

    pos = np.searchsorted(sorted_codes, w_norm)

    left_pos = np.clip(pos - 1, 0, len(sorted_codes) - 1)
    right_pos = np.clip(pos, 0, len(sorted_codes) - 1)

    left_val = sorted_codes[left_pos]
    right_val = sorted_codes[right_pos]

    choose_right = np.abs(right_val - w_norm) < np.abs(left_val - w_norm)

    nearest_sorted_pos = np.where(choose_right, right_pos, left_pos)
    nearest_original_idx = order[nearest_sorted_pos]

    return nearest_original_idx


def map_weight_array_once(
    weight_array,
    codebook_df,
    I_min,
    I_max,
    I_range,
    alpha,
    rng
):
    """
    对一个权重矩阵做一次映射：
    原始权重 -> 归一化 -> 最近差分码本 -> I_pos/I_neg 加 alpha 噪声 -> 实际差分权重
    """

    original_shape = weight_array.shape

    w = weight_array.reshape(-1).astype(np.float64)

    scale = np.max(np.abs(w))

    if scale < 1e-12:
        return weight_array.copy()

    # 1. 权重归一化到 [-1, 1]
    w_norm = w / scale
    w_norm = np.clip(w_norm, -1.0, 1.0)

    code_values = codebook_df["code_value"].to_numpy(dtype=np.float64)

    nearest_idx = find_nearest_indices(w_norm, code_values)

    I_pos = codebook_df["I_pos"].to_numpy(dtype=np.float64)[nearest_idx]
    I_neg = codebook_df["I_neg"].to_numpy(dtype=np.float64)[nearest_idx]

    # 2. 对两个电流状态分别加状态比例误差
    # epsilon ~ N(0, (alpha * abs(I))^2)
    noise_pos = rng.normal(
        loc=0.0,
        scale=alpha * np.abs(I_pos)
    )

    noise_neg = rng.normal(
        loc=0.0,
        scale=alpha * np.abs(I_neg)
    )

    I_pos_actual = I_pos + noise_pos
    I_neg_actual = I_neg + noise_neg

    if CLIP_CURRENT:
        I_pos_actual = np.clip(I_pos_actual, I_min, I_max)
        I_neg_actual = np.clip(I_neg_actual, I_min, I_max)

    # 3. 重新计算实际差分码本值
    code_actual = (I_pos_actual - I_neg_actual) / I_range

    # 4. 恢复到原权重尺度
    mapped_w = code_actual * scale

    return mapped_w.reshape(original_shape).astype(weight_array.dtype)


def map_model_once(
    clean_model,
    codebook_df,
    I_min,
    I_max,
    I_range,
    alpha,
    seed
):
    """
    复制 clean best model，然后对所有 weight 做一次映射。
    bias 不做映射。
    """

    mapped_model = copy.deepcopy(clean_model).to(device)
    mapped_model.eval()

    rng = np.random.default_rng(seed)

    with torch.no_grad():
        for name, param in mapped_model.named_parameters():
            if "weight" not in name:
                continue

            weight_np = param.detach().cpu().numpy()

            mapped_weight_np = map_weight_array_once(
                weight_array=weight_np,
                codebook_df=codebook_df,
                I_min=I_min,
                I_max=I_max,
                I_range=I_range,
                alpha=alpha,
                rng=rng
            )

            param.copy_(
                torch.tensor(
                    mapped_weight_np,
                    dtype=param.dtype,
                    device=param.device
                )
            )

    return mapped_model


# ============================================================
# 9. 阶段2：固定 best model，扫描 alpha，只做映射推理
# ============================================================

def run_alpha_inference(clean_model, test_loader, codebook_df, I_min, I_max, I_range):
    results = []

    for alpha in ALPHA_LIST:
        print(f"\nMapping and inference for alpha = {alpha:.10f}")

        mapped_model = map_model_once(
            clean_model=clean_model,
            codebook_df=codebook_df,
            I_min=I_min,
            I_max=I_max,
            I_range=I_range,
            alpha=alpha,
            seed=SEED + int(alpha * 1_000_000)
        )

        test_acc = evaluate(mapped_model, test_loader, criterion=None)

        results.append({
            "alpha": alpha,
            "test_accuracy": test_acc,
            "is_own_alpha": abs(alpha - OWN_ALPHA) < 1e-12
        })

        print(f"alpha={alpha:.10f}, test_accuracy={test_acc:.6f}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(ALPHA_ACC_CSV, index=False, encoding="utf-8-sig")

    print("\nAlpha-Accuracy results saved to:", ALPHA_ACC_CSV)

    return result_df


# ============================================================
# 10. 画 Accuracy-alpha 折线图，自己的 alpha 标红点
# ============================================================

def plot_alpha_accuracy(result_df):
    result_df = result_df.sort_values("alpha").reset_index(drop=True)

    x = result_df["alpha"].to_numpy(dtype=float)
    y = result_df["test_accuracy"].to_numpy(dtype=float)

    plt.figure(figsize=(6.5, 4.5), dpi=300)

    plt.plot(
        x,
        y,
        marker="o",
        linestyle="-",
        linewidth=1.5,
        label="Mapped inference accuracy"
    )

    own_df = result_df[result_df["is_own_alpha"] == True]

    if len(own_df) > 0:
        own_x = own_df["alpha"].iloc[0]
        own_y = own_df["test_accuracy"].iloc[0]

        plt.scatter(
            [own_x],
            [own_y],
            color="red",
            s=80,
            zorder=5,
            label=f"Our alpha = {OWN_ALPHA:.5f}"
        )

        plt.axvline(
            own_x,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.7
        )

        plt.text(
            own_x,
            own_y,
            f"  α={OWN_ALPHA:.4f}",
            color="red",
            fontsize=9,
            va="bottom"
        )

    plt.xlabel("State-proportional coefficient α")
    plt.ylabel("MNIST test accuracy")
    plt.title("MNIST Accuracy vs α after One-time Mapping")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(ALPHA_ACC_FIG, dpi=300, format="tiff")
    plt.close()

    print("Figure saved to:", ALPHA_ACC_FIG)


# ============================================================
# 11. 主函数
# ============================================================

def main():
    config = {
        "base_dir": BASE_DIR,
        "data_dir": DATA_DIR,
        "current_csv": CURRENT_CSV,
        "output_dir": OUTPUT_DIR,
        "own_alpha": OWN_ALPHA,
        "alpha_list": ALPHA_LIST,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "download_mnist": DOWNLOAD_MNIST,
        "train_clean_model": TRAIN_CLEAN_MODEL,
        "clip_current": CLIP_CURRENT,
        "device": str(device),
        "experiment_logic": (
            "Train clean MNIST model for 50 epochs once, save best validation model, "
            "then fix best model and perform one-time weight mapping for each alpha only during inference."
        )
    }

    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print("Experiment config saved to:", CONFIG_JSON)

    train_loader, val_loader, test_loader = get_dataloaders()

    # -----------------------------
    # 阶段1：先正常训练 MNIST
    # -----------------------------
    if TRAIN_CLEAN_MODEL or not os.path.exists(BEST_MODEL_PATH):
        train_clean_mnist_model(train_loader, val_loader)
    else:
        print("Skip clean training, use existing best model:", BEST_MODEL_PATH)

    # 加载最优 clean model
    clean_model = load_best_clean_model()

    clean_test_acc = evaluate(clean_model, test_loader, criterion=None)

    print("\nClean best model test accuracy:", clean_test_acc)

    with open(os.path.join(OUTPUT_DIR, "clean_best_model_test_accuracy.txt"), "w", encoding="utf-8") as f:
        f.write(f"clean_best_model_test_accuracy={clean_test_acc}\n")

    # -----------------------------
    # 阶段2：构建差分码本
    # -----------------------------
    codebook_df, I_min, I_max, I_range = build_differential_codebook(CURRENT_CSV)

    # -----------------------------
    # 阶段3：固定最优模型，扫描 alpha 做一次映射推理
    # -----------------------------
    result_df = run_alpha_inference(
        clean_model=clean_model,
        test_loader=test_loader,
        codebook_df=codebook_df,
        I_min=I_min,
        I_max=I_max,
        I_range=I_range
    )

    # -----------------------------
    # 阶段4：画图
    # -----------------------------
    plot_alpha_accuracy(result_df)

    print("\nAll results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()