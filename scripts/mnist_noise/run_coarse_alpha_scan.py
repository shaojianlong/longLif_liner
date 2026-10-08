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
# 1. 路径与全局参数
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "datasets")

CURRENT_CSV = os.path.join(PROJECT_ROOT, "results", "current_analysis", "modeling", "alpha_sigma_analysis", "alpha_sigma_per_target.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "mnist_noise", "coarse_scan")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_CLEAN_MODEL_PATH = os.path.join(OUTPUT_DIR, "mnist_clean_best_model.pth")
CODEBOOK_CSV = os.path.join(OUTPUT_DIR, "differential_current_codebook.csv")
RAW_RESULT_CSV = os.path.join(OUTPUT_DIR, "alpha_accuracy_raw_results.csv")
SUMMARY_RESULT_CSV = os.path.join(OUTPUT_DIR, "alpha_accuracy_summary.csv")
FIG_PATH = os.path.join(OUTPUT_DIR, "alpha_accuracy_curve.tif")
CONFIG_JSON = os.path.join(OUTPUT_DIR, "experiment_config.json")


# 你自己从 sigma = alpha * abs(target_current) 拟合得到的 alpha
OWN_ALPHA = 0.037749195736928795

# alpha 扫描范围
# 会自动把 OWN_ALPHA 加进去，并排序
ALPHA_LIST = [
    0.00,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
]

# 每个 alpha 重复次数
# 如果你只想每个 alpha 映射一次，就设为 1
# 为了曲线更稳定，建议设为 5 或 10
REPEAT_TIMES = 5

# clean model 训练 epoch
# 注意：这里是 clean model 只训练一次，不是每个 alpha 训练一次
CLEAN_EPOCHS = 50

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
VAL_RATIO = 10000 / 60000

SEED = 42

# MNIST 如果已经在 data 文件夹中，保持 False
# 如果本地没有 MNIST，可以改成 True 自动下载
DOWNLOAD_MNIST = False

# 是否重新训练 clean model
# 如果 False 且 BEST_CLEAN_MODEL_PATH 存在，则直接加载
TRAIN_CLEAN_MODEL = True

# 是否对加入噪声后的电流限制在 [I_min, I_max]
# 物理上更合理；如果你想看更强的误差破坏效果，可以改成 False
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
# 3. 定义 MNIST CNN 模型
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================
# 4. 读取 MNIST 数据
# ============================================================

def get_mnist_loaders():
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

    val_size = 10000
    train_size = len(full_train_dataset) - val_size

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
# 5. 训练和测试函数
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

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
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        if criterion is not None:
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total

    if criterion is not None:
        avg_loss = total_loss / total
        return avg_loss, acc

    return acc


def train_clean_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_epoch = 0
    train_log = []

    for epoch in range(1, CLEAN_EPOCHS + 1):
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

        epoch_time = time.time() - start_time

        train_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "time_sec": epoch_time
        })

        print(
            f"Epoch [{epoch:03d}/{CLEAN_EPOCHS}] "
            f"train_loss={train_loss:.6f}, "
            f"train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.6f}, "
            f"val_acc={val_acc:.4f}, "
            f"time={epoch_time:.2f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), BEST_CLEAN_MODEL_PATH)

    log_df = pd.DataFrame(train_log)
    log_df.to_csv(
        os.path.join(OUTPUT_DIR, "clean_training_log.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\nClean model training finished.")
    print("Best epoch:", best_epoch)
    print("Best val acc:", best_val_acc)
    print("Best model saved to:", BEST_CLEAN_MODEL_PATH)

    return best_val_acc, best_epoch


# ============================================================
# 6. 读取目标电流并构建差分码本
# ============================================================

def find_target_current_column(df):
    candidate_cols = [
        "target",
        "target_current",
        "Target",
        "Target_Current",
        "abs_target_current"
    ]

    for col in candidate_cols:
        if col in df.columns:
            return col

    # 如果找不到，默认用第二列
    return df.columns[1]


def build_differential_codebook(current_csv, own_alpha):
    df = pd.read_csv(current_csv)

    target_col = find_target_current_column(df)

    current_states = pd.to_numeric(
        df[target_col],
        errors="coerce"
    ).dropna().to_numpy(dtype=float)

    # 去重并排序
    current_states = np.unique(current_states)
    current_states = np.sort(current_states)

    if len(current_states) < 2:
        raise ValueError("目标电流状态数量太少，无法构建差分码本。")

    I_min = np.min(current_states)
    I_max = np.max(current_states)
    I_range = I_max - I_min

    if I_range <= 0:
        raise ValueError("I_max - I_min 必须大于 0。")

    g_states = (current_states - I_min) / I_range

    rows = []

    for i in range(len(current_states)):
        for j in range(len(current_states)):
            I_pos = current_states[i]
            I_neg = current_states[j]

            g_pos = g_states[i]
            g_neg = g_states[j]

            code_value = g_pos - g_neg

            # 差分码本对应的理论标准差
            # sigma_code = alpha * sqrt(I_pos^2 + I_neg^2) / I_range
            sigma_code = own_alpha * np.sqrt(I_pos ** 2 + I_neg ** 2) / I_range

            rows.append({
                "pos_index": i,
                "neg_index": j,
                "I_pos": I_pos,
                "I_neg": I_neg,
                "g_pos": g_pos,
                "g_neg": g_neg,
                "code_value": code_value,
                "sigma_code_at_own_alpha": sigma_code
            })

    codebook_df = pd.DataFrame(rows)
    codebook_df = codebook_df.sort_values("code_value").reset_index(drop=True)

    codebook_df.to_csv(
        CODEBOOK_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDifferential codebook built.")
    print("Target current column:", target_col)
    print("Number of current states:", len(current_states))
    print("Number of differential codes:", len(codebook_df))
    print("I_min:", I_min)
    print("I_max:", I_max)
    print("Codebook saved to:", CODEBOOK_CSV)

    return codebook_df, I_min, I_max, I_range, target_col


# ============================================================
# 7. 权重映射函数：差分码本 + alpha 电流噪声
# ============================================================

def find_nearest_code_indices(w_norm, code_values):
    """
    对每个归一化权重 w_norm，找到最近的 code_value。
    使用排序 + searchsorted，比直接构造巨大距离矩阵更稳。
    """
    order = np.argsort(code_values)
    sorted_codes = code_values[order]

    positions = np.searchsorted(sorted_codes, w_norm)

    positions_left = np.clip(positions - 1, 0, len(sorted_codes) - 1)
    positions_right = np.clip(positions, 0, len(sorted_codes) - 1)

    left_codes = sorted_codes[positions_left]
    right_codes = sorted_codes[positions_right]

    choose_right = np.abs(right_codes - w_norm) < np.abs(left_codes - w_norm)

    nearest_sorted_positions = np.where(
        choose_right,
        positions_right,
        positions_left
    )

    nearest_original_indices = order[nearest_sorted_positions]

    return nearest_original_indices


def map_weight_array_with_alpha(
    weight_array,
    codebook_df,
    I_min,
    I_max,
    I_range,
    alpha,
    rng,
    clip_current=True
):
    original_shape = weight_array.shape
    w = weight_array.reshape(-1).astype(np.float64)

    weight_scale = np.max(np.abs(w))

    if weight_scale < 1e-12:
        return weight_array.copy()

    # 1. 权重归一化到 [-1, 1]
    w_norm = w / weight_scale
    w_norm = np.clip(w_norm, -1.0, 1.0)

    code_values = codebook_df["code_value"].to_numpy(dtype=np.float64)

    nearest_indices = find_nearest_code_indices(w_norm, code_values)

    I_pos = codebook_df["I_pos"].to_numpy(dtype=np.float64)[nearest_indices]
    I_neg = codebook_df["I_neg"].to_numpy(dtype=np.float64)[nearest_indices]

    # 2. 对差分对中的两个目标电流分别加入 state-proportional noise
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

    # 3. 可选：限制实际电流不超出器件范围
    if clip_current:
        I_pos_actual = np.clip(I_pos_actual, I_min, I_max)
        I_neg_actual = np.clip(I_neg_actual, I_min, I_max)

    # 4. 重新计算实际差分码本值
    code_actual = (I_pos_actual - I_neg_actual) / I_range

    # 5. 映射回原权重尺度
    mapped_w = code_actual * weight_scale

    return mapped_w.reshape(original_shape).astype(weight_array.dtype)


def map_model_once_for_inference(
    clean_model,
    codebook_df,
    I_min,
    I_max,
    I_range,
    alpha,
    seed,
    clip_current=True
):
    mapped_model = copy.deepcopy(clean_model)
    mapped_model.to(device)
    mapped_model.eval()

    rng = np.random.default_rng(seed)

    with torch.no_grad():
        for name, param in mapped_model.named_parameters():
            # 一般只映射 weight，不映射 bias
            if "weight" not in name:
                continue

            weight_np = param.detach().cpu().numpy()

            mapped_weight_np = map_weight_array_with_alpha(
                weight_array=weight_np,
                codebook_df=codebook_df,
                I_min=I_min,
                I_max=I_max,
                I_range=I_range,
                alpha=alpha,
                rng=rng,
                clip_current=clip_current
            )

            mapped_weight_tensor = torch.tensor(
                mapped_weight_np,
                dtype=param.dtype,
                device=param.device
            )

            param.copy_(mapped_weight_tensor)

    return mapped_model


# ============================================================
# 8. alpha-Accuracy 推理实验
# ============================================================

def run_alpha_accuracy_experiment(
    clean_model,
    test_loader,
    codebook_df,
    I_min,
    I_max,
    I_range
):
    alpha_list = sorted(set(ALPHA_LIST + [OWN_ALPHA]))

    raw_results = []

    for alpha in alpha_list:
        print(f"\nRunning alpha = {alpha:.10f}")

        for repeat in range(REPEAT_TIMES):
            seed = SEED + repeat + int(alpha * 1_000_000)

            mapped_model = map_model_once_for_inference(
                clean_model=clean_model,
                codebook_df=codebook_df,
                I_min=I_min,
                I_max=I_max,
                I_range=I_range,
                alpha=alpha,
                seed=seed,
                clip_current=CLIP_CURRENT
            )

            test_acc = evaluate(mapped_model, test_loader, criterion=None)

            raw_results.append({
                "alpha": alpha,
                "repeat": repeat,
                "test_accuracy": test_acc,
                "is_own_alpha": abs(alpha - OWN_ALPHA) < 1e-12
            })

            print(
                f"alpha={alpha:.10f}, "
                f"repeat={repeat}, "
                f"test_acc={test_acc:.6f}"
            )

    raw_df = pd.DataFrame(raw_results)
    raw_df.to_csv(
        RAW_RESULT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    summary_df = raw_df.groupby("alpha", as_index=False).agg(
        mean_accuracy=("test_accuracy", "mean"),
        std_accuracy=("test_accuracy", "std"),
        max_accuracy=("test_accuracy", "max"),
        min_accuracy=("test_accuracy", "min"),
        repeat_count=("test_accuracy", "count")
    )

    summary_df["is_own_alpha"] = summary_df["alpha"].apply(
        lambda x: abs(x - OWN_ALPHA) < 1e-12
    )

    summary_df.to_csv(
        SUMMARY_RESULT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nAlpha-Accuracy experiment finished.")
    print("Raw results saved to:", RAW_RESULT_CSV)
    print("Summary results saved to:", SUMMARY_RESULT_CSV)

    return raw_df, summary_df


# ============================================================
# 9. 绘制 alpha-Accuracy 折线图
# ============================================================

def plot_alpha_accuracy(summary_df):
    summary_df = summary_df.sort_values("alpha").reset_index(drop=True)

    x = summary_df["alpha"].to_numpy(dtype=float)
    y = summary_df["mean_accuracy"].to_numpy(dtype=float)
    yerr = summary_df["std_accuracy"].fillna(0).to_numpy(dtype=float)

    plt.figure(figsize=(6.5, 4.5), dpi=300)

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        linestyle="-",
        capsize=3,
        label="Mean accuracy"
    )

    own_row = summary_df[summary_df["is_own_alpha"] == True]

    if len(own_row) > 0:
        own_x = own_row["alpha"].iloc[0]
        own_y = own_row["mean_accuracy"].iloc[0]

        plt.scatter(
            [own_x],
            [own_y],
            color="red",
            s=70,
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
    plt.title("MNIST Accuracy vs α after One-time Weight Mapping")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(FIG_PATH, dpi=300, format="tiff")
    plt.close()

    print("Figure saved to:", FIG_PATH)


# ============================================================
# 10. 主程序
# ============================================================

def main():
    config = {
        "data_dir": DATA_DIR,
        "current_csv": CURRENT_CSV,
        "output_dir": OUTPUT_DIR,
        "own_alpha": OWN_ALPHA,
        "alpha_list": sorted(set(ALPHA_LIST + [OWN_ALPHA])),
        "repeat_times": REPEAT_TIMES,
        "clean_epochs": CLEAN_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "clip_current": CLIP_CURRENT,
        "seed": SEED,
        "train_clean_model": TRAIN_CLEAN_MODEL,
        "download_mnist": DOWNLOAD_MNIST,
        "device": str(device)
    }

    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print("Config saved to:", CONFIG_JSON)

    train_loader, val_loader, test_loader = get_mnist_loaders()

    # 1. 训练或加载 clean model
    clean_model = SimpleCNN().to(device)

    if TRAIN_CLEAN_MODEL or not os.path.exists(BEST_CLEAN_MODEL_PATH):
        train_clean_model(clean_model, train_loader, val_loader)
    else:
        print("Skip clean training. Loading:", BEST_CLEAN_MODEL_PATH)

    clean_model.load_state_dict(
        torch.load(BEST_CLEAN_MODEL_PATH, map_location=device)
    )
    clean_model.to(device)
    clean_model.eval()

    clean_test_acc = evaluate(clean_model, test_loader, criterion=None)
    print("\nClean best model test accuracy:", clean_test_acc)

    with open(
        os.path.join(OUTPUT_DIR, "clean_test_accuracy.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(f"clean_test_accuracy={clean_test_acc}\n")

    # 2. 构建差分码本
    codebook_df, I_min, I_max, I_range, target_col = build_differential_codebook(
        current_csv=CURRENT_CSV,
        own_alpha=OWN_ALPHA
    )

    # 3. alpha-Accuracy 实验
    raw_df, summary_df = run_alpha_accuracy_experiment(
        clean_model=clean_model,
        test_loader=test_loader,
        codebook_df=codebook_df,
        I_min=I_min,
        I_max=I_max,
        I_range=I_range
    )

    # 4. 画图
    plot_alpha_accuracy(summary_df)

    print("\nAll results have been saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()