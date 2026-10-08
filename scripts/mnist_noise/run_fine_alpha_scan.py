import os
import copy
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. 路径设置
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data", "datasets")

BEST_MODEL_PATH = os.path.join(BASE_DIR, "results", "mnist_noise", "inference_scan", "mnist_clean_best_model.pth")

CURRENT_CSV = os.path.join(BASE_DIR, "results", "current_analysis", "modeling", "alpha_sigma_analysis", "alpha_sigma_per_target.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "results", "mnist_noise", "fine_scan")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CODEBOOK_CSV = os.path.join(OUTPUT_DIR, "differential_current_codebook.csv")
ALPHA_ACC_CSV = os.path.join(OUTPUT_DIR, "alpha_accuracy_fine_results.csv")
ALPHA_ACC_FIG = os.path.join(OUTPUT_DIR, "alpha_accuracy_fine_curve.tif")
CONFIG_JSON = os.path.join(OUTPUT_DIR, "alpha_mapping_fine_config.json")


# ============================================================
# 2. 实验参数
# ============================================================

# 你自己实验拟合得到的 alpha
OWN_ALPHA = 0.037749195736928795

# alpha 从 0 到 1，每 0.005 一个档位
ALPHA_STEP = 0.005
ALPHA_GRID = np.arange(0.0, 1.0 + ALPHA_STEP / 2, ALPHA_STEP).tolist()

# 额外把自己的 alpha 加进去，便于图中标红点
ALPHA_LIST = sorted(set([round(a, 10) for a in ALPHA_GRID] + [OWN_ALPHA]))

BATCH_SIZE = 128
SEED = 42
DOWNLOAD_MNIST = False

# 是否将加入噪声后的电流限制在 [I_min, I_max]
# True 更符合器件电流状态范围；False 可以看到更强的噪声破坏效果
CLIP_CURRENT = True

# 每个 alpha 做几次随机映射
# 如果只想严格“一次映射推理”，保持 1
# 如果想让曲线更平滑，可以改成 3 或 5
REPEAT_TIMES = 1


# ============================================================
# 3. 固定随机种子
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
# 4. 模型结构
#    注意：这里必须和你训练 mnist_clean_best_model.pth 时的模型结构一致
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
# 5. 加载 MNIST test set
# ============================================================

def get_test_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        transform=transform,
        download=DOWNLOAD_MNIST
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return test_loader


# ============================================================
# 6. 加载最优模型
# ============================================================

def load_best_model():
    model = MNISTCNN().to(device)

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)

    # 兼容两种保存方式：
    # 1. torch.save(model.state_dict(), path)
    # 2. torch.save({"state_dict": model.state_dict()}, path)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Best model loaded from:", BEST_MODEL_PATH)

    return model


# ============================================================
# 7. 测试函数
# ============================================================

@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total_correct = 0
    total_num = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        total_correct += (preds == labels).sum().item()
        total_num += labels.size(0)

    acc = total_correct / total_num

    return acc


# ============================================================
# 8. 构建归一化电流全组合差分码本
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
# 9. 最近码本查找
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


# ============================================================
# 10. 权重映射函数
# ============================================================

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
    原始权重
        ↓
    归一化到 [-1, 1]
        ↓
    找最近差分码本 code_value
        ↓
    找到对应 I_pos 和 I_neg
        ↓
    对 I_pos 和 I_neg 分别加 alpha 电流噪声
        ↓
    重新计算实际差分权重
        ↓
    恢复到原权重尺度
    """

    original_shape = weight_array.shape
    w = weight_array.reshape(-1).astype(np.float64)

    scale = np.max(np.abs(w))

    if scale < 1e-12:
        return weight_array.copy()

    # 1. 权重归一化
    w_norm = w / scale
    w_norm = np.clip(w_norm, -1.0, 1.0)

    code_values = codebook_df["code_value"].to_numpy(dtype=np.float64)

    # 2. 找最近的差分码本值
    nearest_idx = find_nearest_indices(w_norm, code_values)

    I_pos_all = codebook_df["I_pos"].to_numpy(dtype=np.float64)
    I_neg_all = codebook_df["I_neg"].to_numpy(dtype=np.float64)

    I_pos = I_pos_all[nearest_idx]
    I_neg = I_neg_all[nearest_idx]

    # 3. 对两个电流状态分别加入状态比例误差
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

    # 4. 可选：限制在器件电流范围内
    if CLIP_CURRENT:
        I_pos_actual = np.clip(I_pos_actual, I_min, I_max)
        I_neg_actual = np.clip(I_neg_actual, I_min, I_max)

    # 5. 重新计算实际差分码本值
    code_actual = (I_pos_actual - I_neg_actual) / I_range

    # 6. 恢复到原权重尺度
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
    bias 不映射。
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
# 11. alpha-Accuracy 推理实验
# ============================================================

def run_alpha_mapping_inference(
    clean_model,
    test_loader,
    codebook_df,
    I_min,
    I_max,
    I_range
):
    results = []

    for alpha in ALPHA_LIST:
        acc_list = []

        print(f"\nAlpha = {alpha:.10f}")

        for repeat in range(REPEAT_TIMES):
            seed = SEED + repeat + int(alpha * 1_000_000)

            mapped_model = map_model_once(
                clean_model=clean_model,
                codebook_df=codebook_df,
                I_min=I_min,
                I_max=I_max,
                I_range=I_range,
                alpha=alpha,
                seed=seed
            )

            acc = evaluate(mapped_model, test_loader)
            acc_list.append(acc)

            print(
                f"alpha={alpha:.10f}, "
                f"repeat={repeat}, "
                f"test_accuracy={acc:.6f}"
            )

        mean_acc = float(np.mean(acc_list))
        std_acc = float(np.std(acc_list, ddof=1)) if len(acc_list) > 1 else 0.0

        results.append({
            "alpha": alpha,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "repeat_times": REPEAT_TIMES,
            "is_own_alpha": abs(alpha - OWN_ALPHA) < 1e-12
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(ALPHA_ACC_CSV, index=False, encoding="utf-8-sig")

    print("\nAlpha-Accuracy results saved to:", ALPHA_ACC_CSV)

    return result_df


# ============================================================
# 12. 绘制 Accuracy-alpha 折线图
# ============================================================

def plot_alpha_accuracy(result_df):
    result_df = result_df.sort_values("alpha").reset_index(drop=True)

    x = result_df["alpha"].to_numpy(dtype=float)
    y = result_df["mean_accuracy"].to_numpy(dtype=float)
    yerr = result_df["std_accuracy"].to_numpy(dtype=float)

    plt.figure(figsize=(7.0, 4.8), dpi=300)

    if REPEAT_TIMES > 1:
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            markersize=2.5,
            linestyle="-",
            linewidth=1.2,
            capsize=2,
            label="Mapped inference accuracy"
        )
    else:
        plt.plot(
            x,
            y,
            marker="o",
            markersize=2.5,
            linestyle="-",
            linewidth=1.2,
            label="Mapped inference accuracy"
        )

    own_df = result_df[result_df["is_own_alpha"] == True]

    if len(own_df) > 0:
        own_x = own_df["alpha"].iloc[0]
        own_y = own_df["mean_accuracy"].iloc[0]

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
    plt.title("MNIST Accuracy vs α after One-time Weight Mapping")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(ALPHA_ACC_FIG, dpi=300, format="tiff")
    plt.close()

    print("Figure saved to:", ALPHA_ACC_FIG)


# ============================================================
# 13. 主函数
# ============================================================

def main():
    config = {
        "base_dir": BASE_DIR,
        "data_dir": DATA_DIR,
        "best_model_path": BEST_MODEL_PATH,
        "current_csv": CURRENT_CSV,
        "output_dir": OUTPUT_DIR,
        "own_alpha": OWN_ALPHA,
        "alpha_step": ALPHA_STEP,
        "alpha_min": 0.0,
        "alpha_max": 1.0,
        "alpha_count": len(ALPHA_LIST),
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "download_mnist": DOWNLOAD_MNIST,
        "clip_current": CLIP_CURRENT,
        "repeat_times": REPEAT_TIMES,
        "device": str(device),
        "experiment_logic": (
            "Load trained best MNIST model directly, "
            "construct differential current codebook from target current, "
            "map model weights once for each alpha, "
            "and evaluate MNIST test accuracy without retraining."
        )
    }

    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print("Config saved to:", CONFIG_JSON)

    # 1. 加载 MNIST test set
    test_loader = get_test_loader()

    # 2. 加载最优模型
    clean_model = load_best_model()

    # 3. 测试 clean model 原始准确率
    clean_acc = evaluate(clean_model, test_loader)
    print("\nClean best model test accuracy:", clean_acc)

    with open(os.path.join(OUTPUT_DIR, "clean_best_model_test_accuracy.txt"), "w", encoding="utf-8") as f:
        f.write(f"clean_best_model_test_accuracy={clean_acc}\n")

    # 4. 构建差分码本
    codebook_df, I_min, I_max, I_range = build_differential_codebook(CURRENT_CSV)

    # 5. alpha 扫描映射推理
    result_df = run_alpha_mapping_inference(
        clean_model=clean_model,
        test_loader=test_loader,
        codebook_df=codebook_df,
        I_min=I_min,
        I_max=I_max,
        I_range=I_range
    )

    # 6. 画图
    plot_alpha_accuracy(result_df)

    print("\nAll results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()