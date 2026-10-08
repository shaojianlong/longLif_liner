import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "results" / "current_analysis" / "modeling" / "curve1_x_replaced_points_with_target.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "current_analysis" / "modeling" / "target_current_distribution_modeling"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HIST_DIR = os.path.join(OUTPUT_DIR, "histograms")
os.makedirs(HIST_DIR, exist_ok=True)

STATS_CSV = os.path.join(OUTPUT_DIR, "target_current_distribution_stats.csv")
HIST_DATA_CSV = os.path.join(OUTPUT_DIR, "target_current_histogram_bin_data.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "target_current_distribution_summary.json")


# ============================================================
# 2. 参数设置
# ============================================================

TARGET_COLUMN_CANDIDATES = [
    "target",
    "target_current",
    "Target",
    "Target_Current",
    "abs_target_current"
]

# 直方图分成 5 个小段
BIN_COUNT = 5

# 样本标准差，ddof=1
STD_DDOF = 1

# 防止 target = 0 时除零
EPS = 1e-30


# ============================================================
# 3. 自动识别目标电流列和实际电流列
# ============================================================

def find_target_column(df):
    for col in TARGET_COLUMN_CANDIDATES:
        if col in df.columns:
            return col

    # 如果找不到，默认第二列是目标电流
    if len(df.columns) >= 2:
        return df.columns[1]

    raise ValueError("无法识别目标电流列。")


def find_actual_current_columns(df, target_col):
    """
    优先识别 Curve*_Y 作为实际电流列。
    如果没有 Curve*_Y，则尝试识别 actual 相关列。
    """
    curve_cols = [
        col for col in df.columns
        if col.startswith("Curve") and col.endswith("_Y")
    ]

    if len(curve_cols) > 0:
        return curve_cols

    actual_cols = [
        col for col in df.columns
        if "actual" in col.lower()
        or "current_" in col.lower()
        or "device" in col.lower()
    ]

    # 排除目标列和已经计算出来的统计列
    exclude_cols = {
        target_col,
        "mean_actual_current",
        "sigma",
        "mean_bias",
        "alpha_i",
        "abs_target_current",
        "X"
    }

    actual_cols = [col for col in actual_cols if col not in exclude_cols]

    if len(actual_cols) > 0:
        return actual_cols

    raise ValueError(
        "没有找到 24 个实际电流列。请确认输入 CSV 中包含 Curve1_Y、Curve2_Y ... 这类实际电流列。"
    )


# ============================================================
# 4. 主计算函数
# ============================================================

def main():
    df = pd.read_csv(INPUT_CSV)

    target_col = find_target_column(df)
    actual_cols = find_actual_current_columns(df, target_col)

    print("Input CSV:", INPUT_CSV)
    print("Target current column:", target_col)
    print("Actual current columns:", actual_cols)
    print("Actual current column count:", len(actual_cols))

    # 转换为数值
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    for col in actual_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    stats_rows = []
    hist_rows = []

    for row_idx, row in df.iterrows():
        target_current = row[target_col]

        actual_values = row[actual_cols].to_numpy(dtype=float)
        actual_values = actual_values[~np.isnan(actual_values)]

        if len(actual_values) == 0:
            continue

        # ====================================================
        # 4.1 计算 μ_i、σ_i、Bias_i、alpha_i
        # ====================================================

        mu_i = np.mean(actual_values)

        if len(actual_values) > 1:
            sigma_i = np.std(actual_values, ddof=STD_DDOF)
        else:
            sigma_i = 0.0

        bias_i = mu_i - target_current

        abs_target = abs(target_current)

        if abs_target > EPS:
            alpha_i = sigma_i / abs_target
        else:
            alpha_i = np.nan

        actual_min = np.min(actual_values)
        actual_max = np.max(actual_values)
        actual_range = actual_max - actual_min

        stats_rows.append({
            "row_index": row_idx,
            "target_current": target_current,
            "actual_count": len(actual_values),
            "mu_i": mu_i,
            "sigma_i": sigma_i,
            "alpha_i": alpha_i,
            "alpha_i_percent": alpha_i * 100 if not np.isnan(alpha_i) else np.nan,
            "bias_i": bias_i,
            "abs_bias_i": abs(bias_i),
            "actual_min": actual_min,
            "actual_max": actual_max,
            "actual_range": actual_range,
            "actual_median": np.median(actual_values),
            "actual_q25": np.percentile(actual_values, 25),
            "actual_q75": np.percentile(actual_values, 75)
        })

        # ====================================================
        # 4.2 构建 5 段直方图
        # ====================================================

        if actual_max == actual_min:
            # 如果 24 个值完全一样，手动扩展一个极小范围，避免 histogram 报错
            delta = abs(actual_min) * 1e-6
            if delta == 0:
                delta = 1e-12

            hist_min = actual_min - delta
            hist_max = actual_max + delta
        else:
            hist_min = actual_min
            hist_max = actual_max

        bin_edges = np.linspace(hist_min, hist_max, BIN_COUNT + 1)
        counts, edges = np.histogram(actual_values, bins=bin_edges)

        probabilities = counts / len(actual_values)

        bin_widths = edges[1:] - edges[:-1]
        densities = counts / (len(actual_values) * bin_widths)

        for bin_idx in range(BIN_COUNT):
            hist_rows.append({
                "row_index": row_idx,
                "target_current": target_current,
                "bin_index": bin_idx + 1,
                "bin_left": edges[bin_idx],
                "bin_right": edges[bin_idx + 1],
                "bin_center": (edges[bin_idx] + edges[bin_idx + 1]) / 2,
                "bin_width": bin_widths[bin_idx],
                "count": int(counts[bin_idx]),
                "probability": probabilities[bin_idx],
                "density": densities[bin_idx]
            })

        # ====================================================
        # 4.3 绘制每个目标电流对应的直方图
        # ====================================================

        plt.figure(figsize=(5.5, 4.0), dpi=300)

        plt.hist(
            actual_values,
            bins=bin_edges,
            edgecolor="black",
            alpha=0.75
        )

        plt.axvline(
            target_current,
            linestyle="--",
            linewidth=1.2,
            label="Target current"
        )

        plt.axvline(
            mu_i,
            linestyle="-",
            linewidth=1.2,
            label="Mean actual current"
        )

        plt.xlabel("Actual current")
        plt.ylabel("Count")
        plt.title(
            f"Target current distribution\n"
            f"target={target_current:.4e}, "
            f"mu={mu_i:.4e}, "
            f"sigma={sigma_i:.4e}"
        )
        plt.legend()
        plt.tight_layout()

        hist_fig_name = f"target_{row_idx:03d}_histogram.tif"
        hist_fig_path = os.path.join(HIST_DIR, hist_fig_name)

        plt.savefig(hist_fig_path, dpi=300, format="tiff")
        plt.close()

    # ========================================================
    # 5. 保存统计结果
    # ========================================================

    stats_df = pd.DataFrame(stats_rows)
    hist_df = pd.DataFrame(hist_rows)

    stats_df.to_csv(STATS_CSV, index=False, encoding="utf-8-sig")
    hist_df.to_csv(HIST_DATA_CSV, index=False, encoding="utf-8-sig")

    # ========================================================
    # 6. 计算整体 summary
    # ========================================================

    if len(stats_df) > 0:
        valid_alpha = stats_df["alpha_i"].dropna()

        summary = {
            "input_csv": str(INPUT_CSV),
            "output_dir": str(OUTPUT_DIR),
            "target_current_column": target_col,
            "actual_current_columns": actual_cols,
            "target_state_count": int(len(stats_df)),
            "actual_current_column_count": int(len(actual_cols)),
            "histogram_bin_count": BIN_COUNT,
            "std_definition": f"sample standard deviation with ddof={STD_DDOF}",
            "mu_i_definition": "mean of actual currents for each target current",
            "sigma_i_definition": "standard deviation of actual currents for each target current",
            "alpha_i_definition": "alpha_i = sigma_i / abs(target_current_i)",
            "bias_i_definition": "bias_i = mu_i - target_current_i",
            "summary_statistics": {
                "mean_alpha_i": float(valid_alpha.mean()) if len(valid_alpha) > 0 else None,
                "median_alpha_i": float(valid_alpha.median()) if len(valid_alpha) > 0 else None,
                "std_alpha_i": float(valid_alpha.std(ddof=1)) if len(valid_alpha) > 1 else None,
                "min_alpha_i": float(valid_alpha.min()) if len(valid_alpha) > 0 else None,
                "max_alpha_i": float(valid_alpha.max()) if len(valid_alpha) > 0 else None,
                "mean_sigma_i": float(stats_df["sigma_i"].mean()),
                "median_sigma_i": float(stats_df["sigma_i"].median()),
                "mean_abs_bias_i": float(stats_df["abs_bias_i"].mean()),
                "median_abs_bias_i": float(stats_df["abs_bias_i"].median())
            }
        }
    else:
        summary = {
            "input_csv": INPUT_CSV,
            "error": "No valid target-current rows were processed."
        }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n========== Finished ==========")
    print("Per-target statistics saved to:")
    print(STATS_CSV)

    print("\nHistogram bin data saved to:")
    print(HIST_DATA_CSV)

    print("\nHistogram figures saved to:")
    print(HIST_DIR)

    print("\nSummary JSON saved to:")
    print(SUMMARY_JSON)


if __name__ == "__main__":
    main()