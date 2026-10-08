import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. 全局路径设置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "results" / "current_analysis" / "modeling" / "curve1_x_replaced_points_with_target.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "current_analysis" / "modeling" / "alpha_sigma_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_DETAIL_CSV = OUTPUT_DIR / "alpha_sigma_per_target.csv"
OUTPUT_JSON = OUTPUT_DIR / "alpha_sigma_fit_result.json"
OUTPUT_FIG = OUTPUT_DIR / "sigma_vs_target_current.png"


# =========================
# 2. 读取数据
# =========================
df = pd.read_csv(INPUT_CSV)

# 第一列：横坐标
x_col = df.columns[0]

# 第二列：目标电流
target_col = df.columns[1]

# 第三列到最后：实际电流
actual_cols = list(df.columns[2:])

print("X column:", x_col)
print("Target current column:", target_col)
print("Actual current columns:", actual_cols)
print("Number of actual-current columns:", len(actual_cols))


# =========================
# 3. 转换为数值类型
# =========================
df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

for col in actual_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# =========================
# 4. 逐行计算 mean、sigma、bias、alpha_i
# =========================
actual_values = df[actual_cols]

# 每个目标电流下，24个实际电流的均值
df["mean_actual_current"] = actual_values.mean(axis=1, skipna=True)

# 每个目标电流下，24个实际电流的标准差 sigma
# ddof=1 表示样本标准差，和论文中 repeated tests 的统计方式更一致
df["sigma"] = actual_values.std(axis=1, ddof=1, skipna=True)

# 系统偏差：实际电流均值 - 目标电流
df["mean_bias"] = df["mean_actual_current"] - df[target_col]

# 绝对目标电流，用于计算 alpha
# 因为 sigma 一定为正，若 target_current 可能有极小负值，直接除会不合理
df["abs_target_current"] = df[target_col].abs()

# 避免 target_current 为 0 或极接近 0 时除零
EPS = 1e-30
valid_alpha_mask = df["abs_target_current"] > EPS

df["alpha_i"] = np.nan
df.loc[valid_alpha_mask, "alpha_i"] = (
    df.loc[valid_alpha_mask, "sigma"] /
    df.loc[valid_alpha_mask, "abs_target_current"]
)


# =========================
# 5. 全局拟合 sigma = alpha * target_current
#    强制过原点拟合
# =========================
fit_df = df[
    valid_alpha_mask &
    df["sigma"].notna() &
    df["abs_target_current"].notna()
].copy()

x = fit_df["abs_target_current"].to_numpy(dtype=float)
y = fit_df["sigma"].to_numpy(dtype=float)

# 强制过原点拟合：
# alpha = sum(x*y) / sum(x^2)
alpha_forced_origin = np.sum(x * y) / np.sum(x ** 2)

# 拟合得到的 sigma
y_pred_forced = alpha_forced_origin * x

# centered R^2
ss_res = np.sum((y - y_pred_forced) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

# uncentered R^2，更适合强制过原点拟合
ss_tot_uncentered = np.sum(y ** 2)
r_squared_uncentered = 1 - ss_res / ss_tot_uncentered if ss_tot_uncentered != 0 else np.nan


# =========================
# 6. 额外计算：普通线性拟合 sigma = k * target + b
#    仅用于对比，不作为最终 alpha
# =========================
k_free, b_free = np.polyfit(x, y, deg=1)
y_pred_free = k_free * x + b_free

ss_res_free = np.sum((y - y_pred_free) ** 2)
r_squared_free = 1 - ss_res_free / ss_tot if ss_tot != 0 else np.nan


# =========================
# 7. 保存逐目标统计结果
# =========================
detail_cols = [
    x_col,
    target_col,
    "abs_target_current",
    "mean_actual_current",
    "sigma",
    "mean_bias",
    "alpha_i"
]

df[detail_cols].to_csv(OUTPUT_DETAIL_CSV, index=False, encoding="utf-8-sig")


# =========================
# 8. 保存 JSON 汇总结果
# =========================
result = {
    "source_csv": str(INPUT_CSV),
    "x_column": x_col,
    "target_current_column": target_col,
    "actual_current_columns": actual_cols,
    "target_state_count": int(len(df)),
    "actual_current_columns_count": int(len(actual_cols)),
    "total_actual_point_count": int(df[actual_cols].count().sum()),

    "sigma_definition": "sample_std_of_actual_currents_under_the_same_target_current",
    "alpha_definition": "sigma = alpha * abs(target_current)",

    "forced_origin_fit": {
        "fit_equation": "sigma = alpha * abs(target_current)",
        "alpha": float(alpha_forced_origin),
        "alpha_percent": float(alpha_forced_origin * 100),
        "bias": 0.0,
        "r_squared": float(r_squared),
        "r_squared_uncentered": float(r_squared_uncentered)
    },

    "ordinary_linear_fit_for_comparison": {
        "fit_equation": "sigma = k * abs(target_current) + b",
        "k": float(k_free),
        "b": float(b_free),
        "r_squared": float(r_squared_free)
    },

    "per_target_summary": {
        "mean_alpha_i": float(df["alpha_i"].mean(skipna=True)),
        "median_alpha_i": float(df["alpha_i"].median(skipna=True)),
        "std_alpha_i": float(df["alpha_i"].std(skipna=True, ddof=1)),
        "min_alpha_i": float(df["alpha_i"].min(skipna=True)),
        "max_alpha_i": float(df["alpha_i"].max(skipna=True)),
        "mean_sigma": float(df["sigma"].mean(skipna=True)),
        "median_sigma": float(df["sigma"].median(skipna=True)),
        "mean_abs_bias": float(df["mean_bias"].abs().mean(skipna=True))
    }
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)


# =========================
# 9. 绘制 sigma-target 拟合图
# =========================
plt.figure(figsize=(6, 4), dpi=300)

plt.scatter(x, y, label="Calculated sigma")

x_line = np.linspace(0, np.max(x), 200)
y_line = alpha_forced_origin * x_line

plt.plot(
    x_line,
    y_line,
    label=f"Forced-origin fit: alpha = {alpha_forced_origin:.6g}"
)

plt.xlabel("Abs target current")
plt.ylabel("Sigma")
plt.title("Sigma vs Target Current")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_FIG)
plt.close()


# =========================
# 10. 输出结果
# =========================
print("\n========== Alpha / Sigma Analysis Finished ==========")
print(f"Input CSV: {INPUT_CSV}")
print(f"Per-target result saved to: {OUTPUT_DETAIL_CSV}")
print(f"Fit result JSON saved to: {OUTPUT_JSON}")
print(f"Figure saved to: {OUTPUT_FIG}")

print("\n========== Main Result ==========")
print(f"Global alpha forced origin = {alpha_forced_origin}")
print(f"Global alpha percent       = {alpha_forced_origin * 100:.4f}%")
print(f"R^2 centered               = {r_squared}")
print(f"R^2 uncentered             = {r_squared_uncentered}")

print("\n========== Per-target Alpha Summary ==========")
print(f"Mean alpha_i   = {df['alpha_i'].mean(skipna=True)}")
print(f"Median alpha_i = {df['alpha_i'].median(skipna=True)}")
print(f"Min alpha_i    = {df['alpha_i'].min(skipna=True)}")
print(f"Max alpha_i    = {df['alpha_i'].max(skipna=True)}")