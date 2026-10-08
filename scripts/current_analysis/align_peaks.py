from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 文件与输出参数
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "results" / "current_analysis" / "peaks" / "selected_peaks_by_curve.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "current_analysis" / "peaks"

# 原始峰横坐标散点图
OUTPUT_TIF_ORIGINAL = OUTPUT_DIR / "selected_peaks_scatter_filtered.tif"

# 所有曲线横坐标替换为 Curve 1 横坐标后的散点图
OUTPUT_TIF_CURVE1_X = OUTPUT_DIR / "selected_peaks_scatter_curve1_x.tif"

# 新增：横坐标替换后的数据与拟合结果保存目录
REPLACED_DATA_DIR = OUTPUT_DIR / "curve1_x_replaced"

# 新增：横坐标替换后的坐标表
REPLACED_POINTS_CSV = REPLACED_DATA_DIR / "curve1_x_replaced_points.csv"

# 新增：线性拟合系数与偏置 JSON
LINEAR_FIT_JSON = REPLACED_DATA_DIR / "curve1_x_linear_fit.json"

DPI = 400
POINT_SIZE = 45


# ============================================================
# 剔除的曲线类别
# ============================================================
EXCLUDED_CURVES = {15, 16, 18, 19, 23, 30, 31, 32}


# ============================================================
# 坐标轴显示范围
# ============================================================
X_AXIS_MIN = 0
X_AXIS_MAX = 100

Y_AXIS_MIN = 0
Y_AXIS_MAX = 1e-6


# ============================================================
# 使用哪条曲线的横坐标作为统一横坐标
# ============================================================
REFERENCE_CURVE_NUMBER = 1


# ============================================================
# 图形显示参数
# ============================================================
MARKERS = [
    "o", "s", "^", "v", "D", "P", "X", "*",
    "<", ">", "h", "H", "d", "p", "8"
]


def get_curve_number(name, default_number):
    """
    从表头提取曲线编号。
    例如：Curve1 -> 1，Curve12 -> 12。
    """
    match = re.search(r"(\d+)", str(name))

    if match:
        return int(match.group(1))

    return default_number


def read_curve_data(csv_path):
    """
    读取 selected_peaks_by_curve.csv。

    CSV 格式：
    第 1 行：Curve1, Curve1, Curve2, Curve2, ...
    第 2 行：X,      Y,      X,      Y,      ...
    第 3 行起：实际峰坐标数据。

    返回格式：
    {
        1: {"x": ndarray, "y": ndarray},
        2: {"x": ndarray, "y": ndarray},
        ...
    }
    """
    raw_data = pd.read_csv(
        csv_path,
        header=None,
        encoding="utf-8-sig"
    )

    if raw_data.shape[0] < 3:
        raise ValueError("CSV 中没有足够的数据行。")

    if raw_data.shape[1] < 2:
        raise ValueError("CSV 中列数不足，无法读取 X-Y 坐标。")

    curve_data = {}

    # 每两列对应一条曲线：X、Y
    for col_index in range(0, raw_data.shape[1] - 1, 2):
        curve_name = raw_data.iloc[0, col_index]

        curve_number = get_curve_number(
            curve_name,
            default_number=col_index // 2 + 1
        )

        # 第 3 行起为数据
        x_values = pd.to_numeric(
            raw_data.iloc[2:, col_index],
            errors="coerce"
        ).to_numpy(dtype=float)

        y_values = pd.to_numeric(
            raw_data.iloc[2:, col_index + 1],
            errors="coerce"
        ).to_numpy(dtype=float)

        valid_mask = np.isfinite(x_values) & np.isfinite(y_values)

        curve_data[curve_number] = {
            "x": x_values[valid_mask],
            "y": y_values[valid_mask]
        }

    return curve_data


def plot_all_curves(curve_data, output_path, title, x_axis_label):
    """
    将所有曲线散点绘制在同一张图中。
    """
    fig, ax = plt.subplots(figsize=(22, 8))

    plotted_curve_count = 0
    total_points = 0

    for data_index, curve_number in enumerate(sorted(curve_data.keys())):
        x = curve_data[curve_number]["x"]
        y = curve_data[curve_number]["y"]

        if len(x) == 0:
            print(f"Curve {curve_number:02d}：没有有效峰值点，跳过。")
            continue

        marker_style = MARKERS[data_index % len(MARKERS)]

        ax.scatter(
            x,
            y,
            s=POINT_SIZE,
            alpha=0.85,
            marker=marker_style,
            label=f"Curve {curve_number}"
        )

        plotted_curve_count += 1
        total_points += len(x)

    if plotted_curve_count == 0:
        plt.close(fig)
        raise ValueError("没有可绘制的有效峰值点。")

    ax.set_xlim(X_AXIS_MIN, X_AXIS_MAX)
    ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_axis_label, fontsize=13)
    ax.set_ylabel("Y", fontsize=13)

    ax.grid(True, linestyle="--", alpha=0.4)

    ax.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0)
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        ncol=2,
        title="Remaining Curves"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        format="tiff",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"已保存图片：{output_path.resolve()} | "
        f"曲线数：{plotted_curve_count} | "
        f"散点数：{total_points}"
    )


def replace_x_with_reference_x(curve_data, reference_curve_number):
    """
    将每条曲线的横坐标替换为参考曲线的横坐标。

    对应规则：
    当前曲线第 1 个峰 -> 参考曲线第 1 个峰的 X
    当前曲线第 2 个峰 -> 参考曲线第 2 个峰的 X
    ...

    Y 值保持为当前曲线原始 Y。
    """
    if reference_curve_number not in curve_data:
        raise ValueError(
            f"找不到 Curve {reference_curve_number}，无法作为横坐标参考。"
        )

    reference_x = curve_data[reference_curve_number]["x"]

    if len(reference_x) == 0:
        raise ValueError(
            f"Curve {reference_curve_number} 没有有效峰坐标，无法作为参考。"
        )

    replaced_data = {}

    for curve_number, data in curve_data.items():
        original_y = data["y"]

        common_count = min(len(reference_x), len(original_y))

        replaced_data[curve_number] = {
            "x": reference_x[:common_count].copy(),
            "y": original_y[:common_count].copy()
        }

        print(
            f"Curve {curve_number:02d}："
            f"原始峰数 {len(original_y)}，"
            f"替换后保留 {common_count} 个点。"
        )

    return replaced_data


def save_replaced_points_to_csv(replaced_data, reference_curve_number, output_path):
    """
    保存替换横坐标后的点。

    输出格式：
    第一列：统一横坐标 X（来自 Curve 1）
    后续每列：不同 Curve 对应的 Y 值

    示例：
    X,Curve1_Y,Curve2_Y,Curve3_Y,...
    1.2e+00,1.1e-07,1.3e-07,...
    2.0e+00,1.6e-07,1.4e-07,...
    """
    if reference_curve_number not in replaced_data:
        raise ValueError(
            f"找不到 Curve {reference_curve_number} 的替换后数据。"
        )

    reference_x = replaced_data[reference_curve_number]["x"]

    output_dict = {
        "X": reference_x
    }

    for curve_number in sorted(replaced_data.keys()):
        y_values = replaced_data[curve_number]["y"]

        # 补齐到与参考横坐标相同长度；缺少的峰用 NaN 留空
        padded_y = np.full(len(reference_x), np.nan, dtype=float)
        padded_y[:len(y_values)] = y_values

        output_dict[f"Curve{curve_number}_Y"] = padded_y

    output_df = pd.DataFrame(output_dict)

    output_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.12e"
    )

    print(f"已保存替换横坐标后的坐标表：{output_path.resolve()}")


def linear_fit_one_curve(x, y):
    """
    对单条曲线进行线性拟合：

        y = coefficient * x + bias

    返回：
        coefficient: 斜率
        bias: 截距
        r_squared: 决定系数
        point_count: 有效点数
    """
    valid_mask = np.isfinite(x) & np.isfinite(y)

    valid_x = x[valid_mask]
    valid_y = y[valid_mask]

    if len(valid_x) < 2:
        return {
            "point_count": int(len(valid_x)),
            "coefficient": None,
            "bias": None,
            "r_squared": None,
            "status": "insufficient_points"
        }

    coefficient, bias = np.polyfit(valid_x, valid_y, deg=1)

    predicted_y = coefficient * valid_x + bias

    ss_res = np.sum((valid_y - predicted_y) ** 2)
    ss_tot = np.sum((valid_y - np.mean(valid_y)) ** 2)

    if ss_tot == 0:
        r_squared = None
    else:
        r_squared = 1 - ss_res / ss_tot

    return {
        "point_count": int(len(valid_x)),
        "coefficient": float(coefficient),
        "bias": float(bias),
        "r_squared": None if r_squared is None else float(r_squared),
        "status": "success"
    }


def fit_all_replaced_curves(replaced_data, output_path):
    """
    对替换横坐标后的每一条曲线执行线性拟合，
    并将拟合系数、偏置等结果保存为 JSON。
    """
    fit_results = {}

    for curve_number in sorted(replaced_data.keys()):
        x = replaced_data[curve_number]["x"]
        y = replaced_data[curve_number]["y"]

        result = linear_fit_one_curve(x, y)

        fit_results[f"Curve{curve_number}"] = result

        if result["status"] == "success":
            print(
                f"Curve {curve_number:02d} 拟合完成："
                f"k = {result['coefficient']:.12e}, "
                f"b = {result['bias']:.12e}, "
                f"R² = {result['r_squared']:.6f}"
                if result["r_squared"] is not None
                else
                f"Curve {curve_number:02d} 拟合完成："
                f"k = {result['coefficient']:.12e}, "
                f"b = {result['bias']:.12e}, "
                f"R² = 无法计算"
            )
        else:
            print(
                f"Curve {curve_number:02d}："
                f"有效点数不足，无法进行线性拟合。"
            )

    json_data = {
        "fit_equation": "y = coefficient * x + bias",
        "reference_x_curve": REFERENCE_CURVE_NUMBER,
        "excluded_curves": sorted(EXCLUDED_CURVES),
        "fit_results": fit_results
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            json_data,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(f"已保存线性拟合 JSON：{output_path.resolve()}")


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"未找到 CSV 文件：\n{INPUT_CSV.resolve()}\n"
            "请先运行峰值筛选程序，生成 selected_peaks_by_curve.csv。"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPLACED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 读取全部峰坐标
    all_curve_data = read_curve_data(INPUT_CSV)

    # 剔除指定类别
    remaining_curve_data = {
        curve_number: data
        for curve_number, data in all_curve_data.items()
        if curve_number not in EXCLUDED_CURVES
    }

    if not remaining_curve_data:
        raise ValueError("剔除指定类别后，没有曲线可绘制。")

    if REFERENCE_CURVE_NUMBER not in remaining_curve_data:
        raise ValueError(
            f"参考 Curve {REFERENCE_CURVE_NUMBER} 被剔除或不存在，"
            "无法进行横坐标替换。"
        )

    print("=" * 80)
    print(f"输入 CSV：{INPUT_CSV.resolve()}")
    print(f"剔除类别：{sorted(EXCLUDED_CURVES)}")
    print(f"参考横坐标：Curve {REFERENCE_CURVE_NUMBER}")
    print(f"新数据目录：{REPLACED_DATA_DIR.resolve()}")
    print("=" * 80)

    # ============================================================
    # 图 1：使用原始峰横坐标
    # ============================================================
    plot_all_curves(
        curve_data=remaining_curve_data,
        output_path=OUTPUT_TIF_ORIGINAL,
        title="Selected Local Maxima of Remaining Curves",
        x_axis_label="Original X"
    )

    # ============================================================
    # 替换横坐标
    # ============================================================
    replaced_curve_data = replace_x_with_reference_x(
        curve_data=remaining_curve_data,
        reference_curve_number=REFERENCE_CURVE_NUMBER
    )

    # ============================================================
    # 图 2：使用 Curve 1 横坐标
    # ============================================================
    plot_all_curves(
        curve_data=replaced_curve_data,
        output_path=OUTPUT_TIF_CURVE1_X,
        title="Selected Local Maxima Using Curve 1 X Coordinates",
        x_axis_label=f"X Coordinates of Curve {REFERENCE_CURVE_NUMBER}"
    )

    # ============================================================
    # 保存替换横坐标后的新点
    # ============================================================
    save_replaced_points_to_csv(
        replaced_data=replaced_curve_data,
        reference_curve_number=REFERENCE_CURVE_NUMBER,
        output_path=REPLACED_POINTS_CSV
    )

    # ============================================================
    # 线性拟合并保存 JSON
    # ============================================================
    fit_all_replaced_curves(
        replaced_data=replaced_curve_data,
        output_path=LINEAR_FIT_JSON
    )

    print("=" * 80)
    print("全部处理完成。")
    print(f"原始散点图：{OUTPUT_TIF_ORIGINAL.resolve()}")
    print(f"替换横坐标散点图：{OUTPUT_TIF_CURVE1_X.resolve()}")
    print(f"替换后坐标表：{REPLACED_POINTS_CSV.resolve()}")
    print(f"拟合结果 JSON：{LINEAR_FIT_JSON.resolve()}")


if __name__ == "__main__":
    main()