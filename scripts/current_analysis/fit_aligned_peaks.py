from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 输入与输出路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "results" / "current_analysis" / "peaks" / "curve1_x_replaced" / "curve1_x_replaced_points.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "current_analysis" / "peaks" / "curve1_x_replaced"
OUTPUT_JSON = OUTPUT_DIR / "global_linear_fit_all_points.json"
OUTPUT_TIF = OUTPUT_DIR / "global_linear_fit_all_points.tif"

DPI = 400
POINT_SIZE = 42

# 图像显示范围
X_AXIS_MIN = 0
X_AXIS_MAX = 20

Y_AXIS_MIN = 0
Y_AXIS_MAX = 0.4*1e-6


def calculate_global_linear_fit(dataframe: pd.DataFrame):
    """
    第一列为统一横坐标 X；
    后续每列为不同 Curve 的 Y 值。

    将所有 Curve 的有效 (X, Y) 点合并，
    对全部点共同进行一次线性拟合：
        y = coefficient * x + bias
    """

    x_values = pd.to_numeric(
        dataframe.iloc[:, 0],
        errors="coerce"
    ).to_numpy(dtype=float)

    all_x = []
    all_y = []
    curve_point_counts = {}

    # 从第二列开始，逐列读取 Curve 的 Y
    for column_name in dataframe.columns[1:]:
        y_values = pd.to_numeric(
            dataframe[column_name],
            errors="coerce"
        ).to_numpy(dtype=float)

        valid_mask = np.isfinite(x_values) & np.isfinite(y_values)

        current_x = x_values[valid_mask]
        current_y = y_values[valid_mask]

        all_x.append(current_x)
        all_y.append(current_y)

        curve_point_counts[str(column_name)] = int(len(current_x))

    if len(all_x) == 0:
        raise ValueError("没有读取到任何曲线数据。")

    combined_x = np.concatenate(all_x)
    combined_y = np.concatenate(all_y)

    if len(combined_x) < 2:
        raise ValueError("有效点数量不足，无法进行线性拟合。")

    # 全部点一起拟合：y = kx + b
    coefficient, bias = np.polyfit(combined_x, combined_y, deg=1)

    predicted_y = coefficient * combined_x + bias

    ss_res = np.sum((combined_y - predicted_y) ** 2)
    ss_tot = np.sum((combined_y - np.mean(combined_y)) ** 2)

    if ss_tot == 0:
        r_squared = None
    else:
        r_squared = 1 - ss_res / ss_tot

    return {
        "combined_x": combined_x,
        "combined_y": combined_y,
        "coefficient": float(coefficient),
        "bias": float(bias),
        "r_squared": None if r_squared is None else float(r_squared),
        "total_point_count": int(len(combined_x)),
        "curve_point_counts": curve_point_counts
    }


def plot_all_points_and_fit_line(
    dataframe: pd.DataFrame,
    fit_result: dict,
    output_path: Path
):
    """
    绘制所有 Curve 的散点，并叠加一条由全部点共同拟合得到的直线。
    """
    x_values = pd.to_numeric(
        dataframe.iloc[:, 0],
        errors="coerce"
    ).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(22, 8))

    # 每列 Curve_Y 单独画一组散点，便于观察分布
    for column_name in dataframe.columns[1:]:
        y_values = pd.to_numeric(
            dataframe[column_name],
            errors="coerce"
        ).to_numpy(dtype=float)

        valid_mask = np.isfinite(x_values) & np.isfinite(y_values)

        x = x_values[valid_mask]
        y = y_values[valid_mask]

        if len(x) == 0:
            continue

        ax.scatter(
            x,
            y,
            s=POINT_SIZE,
            alpha=0.75,
            label=str(column_name)
        )

    # 所有点共同拟合出的唯一一条直线
    coefficient = fit_result["coefficient"]
    bias = fit_result["bias"]

    x_line = np.linspace(X_AXIS_MIN, X_AXIS_MAX, 500)
    y_line = coefficient * x_line + bias

    r_squared_text = (
        "N/A"
        if fit_result["r_squared"] is None
        else f"{fit_result['r_squared']:.6f}"
    )

    ax.plot(
        x_line,
        y_line,
        linestyle="--",
        linewidth=2.2,
        label=(
            f"Global fit: y = {coefficient:.4e}x + {bias:.4e}, "
            f"R² = {r_squared_text}"
        )
    )

    ax.set_xlim(X_AXIS_MIN, X_AXIS_MAX)
    ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)

    ax.set_title(
        "Global Linear Fit Using All Curve Points",
        fontsize=16
    )
    ax.set_xlabel("X Coordinates of Curve 1", fontsize=13)
    ax.set_ylabel("Y", fontsize=13)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        ncol=2
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        format="tiff",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"未找到输入文件：\n{INPUT_CSV.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 新文件第一行为：X, Curve1_Y, Curve2_Y, ...
    dataframe = pd.read_csv(
        INPUT_CSV,
        encoding="utf-8-sig"
    )

    if dataframe.shape[1] < 2:
        raise ValueError(
            "CSV 至少应包含两列：第一列 X，以及至少一列 Curve_Y。"
        )

    # 确认第一列名称，但即使不是 X 也会按第一列处理
    x_column_name = dataframe.columns[0]

    print("=" * 75)
    print(f"读取文件：{INPUT_CSV.resolve()}")
    print(f"横坐标列：{x_column_name}")
    print(f"Y 数据列数：{dataframe.shape[1] - 1}")
    print("=" * 75)

    # 对所有 Curve 的全部点进行一次全局拟合
    fit_result = calculate_global_linear_fit(dataframe)

    print(f"总参与拟合点数：{fit_result['total_point_count']}")
    print(f"拟合系数 k：{fit_result['coefficient']:.12e}")
    print(f"拟合偏置 b：{fit_result['bias']:.12e}")

    if fit_result["r_squared"] is not None:
        print(f"R²：{fit_result['r_squared']:.8f}")
    else:
        print("R²：无法计算，因为所有 Y 值完全相同。")

    # 保存 JSON
    json_result = {
        "fit_equation": "y = coefficient * x + bias",
        "fit_type": "global_linear_fit_all_curves_combined",
        "source_csv": str(INPUT_CSV),
        "x_column": str(x_column_name),
        "y_columns": [str(column) for column in dataframe.columns[1:]],
        "total_point_count": fit_result["total_point_count"],
        "coefficient": fit_result["coefficient"],
        "bias": fit_result["bias"],
        "r_squared": fit_result["r_squared"],
        "curve_point_counts": fit_result["curve_point_counts"]
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(
            json_result,
            file,
            ensure_ascii=False,
            indent=4
        )

    # 保存散点图及唯一的全局拟合线
    plot_all_points_and_fit_line(
        dataframe=dataframe,
        fit_result=fit_result,
        output_path=OUTPUT_TIF
    )

    print("=" * 75)
    print(f"拟合结果 JSON：{OUTPUT_JSON.resolve()}")
    print(f"拟合可视化图：{OUTPUT_TIF.resolve()}")


if __name__ == "__main__":
    main()