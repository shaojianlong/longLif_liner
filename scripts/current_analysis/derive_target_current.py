from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 输入文件
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "results" / "current_analysis" / "peaks" / "curve1_x_replaced" / "curve1_x_replaced_points.csv"

# ============================================================
# 输出文件夹
# ============================================================
RESULT_DIR = PROJECT_ROOT / "results" / "current_analysis" / "modeling"

OUTPUT_JSON = RESULT_DIR / "global_linear_fit_with_line_data.json"
OUTPUT_POINTS_WITH_TARGET_CSV = RESULT_DIR / "curve1_x_replaced_points_with_target.csv"
OUTPUT_TIF = RESULT_DIR / "global_linear_fit_all_points.tif"

# ============================================================
# 图像参数
# ============================================================
DPI = 400
POINT_SIZE = 42

X_AXIS_MIN = 0
X_AXIS_MAX = 20

Y_AXIS_MIN = 0
Y_AXIS_MAX = 1e-6

# 拟合直线保存到 JSON 时的采样点数量
FIT_LINE_POINT_COUNT = 500


def read_replaced_points(input_csv: Path) -> pd.DataFrame:
    """
    读取横坐标已经替换后的坐标文件。

    文件格式应为：
        X, Curve1_Y, Curve2_Y, Curve3_Y, ...
    """
    if not input_csv.exists():
        raise FileNotFoundError(
            f"未找到输入文件：\n{input_csv.resolve()}"
        )

    dataframe = pd.read_csv(
        input_csv,
        encoding="utf-8-sig"
    )

    if dataframe.shape[1] < 2:
        raise ValueError(
            "输入 CSV 至少应包含两列：第一列 X，后面至少一列 Curve*_Y。"
        )

    return dataframe


def collect_all_points_for_global_fit(dataframe: pd.DataFrame):
    """
    将所有 Curve 的点合并，用于全局拟合。

    第一列为 X；
    后续每一列为不同 Curve 的 Y。
    """
    x_values = pd.to_numeric(
        dataframe.iloc[:, 0],
        errors="coerce"
    ).to_numpy(dtype=float)

    all_x = []
    all_y = []

    point_count_by_curve = {}

    for column_name in dataframe.columns[1:]:
        y_values = pd.to_numeric(
            dataframe[column_name],
            errors="coerce"
        ).to_numpy(dtype=float)

        valid_mask = np.isfinite(x_values) & np.isfinite(y_values)

        current_x = x_values[valid_mask]
        current_y = y_values[valid_mask]

        if len(current_x) > 0:
            all_x.append(current_x)
            all_y.append(current_y)

        point_count_by_curve[str(column_name)] = int(len(current_x))

    if len(all_x) == 0:
        raise ValueError("没有找到任何有效的 X-Y 点，无法拟合。")

    combined_x = np.concatenate(all_x)
    combined_y = np.concatenate(all_y)

    if len(combined_x) < 2:
        raise ValueError("有效点数量不足 2 个，无法进行线性拟合。")

    return combined_x, combined_y, point_count_by_curve


def global_linear_fit(combined_x: np.ndarray, combined_y: np.ndarray):
    """
    对全部点共同拟合一条直线：

        y = coefficient * x + bias
    """
    coefficient, bias = np.polyfit(combined_x, combined_y, deg=1)

    predicted_y = coefficient * combined_x + bias

    ss_res = np.sum((combined_y - predicted_y) ** 2)
    ss_tot = np.sum((combined_y - np.mean(combined_y)) ** 2)

    if ss_tot == 0:
        r_squared = None
    else:
        r_squared = 1 - ss_res / ss_tot

    return {
        "coefficient": float(coefficient),
        "bias": float(bias),
        "r_squared": None if r_squared is None else float(r_squared),
        "total_point_count": int(len(combined_x))
    }


def save_points_with_target(
    dataframe: pd.DataFrame,
    coefficient: float,
    bias: float,
    output_csv: Path
):
    """
    将原坐标文件重新保存一份，并在 X 后面插入 target 列。

    target 的来源：
        target = coefficient * X + bias

    输出格式：
        X, target, Curve1_Y, Curve2_Y, Curve3_Y, ...
    """
    output_df = dataframe.copy()

    x_values = pd.to_numeric(
        output_df.iloc[:, 0],
        errors="coerce"
    ).to_numpy(dtype=float)

    target_values = coefficient * x_values + bias

    # 在第一列 X 后面插入 target 列
    output_df.insert(
        loc=1,
        column="target",
        value=target_values
    )

    output_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
        float_format="%.12e"
    )

    print(f"已保存带 target 的坐标文件：{output_csv.resolve()}")


def build_fit_line_data(coefficient: float, bias: float):
    """
    生成拟合直线的采样数据，用于保存到 JSON。

    注意：
    这里的直线采样范围使用 X_AXIS_MIN 到 X_AXIS_MAX，
    即当前图像显示范围 0 到 20。
    """
    x_line = np.linspace(
        X_AXIS_MIN,
        X_AXIS_MAX,
        FIT_LINE_POINT_COUNT
    )

    y_line = coefficient * x_line + bias

    line_points = []

    for x_value, y_value in zip(x_line, y_line):
        line_points.append({
            "x": float(x_value),
            "y": float(y_value)
        })

    return line_points


def save_fit_json(
    fit_result: dict,
    line_points: list,
    point_count_by_curve: dict,
    dataframe: pd.DataFrame,
    output_json: Path
):
    """
    保存全局拟合参数、拟合直线数据和基础信息到 JSON。
    """
    json_data = {
        "fit_equation": "y = coefficient * x + bias",
        "fit_type": "global_linear_fit_all_curves_combined",
        "source_csv": str(INPUT_CSV),
        "output_points_with_target_csv": str(OUTPUT_POINTS_WITH_TARGET_CSV),
        "x_column": str(dataframe.columns[0]),
        "y_columns": [str(column) for column in dataframe.columns[1:]],
        "total_point_count": fit_result["total_point_count"],
        "coefficient": fit_result["coefficient"],
        "bias": fit_result["bias"],
        "r_squared": fit_result["r_squared"],
        "point_count_by_curve": point_count_by_curve,
        "fit_line_data": {
            "x_min": X_AXIS_MIN,
            "x_max": X_AXIS_MAX,
            "point_count": FIT_LINE_POINT_COUNT,
            "points": line_points
        }
    }

    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(
            json_data,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(f"已保存拟合结果 JSON：{output_json.resolve()}")


def plot_global_fit(
    dataframe: pd.DataFrame,
    coefficient: float,
    bias: float,
    r_squared,
    output_tif: Path
):
    """
    绘制所有散点，并叠加一条全局拟合直线。
    """
    x_values = pd.to_numeric(
        dataframe.iloc[:, 0],
        errors="coerce"
    ).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(18, 8))

    total_points = 0

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

        total_points += len(x)

    x_line = np.linspace(X_AXIS_MIN, X_AXIS_MAX, FIT_LINE_POINT_COUNT)
    y_line = coefficient * x_line + bias

    r_squared_text = "N/A" if r_squared is None else f"{r_squared:.6f}"

    ax.plot(
        x_line,
        y_line,
        linestyle="--",
        linewidth=2.4,
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
    ax.set_xlabel("X", fontsize=13)
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
        ncol=2
    )

    fig.tight_layout()

    fig.savefig(
        output_tif,
        format="tiff",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"已保存拟合可视化图：{output_tif.resolve()}")
    print(f"图中散点总数：{total_points}")


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"输入文件：{INPUT_CSV.resolve()}")
    print(f"输出文件夹：{RESULT_DIR.resolve()}")
    print("=" * 80)

    dataframe = read_replaced_points(INPUT_CSV)

    combined_x, combined_y, point_count_by_curve = collect_all_points_for_global_fit(
        dataframe
    )

    fit_result = global_linear_fit(
        combined_x=combined_x,
        combined_y=combined_y
    )

    coefficient = fit_result["coefficient"]
    bias = fit_result["bias"]
    r_squared = fit_result["r_squared"]

    print(f"全局拟合点数：{fit_result['total_point_count']}")
    print(f"coefficient：{coefficient:.12e}")
    print(f"bias：{bias:.12e}")

    if r_squared is None:
        print("R²：无法计算")
    else:
        print(f"R²：{r_squared:.8f}")

    # 1. 保存带 target 的新坐标 CSV
    save_points_with_target(
        dataframe=dataframe,
        coefficient=coefficient,
        bias=bias,
        output_csv=OUTPUT_POINTS_WITH_TARGET_CSV
    )

    # 2. 生成并保存拟合直线数据 JSON
    line_points = build_fit_line_data(
        coefficient=coefficient,
        bias=bias
    )

    save_fit_json(
        fit_result=fit_result,
        line_points=line_points,
        point_count_by_curve=point_count_by_curve,
        dataframe=dataframe,
        output_json=OUTPUT_JSON
    )

    # 3. 保存拟合图
    plot_global_fit(
        dataframe=dataframe,
        coefficient=coefficient,
        bias=bias,
        r_squared=r_squared,
        output_tif=OUTPUT_TIF
    )

    print("=" * 80)
    print("全部处理完成。新文件均已保存到 result 文件夹。")
    print(f"带 target 的坐标文件：{OUTPUT_POINTS_WITH_TARGET_CSV.resolve()}")
    print(f"拟合直线 JSON：{OUTPUT_JSON.resolve()}")
    print(f"拟合图：{OUTPUT_TIF.resolve()}")


if __name__ == "__main__":
    main()