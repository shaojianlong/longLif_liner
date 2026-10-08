from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ============================================================
# 文件与基础参数
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "C2C.xlsx"
SHEET_NAME = 0
OUTPUT_DIR = PROJECT_ROOT / "results" / "current_analysis" / "peaks"

NUM_CURVES = 32
MAX_MARKED_PEAKS = 10
DPI = 400


# ============================================================
# 局部极大值纵坐标筛选范围
# ============================================================
Y_MIN = 1e-7
Y_MAX = 3e-7


# ============================================================
# 相邻最终保留峰的最小横坐标间隔
# ============================================================
MIN_X_GAP = 0.5


# ============================================================
# 峰检测参数
# 噪声较多时可自行修改，例如：
# MIN_PEAK_DISTANCE = 20
# MIN_PROMINENCE = 1e-9
# ============================================================
MIN_PEAK_DISTANCE = None
MIN_PROMINENCE = None


def read_one_curve(raw_data: pd.DataFrame, curve_index: int):
    """
    每两列为一条曲线：
    第 1、2 列 -> Curve 1 的 X、Y
    第 3、4 列 -> Curve 2 的 X、Y
    ...

    自动识别首个有效 X-Y 坐标对所在行。
    """
    x_col = curve_index * 2
    y_col = x_col + 1

    x_series = pd.to_numeric(raw_data.iloc[:, x_col], errors="coerce")
    y_series = pd.to_numeric(raw_data.iloc[:, y_col], errors="coerce")

    x_values = x_series.to_numpy(dtype=float)
    y_values = y_series.to_numpy(dtype=float)

    valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
    valid_positions = np.flatnonzero(valid_mask)

    if len(valid_positions) == 0:
        return None, None, None

    start_position = valid_positions[0]

    row_indices = np.arange(len(raw_data))
    keep_mask = valid_mask & (row_indices >= start_position)

    x = x_values[keep_mask]
    y = y_values[keep_mask]

    # Excel 行号从 1 开始
    start_excel_row = start_position + 1

    return x, y, start_excel_row


def filter_peaks_by_x_gap(candidate_peak_indices: np.ndarray, x: np.ndarray):
    """
    保留第一个候选峰；
    后续峰只有与前一个“已保留峰”的横坐标间隔 >= MIN_X_GAP 时才保留。
    最多保留 MAX_MARKED_PEAKS 个。
    """
    if len(candidate_peak_indices) == 0:
        return np.array([], dtype=int)

    selected_peaks = [candidate_peak_indices[0]]
    last_selected_peak = candidate_peak_indices[0]

    for current_peak in candidate_peak_indices[1:]:
        x_gap = abs(x[current_peak] - x[last_selected_peak])

        if x_gap >= MIN_X_GAP:
            selected_peaks.append(current_peak)
            last_selected_peak = current_peak

        if len(selected_peaks) >= MAX_MARKED_PEAKS:
            break

    return np.array(selected_peaks, dtype=int)


def detect_valid_peaks(x: np.ndarray, y: np.ndarray):
    """
    1. 找全部局部极大值；
    2. 筛选 y 在 [Y_MIN, Y_MAX] 的峰；
    3. 保留相邻峰横坐标间隔 >= MIN_X_GAP 的峰；
    4. 最多保留前 10 个。
    """
    all_peak_indices, _ = find_peaks(
        y,
        distance=MIN_PEAK_DISTANCE,
        prominence=MIN_PROMINENCE
    )

    y_mask = (
        (y[all_peak_indices] >= Y_MIN)
        & (y[all_peak_indices] <= Y_MAX)
    )

    candidate_peak_indices = all_peak_indices[y_mask]

    marked_peak_indices = filter_peaks_by_x_gap(
        candidate_peak_indices,
        x
    )

    return candidate_peak_indices, marked_peak_indices


def plot_one_curve(
    curve_number: int,
    x: np.ndarray,
    y: np.ndarray,
    marked_peak_indices: np.ndarray
):
    """
    绘制曲线，并将最终筛选出的极大值标红。
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        x,
        y,
        linewidth=1.0,
        label=f"Curve {curve_number}"
    )

    if len(marked_peak_indices) > 0:
        ax.scatter(
            x[marked_peak_indices],
            y[marked_peak_indices],
            color="red",
            s=40,
            zorder=4,
            label=f"Selected peaks ({len(marked_peak_indices)})"
        )

        # 红点编号
        for rank, peak_index in enumerate(marked_peak_indices, start=1):
            ax.annotate(
                str(rank),
                xy=(x[peak_index], y[peak_index]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
                color="red"
            )

    ax.axhline(
        Y_MIN,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        label=f"Y min = {Y_MIN:.1e}"
    )

    ax.axhline(
        Y_MAX,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        label=f"Y max = {Y_MAX:.1e}"
    )

    ax.set_title(f"Curve {curve_number}: Local Maxima Selection")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)

    fig.tight_layout()

    output_file = OUTPUT_DIR / f"curve_{curve_number:02d}.tif"

    fig.savefig(
        output_file,
        format="tiff",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_file


def save_peaks_by_curve_to_csv(curve_peaks: dict, output_path: Path):
    """
    输出格式：
    第 1 行：Curve1, Curve1, Curve2, Curve2, ...
    第 2 行：X,      Y,      X,      Y,      ...

    每条曲线占两列。不同曲线峰数不足的部分自动留空。
    """
    first_header = []
    second_header = []

    for curve_number in range(1, NUM_CURVES + 1):
        first_header.extend([f"Curve{curve_number}", f"Curve{curve_number}"])
        second_header.extend(["X", "Y"])

    max_peak_count = max(
        (len(curve_peaks[curve_number][0]) for curve_number in curve_peaks),
        default=0
    )

    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)

        writer.writerow(first_header)
        writer.writerow(second_header)

        for peak_rank in range(max_peak_count):
            row = []

            for curve_number in range(1, NUM_CURVES + 1):
                x_peaks, y_peaks = curve_peaks.get(
                    curve_number,
                    (np.array([]), np.array([]))
                )

                if peak_rank < len(x_peaks):
                    row.extend([
                        f"{x_peaks[peak_rank]:.12e}",
                        f"{y_peaks[peak_rank]:.12e}"
                    ])
                else:
                    row.extend(["", ""])

            writer.writerow(row)


def main():
    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(
            f"未找到 Excel 文件：\n{input_path.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = pd.read_excel(
        input_path,
        sheet_name=SHEET_NAME,
        header=None
    )

    required_columns = NUM_CURVES * 2

    if raw_data.shape[1] < required_columns:
        raise ValueError(
            f"Excel 当前只有 {raw_data.shape[1]} 列，"
            f"但前 {NUM_CURVES} 条曲线至少需要 {required_columns} 列。"
        )

    print("=" * 85)
    print(f"输入文件：{input_path.resolve()}")
    print(f"输出文件夹：{OUTPUT_DIR.resolve()}")
    print(f"峰高条件：{Y_MIN:.1e} <= y_peak <= {Y_MAX:.1e}")
    print(f"相邻保留峰条件：Δx >= {MIN_X_GAP}")
    print("=" * 85)

    # 保存格式：{曲线编号: (峰X坐标数组, 峰Y坐标数组)}
    curve_peaks = {}

    saved_count = 0

    for curve_index in range(NUM_CURVES):
        curve_number = curve_index + 1

        x, y, start_excel_row = read_one_curve(raw_data, curve_index)

        if x is None or len(x) < 3:
            print(f"曲线 {curve_number:02d}：有效数据不足，已跳过。")
            curve_peaks[curve_number] = (np.array([]), np.array([]))
            continue

        candidate_peak_indices, marked_peak_indices = detect_valid_peaks(x, y)

        output_file = plot_one_curve(
            curve_number=curve_number,
            x=x,
            y=y,
            marked_peak_indices=marked_peak_indices
        )

        # 保存该条曲线最终标红峰的 X、Y 坐标
        curve_peaks[curve_number] = (
            x[marked_peak_indices],
            y[marked_peak_indices]
        )

        print(
            f"曲线 {curve_number:02d} | "
            f"起始行：Excel 第 {start_excel_row} 行 | "
            f"有效点：{len(x)} | "
            f"候选峰：{len(candidate_peak_indices)} | "
            f"最终标红峰：{len(marked_peak_indices)} | "
            f"保存：{output_file.name}"
        )

        saved_count += 1

    # 保存为“每条曲线两列”的 CSV
    csv_path = OUTPUT_DIR / "selected_peaks_by_curve.csv"

    save_peaks_by_curve_to_csv(
        curve_peaks=curve_peaks,
        output_path=csv_path
    )

    print("=" * 85)
    print(f"完成，共保存 {saved_count} 张 TIFF 图片。")
    print(f"峰值坐标 CSV：{csv_path.resolve()}")


if __name__ == "__main__":
    main()