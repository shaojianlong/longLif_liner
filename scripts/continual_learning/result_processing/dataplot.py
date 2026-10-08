import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# 1. CSV 路径
# =========================
csv_path = Path(__file__).resolve().parents[3] / "results" / "continual_learning" / "continual_eval.csv"

# 输出图片路径
run_dir = csv_path.parent.parent
fig_dir = run_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

save_path = fig_dir / "continual_eval_acc_loss_same_figure.png"


# =========================
# 2. 读取数据
# =========================
df = pd.read_csv(csv_path)

# 如果没有 test_acc_percent，就用 test_acc * 100 生成
if "test_acc_percent" not in df.columns:
    df["test_acc_percent"] = df["test_acc"] * 100.0

# 按全局 epoch 排序
df = df.sort_values(["global_epoch", "evaluated_task_index"])


# =========================
# 3. 画图
# =========================
fig, ax_acc = plt.subplots(figsize=(16, 7), dpi=150)
ax_loss = ax_acc.twinx()

tasks = (
    df[["evaluated_task_index", "evaluated_task_name"]]
    .drop_duplicates()
    .sort_values("evaluated_task_index")
)

cmap = plt.get_cmap("tab10")

acc_lines = []
loss_lines = []

for i, row in tasks.iterrows():
    task_idx = int(row["evaluated_task_index"])
    task_name = row["evaluated_task_name"]

    sub = df[df["evaluated_task_name"] == task_name].sort_values("global_epoch")
    color = cmap((task_idx - 1) % 10)

    # Accuracy：实线
    acc_line, = ax_acc.plot(
        sub["global_epoch"],
        sub["test_acc_percent"],
        linestyle="-",
        linewidth=2.2,
        color=color,
        label=f"{task_name} Acc"
    )

    # Loss：虚线
    loss_line, = ax_loss.plot(
        sub["global_epoch"],
        sub["test_loss"],
        linestyle="--",
        linewidth=1.8,
        color=color,
        alpha=0.85,
        label=f"{task_name} Loss"
    )

    acc_lines.append(acc_line)
    loss_lines.append(loss_line)


# =========================
# 4. 添加任务切换竖线和文字
# =========================
task_segments = (
    df.groupby(["learning_task_index", "learning_task_name"])["global_epoch"]
    .agg(["min", "max"])
    .reset_index()
    .sort_values("learning_task_index")
)

for _, row in task_segments.iterrows():
    start_ep = row["min"]
    end_ep = row["max"]
    task_idx = int(row["learning_task_index"])
    task_name = row["learning_task_name"]

    # 竖虚线
    ax_acc.axvline(
        x=start_ep,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7
    )

    # 在每个任务阶段中间写 Learning task
    mid_ep = (start_ep + end_ep) / 2
    ax_acc.text(
        mid_ep,
        8,
        f"Learning task {task_idx}\n{task_name}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
        alpha=0.85
    )


# =========================
# 5. 坐标轴和标题
# =========================
ax_acc.set_xlabel("Global Epoch", fontsize=12)
ax_acc.set_ylabel("Test Accuracy (%)", fontsize=12)
ax_loss.set_ylabel("Test Loss", fontsize=12)

ax_acc.set_ylim(0, 102)

ax_acc.set_title(
    "Continual Evaluation: Accuracy and Loss for Each Task",
    fontsize=14
)

ax_acc.grid(alpha=0.25)


# =========================
# 6. 合并图例
# =========================
all_lines = acc_lines + loss_lines
all_labels = [line.get_label() for line in all_lines]

ax_acc.legend(
    all_lines,
    all_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.13),
    ncol=3,
    fontsize=8,
    frameon=False
)

fig.tight_layout()
fig.savefig(save_path, bbox_inches="tight")
plt.close(fig)

print(f"Figure saved to: {save_path}")