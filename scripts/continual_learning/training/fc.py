#!/usr/bin/env python3
"""
Sequential continual learning — exclusive FC weight allocation (zero forgetting).

Each task permanently owns a disjoint subset of shared-FC weights.
Once frozen, a task's weights are never touched by any later task,
so old-task accuracy does not degrade.

Task order:
    MNIST → QMNIST → EMNIST-Digits → Fashion-MNIST → EMNIST-Letters → EMNIST-Balanced
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data" / "datasets"
OUTPUT = PROJECT / "results" / "continual_learning"


# ====================================================================
#  Model — exclusive-masked MLP
# ====================================================================

class MaskedExclusiveMLP(nn.Module):
    """Shared FC backbone with per-task exclusive weight masks.

    Each layer stores two boolean flat buffers per task lifecycle:
      ``mask_{name}_frozen``  — union of ALL past tasks' allocations (permanent)
      ``mask_{name}_task``    — current task's allocation
    A registered gradient hook zeros out frozen positions and, when
    ``_task_active`` is True, also zeros out non-current-task positions.
    """

    def __init__(self, width: int = 512) -> None:
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(784, width, bias=False)
        self.fc2 = nn.Linear(width, width, bias=False)
        self.fc3 = nn.Linear(width, 256, bias=False)
        self.heads: nn.ModuleDict = nn.ModuleDict()

        self._layers: List[Tuple[str, nn.Linear]] = [
            ("fc1", self.fc1), ("fc2", self.fc2), ("fc3", self.fc3)
        ]
        self._task_names: List[str] = []
        self._task_active: bool = False

        for name, layer in self._layers:
            n = layer.weight.numel()
            self.register_buffer(f"mask_{name}_frozen", torch.zeros(n, dtype=torch.bool))
            self.register_buffer(f"mask_{name}_task", torch.zeros(n, dtype=torch.bool))
            layer.weight.register_hook(self._make_hook(name, n))

    # --- internal helpers ---

    def _mf(self, name: str) -> torch.Tensor:
        """Return frozen mask buffer for *name*."""
        return getattr(self, f"mask_{name}_frozen")

    def _mt(self, name: str) -> torch.Tensor:
        """Return current-task mask buffer for *name*."""
        return getattr(self, f"mask_{name}_task")

    def _make_hook(self, name: str, _numel: int):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            g = grad
            frozen = self._mf(name)
            if frozen.any():
                g = g.masked_fill(frozen.view_as(g), 0)
            if self._task_active:
                task = self._mt(name)
                if task.any():
                    g = g.masked_fill(~task.view_as(g), 0)
            return g
        return hook

    # --- allocation ---

    def allocate(self, task_name: str, num_classes: int,
                 total_tasks: int, device: torch.device) -> None:
        """Reserve an equal share of *remaining* weights for *task_name*."""
        if task_name not in self.heads:
            self.heads[task_name] = nn.Linear(256, num_classes).to(device)
        if task_name not in self._task_names:
            self._task_names.append(task_name)

        for name, layer in self._layers:
            total = layer.weight.numel()
            frozen = self._mf(name)
            budget = total // total_tasks
            available = (~frozen).nonzero(as_tuple=False).flatten()
            k = min(budget, available.numel())
            if k <= 0:
                torch.zeros(total, dtype=torch.bool, device=frozen.device,
                            out=getattr(self, f"mask_{name}_task"))
                continue
            scores = layer.weight.detach().abs().flatten()[available]
            _, idx = torch.topk(scores, k)
            sel = available[idx]
            m = torch.zeros(total, dtype=torch.bool, device=frozen.device)
            m[sel] = True
            getattr(self, f"mask_{name}_task").copy_(m)

    def freeze(self) -> None:
        """Permanently freeze current task's weight allocation."""
        for name, _ in self._layers:
            frozen = self._mf(name)
            frozen.copy_(frozen | self._mt(name))

    def store_task_masks(self) -> Dict[str, torch.Tensor]:
        """Snapshot current task masks (for later per-task eval)."""
        return {name: self._mt(name).detach().cpu().clone()
                for name, _ in self._layers}

    def apply_task_masks(self, masks: Dict[str, torch.Tensor], device: torch.device) -> None:
        """Restore per-task masks for inference."""
        for name, _ in self._layers:
            self._mt(name).copy_(masks[name].to(device))

    def restore_frozen_weights(self, snapshot: Dict[str, torch.Tensor]) -> float:
        """Restore frozen weights after optimizer updates and return max pre-restore drift."""
        max_drift = 0.0
        with torch.no_grad():
            for name, layer in self._layers:
                frozen = self._mf(name).view_as(layer.weight)
                if not frozen.any():
                    continue
                before = snapshot[name].to(layer.weight.device)
                drift = (layer.weight.detach()[frozen] - before).abs()
                if drift.numel() > 0:
                    max_drift = max(max_drift, float(drift.max().item()))
                layer.weight.data[frozen] = before
        return max_drift

    def assert_task_masks(self, expected: Dict[str, torch.Tensor], device: torch.device) -> None:
        """Fail fast if the active task mask was accidentally replaced."""
        for name, _ in self._layers:
            if not torch.equal(self._mt(name), expected[name].to(device)):
                raise RuntimeError(f"Active mask for {name} changed unexpectedly.")

    # --- forward ---

    def forward(self, x: torch.Tensor, task_name: str):
        for name, layer in self._layers:
            w = layer.weight * self._mt(name).view_as(layer.weight)
            x = F.relu(F.linear(x, w, None))
        return self.heads[task_name](x)

    def feature_dim(self) -> int:
        return 256

    # --- stats ---

    def mask_stats(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for name, layer in self._layers:
            t = layer.weight.numel()
            f = int(self._mf(name).sum().item())
            out[name] = {"total": t, "frozen": f, "free": t - f}
        return out


# ====================================================================
#  Dataset — IDX reader
# ====================================================================

class FlatImageDataset(Dataset):
    """Reads [N,H,W] images, normalises, flattens to [N, H*W]."""

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 mean: float = 0.1307, std: float = 0.3081,
                 fix_orientation: bool = False) -> None:
        self.images = images
        self.labels = labels.astype(np.int64)
        self.mean, self.std = mean, std
        self.fix = fix_orientation

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        if self.fix:
            img = np.fliplr(np.rot90(img, k=-1)).copy()
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return torch.from_numpy(img).view(-1), \
               torch.tensor(int(self.labels[idx]), dtype=torch.long)


def _open(path: Path):
    return gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")


def _read_idx(path: Path) -> np.ndarray:
    dt = {8: np.uint8, 9: np.int8, 11: np.int16, 12: np.int32, 13: np.float32, 14: np.float64}
    with _open(path) as f:
        m = int.from_bytes(f.read(4), "big")
        dc, nd = (m >> 8) & 0xFF, m & 0xFF
        shape = [int.from_bytes(f.read(4), "big") for _ in range(nd)]
        data = f.read()
    a = np.frombuffer(data, dtype=np.dtype(dt[dc]).newbyteorder(">")).reshape(shape)
    return a.astype(dt[dc], copy=False)


def _idx_file(d: Path, name: str) -> Path:
    for p in [d / name, d / f"{name}.gz"]:
        if p.exists() and p.stat().st_size > 0:
            return p
    raise FileNotFoundError(f"{name} in {d}")


def _ld_mnist(root, folder, name):
    d = root / folder / "raw"
    return dict(name=name, nc=10,
                tr=FlatImageDataset(_read_idx(_idx_file(d, "train-images-idx3-ubyte")),
                                    _read_idx(_idx_file(d, "train-labels-idx1-ubyte"))),
                te=FlatImageDataset(_read_idx(_idx_file(d, "t10k-images-idx3-ubyte")),
                                    _read_idx(_idx_file(d, "t10k-labels-idx1-ubyte"))))


def _ld_qmnist(root):
    d = root / "QMNIST" / "raw"
    tl = _read_idx(_idx_file(d, "qmnist-train-labels-idx2-int"))
    vl = _read_idx(_idx_file(d, "qmnist-test-labels-idx2-int"))
    if tl.ndim == 2: tl = tl[:, 0]
    if vl.ndim == 2: vl = vl[:, 0]
    return dict(name="QMNIST", nc=10,
                tr=FlatImageDataset(_read_idx(_idx_file(d, "qmnist-train-images-idx3-ubyte")), tl),
                te=FlatImageDataset(_read_idx(_idx_file(d, "qmnist-test-images-idx3-ubyte")), vl))


def _ld_emnist(root, split):
    cc = {"digits": 10, "letters": 26, "balanced": 47}
    dn = {"digits": "EMNIST-Digits", "letters": "EMNIST-Letters", "balanced": "EMNIST-Balanced"}
    d = root / "EMNIST" / "raw"
    tl = _read_idx(_idx_file(d, f"emnist-{split}-train-labels-idx1-ubyte"))
    vl = _read_idx(_idx_file(d, f"emnist-{split}-test-labels-idx1-ubyte"))
    if split == "letters":
        tl, vl = tl - 1, vl - 1
    return dict(name=dn[split], nc=cc[split],
                tr=FlatImageDataset(
                    _read_idx(_idx_file(d, f"emnist-{split}-train-images-idx3-ubyte")), tl,
                    fix_orientation=True),
                te=FlatImageDataset(
                    _read_idx(_idx_file(d, f"emnist-{split}-test-images-idx3-ubyte")), vl,
                    fix_orientation=True))


def build_tasks(root: Path) -> List[dict]:
    return [
        _ld_mnist(root, "MNIST", "MNIST"),
        _ld_qmnist(root),
        _ld_emnist(root, "digits"),
        _ld_mnist(root, "FashionMNIST", "Fashion-MNIST"),
        _ld_emnist(root, "letters"),
        _ld_emnist(root, "balanced"),
    ]


# ====================================================================
#  Training / evaluation helpers
# ====================================================================

def make_loaders(task: dict, args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    vs = int(len(task["tr"]) * args.val_ratio)
    ts = len(task["tr"]) - vs
    gen = torch.Generator().manual_seed(args.seed)
    tr_sub, va_sub = random_split(task["tr"], [ts, vs], generator=gen)
    kw = dict(num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    return (DataLoader(tr_sub, batch_size=args.batch_size, shuffle=True, **kw),
            DataLoader(va_sub, batch_size=args.test_batch_size, **kw),
            DataLoader(task["te"], batch_size=args.test_batch_size, **kw))


def _run_epoch(model, tname, loader, optimizer, criterion, device, epoch, epochs, training):
    """One epoch of training or evaluation.  Returns (loss, acc, elapsed_s)."""
    if training:
        model.train()
        model._task_active = True
    else:
        model.eval()
        model._task_active = False

    t0 = time.time()
    tot_loss = tot_cor = tot_n = 0

    it = (tqdm(loader, desc=f"    epoch {epoch}/{epochs}", leave=False, ncols=95)
          if (training and tqdm) else loader)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labs in it:
            imgs, labs = imgs.to(device), labs.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(imgs, tname)
            loss = criterion(logits, labs)
            if training:
                frozen_backup = {}
                with torch.no_grad():
                    for lname, layer in model._layers:
                        fm = model._mf(lname).view_as(layer.weight)
                        if fm.any():
                            frozen_backup[lname] = layer.weight.detach()[fm].cpu().clone()
                loss.backward()
                optimizer.step()
                model.restore_frozen_weights(frozen_backup)
                # clear AdamW internal state (exp_avg, exp_avg_sq) for frozen weights
                for lname, layer in model._layers:
                    state = optimizer.state.get(layer.weight)
                    if not state:
                        continue
                    fm = model._mf(lname).view_as(layer.weight)
                    if fm.any():
                        for v in state.values():
                            if torch.is_tensor(v) and v.shape == layer.weight.shape:
                                v.data.masked_fill_(fm, 0)

            n = labs.size(0)
            tot_loss += loss.item() * n
            tot_cor += int(logits.argmax(1).eq(labs).sum().item())
            tot_n += n
            if training and tqdm and hasattr(it, "set_postfix"):
                it.set_postfix(loss=f"{tot_loss/tot_n:.4f}", acc=f"{tot_cor/tot_n:.4f}")

    model._task_active = False
    return tot_loss / tot_n, tot_cor / tot_n, time.time() - t0


def _continual_eval(model, all_tasks, task_masks, test_loaders,
                    criterion, device, ce_csv, global_ep, cur_idx, cur_name, epoch):
    """Evaluate every previously-learned task and append to continual-eval csv."""
    rows = []
    current_masks = model.store_task_masks()
    for prev_idx in range(1, cur_idx + 1):
        pt = all_tasks[prev_idx - 1]
        pn = pt["name"]
        if pn in task_masks:
            model.apply_task_masks(task_masks[pn], device)
        # else: current task — model._mt() already has the right masks
        if pn not in task_masks:
            model.apply_task_masks(current_masks, device)
        loader = test_loaders[pn]
        loss, acc, _ = _run_epoch(model, pn, loader, None, criterion, device, 0, 0, False)
        rows.append({"global_epoch": global_ep, "learning_task_index": cur_idx,
                      "learning_task_name": cur_name, "epoch_in_task": epoch,
                      "evaluated_task_index": prev_idx, "evaluated_task_name": pn,
                      "test_acc": acc, "test_acc_percent": acc * 100.0, "test_loss": loss})
    model.apply_task_masks(current_masks, device)
    write_h = not ce_csv.exists()
    with open(ce_csv, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_h:
            w.writeheader()
        w.writerows(rows)
    return rows


# ====================================================================
#  CSV / log / plot helpers
# ====================================================================

def _append_csv(path: Path, row: dict):
    h = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if h:
            w.writeheader()
        w.writerow(row)


def _plot_acc(rows, tname, path):
    r = [x for x in rows if x["task_name"] == tname]
    if not r:
        return
    ep = [x["epoch"] for x in r]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(ep, [x["train_acc"] for x in r], label="train acc", linewidth=1.8)
    ax.plot(ep, [x["val_acc"] for x in r], label="val acc", linewidth=1.8)
    ax.set(xlabel="Epoch", ylabel="Accuracy", title=f"{tname}  Acc-Epoch", ylim=(0, 1.02))
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(path); plt.close(fig)


def _plot_ce(rows, tasks, epochs_per_task, path):
    """Continual-evaluation figure (like the reference screenshot)."""
    if not rows:
        return
    names = [t["name"] for t in tasks]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(14, 5.4), dpi=150)
    ax.axhspan(0, 20, color="#f1f1f1", zorder=0)
    for idx, nm in enumerate(names, 1):
        tr = [r for r in rows if r["evaluated_task_name"] == nm]
        if not tr:
            continue
        xs = [float(r["global_epoch"]) for r in tr]
        ys = [float(r["test_acc_percent"]) for r in tr]
        ax.plot(xs, ys, linewidth=2.1, color=cmap((idx - 1) % 10), label=nm)

    total = epochs_per_task * len(tasks)
    for idx, t in enumerate(tasks, 1):
        sx = (idx - 1) * epochs_per_task + 1
        ax.axvline(sx, color="#999", ls="--", lw=1.2, alpha=.75)
        ax.text(sx + epochs_per_task * 0.46, 41, f"Learning\n{t['name']}",
                ha="center", va="center", fontsize=8, color="#222")
    ax.set_xlim(1, total); ax.set_ylim(0, 102)
    ax.set_xlabel("Global epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Continual evaluation on learned tasks")
    ax.grid(alpha=.18); ax.legend(loc="lower left", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _tsne(features, labels, preds, nc, seed, fig_path, csv_class_path, csv_flat_path):
    """t-SNE scatter + two CSV outputs (per-class columns + flat)."""
    n = len(labels)
    if n < 6:
        return
    perp = min(30, max(5, (n - 1) // 3))
    coords = TSNE(n_components=2, init="pca", learning_rate="auto",
                  perplexity=perp, random_state=seed).fit_transform(features)

    # flat CSV
    with open(csv_flat_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["x", "y", "label", "pred"])
        for (x, y), la, pr in zip(coords, labels, preds):
            w.writerow([float(x), float(y), int(la), int(pr)])

    # per-class column CSV
    cls_pts = {c: coords[labels == c] for c in range(nc)}
    mx = max((len(v) for v in cls_pts.values()), default=0)
    fields = []
    for c in range(nc):
        fields.extend([f"class_{c}_x", f"class_{c}_y"])
    with open(csv_class_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for ri in range(mx):
            row = {}
            for c in range(nc):
                pts = cls_pts[c]
                if ri < len(pts):
                    row[f"class_{c}_x"] = float(pts[ri, 0])
                    row[f"class_{c}_y"] = float(pts[ri, 1])
                else:
                    row[f"class_{c}_x"] = row[f"class_{c}_y"] = ""
            w.writerow(row)

    # figure
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    cm = plt.get_cmap("tab20" if nc <= 20 else "gist_ncar")
    for c in range(nc):
        p = coords[labels == c]
        if len(p) == 0:
            continue
        ax.scatter(p[:, 0], p[:, 1], s=8, alpha=.75,
                   color=cm(c / max(1, nc - 1)), label=str(c))
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.set_title(f"t-SNE  ({nc} classes)")
    if nc <= 15:
        ax.legend(markerscale=2, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(fig_path); plt.close(fig)


# ====================================================================
#  Argparse
# ====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Continual learning — exclusive FC weight allocation (zero forgetting).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--project-root", type=Path, default=PROJECT)
    p.add_argument("--data-root", type=Path, default=DATA)
    p.add_argument("--output-root", type=Path, default=OUTPUT)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


# ====================================================================
#  Main
# ====================================================================

def main():
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
                   or args.device == "cuda" else "cpu")

    # ---- output dirs ----
    run_name = datetime.now().strftime("output_fc_%Y%m%d_%H%M%S")
    dirs = {k: args.output_root / run_name / k
            for k in ("checkpoints", "logs", "metrics", "masks", "figures")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ---- logger ----
    logger = logging.getLogger("excl"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in [logging.FileHandler(dirs["logs"] / "train.log", encoding="utf-8"),
              logging.StreamHandler()]:
        h.setFormatter(fmt); logger.addHandler(h)

    logger.info("Device: %s  width: %d", device, args.width)

    # ---- save training args to txt ----
    cfg_path = dirs["metrics"] / "training_args.txt"
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write("Training Configuration\n" + "=" * 60 + "\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")
        f.write(f"  device: {device}\n")
        f.write(f"  run_dir: {dirs['checkpoints'].parent}\n")
    logger.info("Args saved to %s", cfg_path)

    # ---- build tasks ----
    tasks = build_tasks(args.data_root)
    logger.info("Tasks (%d): %s", len(tasks), " -> ".join(t["name"] for t in tasks))

    # ---- model ----
    model = MaskedExclusiveMLP(width=args.width).to(device)
    criterion = nn.CrossEntropyLoss()

    # ---- CSV paths ----
    epoch_csv = dirs["metrics"] / "epoch_metrics.csv"
    ce_csv = dirs["metrics"] / "continual_eval.csv"

    all_epoch_rows: List[dict] = []
    all_ce_rows: List[dict] = []
    task_masks: Dict[str, Dict[str, torch.Tensor]] = {}
    test_loaders: Dict[str, DataLoader] = {}
    global_epoch = 0
    total_tasks = len(tasks)

    # ==================================================================
    #  Task loop
    # ==================================================================
    for ti, task in enumerate(tasks, 1):
        tname = task["name"]
        nc = task["nc"]
        logger.info("=" * 70)
        logger.info("Task %d/%d: %s  (%d classes)", ti, total_tasks, tname, nc)

        # --- 1. allocate exclusive weights ---
        model.allocate(tname, nc, total_tasks, device)
        stats = model.mask_stats()
        for ln, s in stats.items():
            logger.info("  %s  total=%d  frozen=%d  free=%d",
                        ln, s["total"], s["frozen"], s["free"])

        # --- 2. optimizer (all params; hook handles masking) ---
        params = [p for _, l in model._layers for p in l.parameters()]
        params.extend(model.heads[tname].parameters())
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

        # --- 3. data loaders ---
        tr_loader, va_loader, te_loader = make_loaders(task, args)
        test_loaders[tname] = te_loader

        best_va = -1.0
        ckpt_path = dirs["checkpoints"] / f"task_{ti}_{tname}_best.pt"

        # --- 4. train epochs (with frozen-weight drift monitor) ---
        # snapshot all previous tasks' weights so we can detect any drift
        frozen_snapshot: Dict[str, Dict[str, torch.Tensor]] = {}
        if ti > 1:
            with torch.no_grad():
                for prev_name in [t["name"] for t in tasks[:ti - 1]]:
                    pm = task_masks[prev_name]
                    frozen_snapshot[prev_name] = {}
                    for lname, layer in model._layers:
                        task_m = pm[lname].view_as(layer.weight)
                        frozen_snapshot[prev_name][lname] = layer.weight.detach()[task_m].cpu().clone()

        for ep in range(1, args.epochs + 1):
            global_epoch += 1

            expected_task_masks = model.store_task_masks()
            tr_loss, tr_acc, dt = _run_epoch(
                model, tname, tr_loader, optimizer, criterion,
                device, ep, args.epochs, training=True)
            model.assert_task_masks(expected_task_masks, device)

            # --- frozen weight drift check ---
            if frozen_snapshot:
                max_drift = 0.0
                with torch.no_grad():
                    for prev_name, layers in frozen_snapshot.items():
                        pm = task_masks[prev_name]
                        for lname, orig_w in layers.items():
                            layer = dict(model._layers)[lname]
                            task_m = pm[lname].view_as(layer.weight)
                            cur_w = layer.weight.detach()[task_m].cpu()
                            drift = (cur_w - orig_w).abs().max().item()
                            max_drift = max(max_drift, drift)
                if max_drift > 1e-10:
                    raise RuntimeError(f"Frozen weights drifted at epoch {ep}: max_delta={max_drift:.6e}")

            va_loss, va_acc, _ = _run_epoch(
                model, tname, va_loader, None, criterion,
                device, ep, args.epochs, training=False)
            model.assert_task_masks(expected_task_masks, device)

            improved = va_acc > best_va
            if improved:
                best_va = va_acc
                torch.save({"ti": ti, "name": tname, "epoch": ep,
                            "sd": model.state_dict(), "va": va_acc,
                            "task_masks": model.store_task_masks()}, ckpt_path)

            row = {"global_epoch": global_epoch, "task_index": ti,
                   "task_name": tname, "epoch": ep,
                   "train_loss": tr_loss, "val_loss": va_loss,
                   "train_acc": tr_acc, "val_acc": va_acc,
                   "time_s": round(dt, 2), "best_val_acc": best_va,
                   "improved": improved}
            _append_csv(epoch_csv, row)
            all_epoch_rows.append(row)

            marker = "*" if improved else " "
            logger.info("  epoch %03d/%03d  train %.4f / %.4f  val %.4f / %.4f  %s  %.1fs",
                        ep, args.epochs, tr_loss, tr_acc, va_loss, va_acc, marker, dt)

            # --- continual eval ---
            ce_rows = _continual_eval(model, tasks, task_masks, test_loaders,
                                       criterion, device, ce_csv,
                                       global_epoch, ti, tname, ep)
            model.assert_task_masks(expected_task_masks, device)
            all_ce_rows.extend(ce_rows)

        # --- 5. load best, freeze, store masks ---
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["sd"])
        model.apply_task_masks(ckpt["task_masks"], device)
        model.freeze()
        task_masks[tname] = model.store_task_masks()

        torch.save({"ti": ti, "name": tname, "frozen_masks": {
            n: model._mf(n).cpu() for n, _ in model._layers}},
            dirs["masks"] / f"task_{ti}_{tname}_frozen.pt")

        # --- 6. test with best model ---
        te_loss, te_acc, yt, yp = 0.0, 0.0, [], []
        model.eval()
        with torch.no_grad():
            for imgs, labs in te_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                logits = model(imgs, tname)
                te_loss += criterion(logits, labs).item() * labs.size(0)
                preds = logits.argmax(1)
                yt.append(labs.cpu().numpy()); yp.append(preds.cpu().numpy())
        te_loss /= len(te_loader.dataset)
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        te_acc = float((yt == yp).mean())

        # confusion matrix
        cm = confusion_matrix(yt, yp, labels=list(range(nc)))
        cm_path = dirs["figures"] / f"{tname}_confusion_matrix.csv"
        with open(cm_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred"] + [f"p{i}" for i in range(nc)])
            for i, r in enumerate(cm):
                w.writerow([f"t{i}"] + r.tolist())
        logger.info("  TEST  loss=%.6f  acc=%.4f  cm=%s", te_loss, te_acc, cm_path)

        with open(dirs["metrics"] / f"{tname}_test.json", "w", encoding="utf-8") as f:
            json.dump({"task_index": ti, "task_name": tname,
                        "test_loss": te_loss, "test_acc": te_acc,
                        "confusion_csv": str(cm_path)}, f, indent=2)

        # --- 7. t-SNE ---
        all_feats: list = []
        model.eval()
        with torch.no_grad():
            for imgs, labs in te_loader:
                imgs = imgs.to(device)
                for name, layer in model._layers:
                    w = layer.weight * model._mt(name).view_as(layer.weight)
                    imgs = F.relu(F.linear(imgs, w, None))
                all_feats.append(imgs.cpu().numpy())
        feats = np.concatenate(all_feats)

        _tsne(feats, yt, yp, nc, args.seed,
              dirs["figures"] / f"{tname}_tsne.png",
              dirs["figures"] / f"{tname}_tsne_class_points.csv",
              dirs["figures"] / f"{tname}_tsne_flat.csv")
        logger.info("  t-SNE saved for %s", tname)

        # --- 8. Acc-Epoch plot ---
        _plot_acc(all_epoch_rows, tname, dirs["figures"] / f"{tname}_acc_epoch.png")

        logger.info("  Done %s  frozen: %s", tname,
                     {n: s["frozen"] for n, s in model.mask_stats().items()})

    # ==================================================================
    #  Post-training
    # ==================================================================

    # continual-eval figure
    _plot_ce(all_ce_rows, tasks, args.epochs,
             dirs["figures"] / "continual_eval.png")

    # final summary
    logger.info("=" * 70)
    logger.info("FINAL frozen weights:")
    for n, s in model.mask_stats().items():
        logger.info("  %s  total=%d  frozen=%d  free=%d", n, s["total"], s["frozen"], s["free"])
    logger.info("Outputs: %s", dirs["checkpoints"].parent)
    print(f"\nDone.  Outputs: {dirs['checkpoints'].parent}")


if __name__ == "__main__":
    main()
