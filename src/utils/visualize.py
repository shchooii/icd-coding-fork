import json, numpy as np, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from math import ceil
from typing import Optional
from matplotlib.patches import Patch
import pandas as pd

@torch.no_grad()
def plot_per_class_best_threshold(
    logits: torch.Tensor,            # (N, C) – GPU tensor
    targets: torch.Tensor,           # (N, C) – GPU tensor, {0,1}
    label_transform,
    json_path: str = "/home/mixlab/tabular/icd-coding/files/data/mimiciv_icd10/icd10_longtail_split.json",
    save_path: str = "class_wise_best_threshold.svg",
    csv_path:  Optional[str] = "threshold_curve.csv",
    # ── 그래프 옵션 ───────────────────────────────────────────────
    point_size: float = 1.3,
    smooth_prop: float = 0.10,        # 전체의 10 % 창
    line_width: float = 1.2,          # 평활선 두께
    chunk: Optional[int] = 1024,
    # ── 색상 팔레트 ────────────────────────────────────────────────
    scatter_color: str = "#1f77b4",   # 산점도
    line_color: str    = "#d62728",   # 평활선 (crimson)
    head_color: str    = "#ffe6e6",   # Head 영역
    med_color: str     = "#fff6cc",   # Medium 영역
    tail_color: str    = "#e6ecff",   # Tail 영역
):
    # ── 1. JSON → 인덱스 ──────────────────────────────────────────
    with open(json_path, encoding="utf-8") as f:
        split = json.load(f)
    ordered_codes = list(split["head"]) + list(split["medium"]) + list(split["tail"])
    ordered_idx = torch.tensor(label_transform.get_indices(ordered_codes),
                               device=logits.device)

    # ── 2. threshold 후보 ─────────────────────────────────────────
    thrs = torch.linspace(0., 1., 101, device=logits.device)          # (T,)
    best_thr = torch.empty(len(ordered_idx), device=logits.device)

    # ── 3. chunk 단위로 F1 최대값 계산 (GPU) ─────────────────────
    for s in range(0, len(ordered_idx), chunk or len(ordered_idx)):
        idx = ordered_idx[s:s + (chunk or len(ordered_idx))]
        logit_c = logits[:, idx]                   # (N, C')
        true_c  = targets[:, idx].bool()           # (N, C')

        preds = logit_c.unsqueeze(2) >= thrs       # (N, C', T)
        true  = true_c.unsqueeze(2)                # (N, C', 1)

        tp = (preds &  true).sum(0).float()
        fp = (preds & ~true).sum(0).float()
        fn = (~preds &  true).sum(0).float()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)    # (C', T)

        best_thr[s:s+len(idx)] = thrs[torch.argmax(f1, dim=1)]

    best_thr_cpu = best_thr.cpu().numpy()

    # ── 4. 강한 평활(MA) ─────────────────────────────────────────
    if 0. < smooth_prop < 1.:
        win = max(2, ceil(len(best_thr) * smooth_prop))
        pad = win // 2
        padded = F.pad(best_thr.view(1, 1, -1), (pad, pad), mode="replicate")
        smoothed = F.avg_pool1d(padded, kernel_size=win, stride=1)\
                     .squeeze().cpu().numpy()
    else:
        smoothed = np.full_like(best_thr_cpu, np.nan)

    # ── 5. (선택) CSV 저장 ───────────────────────────────────────
    if csv_path:
        pd.DataFrame({
            "index":    np.arange(len(best_thr_cpu)),
            "best_thr": best_thr_cpu,
            "smoothed": smoothed
        }).to_csv(csv_path, index=False)
        print(f"💾 CSV 저장 완료: {csv_path}")

    # ── 6. 시각화 ───────────────────────────────────────────────
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(14, 4))

    x = np.arange(len(best_thr_cpu))
    scat = plt.scatter(x, best_thr_cpu, s=point_size,
                       alpha=0.6, color=scatter_color,
                       label="Best threshold")
    line, = plt.plot(x, smoothed, lw=line_width, color=line_color,
                     ls="--", label="Moving Avg")

    head_end = len(split["head"]) - 1
    med_end  = head_end + len(split["medium"])
    plt.axvspan(-.5, head_end+.5,            alpha=.15, color=head_color)
    plt.axvspan(head_end+.5, med_end+.5,     alpha=.15, color=med_color)
    plt.axvspan(med_end+.5, len(best_thr)-.5,alpha=.15, color=tail_color)

    # 범례 (흰 배경)
    legend_patches = [
        Patch(facecolor=head_color, edgecolor='none', alpha=.7, label='Head'),
        Patch(facecolor=med_color,  edgecolor='none', alpha=.7, label='Medium'),
        Patch(facecolor=tail_color, edgecolor='none', alpha=.7, label='Tail')
    ]
    spacer = Patch(fc="none", ec="none", label="")
    handles = [scat, line] + [spacer] + legend_patches
    labels  = ["Best threshold", "Moving Avg", "Head", "Medium", "Tail"]
    leg = plt.legend(handles, labels, loc='upper right',
                     ncol=2, fontsize=9, frameon=True)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#dddddd")
    leg.get_frame().set_alpha(1.0)

    plt.title("Class‑wise Best Thresholds L0: BCE", fontsize=15)
    plt.xlabel("Index"); plt.ylabel("Best Threshold"); plt.ylim(0, 1)
    plt.tight_layout(); plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 그래프 저장: {save_path}")

