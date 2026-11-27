r"""
Generate all upgraded publication-grade figures matching Journal of Electronic Imaging
(JEI, SPIE/IS&T) Volume 35 (2026) standards and Chris Mack (2018) Ch. 4.4 guidelines:
  - Pure RGB PNGs at 300 DPI, crisp white backgrounds
  - Zero embedded top chart titles (LaTeX captions serve as titles)
  - Clean (a), (b), (c) sub-panel labels, minimal/no background grid clutter
  - Grayscale-distinguishable markers, line styles, and bar hatching
  - 100% symbol consistency with manuscript equations (z_i^(R), z_i^(V), p_k^(R), p_k^(V))

Generates / refreshes in SPIE_JEI_Submission/:
  1. 1.ARCHITECTURE.png             (symbol-synced overall architecture, no top title)
  2. fig_fusion_modules.png         (3-panel Concat vs Gated vs Weighted Score module diagram)
  3. fig_dataset_protocols.png      (real ISL sample grid + 4-protocol disjoint/enrollment map)
  4. 14.png                         (grayscale-safe hatched supervised vs meta-learning bar)
  5. fig_fusion_ablation.png        (grayscale-safe hatched source-of-truth fusion ablation)
  6. fig_loso_and_simultaneous.png  (6-fold MD5-sanitized LOSO + Simultaneous Enrollment)
  7. fig_weight_sensitivity_pareto.png (real w-sweep curve + Accuracy vs GPU/CPU FPS Pareto)
  8. fig_tsne_confusion_error.png   (real fused t-SNE + confusion heatmap + signer errors)
  9. fig_qualitative_failure_cases.png (real ResNet-12 Grad-CAM + ViT Attention Rollout: Success & Failure)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from sklearn.manifold import TSNE

SPIE_DIR = Path(r"C:\Users\SYLA\Desktop\RESEARCH\SPIE_JEI_Submission")
OUT_DIR = Path(r"C:\Users\SYLA\Desktop\RESEARCH\REVIEW1\experiments\revision_outputs")
GIT_RES = Path(r"C:\Users\SYLA\Desktop\RESEARCH\GIT\revision_experiments\results")
DATA_ROOT = Path(r"C:\Users\SYLA\Desktop\RESEARCH\META1\data")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#222222",
    "axes.linewidth": 0.9,
    "hatch.linewidth": 0.7,
})


def save_rgb_300(fig, paths):
    first = paths[0]
    fig.savefig(first, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with Image.open(first) as im:
        rgb = im.convert("RGB")
        for p in paths:
            if p.parent.exists():
                rgb.save(p, format="PNG", dpi=(300, 300))


def draw_box(ax, x, y, w, h, text, fc="#EDF2F8", ec="#1D3557", lw=1.2, fontsize=8.0,
             fontweight="normal", textcolor="#111827", radius=0.015, ls="-", va="center"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3,
    )
    ax.add_patch(box)
    if text:
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va=va, fontsize=fontsize, fontweight=fontweight,
            color=textcolor, linespacing=1.28, zorder=4,
        )
    return box


def draw_arrow(ax, x1, y1, x2, y2, color="#1D3557", lw=1.4, rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_length=5.5,head_width=3.2",
        color=color, linewidth=lw,
        connectionstyle=f"arc3,rad={rad}", zorder=5,
    )
    ax.add_patch(arr)


# =====================================================================
# 1. FIG. 1 — Symbol-Synced Overall Architecture (NO Embedded Title)
# =====================================================================
def make_fig1_architecture():
    fig, ax = plt.subplots(figsize=(13.2, 5.4), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Outer dashed frame
    outer = FancyBboxPatch(
        (0.012, 0.03), 0.976, 0.94,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="#FFFFFF", edgecolor="#475569", linewidth=1.2, linestyle="--", zorder=1,
    )
    ax.add_patch(outer)

    # Left container: Episodic ISL Input
    draw_box(ax, 0.028, 0.14, 0.165, 0.79, "", fc="#FFFFFF", ec="#1D3557", lw=1.3)
    ax.text(0.1105, 0.895, "Episodic ISL Input", ha="center", va="top", fontsize=9.8, fontweight="bold", color="#1D3557", zorder=4)
    ax.text(0.1105, 0.835, "Support $\\mathcal{S}$ & Query $\\mathcal{Q}$\n($N$-way $K$-shot)", ha="center", va="top", fontsize=7.8, color="#334155", zorder=4)

    # 4 real ISL image thumbnails
    thumb_specs = [
        ("pain", "User_1", 0.072, 0.645),
        ("bad", "User_4", 0.149, 0.645),
        ("afraid", "User_3", 0.072, 0.435),
        ("doctor", "User_2", 0.149, 0.435),
    ]
    for cls, sgn, cx, cy in thumb_specs:
        sdir = DATA_ROOT / cls / sgn
        img_file = sorted(sdir.glob("*.jpg"))[10]
        with Image.open(img_file) as im:
            arr = np.asarray(im.convert("RGB").resize((84, 84), Image.Resampling.BILINEAR))
        oi = OffsetImage(arr, zoom=0.50)
        ab = AnnotationBbox(oi, (cx, cy), frameon=True, pad=0.10,
                            bboxprops=dict(edgecolor="#64748B", linewidth=0.8, facecolor="white"))
        ab.set_zorder(5)
        ax.add_artist(ab)
        ax.text(cx, cy - 0.084, cls, ha="center", va="top", fontsize=7.2, style="italic", color="#1E293B", zorder=5)

    draw_box(
        ax, 0.036, 0.165, 0.149, 0.125,
        "Dual-Res Inputs\n84×84 & 224×224\nLinear Scale [0, 1]",
        fc="#EEF2F6", ec="#334155", lw=1.0, fontsize=6.9, fontweight="bold",
    )

    # Branch 1 (Top): Local Spatial CNN Path (ResNet-12)
    draw_box(ax, 0.218, 0.56, 0.462, 0.37, "", fc="#FFFFFF", ec="#1D3557", lw=1.3)
    ax.text(0.449, 0.905, "Branch 1: Local Spatial CNN Path (ResNet-12)", ha="center", va="top",
            fontsize=9.3, fontweight="bold", color="#1D3557", zorder=4)
    draw_box(ax, 0.230, 0.59, 0.148, 0.25,
             "ResNet-12 Encoder\n$f_{\\theta_r}^{(R)}(x_{i,84})$\n4 ResBlocks (64→640)\nAdaptive Avg Pool",
             fc="#E2ECF7", ec="#1D3557", lw=1.1, fontsize=7.3, fontweight="bold")
    draw_box(ax, 0.402, 0.59, 0.114, 0.25,
             "L2 Normalization\n$\\hat{z}_i^{(R)} \\in \\mathbb{S}^{639}$\n640-D Unit\nEmbedding",
             fc="#EDF3FA", ec="#335C81", lw=1.0, fontsize=7.3)
    draw_box(ax, 0.540, 0.59, 0.128, 0.25,
             "CNN Prototypes\n$p_k^{(R)} = \\mathrm{Norm}(\\bar{p}_k^{(R)})$\nScaled Cosine Logit:\n$s_R(q,k)=\\tau\\,\\hat{z}_q^{(R)\\top}p_k^{(R)}$",
             fc="#FFF7ED", ec="#B45309", lw=1.0, fontsize=7.1)
    draw_arrow(ax, 0.382, 0.715, 0.399, 0.715)
    draw_arrow(ax, 0.520, 0.715, 0.537, 0.715)

    # Branch 2 (Bottom): Global Contextual Transformer Path (ViT-B/16)
    draw_box(ax, 0.218, 0.14, 0.462, 0.37, "", fc="#FFFFFF", ec="#1D3557", lw=1.3)
    ax.text(0.449, 0.485, "Branch 2: Global Contextual Transformer Path (ViT-B/16)", ha="center", va="top",
            fontsize=9.3, fontweight="bold", color="#1D3557", zorder=4)
    draw_box(ax, 0.230, 0.17, 0.148, 0.25,
             "ViT-B/16 Encoder\n$f_{\\theta_v}^{(V)}(x_{i,224})$\n196 Patches (16×16)\n12× (LN + MSA + MLP)",
             fc="#E2ECF7", ec="#1D3557", lw=1.1, fontsize=7.3, fontweight="bold")
    draw_box(ax, 0.402, 0.17, 0.114, 0.25,
             "L2 Normalization\n$\\hat{z}_i^{(V)} \\in \\mathbb{S}^{767}$\n768-D Unit\n[CLS] Embedding",
             fc="#EDF3FA", ec="#335C81", lw=1.0, fontsize=7.3)
    draw_box(ax, 0.540, 0.17, 0.128, 0.25,
             "ViT Prototypes\n$p_k^{(V)} = \\mathrm{Norm}(\\bar{p}_k^{(V)})$\nScaled Cosine Logit:\n$s_V(q,k)=\\tau\\,\\hat{z}_q^{(V)\\top}p_k^{(V)}$",
             fc="#FFF7ED", ec="#B45309", lw=1.0, fontsize=7.1)
    draw_arrow(ax, 0.382, 0.295, 0.399, 0.295)
    draw_arrow(ax, 0.520, 0.295, 0.537, 0.295)

    # Input -> Branches arrows
    draw_arrow(ax, 0.193, 0.715, 0.218, 0.715)
    draw_arrow(ax, 0.193, 0.295, 0.218, 0.295)

    # Right container: Unified Hyperspherical Fusion
    draw_box(ax, 0.706, 0.14, 0.266, 0.79, "", fc="#FFFFFF", ec="#1D3557", lw=1.3)
    ax.text(0.839, 0.905, "Unified Hyperspherical\nScore-Level Fusion", ha="center", va="top",
            fontsize=9.5, fontweight="bold", color="#1D3557", zorder=4)

    draw_box(ax, 0.720, 0.69, 0.238, 0.135,
             "Metric Equivalence on $\\mathbb{S}^{d-1}$\n$\\|\\hat{z}_q^{(b)} - p_k^{(b)}\\|_2^2 = 2(1 - \\hat{z}_q^{(b)\\top} p_k^{(b)})$",
             fc="#F8FAFC", ec="#475569", lw=1.0, fontsize=7.5)
    draw_box(ax, 0.720, 0.47, 0.238, 0.175,
             "Learnable Weighted Score Fusion\n$s_{\\mathrm{fused}}(q,k) = w\\,s_R(q,k) + (1-w)\\,s_V(q,k)$\n$w = \\sigma(\\alpha_w) \\in (0,1)$, shared $\\tau$",
             fc="#DCE8F5", ec="#1D3557", lw=1.4, fontsize=7.7, fontweight="bold")
    draw_box(ax, 0.720, 0.305, 0.238, 0.12,
             "Episodic Softmax Posterior\n$p_\\Theta(y=k \\mid x_q, \\mathcal{S}) = \\mathrm{softmax}_k(s_{\\mathrm{fused}})$",
             fc="#F8FAFC", ec="#475569", lw=1.0, fontsize=7.4)
    draw_box(ax, 0.720, 0.165, 0.238, 0.10,
             "Predicted Sign Class $\\hat{y}_q = \\arg\\max_k s_{\\mathrm{fused}}(q,k)$",
             fc="#E6F0FA", ec="#1D3557", lw=1.2, fontsize=7.6, fontweight="bold")

    draw_arrow(ax, 0.667, 0.715, 0.720, 0.575, rad=-0.10)
    draw_arrow(ax, 0.667, 0.295, 0.720, 0.535, rad=0.10)
    draw_arrow(ax, 0.839, 0.690, 0.839, 0.645)
    draw_arrow(ax, 0.839, 0.470, 0.839, 0.425)
    draw_arrow(ax, 0.839, 0.305, 0.839, 0.265)

    # Bottom training bar note
    ax.text(
        0.50, 0.075,
        "End-to-End Episodic Meta-Training via Adam (Differential LRs: ResNet-12 & Fusion = $10^{-3}$, ViT-B/16 = $10^{-5}$; Frozen at Meta-Testing)",
        ha="center", va="center", fontsize=8.3, style="italic", color="#334155", zorder=4,
    )

    save_rgb_300(fig, [SPIE_DIR / "1.ARCHITECTURE.png"])
    print("[1/9] Saved symbol-synced 1.ARCHITECTURE.png")


# =====================================================================
# 2. FIG. 2 — Proposed Hyperspherical Fusion Module Comparison
# =====================================================================
def make_fig2_fusion_modules():
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3), dpi=300)
    titles = [
        "(a) Concatenation Fusion (+0.50 M Params)",
        "(b) Gated Feature Fusion (+0.50 M Params)",
        "(c) Weighted Score Fusion (+2 Scalars, Proposed)",
    ]

    for idx, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        border_c = "#1D3557" if idx == 2 else "#64748B"
        bg_c = "#F4F8FC" if idx == 2 else "#FFFFFF"
        lw_o = 1.6 if idx == 2 else 1.1
        draw_box(ax, 0.02, 0.03, 0.96, 0.90, "", fc=bg_c, ec=border_c, lw=lw_o, ls="-")
        ax.text(0.50, 0.965, title, ha="center", va="center", fontsize=9.1, fontweight="bold",
                color="#1D3557" if idx == 2 else "#1F2937")

        # Dual raw inputs at top
        draw_box(ax, 0.07, 0.77, 0.38, 0.12, "ResNet-12\n$f_{\\theta_r}^{(R)} \\in \\mathbb{R}^{640}$",
                 fc="#E2ECF7", ec="#1D3557", lw=1.0, fontsize=7.6, fontweight="bold")
        draw_box(ax, 0.55, 0.77, 0.38, 0.12, "ViT-B/16\n$f_{\\theta_v}^{(V)} \\in \\mathbb{R}^{768}$",
                 fc="#FFF3E0", ec="#B45309", lw=1.0, fontsize=7.6, fontweight="bold")

        if idx == 0:
            # Concatenation Fusion
            draw_box(ax, 0.07, 0.56, 0.38, 0.13, "Proj$_R$ (640→256)\nLN + GELU + Drop(0.2)\nL2 → $\\hat{u}^{(R)} \\in \\mathbb{S}^{255}$",
                     fc="#F8FAFC", ec="#475569", lw=0.9, fontsize=6.8)
            draw_box(ax, 0.55, 0.56, 0.38, 0.13, "Proj$_V$ (768→256)\nLN + GELU + Drop(0.2)\nL2 → $\\hat{u}^{(V)} \\in \\mathbb{S}^{255}$",
                     fc="#F8FAFC", ec="#475569", lw=0.9, fontsize=6.8)
            draw_box(ax, 0.12, 0.35, 0.76, 0.13, "Concatenate $[\\hat{u}^{(R)} \\,\\|\\, \\hat{u}^{(V)}] \\in \\mathbb{R}^{512}$\nLinear Projection (512→256) + L2 Norm",
                     fc="#FEF2F2", ec="#991B1B", lw=1.0, fontsize=7.2)
            draw_box(ax, 0.14, 0.17, 0.72, 0.11, "Joint Prototype $p_k \\in \\mathbb{S}^{255}$\nCosine Logit: $s(q,k) = \\tau\\,z_q^\\top p_k$",
                     fc="#FFFFFF", ec="#334155", lw=0.9, fontsize=7.2)
            ax.text(0.50, 0.095, "High projection capacity → overfits 12–14 base classes",
                    ha="center", va="center", fontsize=7.1, style="italic", color="#991B1B")
            draw_arrow(ax, 0.26, 0.77, 0.26, 0.69)
            draw_arrow(ax, 0.74, 0.77, 0.74, 0.69)
            draw_arrow(ax, 0.26, 0.56, 0.38, 0.48)
            draw_arrow(ax, 0.74, 0.56, 0.62, 0.48)
            draw_arrow(ax, 0.50, 0.35, 0.50, 0.28)

        elif idx == 1:
            # Gated Feature Fusion
            draw_box(ax, 0.07, 0.56, 0.38, 0.13, "Proj$_R$ (640→256)\nLN + GELU + Drop(0.2)\nL2 → $\\hat{u}^{(R)} \\in \\mathbb{S}^{255}$",
                     fc="#F8FAFC", ec="#475569", lw=0.9, fontsize=6.8)
            draw_box(ax, 0.55, 0.56, 0.38, 0.13, "Proj$_V$ (768→256)\nLN + GELU + Drop(0.2)\nL2 → $\\hat{u}^{(V)} \\in \\mathbb{S}^{255}$",
                     fc="#F8FAFC", ec="#475569", lw=0.9, fontsize=6.8)
            draw_box(ax, 0.09, 0.35, 0.82, 0.13, "Sigmoid Gate $\\gamma(x)=\\sigma(W_g[\\hat{u}^{(R)}\\|\\hat{u}^{(V)}]+b_g)$\n$z_{\\mathrm{gated}}=\\mathrm{Norm}(\\gamma\\odot\\hat{u}^{(R)}+(1-\\gamma)\\odot\\hat{u}^{(V)})$",
                     fc="#FFFBEB", ec="#B45309", lw=1.0, fontsize=6.9)
            draw_box(ax, 0.14, 0.17, 0.72, 0.11, "Gated Prototype $p_k \\in \\mathbb{S}^{255}$\nCosine Logit: $s(q,k) = \\tau\\,z_{\\mathrm{gated}}(x_q)^\\top p_k$",
                     fc="#FFFFFF", ec="#334155", lw=0.9, fontsize=7.2)
            ax.text(0.50, 0.095, "Adaptive per-sample gate; still pays MLP projection cost",
                    ha="center", va="center", fontsize=7.1, style="italic", color="#92400E")
            draw_arrow(ax, 0.26, 0.77, 0.26, 0.69)
            draw_arrow(ax, 0.74, 0.77, 0.74, 0.69)
            draw_arrow(ax, 0.26, 0.56, 0.38, 0.48)
            draw_arrow(ax, 0.74, 0.56, 0.62, 0.48)
            draw_arrow(ax, 0.50, 0.35, 0.50, 0.28)

        else:
            # Weighted Score Fusion (Proposed)
            draw_box(ax, 0.07, 0.56, 0.38, 0.13, "Parameter-Free L2\n$\\hat{z}_i^{(R)} \\in \\mathbb{S}^{639}$\nPrototype $p_k^{(R)}$",
                     fc="#E2ECF7", ec="#1D3557", lw=1.0, fontsize=7.2)
            draw_box(ax, 0.55, 0.56, 0.38, 0.13, "Parameter-Free L2\n$\\hat{z}_i^{(V)} \\in \\mathbb{S}^{767}$\nPrototype $p_k^{(V)}$",
                     fc="#FFF3E0", ec="#B45309", lw=1.0, fontsize=7.2)
            draw_box(ax, 0.07, 0.37, 0.38, 0.11, "Branch Logit\n$s_R = \\tau\\,\\hat{z}_q^{(R)\\top} p_k^{(R)}$",
                     fc="#FFFFFF", ec="#1D3557", lw=1.0, fontsize=7.2)
            draw_box(ax, 0.55, 0.37, 0.38, 0.11, "Branch Logit\n$s_V = \\tau\\,\\hat{z}_q^{(V)\\top} p_k^{(V)}$",
                     fc="#FFFFFF", ec="#B45309", lw=1.0, fontsize=7.2)
            draw_box(ax, 0.09, 0.17, 0.82, 0.125, "Single-Scalar Score Fusion ($\\alpha_w, \\tau$)\n$s_{\\mathrm{fused}} = w\\,s_R + (1-w)\\,s_V,\\; w=\\sigma(\\alpha_w)$",
                     fc="#DCE8F5", ec="#1D3557", lw=1.4, fontsize=7.4, fontweight="bold")
            ax.text(0.50, 0.095, "Zero projection MLP → preserves pretrained ViT geometry",
                    ha="center", va="center", fontsize=7.1, fontweight="bold", color="#1D3557")
            draw_arrow(ax, 0.26, 0.77, 0.26, 0.69)
            draw_arrow(ax, 0.74, 0.77, 0.74, 0.69)
            draw_arrow(ax, 0.26, 0.56, 0.26, 0.48)
            draw_arrow(ax, 0.74, 0.56, 0.74, 0.48)
            draw_arrow(ax, 0.26, 0.37, 0.38, 0.295)
            draw_arrow(ax, 0.74, 0.37, 0.62, 0.295)

    fig.tight_layout(w_pad=0.8)
    save_rgb_300(fig, [SPIE_DIR / "fig_fusion_modules.png", OUT_DIR / "fig_fusion_modules.png", GIT_RES / "fig_fusion_modules.png"])
    print("[2/9] Saved fig_fusion_modules.png")


# =====================================================================
# 3. FIG. 3 — Dataset Sample Grid + Four Strictly Disjoint Protocols
# =====================================================================
def make_fig3_dataset_protocols():
    fig = plt.figure(figsize=(13.2, 4.9), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.14)

    # Panel (a): Real 3x6 ISL sample grid (unseen test words across all 6 signers)
    sub_gs = gs[0].subgridspec(3, 6, wspace=0.06, hspace=0.08)
    words = ["afraid", "bad", "pain"]
    signers = [f"User_{i}" for i in range(1, 7)]
    for r, wname in enumerate(words):
        for c, sgn in enumerate(signers):
            ax_im = fig.add_subplot(sub_gs[r, c])
            img_file = sorted((DATA_ROOT / wname / sgn).glob("*.jpg"))[25]
            with Image.open(img_file) as im:
                ax_im.imshow(im.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR))
            ax_im.set_xticks([])
            ax_im.set_yticks([])
            for sp in ax_im.spines.values():
                sp.set_edgecolor("#475569")
                sp.set_linewidth(0.7)
            if r == 0:
                ax_im.set_title(sgn, fontsize=8.3, fontweight="bold", pad=3)
            if c == 0:
                ax_im.set_ylabel(wname, fontsize=8.8, fontweight="bold", rotation=90, labelpad=4)

    fig.text(0.29, 0.965, "(a) Representative Mendeley ISL Samples Across All Six Signers",
             ha="center", va="center", fontsize=10.2, fontweight="bold", color="#1A202C")

    # Panel (b): Visual 4-Protocol Disjoint Matrix
    ax_p = fig.add_subplot(gs[1])
    ax_p.set_xlim(0, 1)
    ax_p.set_ylim(0, 1)
    ax_p.axis("off")
    fig.text(0.755, 0.965, "(b) Strictly Disjoint Split & Support-Enrollment Regimes",
             ha="center", va="center", fontsize=10.2, fontweight="bold", color="#1A202C")

    # Helper to draw a 2x2 or 3x1 schematic matrix for each protocol
    protos = [
        ("1. Cross-Signer (User-Disjoint + 6-Fold LOSO)",
         0.76, [("Train\nU1–U4 × 20 cls\n(12,000)", 0.03, 0.56, "#CBD5E1", ""),
                ("Val\nU5\n(3,000)", 0.60, 0.17, "#FDE68A", "///"),
                ("Test\nU6\n(3,000)", 0.78, 0.19, "#1D3557", "")]),
        ("2. Unseen Class (Class-Disjoint, 14 / 3 / 3)",
         0.53, [("Train\nAll 6 Users × 14 cls\n(12,600)", 0.03, 0.56, "#CBD5E1", ""),
                ("Val\n3 cls\n(2,700)", 0.60, 0.17, "#FDE68A", "///"),
                ("Test\n3 cls\n(2,700)", 0.78, 0.19, "#1D3557", "")]),
        ("3. Matched 5-Way Unseen Class (12 / 3 / 5)",
         0.30, [("Train\nAll 6 Users × 12 cls\n(10,800)", 0.03, 0.50, "#CBD5E1", ""),
                ("Val\n3 cls\n(2,700)", 0.54, 0.18, "#FDE68A", "///"),
                ("Test\n5 cls\n(4,500)", 0.73, 0.24, "#1D3557", "")]),
    ]
    for title_p, y0, blocks in protos:
        ax_p.text(0.03, y0 + 0.165, title_p, ha="left", va="bottom", fontsize=8.3, fontweight="bold", color="#1E293B")
        for lbl, x0, bw, col, hatch in blocks:
            tc = "#FFFFFF" if col == "#1D3557" else "#111827"
            rect = FancyBboxPatch(
                (x0, y0), bw, 0.15,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                facecolor=col, edgecolor="#1E293B", linewidth=0.9, hatch=hatch, zorder=3,
            )
            ax_p.add_patch(rect)
            tbbox = dict(boxstyle="round,pad=0.14", fc=col, ec="none", alpha=0.88) if hatch else None
            ax_p.text(x0 + bw / 2, y0 + 0.075, lbl, ha="center", va="center",
                      fontsize=6.8, fontweight="bold", color=tc, bbox=tbbox, zorder=4)

    # Protocol 4: Simultaneous Doubly-Disjoint + Enrollment Distinction
    ax_p.text(0.03, 0.215, "4. Simultaneous Doubly-Disjoint (Zero Signer & Class Overlap)",
              ha="left", va="bottom", fontsize=8.3, fontweight="bold", color="#991B1B")
    blocks4 = [
        ("Train\nU1–U4 × 14 cls\n(8,400)", 0.03, 0.34, "#CBD5E1", ""),
        ("Excluded\nCross-Block\n(8,700)", 0.38, 0.19, "#F1F5F9", "xxx"),
        ("Val: U5\n3 unseen\n(450)", 0.58, 0.18, "#FDE68A", "///"),
        ("Test: U6\n3 unseen\n(450)", 0.77, 0.20, "#1D3557", ""),
    ]
    for lbl, x0, bw, col, hatch in blocks4:
        tc = "#FFFFFF" if col == "#1D3557" else ("#334155" if col == "#F1F5F9" else "#111827")
        rect = FancyBboxPatch(
            (x0, 0.06), bw, 0.145,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor=col, edgecolor="#1E293B", linewidth=0.9, hatch=hatch, zorder=3,
        )
        ax_p.add_patch(rect)
        tbbox = dict(boxstyle="round,pad=0.14", fc=col, ec="none", alpha=0.90) if hatch else None
        ax_p.text(x0 + bw / 2, 0.1325, lbl, ha="center", va="center",
                  fontsize=6.6, fontweight="bold", color=tc, bbox=tbbox, zorder=4)
    ax_p.text(0.50, 0.012,
              "Enrollment Conditions: (i) Same-Signer (U6→U6, U5→U5)  vs.  (ii) Cross-Signer (U5↔U6)",
              ha="center", va="center", fontsize=7.0, fontweight="bold", color="#1D3557")

    save_rgb_300(fig, [SPIE_DIR / "fig_dataset_protocols.png", OUT_DIR / "fig_dataset_protocols.png", GIT_RES / "fig_dataset_protocols.png"])
    print("[3/9] Saved fig_dataset_protocols.png")


# =====================================================================
# 4 & 5. Refresh 14.png & fig_fusion_ablation.png with Grayscale-Safe Hatching
# =====================================================================
def make_fig4_baseline_bar_hatched():
    models = [
        "Conv4\n(Supervised)",
        "ResNet-12\n(Supervised)",
        "ResNet-18\n(Supervised)",
        "ResNet-34\n(Supervised)",
        "ResNet-12 +\nProtoNet",
        "Dual-Backbone\nScore Fusion",
    ]
    accs = [22.0, 38.5, 47.0, 25.0, 99.60, 99.93]
    colors = ["#7AA2D4", "#7AA2D4", "#7AA2D4", "#7AA2D4", "#E59838", "#1D3557"]
    hatches = ["///", "///", "///", "///", "...", ""]

    fig, ax = plt.subplots(figsize=(8.4, 4.3), dpi=300)
    x = np.array([0, 1, 2, 3, 4.35, 5.45])
    bars = ax.bar(x, accs, width=0.70, color=colors, edgecolor="#1A202C", linewidth=0.9, zorder=3)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)

    for b, val in zip(bars, accs):
        label = f"{val:.2f}%" if val > 99.7 else f"{val:.1f}%"
        ax.text(b.get_x() + b.get_width() / 2, val + 1.6, label,
                ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#111111")

    ax.axvline(3.67, color="#666666", linestyle="--", linewidth=1.0, zorder=2)

    legend_handles = [
        patches.Patch(facecolor="#7AA2D4", edgecolor="#1A202C", hatch="///", label="Supervised Baselines (No Meta-Learning)"),
        patches.Patch(facecolor="#E59838", edgecolor="#1A202C", hatch="...", label="Single-Backbone ProtoNet (2-Way 10-Shot)"),
        patches.Patch(facecolor="#1D3557", edgecolor="#1A202C", label="Dual-Backbone Weighted Score Fusion (2-Way 10-Shot)"),
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.8)
    ax.set_ylabel("Test Accuracy on Unseen Signer User_6 (%)", fontsize=10.0)
    ax.set_ylim(0, 113)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(handles=legend_handles, fontsize=8.4, loc="upper left", frameon=True, framealpha=0.96)
    fig.tight_layout()

    save_rgb_300(fig, [SPIE_DIR / "14.png"])
    print("[4/9] Saved grayscale-safe hatched 14.png")


def make_fig7_fusion_ablation_hatched():
    labels = [
        "ResNet-12\n(Single)",
        "Dual\nConcat",
        "Dual\nGated",
        "ViT-B/16\n(Single)",
        "Dual Weighted\nScore (Best)",
    ]
    cs_2w, cs_2w_ci = [95.64, 98.24, 99.53, 99.98, 99.93], [1.42, 0.66, 0.30, 0.04, 0.08]
    cs_3w, cs_3w_ci = [92.83, 98.15, 99.19, 99.84, 99.90], [1.44, 0.57, 0.37, 0.14, 0.09]
    uc_2w, uc_2w_ci = [86.78, 87.36, 91.93, 95.42, 95.69], [2.51, 1.78, 1.17, 0.82, 0.85]
    uc_3w, uc_3w_ci = [84.01, 83.84, 91.30, 94.49, 96.53], [1.34, 1.11, 0.79, 0.64, 0.59]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=300)
    x = np.arange(len(labels))
    w = 0.36
    c_2w, c_3w, c_best_edge = "#8EB1D9", "#1D3557", "#B45309"

    panels = [
        (axes[0], "(a) Cross-Signer (User_6)", cs_2w, cs_2w_ci, cs_3w, cs_3w_ci, (88.0, 102.5)),
        (axes[1], "(b) Unseen Gesture Classes", uc_2w, uc_2w_ci, uc_3w, uc_3w_ci, (78.0, 102.0)),
    ]

    for idx, (ax, subtitle, y2, e2, y3, e3, ylim) in enumerate(panels):
        b1 = ax.bar(
            x - w / 2, y2, width=w, yerr=e2, capsize=3,
            color=c_2w, edgecolor="#243B55", linewidth=0.85, hatch="///",
            error_kw=dict(ecolor="#222222", lw=1.0, capthick=1.0),
            label="2-Way 10-Shot", zorder=3,
        )
        b2 = ax.bar(
            x + w / 2, y3, width=w, yerr=e3, capsize=3,
            color=c_3w, edgecolor="#102136", linewidth=0.85,
            error_kw=dict(ecolor="#222222", lw=1.0, capthick=1.0),
            label="3-Way 10-Shot", zorder=3,
        )
        ax.axvspan(3.52, 4.48, color="#EDF2F8", alpha=0.85, zorder=1)
        b1[-1].set_edgecolor(c_best_edge); b1[-1].set_linewidth(1.5)
        b2[-1].set_edgecolor(c_best_edge); b2[-1].set_linewidth(1.5)

        for bar, val, err in zip(b1, y2, e2):
            ax.text(bar.get_x() + bar.get_width() / 2, min(val + err + 0.30, ylim[1] - 0.8),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#2B4C7E")
        for bar, val, err in zip(b2, y3, e3):
            ax.text(bar.get_x() + bar.get_width() / 2, min(val + err + 0.30, ylim[1] - 0.8),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#102136")

        if idx == 1:
            ax.annotate(
                "+12.52 pp vs. ResNet-12\n+2.04 pp vs. ViT-B/16 (Holm $p < 10^{-9}$)",
                xy=(4 + w / 2, 96.53 + 0.59 + 0.85),
                xytext=(2.05, 99.6),
                ha="center", va="center", fontsize=7.8, fontweight="bold", color="#1D3557",
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec="#355070", lw=0.9),
                arrowprops=dict(arrowstyle="->", color="#355070", lw=1.2, connectionstyle="arc3,rad=-0.12"),
                zorder=5,
            )

        ax.set_title(subtitle, fontsize=10.5, fontweight="bold", color="#1A202C", pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.3)
        ax.set_ylim(*ylim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.legend(fontsize=8.4, loc="lower right", frameon=True, framealpha=0.96)

    axes[0].set_ylabel("Episodic Test Accuracy (% ± 95% CI)", fontsize=9.8)
    fig.tight_layout()
    save_rgb_300(fig, [SPIE_DIR / "fig_fusion_ablation.png", OUT_DIR / "fig_fusion_ablation.png", GIT_RES / "fig_fusion_ablation.png"])
    print("[5/9] Saved grayscale-safe hatched fig_fusion_ablation.png")


# =====================================================================
# 6. FIG. 8 — 6-Fold MD5-Sanitized LOSO & Simultaneous Doubly-Disjoint
# =====================================================================
def make_fig8_loso_and_simultaneous():
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5), dpi=300)

    # Panel (a): 6-Fold MD5-Sanitized LOSO (exact values from Table tab:loso_6fold)
    ax0 = axes[0]
    folds = ["U1", "U2", "U3", "U4", "U5*", "U6*", "Macro"]
    res_loso = [92.09, 84.00, 95.60, 93.87, 96.13, 95.73, 92.90]
    vit_loso = [99.60, 99.69, 100.00, 100.00, 100.00, 99.87, 99.86]
    fus_loso = [99.07, 99.56, 100.00, 99.96, 100.00, 99.96, 99.76]

    x = np.arange(len(folds))
    w = 0.26
    ax0.bar(x - w, res_loso, width=w, color="#94A3B8", edgecolor="#1E293B", hatch="///", lw=0.8, label="ResNet-12 (Single)")
    ax0.bar(x,     vit_loso, width=w, color="#E59838", edgecolor="#1E293B", hatch="...", lw=0.8, label="ViT-B/16 (Single)")
    ax0.bar(x + w, fus_loso, width=w, color="#1D3557", edgecolor="#1E293B", lw=0.8, label="Weighted Score Fusion")
    ax0.axvline(5.5, color="#64748B", linestyle="--", lw=0.9)

    ax0.annotate(
        "+15.56 pp lift on User_2 (84.0% → 99.6%)",
        xy=(1 + w, 99.9), xytext=(1.7, 103.6),
        ha="center", va="center", fontsize=7.6, fontweight="bold", color="#1D3557",
        bbox=dict(boxstyle="round,pad=0.25", fc="#FFFFFF", ec="#1D3557", lw=0.85),
        arrowprops=dict(arrowstyle="->", color="#1D3557", lw=1.1, connectionstyle="arc3,rad=0.10"),
    )
    ax0.set_title("(a) Duplicate-Sanitized 6-Fold LOSO (3-Way 10-Shot)", fontsize=10.2, fontweight="bold", pad=6)
    ax0.set_xticks(x)
    ax0.set_xticklabels(folds, fontsize=8.5)
    ax0.set_ylabel("Test Accuracy (%)", fontsize=9.8)
    ax0.set_ylim(80, 106.0)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.legend(fontsize=7.9, loc="lower right", frameon=True, framealpha=0.96)

    # Panel (b): Simultaneous Doubly-Disjoint (Same-Signer vs Cross-Signer Enrollment)
    ax1 = axes[1]
    regimes = ["Same-Signer\nTest (U6)", "Cross-Signer\nTest (U5→U6)", "Cross-Signer\nTest (U6→U5)", "Cross-Signer\nVal (U5→U6)"]
    r12 = [99.27, 85.02, 81.23, 10.13]
    cnc = [98.99, 93.13, 87.76, 67.42]
    gat = [98.61, 91.17, 85.16, 83.39]
    vit = [100.00, 98.93, 91.27, 78.46]
    wsc = [100.00, 98.80, 93.19, 69.63]

    x2 = np.arange(len(regimes))
    w2 = 0.16
    ax1.bar(x2 - 2 * w2, r12, width=w2, color="#94A3B8", edgecolor="#1E293B", hatch="///", lw=0.75, label="ResNet-12")
    ax1.bar(x2 - 1 * w2, cnc, width=w2, color="#CBD5E1", edgecolor="#1E293B", hatch="xxx", lw=0.75, label="Concat")
    ax1.bar(x2 + 0 * w2, gat, width=w2, color="#93C5FD", edgecolor="#1E293B", hatch="\\\\\\", lw=0.75, label="Gated")
    ax1.bar(x2 + 1 * w2, vit, width=w2, color="#E59838", edgecolor="#1E293B", hatch="...", lw=0.75, label="ViT-B/16")
    ax1.bar(x2 + 2 * w2, wsc, width=w2, color="#1D3557", edgecolor="#1E293B", lw=0.85, label="Weighted Score")

    ax1.annotate(
        "93.19% (+1.91 pp\nvs. ViT, p<10⁻¹⁰)",
        xy=(2 + 2 * w2, 93.8), xytext=(2.05, 110.5),
        ha="center", va="center", fontsize=7.3, fontweight="bold", color="#1D3557",
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec="#1D3557", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#1D3557", lw=1.0),
    )
    ax1.annotate(
        "ResNet-12\ncollapse\n(10.1%)",
        xy=(3 - 2 * w2, 11.5), xytext=(3 - 2 * w2, 42.0),
        ha="center", va="center", fontsize=7.0, fontweight="bold", color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.22", fc="#FEF2F2", ec="#991B1B", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#991B1B", lw=1.0),
    )

    ax1.set_title("(b) Simultaneous Doubly-Disjoint Enrollment (3-Way 10-Shot)", fontsize=10.2, fontweight="bold", pad=6)
    ax1.set_xticks(x2)
    ax1.set_xticklabels(regimes, fontsize=8.1)
    ax1.set_ylim(0, 120)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(fontsize=7.3, ncol=2, loc="lower left", frameon=True, framealpha=0.96)

    fig.tight_layout()
    save_rgb_300(fig, [SPIE_DIR / "fig_loso_and_simultaneous.png", OUT_DIR / "fig_loso_and_simultaneous.png", GIT_RES / "fig_loso_and_simultaneous.png"])
    print("[6/9] Saved fig_loso_and_simultaneous.png")


# =====================================================================
# 7. FIG. 9 — Weight Sensitivity Curve (w in [0,1]) + Accuracy-FPS Pareto
# =====================================================================
def make_fig9_weight_sensitivity_pareto():
    df_w = pd.read_csv(OUT_DIR / "jei_weight_sensitivity_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4), dpi=300)

    # Panel (a): Fusion weight sensitivity sweep
    ax0 = axes[0]
    for proto, label, col, mk, ls in [
        ("cross_signer", "Cross-Signer (3-Way 10-Shot)", "#D97706", "s", "--"),
        ("unseen_class", "Unseen Class (3-Way 10-Shot)", "#1D3557", "o", "-"),
    ]:
        sub = df_w[(df_w["Protocol"] == proto) & (df_w["Is_Learned"] == 0)].sort_values("Weight_w")
        lrn = df_w[(df_w["Protocol"] == proto) & (df_w["Is_Learned"] == 1)].iloc[0]
        w_vals = sub["Weight_w"].to_numpy()
        m_vals = sub["Mean_Acc"].to_numpy()
        ci_vals = sub["CI95"].to_numpy()
        ax0.plot(w_vals, m_vals, marker=mk, linestyle=ls, color=col, linewidth=2.1, markersize=5.8, label=label, zorder=3)
        ax0.fill_between(w_vals, m_vals - ci_vals, m_vals + ci_vals, color=col, alpha=0.14, zorder=2)
        ax0.scatter([lrn["Weight_w"]], [lrn["Mean_Acc"]], s=95, marker="*", color="#DC2626", edgecolor="#111111", lw=0.7, zorder=6)

    w_uc = float(df_w[(df_w["Protocol"] == "unseen_class") & (df_w["Is_Learned"] == 1)]["Weight_w"].iloc[0])
    acc_uc = float(df_w[(df_w["Protocol"] == "unseen_class") & (df_w["Is_Learned"] == 1)]["Mean_Acc"].iloc[0])
    ax0.axvline(w_uc, color="#DC2626", linestyle=":", linewidth=1.3, zorder=4)
    ax0.annotate(
        f"Meta-Learned $w^*={w_uc:.4f}$\n(Unseen: {acc_uc:.2f}%, Peak)",
        xy=(w_uc, acc_uc), xytext=(0.56, 86.2),
        ha="center", va="center", fontsize=8.0, fontweight="bold", color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.28", fc="#FFF5F5", ec="#DC2626", lw=0.9),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2, connectionstyle="arc3,rad=-0.12"),
    )
    ax0.text(0.01, 89.2, "ViT-B/16\nOnly", ha="left", va="center", fontsize=7.6, color="#475569")
    ax0.text(0.99, 84.8, "ResNet-12\nOnly", ha="right", va="center", fontsize=7.6, color="#475569")
    ax0.set_title("(a) Fusion Weight Sensitivity Sweep ($w \\in [0, 1]$)", fontsize=10.3, fontweight="bold", pad=6)
    ax0.set_xlabel("ResNet-12 Score Weight $w$ (ViT-B/16 Weight $= 1 - w$)", fontsize=9.4)
    ax0.set_ylabel("Test Accuracy (% ± 95% CI)", fontsize=9.6)
    ax0.set_xlim(-0.03, 1.03)
    ax0.set_ylim(73.0, 102.0)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.legend(fontsize=8.2, loc="lower left", frameon=True, framealpha=0.96)

    # Panel (b): Accuracy vs. Efficiency Pareto Scatter (Log FPS)
    ax1 = axes[1]
    models_eff = [
        ("Conv4",         84.20, 1375.5, 509.2, 0.11,  "#64748B", "D", (0, -14),   (0, -14)),
        ("ResNet-18",     80.30,  357.8,  75.8, 11.18, "#0284C7", "^", (0, -14),   (0, -14)),
        ("ResNet-34",     82.20,  182.8,  44.1, 21.28, "#7C3AED", "v", (0, -14),   (-14, -14)),
        ("ResNet-12",     84.01,  463.4,  41.1, 12.42, "#16A34A", "s", (0, 11),    (-8, 11)),
        ("ViT-B/16",      94.49,   99.3,   8.4, 85.80, "#D97706", "o", (8, -14),   (8, -14)),
        ("Dual Score",    96.53,   77.4,   6.9, 98.22, "#1D3557", "*", (4, 10),    (8, 10)),
    ]

    for name, acc, gfps, cfps, params, col, mk, off_g, off_c in models_eff:
        sz = 65 + 2.4 * params
        if mk == "*":
            sz = 230
        ax1.scatter([gfps], [acc], s=sz, marker=mk, color=col, edgecolor="#111111", lw=0.9, zorder=5)
        ax1.scatter([cfps], [acc], s=sz * 0.72, marker=mk, facecolor="#FFFFFF", edgecolor=col, lw=1.6, zorder=4)
        ax1.plot([cfps, gfps], [acc, acc], linestyle=":", color=col, alpha=0.55, lw=1.0, zorder=2)
        ax1.annotate(name, (gfps, acc), xytext=off_g, textcoords="offset points",
                     ha="center", va="center", fontsize=7.6, fontweight="bold" if name == "Dual Score" else "normal", color=col)

    ax1.axvline(30.0, color="#DC2626", linestyle="--", lw=1.2, zorder=3)
    ax1.text(26.5, 79.2, "30 FPS\nReal-Time", ha="right", va="bottom", fontsize=7.6, fontweight="bold", color="#DC2626")

    # Legend cleanly placed in the wide open upper-right whitespace (no points obstructed!)
    h_gpu = ax1.scatter([], [], s=65, marker="o", color="#1D3557", edgecolor="#111111", label="RTX 5050 Laptop GPU (Filled)")
    h_cpu = ax1.scatter([], [], s=65, marker="o", facecolor="#FFFFFF", edgecolor="#1D3557", lw=1.5, label="Ryzen AI 7 350 CPU (Hollow)")
    ax1.legend(handles=[h_gpu, h_cpu], fontsize=7.8, loc="upper right", frameon=True, framealpha=0.96)

    ax1.set_xscale("log")
    ax1.set_title("(b) Accuracy–Throughput Pareto Frontier (Batch Size 1, FP32)", fontsize=10.3, fontweight="bold", pad=6)
    ax1.set_xlabel("Single-Image Forward Throughput (FPS, Log Scale)", fontsize=9.4)
    ax1.set_ylabel("Unseen-Class 3-Way 10-Shot Accuracy (%)", fontsize=9.4)
    ax1.set_xlim(4.0, 2600.0)
    ax1.set_ylim(78.5, 99.0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    fig.tight_layout()
    save_rgb_300(fig, [SPIE_DIR / "fig_weight_sensitivity_pareto.png", OUT_DIR / "fig_weight_sensitivity_pareto.png", GIT_RES / "fig_weight_sensitivity_pareto.png"])
    print("[7/9] Saved fig_weight_sensitivity_pareto.png")


# =====================================================================
# 8. FIG. 10 — Real Fused t-SNE + Confusion Matrix Heatmap + Signer Errors
# =====================================================================
def make_fig10_tsne_confusion_error():
    tsne_npz = np.load(OUT_DIR / "jei_unseen_tsne_data.npz")
    emb = tsne_npz["embeddings"]
    classes = tsne_npz["classes"]
    conf = json.loads((OUT_DIR / "reconciled_table13_confusion.json").read_text(encoding="utf-8"))
    f1_d = json.loads((OUT_DIR / "jei_per_class_f1.json").read_text(encoding="utf-8"))

    coords = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42).fit_transform(emb)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), dpi=300, gridspec_kw={"width_ratios": [1.15, 1.0, 0.95]})

    # Panel (a): Fused hyperspherical t-SNE (distinct markers + colorblind-safe colors)
    ax0 = axes[0]
    styles = [
        ("afraid", "#1D3557", "o"),
        ("pain",   "#D97706", "^"),
        ("bad",    "#16A34A", "s"),
    ]
    for cname, col, mk in styles:
        m = classes == cname
        ax0.scatter(coords[m, 0], coords[m, 1], c=col, marker=mk, s=34, alpha=0.84,
                    edgecolors="#111111", linewidths=0.45, label=cname)
    ax0.set_title("(a) Fused Hyperspherical t-SNE", fontsize=10.1, fontweight="bold", pad=6)
    ax0.set_xlabel("t-SNE Dimension 1", fontsize=9.0)
    ax0.set_ylabel("t-SNE Dimension 2", fontsize=9.0)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.legend(title="Unseen Class", title_fontsize=8.2, fontsize=8.0, loc="best", frameon=True, framealpha=0.95)

    # Panel (b): Normalized Confusion Matrix Heatmap (exact 6,750 queries)
    ax1 = axes[1]
    cm = np.array(conf["matrix"], dtype=float)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100.0
    cls_names = conf["classes"]
    im = ax1.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(3):
        for j in range(3):
            val_c = int(cm[i, j])
            val_p = cm_pct[i, j]
            tc = "#FFFFFF" if val_p > 55 else "#111827"
            ax1.text(j, i, f"{val_c:,}\n({val_p:.1f}%)", ha="center", va="center",
                     fontsize=8.3, fontweight="bold" if i == j else "normal", color=tc)
    ax1.set_xticks(range(3)); ax1.set_xticklabels(cls_names, fontsize=8.8)
    ax1.set_yticks(range(3)); ax1.set_yticklabels(cls_names, fontsize=8.8)
    ax1.set_xlabel("Predicted Class", fontsize=9.0)
    ax1.set_ylabel("True Unseen Class", fontsize=9.0)
    ax1.set_title(f"(b) Confusion Matrix (Macro-F1: {f1_d['macro_f1_pct']:.2f}%)", fontsize=10.1, fontweight="bold", pad=6)

    # Panel (c): Signer-Stratified Misclassification Count Breakdown
    ax2 = axes[2]
    signers = [f"User_{i}" for i in range(1, 7)]
    errs = []
    accs = []
    for s in signers:
        m_s = np.array(conf["per_signer_breakdown"][s]["matrix"], dtype=int)
        tot = int(m_s.sum())
        err = tot - int(np.trace(m_s))
        errs.append(err)
        accs.append(conf["per_signer_breakdown"][s]["overall_acc_pct"])

    y_pos = np.arange(len(signers))
    bar_cols = ["#94A3B8" if e < 20 else "#B91C1C" for e in errs]
    hatches = ["///" if e < 20 else "" for e in errs]
    bars = ax2.barh(y_pos, errs, color=bar_cols, edgecolor="#1E293B", lw=0.85, height=0.62)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    for yp, e, ac in zip(y_pos, errs, accs):
        ax2.text(e + 2.2, yp, f"{e} ({ac:.1f}%)", ha="left", va="center", fontsize=7.8, fontweight="bold", color="#1E293B")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(signers, fontsize=8.6)
    ax2.invert_yaxis()
    ax2.set_xlabel("Misclassified Queries (Total = 234 / 6,750)", fontsize=8.8)
    ax2.set_xlim(0, max(errs) * 1.42)
    ax2.set_title("(c) Signer-Localized Error Distribution", fontsize=10.1, fontweight="bold", pad=6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    save_rgb_300(fig, [SPIE_DIR / "fig_tsne_confusion_error.png", OUT_DIR / "fig_tsne_confusion_error.png", GIT_RES / "fig_tsne_confusion_error.png"])
    print("[8/9] Saved fig_tsne_confusion_error.png")


# =====================================================================
# 9. FIG. 11 — Qualitative Interpretability & Explicit Failure-Case Grid
# =====================================================================
def overlay_heatmap(rgb_224: np.ndarray, hmap: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    hm_img = Image.fromarray((np.clip(hmap, 0, 1) * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BICUBIC)
    hm_norm = np.asarray(hm_img, dtype=float) / 255.0
    cmap = plt.get_cmap("inferno")
    hm_rgb = (cmap(hm_norm)[..., :3] * 255.0)
    blended = (1.0 - alpha) * rgb_224.astype(float) + alpha * hm_rgb
    return np.clip(blended, 0, 255).astype(np.uint8)


def make_fig11_qualitative_failure_cases():
    data = np.load(OUT_DIR / "jei_qualitative_maps.npz")
    meta = json.loads((OUT_DIR / "jei_qualitative_meta.json").read_text(encoding="utf-8"))
    rgbs = data["rgb"]
    cams = data["gradcam"]
    atts = data["vit_attn"]

    fig, axes = plt.subplots(3, 6, figsize=(13.2, 6.6), dpi=300)
    row_labels = ["Input RGB\nFrame", "ResNet-12\nGrad-CAM", "ViT-B/16\nAttn. Rollout"]

    for col in range(6):
        m = meta[col]
        is_ok = (m["kind"] == "Success")
        edge_c = "#166534" if is_ok else "#B91C1C"
        imgs = [
            rgbs[col],
            overlay_heatmap(rgbs[col], cams[col], alpha=0.46),
            overlay_heatmap(rgbs[col], atts[col], alpha=0.48),
        ]
        for row in range(3):
            ax = axes[row, col]
            ax.imshow(imgs[row])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(edge_c)
                sp.set_linewidth(1.8 if row == 0 else 1.0)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=8.8, fontweight="bold", color="#1A202C", labelpad=6)
            if row == 0:
                if is_ok:
                    hdr = f"{m['signer']}\nTrue/Pred: {m['true_class']}"
                else:
                    hdr = f"{m['signer']}\nTrue: {m['true_class']} → Pred: {m['pred_class']}"
                ax.set_title(hdr, fontsize=8.1, fontweight="bold", color=edge_c, pad=4)

    # Clean group labels above left 3 and right 3 columns + vertical divider
    fig.text(0.275, 0.985, "(a) Correct Unseen-Class Predictions (Complementary Local + Global Focus)",
             ha="center", va="center", fontsize=10.0, fontweight="bold", color="#166534")
    fig.text(0.745, 0.985, "(b) Misclassified Failure Cases (Overlapping Static Chest-Proximity Postures)",
             ha="center", va="center", fontsize=10.0, fontweight="bold", color="#B91C1C")

    fig.subplots_adjust(left=0.07, right=0.99, top=0.89, bottom=0.02, wspace=0.08, hspace=0.08)
    # Add subtle dashed separator line between col 2 and col 3
    line = plt.Line2D([0.525, 0.525], [0.02, 0.94], transform=fig.transFigure, color="#475569", linestyle="--", linewidth=1.3)
    fig.add_artist(line)

    save_rgb_300(fig, [SPIE_DIR / "fig_qualitative_failure_cases.png", OUT_DIR / "fig_qualitative_failure_cases.png", GIT_RES / "fig_qualitative_failure_cases.png"])
    print("[9/9] Saved fig_qualitative_failure_cases.png")


def main():
    make_fig1_architecture()
    make_fig2_fusion_modules()
    make_fig3_dataset_protocols()
    make_fig4_baseline_bar_hatched()
    make_fig7_fusion_ablation_hatched()
    make_fig8_loso_and_simultaneous()
    make_fig9_weight_sensitivity_pareto()
    make_fig10_tsne_confusion_error()
    make_fig11_qualitative_failure_cases()
    print("\nALL 9 JEI VOL. 35 FIGURES GENERATED AT PURE RGB 300 DPI!")


if __name__ == "__main__":
    main()
