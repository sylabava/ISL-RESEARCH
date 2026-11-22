r"""
Regenerate and polish 7 manuscript figures for IJISA PAPER021185 (First Revision)
at crisp 300 DPI, fixing all 3 critical visual/numeric mismatches and 4 minor
terminology/formatting issues identified during the figure-by-figure audit:

  CRITICAL FIXES:
    1. 1.ARCHITECTURE.png       : Unified Dual-Backbone Hyperspherical Fusion architecture
    2. fig_maml_loss_curves.png : Real RTX 5050 3-way MAML & ProtoMAML convergence curves
                                  (all 4 backbones, 2-panel layout, faint raw + smooth EMA)
    3. 11.png                   : Fix "15.77% Spike!" -> exact "+15.67% Gain" & clean labels

  MINOR POLISH:
    4. 7.jpg                    : Fix "Vit-B/16" -> "ViT-B/16", include Dual-Backbone Fusion
    5. 14.png                   : Standardized ResNet-12/18/34 labels + Dual-Backbone 99.93% bar
    6. 15.png                   : Standardized ResNet-12/18/34 legend + Cross-Signer title
    7. 16.png                   : Standardized ResNet-12/18/34 legend + Unseen-Class title
"""

import gc
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TEX_DIR = SCRIPT_DIR / "results"
OUT_DIR = SCRIPT_DIR / "results"
GIT_RES = SCRIPT_DIR / "results"
DATA_ROOT = SCRIPT_DIR.parent / "data"
TEX_DIR.mkdir(parents=True, exist_ok=True)

# Allow importing exact verified MAML training code from the main suite
sys.path.insert(0, str(SCRIPT_DIR))
import run_all_ijisa_revision_experiments as exp_mod


# =====================================================================
# Helper: Clean Rounded Box & Arrow
# =====================================================================
def draw_box(ax, x, y, w, h, text, fc="#EEF4FB", ec="#2B4C7E", lw=1.4,
             fontsize=9, fontweight="normal", text_color="#111111", radius=0.012):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight=fontweight, color=text_color, zorder=4,
        linespacing=1.25,
    )
    return box


def draw_arrow(ax, x1, y1, x2, y2, color="#222222", lw=1.5, rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_length=6.5,head_width=3.8",
        connectionstyle=f"arc3,rad={rad}",
        color=color, linewidth=lw, zorder=5,
    )
    ax.add_patch(arr)


# =====================================================================
# 1. FIG. 1 — 1.ARCHITECTURE.png (Unified Dual-Backbone Fusion)
# =====================================================================
def make_fig1_architecture():
    fig, ax = plt.subplots(figsize=(14.0, 6.2), dpi=300)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # Outer dashed border matching Figs. 2 & 4
    outer = FancyBboxPatch(
        (0.01, 0.02), 0.98, 0.90,
        boxstyle="round,pad=0.005,rounding_size=0.018",
        facecolor="#FFFFFF", edgecolor="#333333", linewidth=1.3,
        linestyle="--", zorder=1,
    )
    ax.add_patch(outer)
    ax.text(
        0.50, 0.955,
        "Dual-Backbone Meta-Learning Architecture with Unified Hyperspherical Fusion",
        ha="center", va="center", fontsize=13.5, fontweight="bold", color="#111111",
    )

    # --- Left: Input Episodic Task with Real ISL Thumbnails ---
    in_panel = FancyBboxPatch(
        (0.025, 0.14), 0.155, 0.72,
        boxstyle="round,pad=0.004,rounding_size=0.015",
        facecolor="#F7F8FA", edgecolor="#555555", linewidth=1.2, zorder=2,
    )
    ax.add_patch(in_panel)
    ax.text(0.1025, 0.825, "Episodic ISL Input", ha="center", va="center",
            fontsize=9.8, fontweight="bold", color="#1A2B4C", zorder=4)
    ax.text(0.1025, 0.775, r"Support $\mathcal{S}$ & Query $\mathcal{Q}$" + "\n(N-way K-shot)",
            ha="center", va="center", fontsize=8.0, color="#333333", zorder=4)

    # Load 4 real ISL gesture thumbnails
    sample_classes = ["pain", "bad", "afraid", "doctor"]
    thumb_positions = [(0.068, 0.61), (0.137, 0.61), (0.068, 0.43), (0.137, 0.43)]
    for cls_name, (tx, ty) in zip(sample_classes, thumb_positions):
        sdir = DATA_ROOT / cls_name / "User_1"
        imgs = sorted(sdir.glob("*.jpg")) if sdir.exists() else []
        if imgs:
            im = Image.open(imgs[min(10, len(imgs) - 1)]).convert("RGB").resize((90, 90))
            oi = OffsetImage(np.asarray(im), zoom=0.45)
            ab = AnnotationBbox(oi, (tx, ty), frameon=True,
                                bboxprops=dict(edgecolor="#666666", lw=0.8), zorder=4)
            ax.add_artist(ab)
            ax.text(tx, ty - 0.072, cls_name, ha="center", va="top",
                    fontsize=7.2, color="#333333", style="italic", zorder=5)

    draw_box(ax, 0.035, 0.165, 0.135, 0.13,
             "Dual-Resolution\nPreprocessing\n" + r"$x_{84}$ & $x_{224}$",
             fc="#E8EEF7", ec="#2B4C7E", lw=1.1, fontsize=7.8, fontweight="bold")

    # --- Top Branch: ResNet-12 (Local Spatial Branch) ---
    top_bg = FancyBboxPatch(
        (0.205, 0.535), 0.465, 0.325,
        boxstyle="round,pad=0.004,rounding_size=0.015",
        facecolor="#FFF8EE", edgecolor="#C97A16", linewidth=1.2, linestyle="-", zorder=2,
    )
    ax.add_patch(top_bg)
    ax.text(0.4375, 0.835, "Branch 1: Local Spatial CNN Path (ResNet-12)",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color="#9C5700", zorder=4)

    draw_box(ax, 0.218, 0.575, 0.145, 0.215,
             "ResNet-12 Backbone\n" + r"Input: $84\times 84\times 3$" + "\n4 ResBlocks (64→640)\n+ Adaptive Avg Pool",
             fc="#FFE4BD", ec="#B86B0A", lw=1.3, fontsize=8.0, fontweight="bold")
    draw_box(ax, 0.388, 0.575, 0.115, 0.215,
             "L2 Normalization\n" + r"$\hat{h}^{(r)} \in \mathbb{S}^{639}$" + "\n640-D Unit\nEmbedding",
             fc="#FDEBD0", ec="#B86B0A", lw=1.2, fontsize=7.8)
    draw_box(ax, 0.528, 0.575, 0.130, 0.215,
             "CNN Prototypes\n" + r"$c_k^{(r)} = \mathrm{Norm}(\bar{h}_k^{(r)})$" + "\nScaled Logits:\n" + r"$s_k^{(r)} = \tau\,\hat{h}_q^{(r)\top} c_k^{(r)}$",
             fc="#FFFFFF", ec="#B86B0A", lw=1.2, fontsize=7.6)

    draw_arrow(ax, 0.363, 0.682, 0.388, 0.682, color="#9C5700")
    draw_arrow(ax, 0.503, 0.682, 0.528, 0.682, color="#9C5700")

    # --- Bottom Branch: ViT-B/16 (Global Contextual Branch) ---
    bot_bg = FancyBboxPatch(
        (0.205, 0.14), 0.465, 0.325,
        boxstyle="round,pad=0.004,rounding_size=0.015",
        facecolor="#EEF6FC", edgecolor="#1E6091", linewidth=1.2, linestyle="-", zorder=2,
    )
    ax.add_patch(bot_bg)
    ax.text(0.4375, 0.440, "Branch 2: Global Contextual Transformer Path (ViT-B/16)",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color="#164B73", zorder=4)

    draw_box(ax, 0.218, 0.180, 0.145, 0.215,
             "ViT-B/16 Backbone\n" + r"Input: $224\times 224\times 3$" + "\n196 Patches (16×16)\n12× MSA + MLP Layers",
             fc="#CCE4F7", ec="#1E6091", lw=1.3, fontsize=8.0, fontweight="bold")
    draw_box(ax, 0.388, 0.180, 0.115, 0.215,
             "L2 Normalization\n" + r"$\hat{h}^{(v)} \in \mathbb{S}^{767}$" + "\n768-D Unit\nEmbedding",
             fc="#DFEFFB", ec="#1E6091", lw=1.2, fontsize=7.8)
    draw_box(ax, 0.528, 0.180, 0.130, 0.215,
             "ViT Prototypes\n" + r"$c_k^{(v)} = \mathrm{Norm}(\bar{h}_k^{(v)})$" + "\nScaled Logits:\n" + r"$s_k^{(v)} = \tau\,\hat{h}_q^{(v)\top} c_k^{(v)}$",
             fc="#FFFFFF", ec="#1E6091", lw=1.2, fontsize=7.6)

    draw_arrow(ax, 0.363, 0.287, 0.388, 0.287, color="#164B73")
    draw_arrow(ax, 0.503, 0.287, 0.528, 0.287, color="#164B73")

    # Input -> Branches arrows
    draw_arrow(ax, 0.180, 0.682, 0.218, 0.682, color="#333333", lw=1.7)
    draw_arrow(ax, 0.180, 0.287, 0.218, 0.287, color="#333333", lw=1.7)

    # --- Right: Unified Hyperspherical Metric Space & Fusion ---
    fus_bg = FancyBboxPatch(
        (0.695, 0.14), 0.280, 0.72,
        boxstyle="round,pad=0.004,rounding_size=0.015",
        facecolor="#F0F8F2", edgecolor="#1E7A46", linewidth=1.5, zorder=2,
    )
    ax.add_patch(fus_bg)
    ax.text(0.835, 0.828, "Unified Hyperspherical\nDual-Backbone Fusion",
            ha="center", va="center", fontsize=9.8, fontweight="bold", color="#125430", zorder=4)

    draw_box(ax, 0.712, 0.655, 0.246, 0.125,
             "Unified Metric Equivalence\n" + r"$\|\hat{z}_q - c_k\|_2^2 = 2 - 2\cos(\hat{z}_q, c_k)$",
             fc="#FFFFFF", ec="#1E7A46", lw=1.1, fontsize=8.0)

    draw_box(ax, 0.712, 0.445, 0.246, 0.175,
             "Learnable Weighted Score Fusion\n" +
             r"$s_{\mathrm{fused}}(q,k) = w\,s_k^{(r)} + (1-w)\,s_k^{(v)}$" + "\n" +
             r"$w = \sigma(\alpha_w)$  (also ablated:" + "\nConcat & Gated Feature Fusion)",
             fc="#D6F0DF", ec="#1E7A46", lw=1.4, fontsize=7.8, fontweight="bold")

    draw_box(ax, 0.712, 0.295, 0.246, 0.115,
             "Softmax Prediction\n" + r"$\hat{y} = \arg\max_k\; p_\Theta(y=k \mid x_q, \mathcal{S})$",
             fc="#FFFFFF", ec="#1E7A46", lw=1.1, fontsize=8.0)

    draw_box(ax, 0.712, 0.162, 0.246, 0.100,
             "Dual Generalization Output\nCross-Signer (99.93%) | Unseen Class (96.04%)",
             fc="#1E7A46", ec="#125430", lw=1.2, fontsize=7.5, fontweight="bold", text_color="#FFFFFF")

    # Arrows inside Fusion block
    draw_arrow(ax, 0.835, 0.655, 0.835, 0.620, color="#125430")
    draw_arrow(ax, 0.835, 0.445, 0.835, 0.410, color="#125430")
    draw_arrow(ax, 0.835, 0.295, 0.835, 0.262, color="#125430")

    # Arrows from Branch 1 & Branch 2 logits into Weighted Score Fusion
    draw_arrow(ax, 0.658, 0.682, 0.712, 0.560, color="#9C5700", lw=1.8, rad=-0.10)
    draw_arrow(ax, 0.658, 0.287, 0.712, 0.505, color="#164B73", lw=1.8, rad=0.10)

    # Bottom caption note inside border
    ax.text(0.50, 0.068,
            "Joint Episodic Training via Adam (Differential LRs: ResNet-12 & Fusion = 1e-3, ViT-B/16 = 1e-5; Temperature τ = 10)",
            ha="center", va="center", fontsize=8.4, color="#333333", style="italic")

    out_path = TEX_DIR / "1.ARCHITECTURE.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[1/7] Saved crisp 300-DPI {out_path}")


# =====================================================================
# 2. FIG. 5 — 7.jpg (Experimental Pipeline Flowchart)
# =====================================================================
def make_fig5_pipeline():
    fig, ax = plt.subplots(figsize=(12.0, 5.2), dpi=300)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    ax.text(0.50, 0.94, "Dual-Generalization ISL Experimental Pipeline",
            ha="center", va="center", fontsize=13.0, fontweight="bold", color="#111111")

    # Row 1 (Left -> Right)
    draw_box(ax, 0.02, 0.64, 0.285, 0.22,
             "ISL Real-Life Words Dataset\n18,000 RGB Images\n20 Classes | 6 Signers (0 Duplicates)",
             fc="#DCE8F5", ec="#244B7A", lw=1.4, fontsize=8.0, fontweight="bold")
    draw_box(ax, 0.3575, 0.64, 0.285, 0.22,
             "Strictly Disjoint Partitions\nCross-Signer (User 1–4 / 5 / 6)\nUnseen-Class (14/3/3 & 12/3/5)",
             fc="#DCE8F5", ec="#244B7A", lw=1.4, fontsize=8.0, fontweight="bold")
    draw_box(ax, 0.695, 0.64, 0.285, 0.22,
             "Episodic Task Construction\nN-Way K-Shot Sampling\nSupport Set S | Query Set Q (M=15)",
             fc="#DCE8F5", ec="#244B7A", lw=1.4, fontsize=8.0, fontweight="bold")

    # Row 2 (Right -> Left)
    draw_box(ax, 0.695, 0.34, 0.285, 0.22,
             "Feature Extraction Backbones\nConv4 / ResNet-12, 18, 34 (84×84)\nViT-B/16 (224×224) & Dual-Backbone",
             fc="#E6F2EA", ec="#1E6B43", lw=1.4, fontsize=7.9, fontweight="bold")
    draw_box(ax, 0.3575, 0.34, 0.285, 0.22,
             "Meta-Learning & Fusion\nProtoNet | FOMAML | ProtoMAML\nWeighted Score & Gated Fusion",
             fc="#E6F2EA", ec="#1E6B43", lw=1.4, fontsize=7.9, fontweight="bold")
    draw_box(ax, 0.02, 0.34, 0.285, 0.22,
             "Episodic Task Adaptation\nSupport-Only Prototypes (Metric)\nor 5-Step Inner SGD (Gradient/Hybrid)",
             fc="#E6F2EA", ec="#1E6B43", lw=1.4, fontsize=7.9, fontweight="bold")

    # Row 3 (Left -> Right)
    draw_box(ax, 0.02, 0.05, 0.44, 0.20,
             "Disjoint Query Evaluation (3 Seeds × 35–50 Episodes)\nMean Accuracy ± Std, 95% CI, Paired t-test & Wilcoxon",
             fc="#FFF3DF", ec="#B86B0A", lw=1.4, fontsize=8.2, fontweight="bold")
    draw_box(ax, 0.54, 0.05, 0.44, 0.20,
             "Matched SOTA & Efficiency Profiling\n2w-2s, 5w-5s, 5w-10s Benchmarks\nParams, GFLOPs, CPU & RTX 5050 GPU Latency / FPS",
             fc="#FFF3DF", ec="#B86B0A", lw=1.4, fontsize=8.2, fontweight="bold")

    # Arrows
    draw_arrow(ax, 0.305, 0.75, 0.3575, 0.75)
    draw_arrow(ax, 0.6425, 0.75, 0.695, 0.75)
    draw_arrow(ax, 0.8375, 0.64, 0.8375, 0.56)
    draw_arrow(ax, 0.695, 0.45, 0.6425, 0.45)
    draw_arrow(ax, 0.3575, 0.45, 0.305, 0.45)
    draw_arrow(ax, 0.1625, 0.34, 0.1625, 0.25)
    draw_arrow(ax, 0.46, 0.15, 0.54, 0.15)

    out_path = TEX_DIR / "7.jpg"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"[2/7] Saved crisp 300-DPI {out_path}")


# =====================================================================
# 3. FIG. 8 — fig_maml_loss_curves.png (Real GPU 2-Panel Convergence)
# =====================================================================
def make_fig8_maml_loss_curves():
    print("[3/7] Recording real RTX 5050 3-way MAML & ProtoMAML convergence curves (all 4 backbones)...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store84 = exp_mod.FastImageStore(target_size=84)
    (tr_s, tr_map), (va_s, va_map), _ = exp_mod.build_protocol_splits(
        DATA_ROOT, "unseen_class", seed=42, n_test_classes=3
    )
    tr_cm = exp_mod.build_classmap(tr_s, tr_map)
    va_cm = exp_mod.build_classmap(va_s, va_map)

    backbones = [
        ("conv4", "Conv4", "#1f77b4"),
        ("resnet12", "ResNet-12", "#2ca02c"),
        ("resnet18", "ResNet-18", "#ff7f0e"),
        ("resnet34", "ResNet-34", "#d62728"),
    ]

    # Record EVERY single iteration loss for true raw + EMA smoothed curves
    histories = {"MAML": {}, "ProtoMAML": {}}
    use_amp = device.type == "cuda"
    for method, proto_init in [("MAML", False), ("ProtoMAML", True)]:
        for bb_key, bb_label, _ in backbones:
            exp_mod.set_seed(42)
            rng = random.Random(42)
            encoder = exp_mod.get_cnn_backbone(bb_key, pretrained=True).to(device)
            fast_enc = exp_mod.get_cnn_backbone(bb_key, pretrained=False).to(device)
            meta_clf = torch.nn.Linear(encoder.output_dim, 3).to(device)
            fast_clf = torch.nn.Linear(encoder.output_dim, 3).to(device)
            meta_params = list(encoder.parameters()) + ([] if proto_init else list(meta_clf.parameters()))
            meta_opt = torch.optim.Adam(meta_params, lr=1e-3, weight_decay=1e-4)
            raw_losses = []
            for it in range(1, 151):
                s_idx, s_y, q_idx, q_y = exp_mod.sample_episode_indices(tr_cm, 3, 5, 15, rng)
                sx, sy, qx, qy = exp_mod.materialize_episode_fast(
                    tr_s, store84, None, s_idx, s_y, q_idx, q_y, device, dual=False, training=True
                )
                fast_enc.load_state_dict(encoder.state_dict())
                fast_enc.train()
                if proto_init:
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        zs0 = torch.nn.functional.normalize(fast_enc(sx).float(), p=2, dim=-1)
                        for c in range(3):
                            p = zs0[sy == c].mean(0)
                            fast_clf.weight.data[c] = 2.0 * p
                            fast_clf.bias.data[c] = -(p.norm() ** 2)
                else:
                    fast_clf.load_state_dict(meta_clf.state_dict())

                inner_opt = torch.optim.SGD([
                    {"params": fast_enc.parameters(), "lr": 0.01},
                    {"params": fast_clf.parameters(), "lr": 0.1},
                ])
                for _ in range(5):
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        zs = torch.nn.functional.normalize(fast_enc(sx).float(), p=2, dim=-1)
                        loss_s = torch.nn.functional.cross_entropy(10.0 * fast_clf(zs), sy)
                    inner_opt.zero_grad()
                    loss_s.backward()
                    torch.nn.utils.clip_grad_norm_(fast_enc.parameters(), 1.0)
                    inner_opt.step()

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    zq = torch.nn.functional.normalize(fast_enc(qx).float(), p=2, dim=-1)
                    q_loss = torch.nn.functional.cross_entropy(10.0 * fast_clf(zq), qy)

                meta_opt.zero_grad()
                fast_enc.zero_grad()
                fast_clf.zero_grad()
                q_loss.backward()
                with torch.no_grad():
                    for pm, pf in zip(encoder.parameters(), fast_enc.parameters()):
                        if pf.grad is not None:
                            pm.grad = pf.grad.detach().clone()
                    for bm, bf in zip(encoder.buffers(), fast_enc.buffers()):
                        bm.copy_(bf)
                    if not proto_init:
                        for pm, pf in zip(meta_clf.parameters(), fast_clf.parameters()):
                            if pf.grad is not None:
                                pm.grad = pf.grad.detach().clone()
                torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
                meta_opt.step()
                raw_losses.append(float(q_loss.item()))
            histories[method][bb_label] = np.array(raw_losses)
            print(f"    {method:9s} | {bb_label:9s} | start={raw_losses[0]:.2f} -> end_ema={np.mean(raw_losses[-15:]):.2f}", flush=True)
            del encoder, fast_enc, meta_clf, fast_clf, meta_opt
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def ema(vals, alpha=0.12):
        out = np.zeros_like(vals)
        out[0] = vals[0]
        for i in range(1, len(vals)):
            out[i] = alpha * vals[i] + (1.0 - alpha) * out[i - 1]
        return out

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=300, sharey=True)
    x = np.arange(1, 151)

    for ax, (method, title) in zip(axes, [
        ("MAML", "Corrected First-Order MAML (3-Way Unseen)"),
        ("ProtoMAML", "Corrected ProtoMAML (3-Way Unseen)"),
    ]):
        for _, bb_label, col in backbones:
            raw = histories[method][bb_label]
            smooth = ema(raw, alpha=0.10)
            ax.plot(x, raw, color=col, alpha=0.16, linewidth=0.9)
            ax.plot(x, smooth, color=col, linewidth=2.2, label=bb_label)
        ax.axhline(np.log(3.0), color="#555555", linestyle=":", linewidth=1.2,
                   label="Random Chance (ln 3 = 1.10)")
        ax.set_title(title, fontsize=11.0, fontweight="bold", pad=8)
        ax.set_xlabel("Meta-Training Iteration", fontsize=10.0)
        ax.set_xlim(1, 150)
        ax.set_ylim(0, 4.5)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(fontsize=8.5, loc="upper right", frameon=True)

    axes[0].set_ylabel("Query Cross-Entropy Loss (EMA + Raw)", fontsize=10.0)
    fig.suptitle("Meta-Training Convergence Curves Across All Four CNN Backbones (Unseen-Class Split)",
                 fontsize=12.0, fontweight="bold", y=1.01)
    fig.tight_layout()

    for dst in [TEX_DIR / "fig_maml_loss_curves.png", OUT_DIR / "fig_maml_loss_curves.png", GIT_RES / "fig_maml_loss_curves.png"]:
        fig.savefig(dst, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[3/7] Saved crisp 2-panel 300-DPI fig_maml_loss_curves.png")


# =====================================================================
# 4. FIG. 9 — 11.png (Fix "15.77% Spike!" -> exact "+15.67% Gain")
# =====================================================================
def make_fig9_vit_vs_resnet12():
    ks = [1, 5, 10, 20]
    res12 = [74.80, 74.10, 77.00, 79.30]
    vit16 = [77.11, 86.42, 92.67, 92.58]

    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=300)
    ax.plot(ks, res12, "o--", color="#1D4ED8", linewidth=2.0, markersize=7.5,
            label="ProtoNet + ResNet-12 (Baseline)")
    ax.plot(ks, vit16, "s-", color="#DC2626", linewidth=2.6, markersize=8.0,
            label="ProtoNet + ViT-B/16")

    # Clean non-overlapping value labels
    for k, v in zip(ks, vit16):
        dy = 1.6 if k != 1 else 2.0
        ax.text(k, v + dy, f"{v:.2f}%", ha="center", va="bottom",
                fontsize=9.2, fontweight="bold", color="#DC2626")
    for k, r in zip(ks, res12):
        ax.text(k, r - 2.3, f"{r:.2f}%", ha="center", va="top",
                fontsize=8.8, color="#1D4ED8")

    # Vertical double-headed gain bracket at K=10 (92.67 - 77.00 = +15.67%)
    ax.annotate(
        "", xy=(10, 91.6), xytext=(10, 78.0),
        arrowprops=dict(arrowstyle="<->", color="#111111", lw=1.6),
    )
    ax.text(10.55, 84.8, "+15.67% Gain\n(92.67% vs. 77.00%)",
            ha="left", va="center", fontsize=9.5, fontweight="bold", color="#111111",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF7ED", ec="#C2410C", lw=1.0))

    ax.set_title("ViT-B/16 vs. ResNet-12 in Unseen Gesture Class Recognition (3-Way)",
                 fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xlabel("Number of Support Examples (K-Shot)", fontsize=10.5)
    ax.set_ylabel("Test Accuracy (%)", fontsize=10.5)
    ax.set_xticks(ks)
    ax.set_xlim(0, 21.6)
    ax.set_ylim(66, 100)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9.5, loc="lower right", frameon=True)
    fig.tight_layout()

    out_path = TEX_DIR / "11.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[4/7] Saved exact-math 300-DPI {out_path}")


# =====================================================================
# 5. FIG. 12 — 14.png (Supervised Baselines vs. Proposed Meta-Learning)
# =====================================================================
def make_fig12_baseline_bar():
    models = [
        "Conv4\n(Supervised)",
        "ResNet-12\n(Supervised)",
        "ResNet-18\n(Supervised)",
        "ResNet-34\n(Supervised)",
        "ResNet-12 +\nProtoNet",
        "Dual-Backbone\nScore Fusion",
    ]
    accs = [22.0, 38.5, 47.0, 25.0, 99.60, 99.93]
    colors = ["#2563EB", "#2563EB", "#2563EB", "#2563EB", "#F59E0B", "#16A34A"]

    fig, ax = plt.subplots(figsize=(8.6, 4.9), dpi=300)
    x = np.array([0, 1, 2, 3, 4.35, 5.45])
    bars = ax.bar(x, accs, width=0.72, color=colors, edgecolor="#222222", linewidth=0.9, zorder=3)

    for b, val in zip(bars, accs):
        label = f"{val:.2f}%" if val > 99.7 else f"{val:.1f}%"
        ax.text(b.get_x() + b.get_width() / 2, val + 1.5, label,
                ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#111111")

    ax.axvline(3.67, color="#888888", linestyle="--", linewidth=1.1, zorder=2)

    legend_handles = [
        patches.Patch(facecolor="#2563EB", edgecolor="#222222", label="Supervised Baselines (No Meta-Learning)"),
        patches.Patch(facecolor="#F59E0B", edgecolor="#222222", label="Single-Backbone ProtoNet (2-Way 10-Shot)"),
        patches.Patch(facecolor="#16A34A", edgecolor="#222222", label="Proposed Dual-Backbone Score Fusion (2-Way 10-Shot)"),
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.8)
    ax.set_ylabel("Test Accuracy on Unseen Signer User_6 (%)", fontsize=10.0)
    ax.set_title("Supervised Baselines vs. Meta-Learning on Unseen Signer (User_6)",
                 fontsize=11.5, fontweight="bold", pad=10)
    ax.set_ylim(0, 113)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=1)
    ax.legend(handles=legend_handles, fontsize=8.3, loc="upper left", frameon=True)
    fig.tight_layout()

    out_path = TEX_DIR / "14.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[5/7] Saved standardized 300-DPI {out_path}")


# =====================================================================
# 6. FIG. 13 — 15.png (Cross-Signer ProtoNet K-Shot Across Backbones)
# =====================================================================
def make_fig13_cross_signer_kshot():
    ks = [1, 5, 10, 20]
    data = {
        "Conv4":     ([87.5, 97.4, 98.5, 99.3], [82.8, 94.5, 97.8, 98.0], "#3B82F6"),
        "ResNet-12": ([96.7, 98.9, 99.6, 99.2], [93.8, 97.4, 99.2, 98.7], "#16A34A"),
        "ResNet-18": ([92.1, 96.7, 96.9, 97.2], [87.6, 95.1, 95.2, 96.4], "#EA580C"),
        "ResNet-34": ([94.8, 98.1, 98.2, 98.0], [91.0, 96.2, 96.2, 97.6], "#7C3AED"),
    }
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=300)
    for name, (y2, y3, col) in data.items():
        ax.plot(ks, y2, "o-", color=col, linewidth=2.0, markersize=6.5, label=f"{name} (N=2)")
        ax.plot(ks, y3, "s--", color=col, linewidth=1.7, markersize=6.0, alpha=0.85, label=f"{name} (N=3)")

    ax.set_title("ProtoNet Cross-Signer Accuracy Across Backbones (Solid: N=2 | Dashed: N=3)",
                 fontsize=11.0, fontweight="bold", pad=10)
    ax.set_xlabel("Support Shots (K-Shot)", fontsize=10.5)
    ax.set_ylabel("Test Accuracy (%)", fontsize=10.5)
    ax.set_xticks(ks)
    ax.set_ylim(81.5, 101.0)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(fontsize=8.5, ncol=2, loc="lower right", frameon=True)
    fig.tight_layout()

    out_path = TEX_DIR / "15.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[6/7] Saved standardized 300-DPI {out_path}")


# =====================================================================
# 7. FIG. 14 — 16.png (Class-Disjoint ProtoNet K-Shot Across Backbones)
# =====================================================================
def make_fig14_unseen_class_kshot():
    ks = [1, 5, 10, 20]
    data = {
        "Conv4":     ([88.5, 88.6, 92.8, 90.7], [75.4, 86.2, 84.2, 83.0], "#3B82F6"),
        "ResNet-12": ([76.2, 88.2, 95.1, 89.7], [74.8, 74.1, 77.0, 79.3], "#16A34A"),
        "ResNet-18": ([71.7, 74.8, 82.7, 76.5], [60.1, 72.3, 80.3, 91.2], "#EA580C"),
        "ResNet-34": ([71.3, 79.6, 78.5, 81.9], [63.4, 72.5, 82.2, 70.8], "#7C3AED"),
    }
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=300)
    for name, (y2, y3, col) in data.items():
        ax.plot(ks, y2, "o-", color=col, linewidth=2.0, markersize=6.5, label=f"{name} (N=2)")
        ax.plot(ks, y3, "s--", color=col, linewidth=1.7, markersize=6.0, alpha=0.85, label=f"{name} (N=3)")

    ax.set_title("ProtoNet Unseen-Class Accuracy Across Backbones (Solid: N=2 | Dashed: N=3)",
                 fontsize=11.0, fontweight="bold", pad=10)
    ax.set_xlabel("Support Shots (K-Shot)", fontsize=10.5)
    ax.set_ylabel("Test Accuracy (%)", fontsize=10.5)
    ax.set_xticks(ks)
    ax.set_ylim(58.0, 98.0)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(fontsize=8.5, ncol=2, loc="lower right", frameon=True)
    fig.tight_layout()

    out_path = TEX_DIR / "16.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[7/7] Saved standardized 300-DPI {out_path}")


def main():
    make_fig1_architecture()
    make_fig5_pipeline()
    make_fig8_maml_loss_curves()
    make_fig9_vit_vs_resnet12()
    make_fig12_baseline_bar()
    make_fig13_cross_signer_kshot()
    make_fig14_unseen_class_kshot()
    print("\nALL 7 FIGURES REGENERATED AT 300 DPI SUCCESSFULLY!")


if __name__ == "__main__":
    main()
