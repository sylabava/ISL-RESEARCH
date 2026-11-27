r"""
JEI Vol. 35 Standard Upgrade Experiment Suite (RTX 5050 GPU, BF16):
  1. Exact Fusion Weight Sensitivity Sweep (w in {0.0, 0.1, ..., 1.0} + learned w*)
     using the archived immutable Source-of-Truth Dual_Score_Fusion checkpoints
     on the exact 150 paired episodes (unseen_class & cross_signer).
  2. Component-Wise Ablation (for JEI checkmark ablation table):
     - Dual_Unnorm_Euclidean (no L2 hyperspherical normalization; raw -||z - p||^2)
     - Dual_L2_FixedW05_FixedTau (L2 normalized, fixed w=0.50, fixed tau=10.0)
     trained under the identical 200-iter BF16 loop, seeds {42, 100, 2024}, and
     evaluated on the exact same 150 paired episodes as source_of_truth_episodes.csv.
  3. Exact Per-Class Precision, Recall, F1, and Macro-F1 from reconciled_table13_confusion.json.
  4. Real ResNet-12 Grad-CAM + Real ViT-B/16 12-Layer Attention Rollout on 3 verified
     correct unseen-class predictions AND 3 verified misclassified failure cases
     (pulled directly from source_of_truth_predictions_unseen.csv), PLUS real fused
     hyperspherical embeddings [sqrt(w)*z_R, sqrt(1-w)*z_V] for unseen-class t-SNE.
"""

import gc
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import stats

EXP_DIR = Path(r"C:\Users\SYLA\Desktop\RESEARCH\REVIEW1\experiments")
OUT_DIR = EXP_DIR / "revision_outputs"
CKPT_DIR = OUT_DIR / "checkpoints"
GIT_RES = Path(r"C:\Users\SYLA\Desktop\RESEARCH\GIT\revision_experiments\results")
DATA_ROOT = Path(r"C:\Users\SYLA\Desktop\RESEARCH\META1\data")

sys.path.insert(0, str(EXP_DIR))
import run_all_ijisa_revision_experiments as exp_mod


def load_dual_score_ckpt(ckpt_path: Path, device: torch.device):
    model = exp_mod.DualBackboneFusionProtoNet(
        exp_mod.get_cnn_backbone("resnet12", False),
        exp_mod.get_vit_backbone(False),
        fusion_type="score",
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


class ComponentAblationDualNet(exp_mod.DualBackboneFusionProtoNet):
    """Ablated dual-backbone score fusion inheriting DualBackboneFusionProtoNet so
    train_protonet_model applies the EXACT differential learning rates
    (lr_cnn=1e-3 for ResNet-12 & scalars, lr_vit=1e-5 for ViT-B/16)."""
    def __init__(self, resnet: nn.Module, vit: nn.Module, use_l2: bool = True, learn_w: bool = True, learn_tau: bool = True):
        super().__init__(resnet, vit, fusion_type="score")
        self.use_l2 = use_l2
        self.learn_w = learn_w
        self.learn_tau = learn_tau
        if not learn_tau:
            self.tau.requires_grad_(False)
        if not learn_w:
            self.score_logit_weight.requires_grad_(False)

    def forward(self, sx_pair, sy, qx_pair, n_way: int):
        sr = exp_mod.encode_in_chunks(self.resnet, sx_pair[0], 64).float()
        qr = exp_mod.encode_in_chunks(self.resnet, qx_pair[0], 64).float()
        sv = exp_mod.encode_in_chunks(self.vit, sx_pair[1], 32).float()
        qv = exp_mod.encode_in_chunks(self.vit, qx_pair[1], 32).float()

        if self.use_l2:
            sr = F.normalize(sr, p=2, dim=-1)
            qr = F.normalize(qr, p=2, dim=-1)
            pr = F.normalize(torch.stack([sr[sy == c].mean(0) for c in range(n_way)]), p=2, dim=-1)
            logits_r = self.tau * torch.matmul(qr, pr.T)

            sv = F.normalize(sv, p=2, dim=-1)
            qv = F.normalize(qv, p=2, dim=-1)
            pv = F.normalize(torch.stack([sv[sy == c].mean(0) for c in range(n_way)]), p=2, dim=-1)
            logits_v = self.tau * torch.matmul(qv, pv.T)
        else:
            # Unnormalized raw negative squared-Euclidean distances (standard ProtoNet metric)
            pr = torch.stack([sr[sy == c].mean(0) for c in range(n_way)])
            pv = torch.stack([sv[sy == c].mean(0) for c in range(n_way)])
            logits_r = -torch.cdist(qr, pr, p=2.0).pow(2)
            logits_v = -torch.cdist(qv, pv, p=2.0).pow(2)

        w = torch.sigmoid(self.score_logit_weight) if self.learn_w else 0.5
        return w * logits_r + (1.0 - w) * logits_v


@torch.no_grad()
def part1_weight_sensitivity_sweep(store84, store224, device):
    sweep_csv = OUT_DIR / "jei_weight_sensitivity_sweep.csv"
    if sweep_csv.exists():
        print("\n[PART 1] [SKIP DONE] jei_weight_sensitivity_sweep.csv already computed from archived checkpoints!", flush=True)
        return pd.read_csv(sweep_csv)
    print("\n[PART 1] Exact Fusion Weight Sensitivity Sweep (w in [0.0..1.0]) on Archived Source-of-Truth Checkpoints...", flush=True)
    ws = [round(i * 0.1, 1) for i in range(11)]
    seeds = [42, 100, 2024]
    use_amp = device.type == "cuda"
    rows = []

    for protocol in ["unseen_class", "cross_signer"]:
        _, _, (te_s, te_map) = exp_mod.build_protocol_splits(DATA_ROOT, protocol, seed=42, n_test_classes=3)
        te_cm = exp_mod.build_classmap(te_s, te_map)

        accs_by_w = {w: [] for w in ws}
        accs_learned = []
        learned_ws = []

        for seed in seeds:
            ckpt = CKPT_DIR / f"{protocol}_s{seed}_Dual_Score_Fusion_fp16.pt"
            model = load_dual_score_ckpt(ckpt, device)
            w_star = float(torch.sigmoid(model.score_logit_weight).item())
            learned_ws.append(w_star)

            ep_rng = random.Random(2000 + seed + 3 * 17)
            test_eps = [exp_mod.sample_episode_indices(te_cm, 3, 10, 15, ep_rng) for _ in range(50)]

            for s_idx, s_y, q_idx, q_y in test_eps:
                sx, sy, qx, qy = exp_mod.materialize_episode_fast(
                    te_s, store84, store224, s_idx, s_y, q_idx, q_y, device, dual=True, training=False
                )
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    sr = F.normalize(exp_mod.encode_in_chunks(model.resnet, sx[0], 64).float(), p=2, dim=-1)
                    qr = F.normalize(exp_mod.encode_in_chunks(model.resnet, qx[0], 64).float(), p=2, dim=-1)
                    pr = F.normalize(torch.stack([sr[sy == c].mean(0) for c in range(3)]), p=2, dim=-1)
                    sim_r = torch.matmul(qr, pr.T)

                    sv = F.normalize(exp_mod.encode_in_chunks(model.vit, sx[1], 32).float(), p=2, dim=-1)
                    qv = F.normalize(exp_mod.encode_in_chunks(model.vit, qx[1], 32).float(), p=2, dim=-1)
                    pv = F.normalize(torch.stack([sv[sy == c].mean(0) for c in range(3)]), p=2, dim=-1)
                    sim_v = torch.matmul(qv, pv.T)

                for w in ws:
                    preds = (w * sim_r + (1.0 - w) * sim_v).argmax(dim=-1)
                    accs_by_w[w].append(float((preds == qy).float().mean().item() * 100.0))
                preds_star = (w_star * sim_r + (1.0 - w_star) * sim_v).argmax(dim=-1)
                accs_learned.append(float((preds_star == qy).float().mean().item() * 100.0))

            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for w in ws:
            arr = np.array(accs_by_w[w], dtype=float)
            rows.append({
                "Protocol": protocol,
                "Weight_w": w,
                "Is_Learned": 0,
                "Mean_Acc": round(float(arr.mean()), 2),
                "Std_Acc": round(float(arr.std(ddof=1)), 2),
                "CI95": round(float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr))), 2),
            })
        arr_l = np.array(accs_learned, dtype=float)
        rows.append({
            "Protocol": protocol,
            "Weight_w": round(float(np.mean(learned_ws)), 4),
            "Is_Learned": 1,
            "Mean_Acc": round(float(arr_l.mean()), 2),
            "Std_Acc": round(float(arr_l.std(ddof=1)), 2),
            "CI95": round(float(1.96 * arr_l.std(ddof=1) / np.sqrt(len(arr_l))), 2),
        })
        print(f"  [{protocol}] w=0.0 (ViT): {np.mean(accs_by_w[0.0]):.2f}% | w=0.5: {np.mean(accs_by_w[0.5]):.2f}% | "
              f"w*={np.mean(learned_ws):.4f}: {arr_l.mean():.2f}% | w=1.0 (ResNet): {np.mean(accs_by_w[1.0]):.2f}%", flush=True)

    df = pd.DataFrame(rows)
    for dst in [OUT_DIR / "jei_weight_sensitivity_sweep.csv", GIT_RES / "jei_weight_sensitivity_sweep.csv"]:
        df.to_csv(dst, index=False)
    return df


def part2_component_ablation(store84, store224, device):
    print("\n[PART 2] Component-Wise Ablation (Unnormalized Euclidean Dual & L2 Fixed w=0.5/tau=10)...", flush=True)
    seeds = [42, 100, 2024]
    use_amp = device.type == "cuda"
    sot_df = pd.read_csv(OUT_DIR / "source_of_truth_episodes.csv")

    ep_csv = OUT_DIR / "jei_component_ablation_episodes.csv"
    ep_rows = pd.read_csv(ep_csv).to_dict("records") if ep_csv.exists() else []
    done = {(r["Protocol"], int(r["Seed"]), r["Variant"]) for r in ep_rows}

    variants = [
        ("Dual_Unnorm_Euclidean", dict(use_l2=False, learn_w=True, learn_tau=False)),
        ("Dual_L2_FixedW05_FixedTau", dict(use_l2=True, learn_w=False, learn_tau=False)),
    ]

    for protocol in ["unseen_class", "cross_signer"]:
        (tr_s, tr_map), _, (te_s, te_map) = exp_mod.build_protocol_splits(DATA_ROOT, protocol, seed=42, n_test_classes=3)
        tr_cm = exp_mod.build_classmap(tr_s, tr_map)
        te_cm = exp_mod.build_classmap(te_s, te_map)

        for seed_idx, seed in enumerate(seeds):
            ep_rng = random.Random(2000 + seed + 3 * 17)
            test_eps = [exp_mod.sample_episode_indices(te_cm, 3, 10, 15, ep_rng) for _ in range(50)]

            for vname, cfg in variants:
                if (protocol, seed, vname) in done:
                    print(f"  [SKIP DONE] {protocol} s={seed} {vname}", flush=True)
                    continue
                t0 = time.time()
                exp_mod.set_seed(seed)
                model = ComponentAblationDualNet(
                    exp_mod.get_cnn_backbone("resnet12", False),
                    exp_mod.get_vit_backbone(True),
                    **cfg,
                )
                assert isinstance(model, exp_mod.DualBackboneFusionProtoNet), "Must inherit DualBackboneFusionProtoNet for differential LRs!"
                model = exp_mod.train_protonet_model(
                    model, tr_s, tr_cm, store84, store224, 3, 5, 15, device,
                    meta_iters=200, lr_cnn=1e-3, lr_vit=1e-5, dual=True, seed=seed,
                )
                model.eval()
                accs = []
                with torch.no_grad():
                    for ep_i, (s_idx, s_y, q_idx, q_y) in enumerate(test_eps):
                        sx, sy, qx, qy = exp_mod.materialize_episode_fast(
                            te_s, store84, store224, s_idx, s_y, q_idx, q_y, device, dual=True, training=False
                        )
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                            logits = model(sx, sy, qx, 3)
                        acc = float((logits.argmax(dim=-1) == qy).float().mean().item() * 100.0)
                        accs.append(acc)
                        ep_rows.append({
                            "Protocol": protocol,
                            "Variant": vname,
                            "Seed": seed,
                            "Global_Episode_ID": seed_idx * 50 + ep_i,
                            "Accuracy_Pct": round(acc, 4),
                            "Learned_w": round(float(torch.sigmoid(model.score_logit_weight).item()), 4) if cfg["learn_w"] else 0.5,
                        })
                pd.DataFrame(ep_rows).to_csv(ep_csv, index=False)
                print(f"  [{protocol:12s} s={seed:4d}] {vname:26s}: {np.mean(accs):5.2f}% ± {np.std(accs, ddof=1):4.2f}% ({time.time()-t0:.0f}s)", flush=True)
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    df_ep = pd.DataFrame(ep_rows)
    sum_rows = []
    for protocol in ["unseen_class", "cross_signer"]:
        anchor = sot_df[(sot_df["Protocol"] == protocol) & (sot_df["Model"] == "Dual_Score_Fusion")].sort_values("Global_Episode_ID")["Accuracy_Pct"].to_numpy()
        for vname, _ in variants:
            sub = df_ep[(df_ep["Protocol"] == protocol) & (df_ep["Variant"] == vname)].sort_values("Global_Episode_ID")
            vals = sub["Accuracy_Pct"].to_numpy()
            t_stat, t_p = stats.ttest_rel(anchor, vals)
            diff = anchor - vals
            w_p = 1.0 if np.allclose(diff, 0.0) else float(stats.wilcoxon(anchor, vals, zero_method="wilcox")[1])
            sum_rows.append({
                "Protocol": protocol,
                "Variant": vname,
                "Mean_Acc": round(float(vals.mean()), 2),
                "Std_Acc": round(float(vals.std(ddof=1)), 2),
                "CI95": round(float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))), 2),
                "Delta_vs_Proposed": round(float(anchor.mean() - vals.mean()), 2),
                "Paired_t_p": float(t_p),
                "Wilcoxon_p": float(w_p),
                "Mean_w": round(float(sub["Learned_w"].mean()), 4),
            })
    df_sum = pd.DataFrame(sum_rows)
    for dst in [OUT_DIR / "jei_component_ablation_summary.csv", GIT_RES / "jei_component_ablation_summary.csv"]:
        df_sum.to_csv(dst, index=False)
    df_ep.to_csv(GIT_RES / "jei_component_ablation_episodes.csv", index=False)
    print("\nComponent Ablation Summary:")
    print(df_sum.to_string(index=False), flush=True)
    return df_sum


def part3_per_class_f1():
    print("\n[PART 3] Computing Exact Per-Class Precision, Recall, F1, and Macro-F1 (6,750 Unseen Queries)...", flush=True)
    d = json.loads((OUT_DIR / "reconciled_table13_confusion.json").read_text(encoding="utf-8"))
    classes = d["classes"]
    cm = np.array(d["matrix"], dtype=float)
    per_cls = {}
    precs, recs, f1s = [], [], []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) * 100.0
        rec = tp / (tp + fn) * 100.0
        f1 = 2 * prec * rec / (prec + rec)
        per_cls[c] = {
            "precision_pct": round(float(prec), 2),
            "recall_pct": round(float(rec), 2),
            "f1_pct": round(float(f1), 2),
            "support": int(cm[i, :].sum()),
        }
        precs.append(prec); recs.append(rec); f1s.append(f1)
    out = {
        "total_queries": int(cm.sum()),
        "overall_acc_pct": round(float(np.trace(cm) / cm.sum() * 100.0), 2),
        "macro_precision_pct": round(float(np.mean(precs)), 2),
        "macro_recall_pct": round(float(np.mean(recs)), 2),
        "macro_f1_pct": round(float(np.mean(f1s)), 2),
        "per_class": per_cls,
    }
    for dst in [OUT_DIR / "jei_per_class_f1.json", GIT_RES / "jei_per_class_f1.json"]:
        dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)
    return out


def compute_resnet_gradcam(model, x84: torch.Tensor, target_proto_r: torch.Tensor) -> np.ndarray:
    """Compute real Grad-CAM heatmap on ResNet-12 layer4 w.r.t. cosine similarity to target_proto_r."""
    feats = {}
    grads = {}

    def fwd_hook(_m, _inp, out):
        feats["f"] = out

    def bwd_hook(_m, _gin, gout):
        grads["g"] = gout[0]

    h1 = model.resnet.layer4.register_forward_hook(fwd_hook)
    h2 = model.resnet.layer4.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        z = F.normalize(model.resnet(x84).float(), p=2, dim=-1)
        score = (z * target_proto_r.unsqueeze(0)).sum()
        score.backward()
        fmap = feats["f"].detach()[0]   # (C, H, W)
        gmap = grads["g"].detach()[0]   # (C, H, W)
        weights = gmap.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * fmap).sum(dim=0))
        cam = cam.cpu().numpy()
        if cam.max() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam
    finally:
        h1.remove()
        h2.remove()


@torch.no_grad()
def compute_vit_attention_rollout(vit_model, x224: torch.Tensor) -> np.ndarray:
    """Compute real 12-layer ViT-B/16 Attention Rollout from [CLS] to the 14x14 patch grid."""
    vit_model.eval()
    x = vit_model._process_input(x224)
    n = x.shape[0]
    cls_tok = vit_model.class_token.expand(n, -1, -1)
    x = torch.cat([cls_tok, x], dim=1)
    x = x + vit_model.encoder.pos_embedding

    rollout = torch.eye(x.size(1), device=x.device)
    for blk in vit_model.encoder.layers:
        y = blk.ln_1(x)
        attn_out, attn_w = blk.self_attention(y, y, y, need_weights=True, average_attn_weights=True)
        # attn_w: (1, 197, 197)
        A = attn_w[0]
        A_res = 0.5 * A + 0.5 * torch.eye(A.size(0), device=A.device)
        A_res = A_res / A_res.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(A_res, rollout)
        x = x + attn_out
        x = x + blk.mlp(blk.ln_2(x))

    cls_to_patches = rollout[0, 1:].reshape(14, 14).cpu().numpy()
    if cls_to_patches.max() > 1e-8:
        cls_to_patches = (cls_to_patches - cls_to_patches.min()) / (cls_to_patches.max() - cls_to_patches.min())
    return cls_to_patches


def part4_qualitative_and_tsne(store84, store224, device):
    print("\n[PART 4] Extracting Real Grad-CAM, ViT Attention Rollout (Success + Failure Cases) & Fused t-SNE...", flush=True)
    _, _, (te_s, te_map) = exp_mod.build_protocol_splits(DATA_ROOT, "unseen_class", seed=42, n_test_classes=3)
    te_cm = exp_mod.build_classmap(te_s, te_map)
    model = load_dual_score_ckpt(CKPT_DIR / "unseen_class_s42_Dual_Score_Fusion_fp16.pt", device)
    w_star = float(torch.sigmoid(model.score_logit_weight).item())

    # Build representative support prototypes from a fixed balanced 10-shot support draw (seed 42 ep 0)
    ep_rng = random.Random(2000 + 42 + 3 * 17)
    s_idx, s_y, _, _ = exp_mod.sample_episode_indices(te_cm, 3, 10, 15, ep_rng)
    sx, sy, _, _ = exp_mod.materialize_episode_fast(te_s, store84, store224, s_idx, s_y, s_idx[:3], s_y[:3], device, dual=True, training=False)
    idx_to_name = {v: k for k, v in te_map.items()}
    local_to_cls = {int(loc): te_s[idx][1] for idx, loc in zip(s_idx, s_y)}
    cls_to_local = {v: k for k, v in local_to_cls.items()}

    with torch.no_grad():
        sr = F.normalize(model.resnet(sx[0]).float(), p=2, dim=-1)
        sv = F.normalize(model.vit(sx[1]).float(), p=2, dim=-1)
        pr = F.normalize(torch.stack([sr[sy == c].mean(0) for c in range(3)]), p=2, dim=-1)
        pv = F.normalize(torch.stack([sv[sy == c].mean(0) for c in range(3)]), p=2, dim=-1)

    # Select real success & failure cases directly from source_of_truth_predictions_unseen.csv (seed 42)
    qdf = pd.read_csv(OUT_DIR / "source_of_truth_predictions_unseen.csv")
    q42 = qdf[qdf["Seed"] == 42]

    name_to_path = {Path(p).name: (p, cls, signer) for (p, cls, signer) in te_s}

    def pick_most_frequent(sub_df):
        cnt = Counter(sub_df["Image_Name"].tolist())
        for img_name, freq in cnt.most_common():
            if img_name in name_to_path:
                return img_name, freq
        raise RuntimeError("No matching image found!")

    selected_specs = [
        ("Success", "afraid", "afraid", q42[(q42["True_Class"] == "afraid") & (q42["Correct"] == 1) & (q42["Signer"] == "User_1")]),
        ("Success", "pain",   "pain",   q42[(q42["True_Class"] == "pain")   & (q42["Correct"] == 1) & (q42["Signer"] == "User_3")]),
        ("Success", "bad",    "bad",    q42[(q42["True_Class"] == "bad")    & (q42["Correct"] == 1) & (q42["Signer"] == "User_4")]),
        ("Failure", "pain",   "afraid", q42[(q42["True_Class"] == "pain")   & (q42["Pred_Class"] == "afraid") & (q42["Signer"] == "User_2")]),
        ("Failure", "pain",   "bad",    q42[(q42["True_Class"] == "pain")   & (q42["Pred_Class"] == "bad")    & (q42["Signer"] == "User_2")]),
        ("Failure", "bad",    "pain",   q42[(q42["True_Class"] == "bad")    & (q42["Pred_Class"] == "pain")   & (q42["Signer"].isin(["User_5", "User_6"]))]),
    ]

    rgb_list, cam_list, att_list, meta_list = [], [], [], []
    for kind, t_cls, p_cls, sub_df in selected_specs:
        img_name, freq = pick_most_frequent(sub_df)
        fpath, true_c, signer = name_to_path[img_name]
        with Image.open(fpath) as im:
            im224 = im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            rgb_np = np.asarray(im224, dtype=np.uint8)
        x84 = (store84.get(fpath).unsqueeze(0).to(device).float() / 255.0)
        x224 = (store224.get(fpath).unsqueeze(0).to(device).float() / 255.0)

        target_loc = cls_to_local[p_cls]
        cam = compute_resnet_gradcam(model, x84, pr[target_loc])
        att = compute_vit_attention_rollout(model.vit, x224)

        rgb_list.append(rgb_np)
        cam_list.append(cam)
        att_list.append(att)
        meta_list.append({
            "kind": kind,
            "true_class": true_c,
            "pred_class": p_cls,
            "signer": signer,
            "image_name": img_name,
            "episode_occurrences": int(freq),
        })
        print(f"  [{kind:7s}] {signer} | true={true_c:6s} pred={p_cls:6s} | {img_name} (occ={freq})", flush=True)

    np.savez_compressed(
        OUT_DIR / "jei_qualitative_maps.npz",
        rgb=np.stack(rgb_list),
        gradcam=np.stack(cam_list),
        vit_attn=np.stack(att_list),
    )
    (OUT_DIR / "jei_qualitative_meta.json").write_text(json.dumps(meta_list, indent=2), encoding="utf-8")

    # Extract exact fused hyperspherical embeddings [sqrt(w)*z_R, sqrt(1-w)*z_V] for unseen-class t-SNE
    rng_tsne = random.Random(42)
    by_cs = {}
    for idx, (fp, cls, sgn) in enumerate(te_s):
        by_cs.setdefault((cls, sgn), []).append(idx)
    chosen_idxs = []
    for cls in sorted(te_map.keys()):
        for sgn in [f"User_{i}" for i in range(1, 7)]:
            pool = by_cs[(cls, sgn)]
            chosen_idxs.extend(rng_tsne.sample(pool, 20))  # 20 x 6 signers x 3 classes = 360 samples

    x84_all = torch.stack([store84.get(te_s[i][0]) for i in chosen_idxs]).to(device).float() / 255.0
    x224_all = torch.stack([store224.get(te_s[i][0]) for i in chosen_idxs]).to(device).float() / 255.0
    with torch.no_grad():
        zr = F.normalize(exp_mod.encode_in_chunks(model.resnet, x84_all, 64).float(), p=2, dim=-1)
        zv = F.normalize(exp_mod.encode_in_chunks(model.vit, x224_all, 32).float(), p=2, dim=-1)
        zfused = torch.cat([np.sqrt(w_star) * zr, np.sqrt(1.0 - w_star) * zv], dim=-1).cpu().numpy()

    np.savez_compressed(
        OUT_DIR / "jei_unseen_tsne_data.npz",
        embeddings=zfused,
        classes=np.array([te_s[i][1] for i in chosen_idxs]),
        signers=np.array([te_s[i][2] for i in chosen_idxs]),
        w_star=np.array([w_star]),
    )
    print(f"  Saved real fused unseen-class embeddings {zfused.shape} (w*={w_star:.4f}) -> jei_unseen_tsne_data.npz", flush=True)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== JEI VOL. 35 STANDARD UPGRADE SUITE ON {device} ===", flush=True)
    store84 = exp_mod.FastImageStore(target_size=84)
    store224 = exp_mod.FastImageStore(target_size=224)

    part1_weight_sensitivity_sweep(store84, store224, device)
    part2_component_ablation(store84, store224, device)
    part3_per_class_f1()
    part4_qualitative_and_tsne(store84, store224, device)
    print(f"\nALL JEI UPGRADE EXPERIMENTS FINISHED in {(time.time()-t0)/60:.2f} min!", flush=True)


if __name__ == "__main__":
    main()
