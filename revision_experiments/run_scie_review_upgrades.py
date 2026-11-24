"""
Portable SCIE Peer-Review Upgrade Suite (Auto-Resuming):
  1. Direct paired statistical significance anchored to Weighted Score Fusion (Proposed)
     vs. ViT-B/16 Only and ResNet-12 Only (3 seeds x 50 identical paired episodes = 150 episodes;
     Unseen-Class 3w-10s & Cross-Signer 3w-10s), plus exact learned fusion weight
     w = sigmoid(alpha_w) and 3-class unseen confusion matrix.
  2. Full 6-Fold Leave-One-Signer-Out (LOSO) Cross-Signer Validation across User_1..User_6
     (3-way 10-shot, 50 episodes/fold = 300 total test episodes across all 6 signers)
     for ResNet-12 Only, ViT-B/16 Only, and Dual-Backbone Weighted Score Fusion.

Usage:
    python revision_experiments/run_scie_review_upgrades.py --data_root ./data
"""

import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_all_ijisa_revision_experiments as exp_mod


def paired_test(scores_prop, scores_base):
    a = np.asarray(scores_prop, dtype=float) * 100.0
    b = np.asarray(scores_base, dtype=float) * 100.0
    diff = a - b
    t_stat, t_p = stats.ttest_rel(a, b)
    if np.allclose(diff, 0.0):
        w_p = 1.0
    else:
        _, w_p = stats.wilcoxon(a, b, zero_method="wilcox")
    return {
        "mean_prop": round(float(a.mean()), 2),
        "std_prop": round(float(a.std(ddof=1)), 2),
        "mean_base": round(float(b.mean()), 2),
        "std_base": round(float(b.std(ddof=1)), 2),
        "delta": round(float(a.mean() - b.mean()), 2),
        "t_stat": round(float(t_stat), 4),
        "t_pvalue": float(t_p),
        "wilcoxon_pvalue": float(w_p),
    }


@torch.no_grad()
def eval_with_confusion(model, samples, class_to_idx, store84, store224, episodes, n_way, device):
    model.eval()
    accs = []
    cm = np.zeros((n_way, n_way), dtype=int)
    use_amp = device.type == "cuda"
    for s_idx, s_y, q_idx, q_y in episodes:
        sx, sy, qx, qy = exp_mod.materialize_episode_fast(
            samples, store84, store224, s_idx, s_y, q_idx, q_y, device, dual=True, training=False
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(sx, sy, qx, n_way)
        preds = logits.argmax(dim=-1)
        accs.append(float((preds == qy).float().mean().item()))
        local_to_global = {int(loc): class_to_idx[samples[idx][1]] for idx, loc in zip(s_idx, s_y)}
        for t_loc, p_loc in zip(qy.cpu().tolist(), preds.cpu().tolist()):
            cm[local_to_global[t_loc], local_to_global[p_loc]] += 1
    return accs, cm


def build_loso_split(data_root: Path, test_signer: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    class_dirs = sorted(
        [p for p in data_root.iterdir() if p.is_dir() and any(d.is_dir() for d in p.iterdir())]
    )
    class_to_idx = {c.name: i for i, c in enumerate(class_dirs)}
    tr_samples, te_samples = [], []
    for cdir in class_dirs:
        for sdir in sorted([d for d in cdir.iterdir() if d.is_dir()]):
            files = sorted([f for f in sdir.iterdir() if f.is_file() and f.suffix.lower() in exts])
            target = te_samples if sdir.name == test_signer else tr_samples
            for f in files:
                target.append((str(f), cdir.name, sdir.name))
    assert {s[2] for s in tr_samples}.isdisjoint({s[2] for s in te_samples})
    return (tr_samples, class_to_idx), (te_samples, class_to_idx)


def main():
    parser = argparse.ArgumentParser(description="Score-Fusion-Anchored Paired Significance & 6-Fold LOSO Suite")
    parser.add_argument("--data_root", type=str, default=str(SCRIPT_DIR.parent / "data"))
    parser.add_argument("--out_dir", type=str, default=str(SCRIPT_DIR / "results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2024])
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== SCIE REVIEW UPGRADE SUITE ON {device} ===", flush=True)

    store84 = exp_mod.FastImageStore(target_size=84)
    store224 = exp_mod.FastImageStore(target_size=224)
    seeds = args.seeds
    q_train = 5
    q_eval = 15

    # PART 1: Direct Score-Fusion vs ViT-B/16 vs ResNet-12 + Learned w + Confusion
    sig_csv = out_dir / "exp2_score_anchored_significance.csv"
    ana_json = out_dir / "score_fusion_analysis.json"
    sig_rows = pd.read_csv(sig_csv).to_dict("records") if sig_csv.exists() else []
    done_protos = {r["Protocol"] for r in sig_rows}
    analysis_data = json.loads(ana_json.read_text(encoding="utf-8")) if ana_json.exists() else {
        "learned_fusion_weights": {}
    }

    print("\n[PART 1] Direct Weighted-Score-Anchored Paired Tests + Learned w + Confusion...", flush=True)
    for protocol in ["unseen_class", "cross_signer"]:
        if protocol in done_protos:
            print(f"  [SKIP DONE] Part 1 {protocol}", flush=True)
            continue

        (tr_s, tr_map), _, (te_s, te_map) = exp_mod.build_protocol_splits(
            data_root, protocol, seed=42, n_test_classes=3
        )
        tr_cm = exp_mod.build_classmap(tr_s, tr_map)
        te_cm = exp_mod.build_classmap(te_s, te_map)
        unseen_cm_total = np.zeros((3, 3), dtype=int)
        idx_to_name = {v: k for k, v in te_map.items()}
        unseen_class_names = [idx_to_name[i] for i in range(min(3, len(idx_to_name)))]

        paired_scores = defaultdict(list)
        w_vals, tau_vals = [], []

        for seed in seeds:
            exp_mod.set_seed(seed)
            ep_rng = random.Random(2000 + seed + 3 * 17)
            test_eps = [exp_mod.sample_episode_indices(te_cm, 3, 10, q_eval, ep_rng) for _ in range(50)]

            specs = [
                ("ResNet-12_Only", lambda: exp_mod.UnifiedProtoNet(exp_mod.get_cnn_backbone("resnet12", False), use_vit_input=False)),
                ("ViT-B16_Only", lambda: exp_mod.UnifiedProtoNet(exp_mod.get_vit_backbone(True), use_vit_input=True)),
                ("Dual_Score_Fusion", lambda: exp_mod.DualBackboneFusionProtoNet(exp_mod.get_cnn_backbone("resnet12", False), exp_mod.get_vit_backbone(True), fusion_type="score")),
            ]

            for vname, builder in specs:
                tm = time.time()
                model = builder()
                model = exp_mod.train_protonet_model(
                    model, tr_s, tr_cm, store84, store224, 3, 10, q_train, device,
                    meta_iters=120, lr_cnn=1e-3, lr_vit=1e-5, dual=True, seed=seed,
                )
                if vname == "Dual_Score_Fusion":
                    w_learned = float(torch.sigmoid(model.score_logit_weight).item())
                    tau_learned = float(model.tau.item())
                    w_vals.append(w_learned)
                    tau_vals.append(tau_learned)
                    if protocol == "unseen_class":
                        accs, cm = eval_with_confusion(model, te_s, te_map, store84, store224, test_eps, 3, device)
                        unseen_cm_total += cm
                    else:
                        accs = exp_mod.eval_protonet_paired(model, te_s, store84, store224, test_eps, 3, device, dual=True)
                    print(
                        f"    [{protocol} s={seed}] {vname:17s} acc={np.mean(accs)*100:5.2f}% "
                        f"| w_res={w_learned:.4f}, w_vit={1-w_learned:.4f}, tau={tau_learned:.2f} ({time.time()-tm:.0f}s)",
                        flush=True,
                    )
                else:
                    accs = exp_mod.eval_protonet_paired(model, te_s, store84, store224, test_eps, 3, device, dual=True)
                    print(
                        f"    [{protocol} s={seed}] {vname:17s} acc={np.mean(accs)*100:5.2f}% ({time.time()-tm:.0f}s)",
                        flush=True,
                    )
                paired_scores[vname].extend(accs)
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        analysis_data["learned_fusion_weights"][protocol] = {
            "w_per_seed": [round(x, 4) for x in w_vals],
            "w_mean": round(float(np.mean(w_vals)), 4),
            "w_std": round(float(np.std(w_vals, ddof=1)), 4),
            "vit_weight_mean": round(float(1.0 - np.mean(w_vals)), 4),
            "tau_mean": round(float(np.mean(tau_vals)), 2),
        }
        if protocol == "unseen_class":
            analysis_data["unseen_3way_classes"] = unseen_class_names
            analysis_data["unseen_3way_confusion_matrix"] = unseen_cm_total.tolist()
            analysis_data["unseen_3way_per_class_recall_pct"] = {
                unseen_class_names[i]: round(float(unseen_cm_total[i, i] / unseen_cm_total[i].sum() * 100.0), 2)
                for i in range(3)
            }

        prop = paired_scores["Dual_Score_Fusion"]
        for base_name in ["ResNet-12_Only", "ViT-B16_Only"]:
            res = paired_test(prop, paired_scores[base_name])
            res["Protocol"] = protocol
            res["Comparison"] = f"Dual_Score_Fusion vs {base_name}"
            sig_rows.append(res)

        pd.DataFrame(sig_rows).to_csv(sig_csv, index=False)
        ana_json.write_text(json.dumps(analysis_data, indent=2), encoding="utf-8")

    # PART 2: 6-Fold Leave-One-Signer-Out (LOSO) Cross-Signer Validation
    loso_csv = out_dir / "exp5_loso_6fold_cross_signer.csv"
    loso_rows = pd.read_csv(loso_csv).to_dict("records") if loso_csv.exists() else []
    done_signers = {r["Test_Signer"] for r in loso_rows}

    print("\n[PART 2] Running 6-Fold Leave-One-Signer-Out (LOSO) Cross-Signer Validation (3-way 10-shot)...", flush=True)
    for fold_idx, test_signer in enumerate([f"User_{i}" for i in range(1, 7)], 1):
        if test_signer in done_signers:
            print(f"  [SKIP DONE] LOSO {test_signer}", flush=True)
            continue

        (tr_s, tr_map), (te_s, te_map) = build_loso_split(data_root, test_signer)
        tr_cm = exp_mod.build_classmap(tr_s, tr_map)
        te_cm = exp_mod.build_classmap(te_s, te_map)

        exp_mod.set_seed(42 + fold_idx)
        ep_rng = random.Random(7000 + fold_idx * 13)
        test_eps = [exp_mod.sample_episode_indices(te_cm, 3, 10, q_eval, ep_rng) for _ in range(50)]

        fold_res = {"Fold": fold_idx, "Test_Signer": test_signer}
        tf = time.time()
        for vname, builder in [
            ("ResNet-12_Only", lambda: exp_mod.UnifiedProtoNet(exp_mod.get_cnn_backbone("resnet12", False), use_vit_input=False)),
            ("ViT-B16_Only", lambda: exp_mod.UnifiedProtoNet(exp_mod.get_vit_backbone(True), use_vit_input=True)),
            ("Dual_Score_Fusion", lambda: exp_mod.DualBackboneFusionProtoNet(exp_mod.get_cnn_backbone("resnet12", False), exp_mod.get_vit_backbone(True), fusion_type="score")),
        ]:
            model = builder()
            model = exp_mod.train_protonet_model(
                model, tr_s, tr_cm, store84, store224, 3, 10, q_train, device,
                meta_iters=80, lr_cnn=1e-3, lr_vit=1e-5, dual=True, seed=42 + fold_idx,
            )
            accs = exp_mod.eval_protonet_paired(model, te_s, store84, store224, test_eps, 3, device, dual=True)
            fold_res[f"{vname}_Mean"] = round(float(np.mean(accs) * 100.0), 2)
            fold_res[f"{vname}_Std"] = round(float(np.std(accs, ddof=1) * 100.0), 2)
            if vname == "Dual_Score_Fusion":
                fold_res["Learned_w"] = round(float(torch.sigmoid(model.score_logit_weight).item()), 4)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        loso_rows.append(fold_res)
        pd.DataFrame(loso_rows).to_csv(loso_csv, index=False)
        print(
            f"  [LOSO Fold {fold_idx}/6: {test_signer}] "
            f"ResNet-12={fold_res['ResNet-12_Only_Mean']:5.2f}% | "
            f"ViT-B/16={fold_res['ViT-B16_Only_Mean']:5.2f}% | "
            f"ScoreFusion={fold_res['Dual_Score_Fusion_Mean']:5.2f}% (w={fold_res['Learned_w']:.3f}) [{time.time()-tf:.0f}s]",
            flush=True,
        )

    print(f"\nALL UPGRADE EXPERIMENTS FINISHED in {(time.time()-t0)/60:.2f} min!", flush=True)


if __name__ == "__main__":
    main()
