r"""
Simultaneous Cross-Signer and Unseen-Class Evaluation (Doubly-Disjoint Protocol)
================================================================================
Strict doubly-disjoint partitioning on Mendeley ISL Real-Life Words (18,000 imgs):
  - Training   : Signers User_1..User_4 (4)  x 14 seen classes   = 8,400 images
  - Validation : Signer  User_5         (1)  x  3 unseen val cls =   450 images
  - Testing    : Signer  User_6         (1)  x  3 unseen test cls=   450 images
    (Strict zero signer overlap, zero class overlap, and verified zero MD5 overlap;
     all remaining 8,700 cross-block frames are strictly excluded.)

Identical training & paired evaluation setup to Single Source-of-Truth reconciliation:
  - 5 variants: ResNet-12_Only, ViT-B16_Only, Dual_Concat_Fusion, Dual_Gated_Fusion, Dual_Score_Fusion
  - 3 seeds: {42, 100, 2024}, 200 meta-training iters (n_way=3, k_train=5, q_train=15),
    Adam (lr_cnn=1e-3, lr_vit=1e-5, wd=1e-4, grad_clip=2.0), BF16 autocast on RTX 5050 GPU
  - Evaluates ALL 5 variants on 150 identical paired test episodes (3 seeds x 50 episodes)
    across 3-way 10-shot (primary), 3-way 5-shot, and 2-way 10-shot, plus validation monitoring
    on User_5 (3 unseen val classes, 3-way 10-shot, strictly logged without test tuning).
  - Logs immutable source-of-truth episode CSVs, 6,750 query predictions for Dual_Score_Fusion,
    fp16 checkpoints, SHA-256 weight fingerprints, and Holm-Bonferroni paired significance tests.
"""

import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import torch

EXP_DIR = Path(r"C:\Users\SYLA\Desktop\RESEARCH\REVIEW1\experiments")
OUT_DIR = EXP_DIR / "revision_outputs"
CKPT_DIR = OUT_DIR / "checkpoints"
GIT_RES = Path(r"C:\Users\SYLA\Desktop\RESEARCH\GIT\revision_experiments\results")
DATA_ROOT = Path(r"C:\Users\SYLA\Desktop\RESEARCH\META1\data")

OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
GIT_RES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(EXP_DIR))
import run_all_ijisa_revision_experiments as exp_mod
from run_unified_reconciliation import model_weight_sha256, eval_episodes_detailed


def holm_bonferroni(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, float(pvals[idx]) * (m - rank))
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj.tolist()


def build_simultaneous_doubly_disjoint_splits(data_root: Path):
    """
    Reuses the exact same 14/3/3 class partition (seed=42) and exact same
    User_1..4 / User_5 / User_6 signer partition as Protocols 1 and 2.
    """
    (u_tr, tr_map), (u_va, va_map), (u_te, te_map) = exp_mod.build_protocol_splits(
        data_root, "unseen_class", seed=42, n_test_classes=3
    )
    train_signers = {"User_1", "User_2", "User_3", "User_4"}
    val_signers = {"User_5"}
    test_signers = {"User_6"}

    tr_samples = [s for s in u_tr if s[2] in train_signers]
    va_samples = [s for s in u_va if s[2] in val_signers]
    te_samples = [s for s in u_te if s[2] in test_signers]

    # Strict disjointness assertions (signers, classes, file paths, and exact MD5 content)
    assert {s[2] for s in tr_samples} == train_signers
    assert {s[2] for s in va_samples} == val_signers
    assert {s[2] for s in te_samples} == test_signers
    assert set(tr_map.keys()).isdisjoint(set(va_map.keys()))
    assert set(tr_map.keys()).isdisjoint(set(te_map.keys()))
    assert set(va_map.keys()).isdisjoint(set(te_map.keys()))
    assert {s[0] for s in tr_samples}.isdisjoint({s[0] for s in te_samples})
    assert {s[0] for s in tr_samples}.isdisjoint({s[0] for s in va_samples})
    assert {s[0] for s in va_samples}.isdisjoint({s[0] for s in te_samples})

    tr_md5 = {hashlib.md5(Path(s[0]).read_bytes()).hexdigest() for s in tr_samples}
    va_md5 = {hashlib.md5(Path(s[0]).read_bytes()).hexdigest() for s in va_samples}
    te_md5 = {hashlib.md5(Path(s[0]).read_bytes()).hexdigest() for s in te_samples}
    assert tr_md5.isdisjoint(te_md5), "MD5 overlap detected between Train and Test!"
    assert tr_md5.isdisjoint(va_md5), "MD5 overlap detected between Train and Val!"
    assert va_md5.isdisjoint(te_md5), "MD5 overlap detected between Val and Test!"

    return (tr_samples, tr_map), (va_samples, va_map), (te_samples, te_map)


def aggregate_and_save(ep_csv: Path, q_csv: Path, meta_json: Path):
    df_ep = pd.read_csv(ep_csv)
    df_q = pd.read_csv(q_csv)

    model_order = ["ResNet-12_Only", "ViT-B16_Only", "Dual_Concat_Fusion", "Dual_Gated_Fusion", "Dual_Score_Fusion"]
    eval_settings = ["test_3w10s", "test_3w5s", "test_2w10s", "val_3w10s"]

    summary_rows = []
    paired_rows = []
    weights_info = {}

    for setting in eval_settings:
        sub = df_ep[df_ep["Eval_Setting"] == setting]
        if len(sub) == 0:
            continue
        ep_arrays = {}
        for m in model_order:
            ms = sub[sub["Model"] == m].sort_values("Global_Episode_ID")
            assert len(ms) == 150, f"Expected 150 rows for {setting}/{m}, got {len(ms)}"
            vals = ms["Accuracy_Pct"].to_numpy(dtype=float)
            ep_arrays[m] = vals
            seed_means = [round(float(ms[ms["Seed"] == s]["Accuracy_Pct"].mean()), 2) for s in [42, 100, 2024]]
            summary_rows.append({
                "Eval_Setting": setting,
                "Model": m,
                "Episodes": len(vals),
                "Episode_Mean": round(float(vals.mean()), 2),
                "Episode_Std": round(float(vals.std(ddof=1)), 2),
                "Episode_CI95": round(float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))), 2),
                "Seed_Means": str(seed_means),
                "Seed_Mean": round(float(np.mean(seed_means)), 2),
                "Seed_Std": round(float(np.std(seed_means, ddof=1)), 2),
            })
            if m == "Dual_Score_Fusion" and setting == "test_3w10s":
                w_seeds = [float(ms[ms["Seed"] == s]["Learned_w"].iloc[0]) for s in [42, 100, 2024]]
                t_seeds = [float(ms[ms["Seed"] == s]["Learned_tau"].iloc[0]) for s in [42, 100, 2024]]
                weights_info = {
                    "w_per_seed": [round(x, 4) for x in w_seeds],
                    "w_mean": round(float(np.mean(w_seeds)), 4),
                    "w_std": round(float(np.std(w_seeds, ddof=1)), 4),
                    "vit_weight_mean": round(float(1.0 - np.mean(w_seeds)), 4),
                    "tau_mean": round(float(np.mean(t_seeds)), 2),
                }

        # Score-Fusion-anchored paired tests + Holm-Bonferroni
        prop = ep_arrays["Dual_Score_Fusion"]
        baselines = ["ResNet-12_Only", "ViT-B16_Only", "Dual_Concat_Fusion", "Dual_Gated_Fusion"]
        p_rows = []
        for b in baselines:
            base = ep_arrays[b]
            diff = prop - base
            t_stat, t_p = stats.ttest_rel(prop, base)
            w_p = 1.0 if np.allclose(diff, 0.0) else float(stats.wilcoxon(prop, base, zero_method="wilcox")[1])
            p_rows.append({
                "Eval_Setting": setting,
                "Comparison": f"Dual_Score_Fusion vs {b}",
                "Score_Mean": round(float(prop.mean()), 2),
                "Score_Std": round(float(prop.std(ddof=1)), 2),
                "Base_Mean": round(float(base.mean()), 2),
                "Base_Std": round(float(base.std(ddof=1)), 2),
                "Delta": round(float(prop.mean() - base.mean()), 2),
                "t_stat": round(float(t_stat), 4),
                "t_pvalue": float(t_p),
                "wilcoxon_pvalue": w_p,
            })
        t_adj = holm_bonferroni([r["t_pvalue"] for r in p_rows])
        w_adj = holm_bonferroni([r["wilcoxon_pvalue"] for r in p_rows])
        for r, ta, wa in zip(p_rows, t_adj, w_adj):
            r["t_pvalue_holm"] = ta
            r["wilcoxon_pvalue_holm"] = wa
            paired_rows.append(r)

    # Confusion matrix for primary test_3w10s (6,750 queries on User_6 unseen classes)
    classes = ["pain", "afraid", "bad"]
    c2i = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((3, 3), dtype=int)
    for t_c, p_c in zip(df_q["True_Class"], df_q["Pred_Class"]):
        cm[c2i[t_c], c2i[p_c]] += 1

    cm_report = {
        "protocol": "simultaneous_cross_signer_unseen_class",
        "test_signer": "User_6",
        "total_queries": int(len(df_q)),
        "overall_acc_pct": round(float(np.trace(cm) / len(df_q) * 100.0), 2),
        "classes": classes,
        "matrix": cm.tolist(),
        "per_class_recall_pct": {
            c: round(float(cm[i, i] / cm[i].sum() * 100.0), 2) for i, c in enumerate(classes)
        },
        "learned_weights": weights_info,
    }

    for base_dir in [OUT_DIR, GIT_RES]:
        pd.DataFrame(summary_rows).to_csv(base_dir / "simultaneous_summary.csv", index=False)
        pd.DataFrame(paired_rows).to_csv(base_dir / "simultaneous_paired_tests.csv", index=False)
        (base_dir / "simultaneous_confusion.json").write_text(json.dumps(cm_report, indent=2), encoding="utf-8")

    print("\n=== SIMULTANEOUS CROSS-SIGNER + UNSEEN-CLASS SUMMARY ===", flush=True)
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    print("\n=== SIMULTANEOUS PAIRED SIGNIFICANCE TESTS (HOLM-BONFERRONI) ===", flush=True)
    print(pd.DataFrame(paired_rows).to_string(index=False), flush=True)
    print("\n=== SIMULTANEOUS CONFUSION MATRIX (USER_6, UNSEEN CLASSES) ===", flush=True)
    print(json.dumps(cm_report, indent=2), flush=True)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== SIMULTANEOUS CROSS-SIGNER & UNSEEN-CLASS EVALUATION ON {device} ===", flush=True)

    (tr_s, tr_map), (va_s, va_map), (te_s, te_map) = build_simultaneous_doubly_disjoint_splits(DATA_ROOT)
    print(
        f"Doubly-disjoint splits verified (0 signer overlap, 0 class overlap, 0 MD5 overlap):\n"
        f"  Train : {len(tr_s)} imgs | Signers={sorted({s[2] for s in tr_s})} | Classes({len(tr_map)})={sorted(tr_map.keys())}\n"
        f"  Val   : {len(va_s)} imgs | Signers={sorted({s[2] for s in va_s})} | Classes({len(va_map)})={sorted(va_map.keys())}\n"
        f"  Test  : {len(te_s)} imgs | Signers={sorted({s[2] for s in te_s})} | Classes({len(te_map)})={sorted(te_map.keys())}",
        flush=True,
    )

    tr_cm = exp_mod.build_classmap(tr_s, tr_map)
    va_cm = exp_mod.build_classmap(va_s, va_map)
    te_cm = exp_mod.build_classmap(te_s, te_map)

    store84 = exp_mod.FastImageStore(target_size=84)
    store224 = exp_mod.FastImageStore(target_size=224)
    seeds = [42, 100, 2024]

    ep_csv = OUT_DIR / "simultaneous_source_of_truth_episodes.csv"
    q_csv = OUT_DIR / "simultaneous_predictions_test_3w10s.csv"
    meta_json = OUT_DIR / "simultaneous_source_of_truth_metadata.json"

    ep_rows = pd.read_csv(ep_csv).to_dict("records") if ep_csv.exists() else []
    q_rows = pd.read_csv(q_csv).to_dict("records") if q_csv.exists() else []
    meta = json.loads(meta_json.read_text(encoding="utf-8")) if meta_json.exists() else {
        "protocol": "simultaneous_cross_signer_unseen_class",
        "protocol_config": {
            "n_way_train": 3,
            "k_train": 5,
            "q_train": 15,
            "meta_iters": 200,
            "eval_settings": [
                {"name": "test_3w10s", "split": "test", "n_way": 3, "k_shot": 10, "q_eval": 15, "episodes_per_seed": 50},
                {"name": "test_3w5s",  "split": "test", "n_way": 3, "k_shot": 5,  "q_eval": 15, "episodes_per_seed": 50},
                {"name": "test_2w10s", "split": "test", "n_way": 2, "k_shot": 10, "q_eval": 15, "episodes_per_seed": 50},
                {"name": "val_3w10s",  "split": "val",  "n_way": 3, "k_shot": 10, "q_eval": 15, "episodes_per_seed": 50},
            ],
            "seeds": seeds,
            "optimizer": "Adam (weight_decay=1e-4, no LR scheduler, fixed 200-iter budget, no val/test tuning)",
            "lr_cnn_and_fusion": 1e-3,
            "lr_vit": 1e-5,
            "grad_clip_norm": 2.0,
            "mixed_precision": "torch.bfloat16 autocast on CUDA",
        },
        "splits": {
            "train_samples": len(tr_s),
            "val_samples": len(va_s),
            "test_samples": len(te_s),
            "excluded_cross_block_samples": 18000 - (len(tr_s) + len(va_s) + len(te_s)),
            "train_signers": sorted({s[2] for s in tr_s}),
            "val_signers": sorted({s[2] for s in va_s}),
            "test_signers": sorted({s[2] for s in te_s}),
            "train_classes": sorted(tr_map.keys()),
            "val_classes": sorted(va_map.keys()),
            "test_classes": sorted(te_map.keys()),
            "verified_zero_md5_overlap": True,
        },
        "runs": {},
    }
    done_runs = {(int(r["Seed"]), r["Model"]) for r in ep_rows if r["Eval_Setting"] == "test_3w10s"}

    for seed_idx, seed in enumerate(seeds):
        # Fixed paired episodes per seed (shared identically across all 5 models)
        rng_3w10s = random.Random(3000 + seed + 3 * 17)
        eps_test_3w10s = [exp_mod.sample_episode_indices(te_cm, 3, 10, 15, rng_3w10s) for _ in range(50)]

        rng_3w5s = random.Random(3100 + seed + 3 * 17)
        eps_test_3w5s = [exp_mod.sample_episode_indices(te_cm, 3, 5, 15, rng_3w5s) for _ in range(50)]

        rng_2w10s = random.Random(3200 + seed + 2 * 17)
        eps_test_2w10s = [exp_mod.sample_episode_indices(te_cm, 2, 10, 15, rng_2w10s) for _ in range(50)]

        rng_val_3w10s = random.Random(3300 + seed + 3 * 17)
        eps_val_3w10s = [exp_mod.sample_episode_indices(va_cm, 3, 10, 15, rng_val_3w10s) for _ in range(50)]

        specs = [
            ("ResNet-12_Only", lambda: exp_mod.UnifiedProtoNet(exp_mod.get_cnn_backbone("resnet12", False), use_vit_input=False)),
            ("ViT-B16_Only", lambda: exp_mod.UnifiedProtoNet(exp_mod.get_vit_backbone(True), use_vit_input=True)),
            ("Dual_Concat_Fusion", lambda: exp_mod.DualBackboneFusionProtoNet(exp_mod.get_cnn_backbone("resnet12", False), exp_mod.get_vit_backbone(True), fusion_type="concat")),
            ("Dual_Gated_Fusion", lambda: exp_mod.DualBackboneFusionProtoNet(exp_mod.get_cnn_backbone("resnet12", False), exp_mod.get_vit_backbone(True), fusion_type="gated")),
            ("Dual_Score_Fusion", lambda: exp_mod.DualBackboneFusionProtoNet(exp_mod.get_cnn_backbone("resnet12", False), exp_mod.get_vit_backbone(True), fusion_type="score")),
        ]

        for vname, builder in specs:
            if (seed, vname) in done_runs:
                print(f"  [SKIP DONE] s={seed:4d} {vname}", flush=True)
                continue

            tm = time.time()
            exp_mod.set_seed(seed)
            model = builder()
            model = exp_mod.train_protonet_model(
                model, tr_s, tr_cm, store84, store224, 3, 5, 15, device,
                meta_iters=200, lr_cnn=1e-3, lr_vit=1e-5, dual=True, seed=seed,
            )

            sha = model_weight_sha256(model)
            ckpt_rel = ""
            if vname == "Dual_Score_Fusion":
                ckpt_name = f"simultaneous_s{seed}_{vname}_fp16.pt"
                ckpt_path = CKPT_DIR / ckpt_name
                torch.save({k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu() for k, v in model.state_dict().items()}, ckpt_path)
                ckpt_rel = f"checkpoints/{ckpt_name}"
                w_val = round(float(torch.sigmoid(model.score_logit_weight).item()), 4)
                tau_val = round(float(model.tau.item()), 3)
            else:
                w_val = None
                tau_val = round(float(model.tau.item()), 3) if hasattr(model, "tau") else None

            eval_jobs = [
                ("test_3w10s", te_s, te_map, eps_test_3w10s, 3, 3000 + seed + 3 * 17, vname == "Dual_Score_Fusion"),
                ("test_3w5s",  te_s, te_map, eps_test_3w5s,  3, 3100 + seed + 3 * 17, False),
                ("test_2w10s", te_s, te_map, eps_test_2w10s, 2, 3200 + seed + 2 * 17, False),
                ("val_3w10s",  va_s, va_map, eps_val_3w10s,  3, 3300 + seed + 3 * 17, False),
            ]
            run_metrics = {}
            for sname, s_data, s_map, s_eps, n_way_e, ep_seed, rec_q in eval_jobs:
                ep_recs, q_recs = eval_episodes_detailed(
                    model, s_data, s_map, store84, store224, s_eps, n_way_e, device, record_queries=rec_q
                )
                for er in ep_recs:
                    ep_rows.append({
                        "Eval_Setting": sname,
                        "Model": vname,
                        "Seed": seed,
                        "Global_Episode_ID": seed_idx * 50 + er["Local_Episode_ID"],
                        "Local_Episode_ID": er["Local_Episode_ID"],
                        "Episode_Sampler_Seed": ep_seed,
                        "Episode_Classes": er["Episode_Classes"],
                        "Correct": er["Correct"],
                        "Total_Queries": er["Total_Queries"],
                        "Accuracy_Pct": er["Accuracy_Pct"],
                        "Learned_w": w_val if w_val is not None else "",
                        "Learned_tau": tau_val if tau_val is not None else "",
                        "Weight_SHA256": sha,
                        "Checkpoint_File": ckpt_rel,
                    })
                if rec_q:
                    for qr in q_recs:
                        qr["Seed"] = seed
                        qr["Global_Episode_ID"] = seed_idx * 50 + qr["Local_Episode_ID"]
                        q_rows.append(qr)
                run_metrics[sname] = {
                    "mean": round(float(np.mean([er["Accuracy_Pct"] for er in ep_recs])), 2),
                    "std": round(float(np.std([er["Accuracy_Pct"] for er in ep_recs], ddof=1)), 2),
                }

            meta["runs"][f"s{seed}__{vname}"] = {
                "metrics": run_metrics,
                "learned_w": w_val,
                "learned_tau": tau_val,
                "weight_sha256": sha,
                "checkpoint_file": ckpt_rel,
                "elapsed_sec": round(time.time() - tm, 1),
            }

            # Flush immediately after every model
            pd.DataFrame(ep_rows).to_csv(ep_csv, index=False)
            pd.DataFrame(ep_rows).to_csv(GIT_RES / ep_csv.name, index=False)
            if q_rows:
                pd.DataFrame(q_rows).to_csv(q_csv, index=False)
                pd.DataFrame(q_rows).to_csv(GIT_RES / q_csv.name, index=False)
            meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            (GIT_RES / meta_json.name).write_text(json.dumps(meta, indent=2), encoding="utf-8")

            extra = f" | w={w_val:.4f}, tau={tau_val:.2f}" if w_val is not None else ""
            print(
                f"  [SAVED] s={seed:4d} {vname:18s} | "
                f"Test 3w10s={run_metrics['test_3w10s']['mean']:5.2f}% "
                f"3w5s={run_metrics['test_3w5s']['mean']:5.2f}% "
                f"2w10s={run_metrics['test_2w10s']['mean']:5.2f}% | "
                f"Val(U5) 3w10s={run_metrics['val_3w10s']['mean']:5.2f}% "
                f"(sha={sha}{extra}) [{time.time()-tm:.0f}s]",
                flush=True,
            )

            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    aggregate_and_save(ep_csv, q_csv, meta_json)
    print(f"\nSIMULTANEOUS EXPERIMENT COMPLETED in {(time.time()-t0)/60:.2f} min!", flush=True)


if __name__ == "__main__":
    main()
