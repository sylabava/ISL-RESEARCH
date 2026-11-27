r"""
Complete 5-Model Zero-Shortcut Cross-Signer Enrollment Stress Test on Unseen Classes
====================================================================================
Strictly doubly-disjoint training split:
  Train: User_1..User_4 (4 signers) x 14 seen classes = 8,400 images
Evaluation (ZERO training signer overlap, ZERO training class overlap, ZERO support-query signer overlap):
  - Test unseen classes {afraid, bad, pain}:
      * Unseen_Test_SupU5_to_QryU6_3w10s  (Support=User_5 -> Query=User_6, 3w-10s)
      * Unseen_Test_SupU6_to_QryU5_3w10s  (Support=User_6 -> Query=User_5, 3w-10s)
      * Unseen_Test_SupU5_to_QryU6_3w5s   (Support=User_5 -> Query=User_6, 3w-5s)
      * Unseen_Test_SupU6_to_QryU5_3w5s   (Support=User_6 -> Query=User_5, 3w-5s)
  - Val unseen classes {assistance, from, warn}:
      * Unseen_Val_SupU5_to_QryU6_3w10s   (Support=User_5 -> Query=User_6, 3w-10s)
      * Unseen_Val_SupU6_to_QryU5_3w10s   (Support=User_6 -> Query=User_5, 3w-10s)

All 5 models evaluated on IDENTICAL paired episodes (3 seeds {42,100,2024} x 50 = 150 paired episodes/scenario):
  - Dual_Score_Fusion reloads its already-saved fp16 checkpoints (simultaneous_s{seed}_Dual_Score_Fusion_fp16.pt)
  - ResNet-12_Only, ViT-B16_Only, Dual_Concat_Fusion, Dual_Gated_Fusion train with identical seed/schedule & save fp16 ckpts
Computes exact episode mean/std/CI95, seed mean/std, and Score-Fusion-anchored Holm-Bonferroni paired t & Wilcoxon tests.
"""

import gc
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
from run_unified_reconciliation import model_weight_sha256
from run_simultaneous_shift_experiment import build_simultaneous_doubly_disjoint_splits, holm_bonferroni


def sample_cross_episodes(sup_cm, qry_cm, n_way, k_shot, q_query, episodes, rng, sup_offset):
    cls_list = sorted(sup_cm.keys())
    ep_list = []
    for _ in range(episodes):
        chosen = rng.sample(cls_list, n_way)
        s_idx, s_y, q_idx, q_y = [], [], [], []
        for loc, c in enumerate(chosen):
            s_pick = rng.sample(sup_cm[c], k_shot)
            q_pick = [sup_offset + idx for idx in rng.sample(qry_cm[c], q_query)]
            s_idx.extend(s_pick)
            s_y.extend([loc] * k_shot)
            q_idx.extend(q_pick)
            q_y.extend([loc] * q_query)
        ep_list.append((s_idx, s_y, q_idx, q_y))
    return ep_list


@torch.no_grad()
def eval_paired_cross_episodes(model, combined_samples, episodes, store84, store224, n_way, device):
    model.eval()
    use_amp = device.type == "cuda"
    accs = []
    for s_idx, s_y, q_idx, q_y in episodes:
        sx, sy, qx, qy = exp_mod.materialize_episode_fast(
            combined_samples, store84, store224, s_idx, s_y, q_idx, q_y, device, dual=True, training=False
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(sx, sy, qx, n_way)
        acc = round(float((logits.argmax(dim=-1) == qy).float().mean().item() * 100.0), 4)
        accs.append(acc)
    return accs


def aggregate_all5(ep_csv: Path):
    df = pd.read_csv(ep_csv)
    model_order = ["ResNet-12_Only", "ViT-B16_Only", "Dual_Concat_Fusion", "Dual_Gated_Fusion", "Dual_Score_Fusion"]
    scenarios = [
        "Unseen_Test_SupU5_to_QryU6_3w10s",
        "Unseen_Test_SupU6_to_QryU5_3w10s",
        "Unseen_Test_SupU5_to_QryU6_3w5s",
        "Unseen_Test_SupU6_to_QryU5_3w5s",
        "Unseen_Val_SupU5_to_QryU6_3w10s",
        "Unseen_Val_SupU6_to_QryU5_3w10s",
    ]

    sum_rows = []
    pair_rows = []
    for sc in scenarios:
        sub = df[df["Scenario"] == sc]
        ep_arrays = {}
        for m in model_order:
            ms = sub[sub["Model"] == m].sort_values("Global_Episode_ID")
            assert len(ms) == 150, f"Expected 150 rows for {sc}/{m}, got {len(ms)}"
            vals = ms["Accuracy_Pct"].to_numpy(dtype=float)
            ep_arrays[m] = vals
            sm = [round(float(ms[ms["Seed"] == s]["Accuracy_Pct"].mean()), 2) for s in [42, 100, 2024]]
            sum_rows.append({
                "Scenario": sc,
                "Model": m,
                "Episodes": len(vals),
                "Episode_Mean": round(float(vals.mean()), 2),
                "Episode_Std": round(float(vals.std(ddof=1)), 2),
                "Episode_CI95": round(float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))), 2),
                "Seed_Means": str(sm),
                "Seed_Mean": round(float(np.mean(sm)), 2),
                "Seed_Std": round(float(np.std(sm, ddof=1)), 2),
            })

        prop = ep_arrays["Dual_Score_Fusion"]
        baselines = ["ResNet-12_Only", "ViT-B16_Only", "Dual_Concat_Fusion", "Dual_Gated_Fusion"]
        p_rows = []
        for b in baselines:
            base = ep_arrays[b]
            diff = prop - base
            t_stat, t_p = stats.ttest_rel(prop, base)
            w_p = 1.0 if np.allclose(diff, 0.0) else float(stats.wilcoxon(prop, base, zero_method="wilcox")[1])
            p_rows.append({
                "Scenario": sc,
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
            pair_rows.append(r)

    for base_dir in [OUT_DIR, GIT_RES]:
        pd.DataFrame(sum_rows).to_csv(base_dir / "simultaneous_cross_enrollment_all5_summary.csv", index=False)
        pd.DataFrame(pair_rows).to_csv(base_dir / "simultaneous_cross_enrollment_all5_paired.csv", index=False)

    print("\n=== ALL-5 ZERO-SHORTCUT CROSS-SIGNER ENROLLMENT SUMMARY ===", flush=True)
    print(pd.DataFrame(sum_rows).to_string(index=False), flush=True)
    print("\n=== ALL-5 CROSS-SIGNER ENROLLMENT PAIRED TESTS (HOLM-BONFERRONI) ===", flush=True)
    print(pd.DataFrame(pair_rows).to_string(index=False), flush=True)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== ALL-5 ZERO-SHORTCUT CROSS-SIGNER ENROLLMENT BENCHMARK ON {device} ===", flush=True)

    (tr_s, tr_map), (_, _), (_, _) = build_simultaneous_doubly_disjoint_splits(DATA_ROOT)
    tr_cm = exp_mod.build_classmap(tr_s, tr_map)

    (_, _), (u_va, va_map), (u_te, te_map) = exp_mod.build_protocol_splits(
        DATA_ROOT, "unseen_class", seed=42, n_test_classes=3
    )
    te_u5 = [s for s in u_te if s[2] == "User_5"]
    te_u6 = [s for s in u_te if s[2] == "User_6"]
    va_u5 = [s for s in u_va if s[2] == "User_5"]
    va_u6 = [s for s in u_va if s[2] == "User_6"]

    te_u5_cm = exp_mod.build_classmap(te_u5, te_map)
    te_u6_cm = exp_mod.build_classmap(te_u6, te_map)
    va_u5_cm = exp_mod.build_classmap(va_u5, va_map)
    va_u6_cm = exp_mod.build_classmap(va_u6, va_map)

    scenario_defs = [
        ("Unseen_Test_SupU5_to_QryU6_3w10s", te_u5, te_u5_cm, te_u6, te_u6_cm, 3, 10, 15),
        ("Unseen_Test_SupU6_to_QryU5_3w10s", te_u6, te_u6_cm, te_u5, te_u5_cm, 3, 10, 15),
        ("Unseen_Test_SupU5_to_QryU6_3w5s",  te_u5, te_u5_cm, te_u6, te_u6_cm, 3, 5,  15),
        ("Unseen_Test_SupU6_to_QryU5_3w5s",  te_u6, te_u6_cm, te_u5, te_u5_cm, 3, 5,  15),
        ("Unseen_Val_SupU5_to_QryU6_3w10s",  va_u5, va_u5_cm, va_u6, va_u6_cm, 3, 10, 15),
        ("Unseen_Val_SupU6_to_QryU5_3w10s",  va_u6, va_u6_cm, va_u5, va_u5_cm, 3, 10, 15),
    ]

    store84 = exp_mod.FastImageStore(target_size=84)
    store224 = exp_mod.FastImageStore(target_size=224)
    seeds = [42, 100, 2024]

    ep_csv = OUT_DIR / "simultaneous_cross_enrollment_all5_episodes.csv"
    ep_rows = pd.read_csv(ep_csv).to_dict("records") if ep_csv.exists() else []
    done_runs = {(int(r["Seed"]), r["Model"]) for r in ep_rows if r["Scenario"] == "Unseen_Val_SupU6_to_QryU5_3w10s"}

    for seed_idx, seed in enumerate(seeds):
        # Pre-sample IDENTICAL paired episodes for all 5 models in this seed
        paired_eps = {}
        for i, (sc_name, sup_s, sup_cm, qry_s, qry_cm, n_w, k_s, q_q) in enumerate(scenario_defs):
            rng = random.Random(9500 + seed * 37 + i * 13)
            combined = list(sup_s) + list(qry_s)
            eps = sample_cross_episodes(sup_cm, qry_cm, n_w, k_s, q_q, 50, rng, sup_offset=len(sup_s))
            paired_eps[sc_name] = (combined, eps, n_w)

        specs = [
            ("ResNet-12_Only", lambda: exp_mod.UnifiedProtoNet(exp_mod.get_cnn_backbone("resnet12", False), use_vit_input=False)),
            ("ViT-B16_Only", lambda: exp_mod.UnifiedProtoNet(exp_mod.get_vit_backbone(True), use_vit_input=True)),
            ("Dual_Concat_Fusion", lambda: exp_mod.DualBackboneFusionProtoNet(exp_mod.get_cnn_backbone("resnet12", False), exp_mod.get_vit_backbone(True), fusion_type="concat")),
            ("Dual_Gated_Fusion", lambda: exp_mod.DualBackboneFusionProtoNet(exp_mod.get_cnn_backbone("resnet12", False), exp_mod.get_vit_backbone(True), fusion_type="gated")),
            ("Dual_Score_Fusion", lambda: exp_mod.DualBackboneFusionProtoNet(exp_mod.get_cnn_backbone("resnet12", False), exp_mod.get_vit_backbone(False), fusion_type="score")),
        ]

        for vname, builder in specs:
            if (seed, vname) in done_runs:
                print(f"  [SKIP DONE] s={seed:4d} {vname}", flush=True)
                continue

            tm = time.time()
            ckpt_path = CKPT_DIR / f"simultaneous_s{seed}_{vname}_fp16.pt"
            if vname == "Dual_Score_Fusion" and ckpt_path.exists():
                model = builder().to(device)
                sd = torch.load(ckpt_path, map_location=device, weights_only=True)
                model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
                source_tag = "reloaded_ckpt"
            else:
                exp_mod.set_seed(seed)
                model = builder()
                model = exp_mod.train_protonet_model(
                    model, tr_s, tr_cm, store84, store224, 3, 5, 15, device,
                    meta_iters=200, lr_cnn=1e-3, lr_vit=1e-5, dual=True, seed=seed,
                )
                torch.save({k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu() for k, v in model.state_dict().items()}, ckpt_path)
                source_tag = "trained_saved"

            sha = model_weight_sha256(model)
            means = {}
            for sc_name, *_ in scenario_defs:
                combined, eps, n_w = paired_eps[sc_name]
                accs = eval_paired_cross_episodes(model, combined, eps, store84, store224, n_w, device)
                means[sc_name] = round(float(np.mean(accs)), 2)
                for loc_i, a in enumerate(accs):
                    ep_rows.append({
                        "Scenario": sc_name,
                        "Model": vname,
                        "Seed": seed,
                        "Global_Episode_ID": seed_idx * 50 + loc_i,
                        "Local_Episode_ID": loc_i,
                        "Accuracy_Pct": a,
                        "Weight_SHA256": sha,
                    })

            pd.DataFrame(ep_rows).to_csv(ep_csv, index=False)
            pd.DataFrame(ep_rows).to_csv(GIT_RES / ep_csv.name, index=False)

            print(
                f"  [SAVED/{source_tag}] s={seed:4d} {vname:18s} | "
                f"Te U5->U6={means['Unseen_Test_SupU5_to_QryU6_3w10s']:5.2f}% "
                f"U6->U5={means['Unseen_Test_SupU6_to_QryU5_3w10s']:5.2f}% | "
                f"Va U5->U6={means['Unseen_Val_SupU5_to_QryU6_3w10s']:5.2f}% "
                f"U6->U5={means['Unseen_Val_SupU6_to_QryU5_3w10s']:5.2f}% [{time.time()-tm:.0f}s]",
                flush=True,
            )

            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    aggregate_all5(ep_csv)
    print(f"\nALL-5 CROSS-SIGNER ENROLLMENT COMPLETED in {(time.time()-t0)/60:.2f} min!", flush=True)


if __name__ == "__main__":
    main()
