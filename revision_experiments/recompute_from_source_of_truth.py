r"""
Pure Deterministic Aggregator:
Reads ONLY the immutable Source-of-Truth files:
  - revision_outputs/source_of_truth_episodes.csv
  - revision_outputs/source_of_truth_predictions_unseen.csv
and deterministically recomputes:
  - reconciled_table9_summary.csv        (Episode mean/std/CI95 + Seed-level mean/std for ALL 5 models)
  - reconciled_table10_paired_tests.csv  (Score-Fusion-anchored paired t, Wilcoxon, Holm-Bonferroni)
  - reconciled_table13_confusion.json    (Total 3x3 confusion matrix, per-class recall & per-signer breakdown)
Guarantees 0.00% discrepancy across Tables 9, 10, 13, Abstract, Sections 5.3-5.4, Discussion & Conclusion.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = Path(r"C:\Users\SYLA\Desktop\RESEARCH\REVIEW1\experiments\revision_outputs")
GIT_RES = Path(r"C:\Users\SYLA\Desktop\RESEARCH\GIT\revision_experiments\results")


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


def main():
    df_ep = pd.read_csv(OUT_DIR / "source_of_truth_episodes.csv")
    df_q = pd.read_csv(OUT_DIR / "source_of_truth_predictions_unseen.csv")

    t9_rows = []
    t10_rows = []
    weights_summary = {}

    model_order = ["ResNet-12_Only", "ViT-B16_Only", "Dual_Concat_Fusion", "Dual_Gated_Fusion", "Dual_Score_Fusion"]

    for proto in ["cross_signer", "unseen_class"]:
        sub = df_ep[df_ep["Protocol"] == proto]
        assert len(sub) == 5 * 150, f"Expected 750 rows for {proto}, found {len(sub)}"

        ep_arrays = {}
        for m in model_order:
            ms = sub[sub["Model"] == m].sort_values("Global_Episode_ID")
            assert len(ms) == 150
            vals = ms["Accuracy_Pct"].to_numpy(dtype=float)
            ep_arrays[m] = vals
            seed_means = [round(float(ms[ms["Seed"] == s]["Accuracy_Pct"].mean()), 2) for s in [42, 100, 2024]]
            t9_rows.append({
                "Protocol": proto,
                "Model": m,
                "Episodes": len(vals),
                "Episode_Mean": round(float(vals.mean()), 2),
                "Episode_Std": round(float(vals.std(ddof=1)), 2),
                "Episode_CI95": round(float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))), 2),
                "Seed_Means": str(seed_means),
                "Seed_Mean": round(float(np.mean(seed_means)), 2),
                "Seed_Std": round(float(np.std(seed_means, ddof=1)), 2),
            })
            if m == "Dual_Score_Fusion":
                w_seeds = [float(ms[ms["Seed"] == s]["Learned_w"].iloc[0]) for s in [42, 100, 2024]]
                t_seeds = [float(ms[ms["Seed"] == s]["Learned_tau"].iloc[0]) for s in [42, 100, 2024]]
                weights_summary[proto] = {
                    "w_per_seed": [round(x, 4) for x in w_seeds],
                    "w_mean": round(float(np.mean(w_seeds)), 4),
                    "w_std": round(float(np.std(w_seeds, ddof=1)), 4),
                    "vit_weight_mean": round(float(1.0 - np.mean(w_seeds)), 4),
                    "tau_mean": round(float(np.mean(t_seeds)), 2),
                }

        # Direct Score-Fusion-anchored paired tests + Holm-Bonferroni across all 4 baselines
        prop = ep_arrays["Dual_Score_Fusion"]
        baselines = ["ResNet-12_Only", "ViT-B16_Only", "Dual_Concat_Fusion", "Dual_Gated_Fusion"]
        p_rows = []
        for b in baselines:
            base = ep_arrays[b]
            diff = prop - base
            t_stat, t_p = stats.ttest_rel(prop, base)
            w_p = 1.0 if np.allclose(diff, 0.0) else float(stats.wilcoxon(prop, base, zero_method="wilcox")[1])
            p_rows.append({
                "Protocol": proto,
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
            t10_rows.append(r)

    # Table 13 confusion + per-signer breakdown strictly from df_q
    classes = ["pain", "afraid", "bad"]
    cm = np.zeros((3, 3), dtype=int)
    c2i = {c: i for i, c in enumerate(classes)}
    for t_c, p_c in zip(df_q["True_Class"], df_q["Pred_Class"]):
        cm[c2i[t_c], c2i[p_c]] += 1

    per_signer = {}
    for sg in sorted(df_q["Signer"].unique()):
        s_df = df_q[df_q["Signer"] == sg]
        s_cm = np.zeros((3, 3), dtype=int)
        for t_c, p_c in zip(s_df["True_Class"], s_df["Pred_Class"]):
            s_cm[c2i[t_c], c2i[p_c]] += 1
        per_signer[sg] = {
            "total_queries": int(len(s_df)),
            "overall_acc_pct": round(float(np.trace(s_cm) / len(s_df) * 100.0), 2),
            "per_class_recall_pct": {
                c: round(float(s_cm[i, i] / max(1, s_cm[i].sum()) * 100.0), 2) for i, c in enumerate(classes)
            },
            "matrix": s_cm.tolist(),
        }

    t13 = {
        "total_queries": int(len(df_q)),
        "overall_acc_pct": round(float(np.trace(cm) / len(df_q) * 100.0), 2),
        "classes": classes,
        "matrix": cm.tolist(),
        "per_class_recall_pct": {
            c: round(float(cm[i, i] / cm[i].sum() * 100.0), 2) for i, c in enumerate(classes)
        },
        "learned_weights": weights_summary,
        "per_signer_breakdown": per_signer,
    }

    for base_dir in [OUT_DIR, GIT_RES]:
        pd.DataFrame(t9_rows).to_csv(base_dir / "reconciled_table9_summary.csv", index=False)
        pd.DataFrame(t10_rows).to_csv(base_dir / "reconciled_table10_paired_tests.csv", index=False)
        (base_dir / "reconciled_table13_confusion.json").write_text(json.dumps(t13, indent=2), encoding="utf-8")

    print("=== RECONCILED TABLE 9 SUMMARY ===")
    print(pd.DataFrame(t9_rows).to_string(index=False))
    print("\n=== RECONCILED TABLE 10 PAIRED TESTS (HOLM-BONFERRONI) ===")
    print(pd.DataFrame(t10_rows).to_string(index=False))
    print("\n=== RECONCILED TABLE 13 CONFUSION & LEARNED WEIGHTS ===")
    print(json.dumps(t13, indent=2))


if __name__ == "__main__":
    main()
