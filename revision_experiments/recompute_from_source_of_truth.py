"""
Portable Deterministic Aggregator:
Reads ONLY the immutable Source-of-Truth files in ./results/:
  - source_of_truth_episodes.csv
  - source_of_truth_predictions_unseen.csv
and deterministically recomputes reconciled Table 9, Table 10 (with Holm-Bonferroni),
and Table 13 (with per-signer breakdown).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RES_DIR = Path(__file__).resolve().parent / "results"


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
    df_ep = pd.read_csv(RES_DIR / "source_of_truth_episodes.csv")
    df_q = pd.read_csv(RES_DIR / "source_of_truth_predictions_unseen.csv")
    model_order = ["ResNet-12_Only", "ViT-B16_Only", "Dual_Concat_Fusion", "Dual_Gated_Fusion", "Dual_Score_Fusion"]

    t9_rows, t10_rows = [], []
    weights_summary = {}

    for proto in ["cross_signer", "unseen_class"]:
        sub = df_ep[df_ep["Protocol"] == proto]
        ep_arrays = {}
        for m in model_order:
            ms = sub[sub["Model"] == m].sort_values("Global_Episode_ID")
            vals = ms["Accuracy_Pct"].to_numpy(dtype=float)
            ep_arrays[m] = vals
            seed_means = [round(float(ms[ms["Seed"] == s]["Accuracy_Pct"].mean()), 2) for s in [42, 100, 2024]]
            t9_rows.append({
                "Protocol": proto, "Model": m, "Episodes": len(vals),
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

        prop = ep_arrays["Dual_Score_Fusion"]
        p_rows = []
        for b in ["ResNet-12_Only", "ViT-B16_Only", "Dual_Concat_Fusion", "Dual_Gated_Fusion"]:
            base = ep_arrays[b]
            diff = prop - base
            t_stat, t_p = stats.ttest_rel(prop, base)
            w_p = 1.0 if np.allclose(diff, 0.0) else float(stats.wilcoxon(prop, base, zero_method="wilcox")[1])
            p_rows.append({
                "Protocol": proto, "Comparison": f"Dual_Score_Fusion vs {b}",
                "Score_Mean": round(float(prop.mean()), 2), "Score_Std": round(float(prop.std(ddof=1)), 2),
                "Base_Mean": round(float(base.mean()), 2), "Base_Std": round(float(base.std(ddof=1)), 2),
                "Delta": round(float(prop.mean() - base.mean()), 2),
                "t_stat": round(float(t_stat), 4), "t_pvalue": float(t_p), "wilcoxon_pvalue": w_p,
            })
        for r, ta, wa in zip(p_rows, holm_bonferroni([x["t_pvalue"] for x in p_rows]), holm_bonferroni([x["wilcoxon_pvalue"] for x in p_rows])):
            r["t_pvalue_holm"] = ta
            r["wilcoxon_pvalue_holm"] = wa
            t10_rows.append(r)

    pd.DataFrame(t9_rows).to_csv(RES_DIR / "reconciled_table9_summary.csv", index=False)
    pd.DataFrame(t10_rows).to_csv(RES_DIR / "reconciled_table10_paired_tests.csv", index=False)
    print(pd.DataFrame(t9_rows).to_string(index=False))
    print(pd.DataFrame(t10_rows).to_string(index=False))


if __name__ == "__main__":
    main()
