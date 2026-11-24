r"""
Evaluate Dual-Backbone Learnable Weighted Score Fusion (primary proposed fusion method)
on the exact same matched N-way K-shot SOTA settings as EXP 3:
  - 2-way 2-shot  (14/3/3 unseen-class split, 3 seeds x 50 episodes = 150 episodes)
  - 5-way 5-shot  (12/3/5 unseen-class split, 3 seeds x 50 episodes = 150 episodes)
  - 5-way 10-shot (12/3/5 unseen-class split, 3 seeds x 50 episodes = 150 episodes)
"""

import argparse
import gc
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_all_ijisa_revision_experiments as exp_mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--out_csv", type=str, default=str(SCRIPT_DIR / "results" / "exp3_fair_sota_matched.csv"))
    args = parser.parse_args()

    t0 = time.time()
    data_root = exp_mod.auto_detect_data_root(args.data_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Dual-Backbone Weighted Score Fusion matched SOTA on {device}...", flush=True)

    store84 = exp_mod.FastImageStore(target_size=84)
    store224 = exp_mod.FastImageStore(target_size=224)
    seeds = [42, 100, 2024]
    proto_iters = 200
    proto_eval_eps = 50
    q_query = 15

    new_rows = []

    # 1) 2-way 2-shot (14/3/3 split)
    (tr_s, tr_map), _, (te_s, te_map) = exp_mod.build_protocol_splits(
        data_root, "unseen_class", seed=42, n_test_classes=3
    )
    tr_cm = exp_mod.build_classmap(tr_s, tr_map)
    te_cm = exp_mod.build_classmap(te_s, te_map)

    scores_2w2s = []
    for seed in seeds:
        exp_mod.set_seed(seed)
        ep_rng2 = random.Random(3000 + seed)
        eps_2w2s = [exp_mod.sample_episode_indices(te_cm, 2, 2, q_query, ep_rng2) for _ in range(proto_eval_eps)]
        model = exp_mod.DualBackboneFusionProtoNet(
            exp_mod.get_cnn_backbone("resnet12", False),
            exp_mod.get_vit_backbone(True),
            fusion_type="score",
        )
        model = exp_mod.train_protonet_model(
            model, tr_s, tr_cm, store84, store224, 2, 10, q_query, device,
            meta_iters=proto_iters, lr_cnn=1e-3, lr_vit=1e-5, dual=True, seed=seed,
        )
        accs = exp_mod.eval_protonet_paired(model, te_s, store84, store224, eps_2w2s, 2, device, dual=True)
        scores_2w2s.extend(accs)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    s2 = exp_mod.summarize_scores(scores_2w2s)
    new_rows.append({
        "Setting": "2-way 2-shot", "Test_Classes": 3, "Model": "Dual_Score_Fusion_Best",
        "Test_Mean": round(s2["mean"] * 100, 2),
        "Test_Std": round(s2["std"] * 100, 2),
        "Test_CI95": round(s2["ci95"] * 100, 2),
    })

    # 2) 5-way 5-shot & 5-way 10-shot (12/3/5 split)
    (tr_s5, tr_map5), _, (te_s5, te_map5) = exp_mod.build_protocol_splits(
        data_root, "unseen_class", seed=42, n_test_classes=5
    )
    tr_cm5 = exp_mod.build_classmap(tr_s5, tr_map5)
    te_cm5 = exp_mod.build_classmap(te_s5, te_map5)

    scores_5w5s, scores_5w10s = [], []
    for seed in seeds:
        exp_mod.set_seed(seed)
        ep_rng5 = random.Random(4000 + seed)
        ep_rng10 = random.Random(5000 + seed)
        eps_5w5s = [exp_mod.sample_episode_indices(te_cm5, 5, 5, q_query, ep_rng5) for _ in range(proto_eval_eps)]
        eps_5w10s = [exp_mod.sample_episode_indices(te_cm5, 5, 10, q_query, ep_rng10) for _ in range(proto_eval_eps)]
        model = exp_mod.DualBackboneFusionProtoNet(
            exp_mod.get_cnn_backbone("resnet12", False),
            exp_mod.get_vit_backbone(True),
            fusion_type="score",
        )
        model = exp_mod.train_protonet_model(
            model, tr_s5, tr_cm5, store84, store224, 5, 5, q_query, device,
            meta_iters=proto_iters, lr_cnn=1e-3, lr_vit=1e-5, dual=True, seed=seed,
        )
        scores_5w5s.extend(exp_mod.eval_protonet_paired(model, te_s5, store84, store224, eps_5w5s, 5, device, dual=True))
        scores_5w10s.extend(exp_mod.eval_protonet_paired(model, te_s5, store84, store224, eps_5w10s, 5, device, dual=True))
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for setting, bucket in [("5-way 5-shot", scores_5w5s), ("5-way 10-shot", scores_5w10s)]:
        s = exp_mod.summarize_scores(bucket)
        new_rows.append({
            "Setting": setting, "Test_Classes": 5, "Model": "Dual_Score_Fusion_Best",
            "Test_Mean": round(s["mean"] * 100, 2),
            "Test_Std": round(s["std"] * 100, 2),
            "Test_CI95": round(s["ci95"] * 100, 2),
        })

    df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    if not df.empty:
        df = df[df["Model"] != "Dual_Score_Fusion_Best"]
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"Updated {out_csv} in {(time.time()-t0)/60:.2f} min!")


if __name__ == "__main__":
    main()
