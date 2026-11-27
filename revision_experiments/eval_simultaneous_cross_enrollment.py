r"""
Zero-Shortcut Cross-Signer Support->Query Stress Test on Unseen Classes
=======================================================================
Uses the ALREADY-SAVED doubly-disjoint Dual_Score_Fusion checkpoints trained strictly on:
  Train: User_1..User_4 (4 signers) x 14 seen classes (8,400 images)
Evaluates on completely unseen signers {User_5, User_6} and completely unseen classes:
  - Test classes {afraid, bad, pain}:
      (a) Support = User_5  -> Query = User_6  (150 episodes, 3-way 10-shot & 3-way 5-shot)
      (b) Support = User_6  -> Query = User_5  (150 episodes, 3-way 10-shot & 3-way 5-shot)
  - Val classes {assistance, from, warn}:
      (c) Support = User_6  -> Query = User_5  (150 episodes, 3-way 10-shot)
      (d) Support = User_5  -> Query = User_6  (150 episodes, 3-way 10-shot)
Eliminates ALL intra-signer burst-capture shortcuts between support and query!
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

EXP_DIR = Path(r"C:\Users\SYLA\Desktop\RESEARCH\REVIEW1\experiments")
OUT_DIR = EXP_DIR / "revision_outputs"
CKPT_DIR = OUT_DIR / "checkpoints"
GIT_RES = Path(r"C:\Users\SYLA\Desktop\RESEARCH\GIT\revision_experiments\results")
DATA_ROOT = Path(r"C:\Users\SYLA\Desktop\RESEARCH\META1\data")

sys.path.insert(0, str(EXP_DIR))
import run_all_ijisa_revision_experiments as exp_mod


@torch.no_grad()
def eval_cross_signer_support_query(model, sup_samples, sup_cm, qry_samples, qry_cm, store84, store224, n_way, k_shot, q_query, episodes, rng, device):
    model.eval()
    accs = []
    use_amp = device.type == "cuda"
    cls_list = sorted(sup_cm.keys())
    combined = list(sup_samples) + list(qry_samples)
    offset = len(sup_samples)
    for _ in range(episodes):
        chosen = rng.sample(cls_list, n_way)
        s_idx, s_y, q_idx, q_y = [], [], [], []
        for loc, c in enumerate(chosen):
            s_pick = rng.sample(sup_cm[c], k_shot)
            q_pick = [offset + idx for idx in rng.sample(qry_cm[c], q_query)]
            s_idx.extend(s_pick)
            s_y.extend([loc] * k_shot)
            q_idx.extend(q_pick)
            q_y.extend([loc] * q_query)
        sx, sy, qx, qy = exp_mod.materialize_episode_fast(
            combined, store84, store224, s_idx, s_y, q_idx, q_y, device, dual=True, training=False
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(sx, sy, qx, n_way)
        acc = float((logits.argmax(dim=-1) == qy).float().mean().item() * 100.0)
        accs.append(acc)
    return accs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    store84 = exp_mod.FastImageStore(target_size=84)
    store224 = exp_mod.FastImageStore(target_size=224)
    seeds = [42, 100, 2024]

    scenarios = [
        ("Unseen_Test_SupU5_to_QryU6_3w10s", te_u5, te_u5_cm, te_u6, te_u6_cm, 3, 10, 15),
        ("Unseen_Test_SupU6_to_QryU5_3w10s", te_u6, te_u6_cm, te_u5, te_u5_cm, 3, 10, 15),
        ("Unseen_Test_SupU5_to_QryU6_3w5s",  te_u5, te_u5_cm, te_u6, te_u6_cm, 3, 5,  15),
        ("Unseen_Test_SupU6_to_QryU5_3w5s",  te_u6, te_u6_cm, te_u5, te_u5_cm, 3, 5,  15),
        ("Unseen_Val_SupU6_to_QryU5_3w10s",  va_u6, va_u6_cm, va_u5, va_u5_cm, 3, 10, 15),
        ("Unseen_Val_SupU5_to_QryU6_3w10s",  va_u5, va_u5_cm, va_u6, va_u6_cm, 3, 10, 15),
    ]

    results = {}
    for sc_name, *_ in scenarios:
        results[sc_name] = {"all_accs": [], "seed_means": []}

    for seed in seeds:
        ckpt_path = CKPT_DIR / f"simultaneous_s{seed}_Dual_Score_Fusion_fp16.pt"
        model = exp_mod.DualBackboneFusionProtoNet(
            exp_mod.get_cnn_backbone("resnet12", False),
            exp_mod.get_vit_backbone(False),
            fusion_type="score",
        ).to(device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        model.eval()

        for i, (sc_name, sup_s, sup_cm, qry_s, qry_cm, n_w, k_s, q_q) in enumerate(scenarios):
            rng = random.Random(9000 + seed * 37 + i * 13)
            accs = eval_cross_signer_support_query(
                model, sup_s, sup_cm, qry_s, qry_cm, store84, store224, n_w, k_s, q_q, 50, rng, device
            )
            results[sc_name]["all_accs"].extend(accs)
            results[sc_name]["seed_means"].append(round(float(np.mean(accs)), 2))

    summary = {}
    print("=== ZERO-SHORTCUT CROSS-SIGNER SUPPORT->QUERY ON UNSEEN CLASSES (Dual_Score_Fusion) ===")
    for sc_name, d in results.items():
        arr = np.asarray(d["all_accs"], dtype=float)
        s_means = d["seed_means"]
        summary[sc_name] = {
            "episodes": int(len(arr)),
            "episode_mean_pct": round(float(arr.mean()), 2),
            "episode_std_pct": round(float(arr.std(ddof=1)), 2),
            "episode_ci95_pct": round(float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr))), 2),
            "seed_means_pct": s_means,
            "seed_mean_pct": round(float(np.mean(s_means)), 2),
            "seed_std_pct": round(float(np.std(s_means, ddof=1)), 2),
        }
        r = summary[sc_name]
        print(
            f"  {sc_name:34s}: {r['episode_mean_pct']:5.2f}% ± {r['episode_std_pct']:4.2f}% "
            f"(CI95 ±{r['episode_ci95_pct']:.2f}%) | seeds={r['seed_means_pct']}"
        )

    for base_dir in [OUT_DIR, GIT_RES]:
        (base_dir / "simultaneous_cross_signer_enrollment.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
