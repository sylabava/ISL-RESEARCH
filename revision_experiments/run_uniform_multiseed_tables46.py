r"""
Option B — Unified Multi-Seed Mean ± Std Suite for Tables 4 & 6 + Fig. 6 (RTX 5050 GPU, BF16):
  1. Table 6 (Unseen Class): L2-normalized ProtoNet across all 4 CNN backbones
     (Conv4, ResNet-12, ResNet-18, ResNet-34), N in {2, 3}, K in {1, 5, 10, 20},
     3 seeds {42, 100, 2024} x 35 episodes = 105 episodes/cell (matching FOMAML & ProtoMAML).
  2. Table 4 (Cross-Signer, User_6): L2-normalized ProtoNet, FOMAML, and ProtoMAML
     across all 4 CNN backbones, N in {2, 3}, K in {1, 5, 10, 20},
     3 seeds {42, 100, 2024} x 35 episodes = 105 episodes/cell.
  3. Fig. 6 (11.png): L2-normalized ViT-B/16 ProtoNet on Unseen Class 3-way, K in {1, 5, 10, 20},
     3 seeds x 50 episodes = 150 episodes/cell, paired with ResNet-12 for unified 11.png.

Auto-resumes cleanly after every (Protocol, Method, Backbone, N) group.
"""

import gc
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EXP_DIR = Path(r"C:\Users\SYLA\Desktop\RESEARCH\REVIEW1\experiments")
OUT_DIR = EXP_DIR / "revision_outputs"
GIT_RES = Path(r"C:\Users\SYLA\Desktop\RESEARCH\GIT\revision_experiments\results")
DATA_ROOT = Path(r"C:\Users\SYLA\Desktop\RESEARCH\META1\data")

sys.path.insert(0, str(EXP_DIR))
import run_all_ijisa_revision_experiments as exp_mod


def save_csv(rows, name):
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / name, index=False)
    df.to_csv(GIT_RES / name, index=False)


def run_protonet_grid(protocol: str, store84, device, out_csv_name: str, seeds, eval_eps: int = 35):
    csv_path = OUT_DIR / out_csv_name
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["Protocol"], r["Method"], r["Backbone"], int(r["N"])) for r in rows}

    (tr_s, tr_map), _, (te_s, te_map) = exp_mod.build_protocol_splits(
        DATA_ROOT, protocol, seed=42, n_test_classes=3
    )
    tr_cm = exp_mod.build_classmap(tr_s, tr_map)
    te_cm = exp_mod.build_classmap(te_s, te_map)

    for bb in ["conv4", "resnet12", "resnet18", "resnet34"]:
        pretrained = bb in ("resnet18", "resnet34")
        for n_way in [2, 3]:
            if (protocol, "ProtoNet", bb, n_way) in done:
                print(f"  [SKIP DONE] {protocol:12s} | ProtoNet  | {bb:8s} | {n_way}-way", flush=True)
                continue
            t0 = time.time()
            k_scores = {k: [] for k in [1, 5, 10, 20]}
            for seed in seeds:
                exp_mod.set_seed(seed)
                enc = exp_mod.get_cnn_backbone(bb, pretrained=pretrained)
                model = exp_mod.UnifiedProtoNet(enc, tau_init=10.0, use_vit_input=False)
                model = exp_mod.train_protonet_model(
                    model, tr_s, tr_cm, store84, None, n_way, 5, 15, device,
                    meta_iters=200, lr_cnn=1e-3, lr_vit=1e-5, dual=False, seed=seed,
                )
                for k_eval in [1, 5, 10, 20]:
                    ep_rng = random.Random(3000 + seed + n_way * 19 + k_eval * 7)
                    eps = [exp_mod.sample_episode_indices(te_cm, n_way, k_eval, 15, ep_rng) for _ in range(eval_eps)]
                    accs = exp_mod.eval_protonet_paired(model, te_s, store84, None, eps, n_way, device, dual=False)
                    k_scores[k_eval].extend(accs)
                del model, enc
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            for k_eval in [1, 5, 10, 20]:
                arr = np.asarray(k_scores[k_eval], dtype=float) * 100.0
                rows.append({
                    "Protocol": protocol,
                    "Method": "ProtoNet",
                    "Backbone": bb,
                    "N": n_way,
                    "K": k_eval,
                    "Test_Mean": round(float(arr.mean()), 2),
                    "Test_Std": round(float(arr.std(ddof=1)), 2),
                    "Test_CI95": round(float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr))), 2),
                    "N_Episodes": len(arr),
                })
            save_csv(rows, out_csv_name)
            s10 = [r for r in rows if r["Protocol"] == protocol and r["Method"] == "ProtoNet" and r["Backbone"] == bb and r["N"] == n_way and r["K"] == 10][0]
            print(f"  [{protocol:12s}] ProtoNet  | {bb:8s} | {n_way}-way K=10: {s10['Test_Mean']:5.2f}% ± {s10['Test_Std']:4.2f}% ({time.time()-t0:.0f}s)", flush=True)


def run_cross_signer_maml_grid(store84, device, out_csv_name: str, seeds, eval_eps: int = 35):
    csv_path = OUT_DIR / out_csv_name
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["Protocol"], r["Method"], r["Backbone"], int(r["N"])) for r in rows}

    (tr_s, tr_map), (va_s, va_map), (te_s, te_map) = exp_mod.build_protocol_splits(
        DATA_ROOT, "cross_signer", seed=42, n_test_classes=3
    )
    tr_cm = exp_mod.build_classmap(tr_s, tr_map)
    va_cm = exp_mod.build_classmap(va_s, va_map)
    te_cm = exp_mod.build_classmap(te_s, te_map)

    for method, proto_init in [("MAML", False), ("ProtoMAML", True)]:
        for bb in ["conv4", "resnet12", "resnet18", "resnet34"]:
            pretrained = bb in ("resnet18", "resnet34")
            for n_way in [2, 3]:
                if ("cross_signer", method, bb, n_way) in done:
                    print(f"  [SKIP DONE] cross_signer | {method:9s} | {bb:8s} | {n_way}-way", flush=True)
                    continue
                t0 = time.time()
                k_scores = {k: [] for k in [1, 5, 10, 20]}
                for seed in seeds:
                    exp_mod.set_seed(seed)
                    enc, mclf, fenc, fclf, _ = exp_mod.train_fomaml_verified(
                        bb, pretrained, tr_s, tr_cm, va_s, va_cm, store84,
                        n_way, 5, 15, device,
                        meta_iters=150, inner_steps=5,
                        inner_lr_enc=0.01, inner_lr_clf=0.1, meta_lr=1e-3,
                        grad_clip=1.0, proto_init=proto_init, seed=seed, track_curve=False,
                    )
                    for k_eval in [1, 5, 10, 20]:
                        accs = exp_mod.eval_fomaml_verified(
                            enc, mclf, fenc, fclf, te_s, te_cm, store84,
                            n_way, k_eval, 15, device,
                            inner_steps=5, inner_lr_enc=0.01, inner_lr_clf=0.1,
                            grad_clip=1.0, proto_init=proto_init,
                            episodes=eval_eps, seed=4000 + seed + n_way * 19 + k_eval * 7,
                        )
                        k_scores[k_eval].extend(accs)
                    del enc, mclf, fenc, fclf
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                for k_eval in [1, 5, 10, 20]:
                    arr = np.asarray(k_scores[k_eval], dtype=float) * 100.0
                    rows.append({
                        "Protocol": "cross_signer",
                        "Method": method,
                        "Backbone": bb,
                        "N": n_way,
                        "K": k_eval,
                        "Test_Mean": round(float(arr.mean()), 2),
                        "Test_Std": round(float(arr.std(ddof=1)), 2),
                        "Test_CI95": round(float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr))), 2),
                        "N_Episodes": len(arr),
                    })
                save_csv(rows, out_csv_name)
                s10 = [r for r in rows if r["Protocol"] == "cross_signer" and r["Method"] == method and r["Backbone"] == bb and r["N"] == n_way and r["K"] == 10][0]
                print(f"  [cross_signer] {method:9s} | {bb:8s} | {n_way}-way K=10: {s10['Test_Mean']:5.2f}% ± {s10['Test_Std']:4.2f}% ({time.time()-t0:.0f}s)", flush=True)


def run_vit_vs_resnet12_kshot(store84, store224, device, seeds):
    out_csv_name = "jei_unified_vit_vs_resnet12_3way_kshot.csv"
    csv_path = OUT_DIR / out_csv_name
    if csv_path.exists():
        print(f"  [SKIP DONE] {out_csv_name} already exists!", flush=True)
        return
    print("\n[PART 3] Running Unified L2-Normalized ViT-B/16 vs. ResNet-12 3-Way K-Shot Curve (150 eps/K)...", flush=True)
    (tr_s, tr_map), _, (te_s, te_map) = exp_mod.build_protocol_splits(
        DATA_ROOT, "unseen_class", seed=42, n_test_classes=3
    )
    tr_cm = exp_mod.build_classmap(tr_s, tr_map)
    te_cm = exp_mod.build_classmap(te_s, te_map)

    rows = []
    for bb_label, is_vit in [("ResNet-12", False), ("ViT-B/16", True)]:
        t0 = time.time()
        k_scores = {k: [] for k in [1, 5, 10, 20]}
        for seed in seeds:
            exp_mod.set_seed(seed)
            enc = exp_mod.get_vit_backbone(True) if is_vit else exp_mod.get_cnn_backbone("resnet12", False)
            model = exp_mod.UnifiedProtoNet(enc, tau_init=10.0, use_vit_input=is_vit)
            model = exp_mod.train_protonet_model(
                model, tr_s, tr_cm, store84, store224, 3, 5, 15, device,
                meta_iters=200, lr_cnn=1e-3, lr_vit=1e-5, dual=is_vit, seed=seed,
            )
            for k_eval in [1, 5, 10, 20]:
                ep_rng = random.Random(2000 + seed + 3 * 17 + (0 if k_eval == 10 else k_eval * 31))
                eps = [exp_mod.sample_episode_indices(te_cm, 3, k_eval, 15, ep_rng) for _ in range(50)]
                accs = exp_mod.eval_protonet_paired(model, te_s, store84, store224, eps, 3, device, dual=is_vit)
                k_scores[k_eval].extend(accs)
            del model, enc
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for k_eval in [1, 5, 10, 20]:
            arr = np.asarray(k_scores[k_eval], dtype=float) * 100.0
            rows.append({
                "Backbone": bb_label,
                "N": 3,
                "K": k_eval,
                "Test_Mean": round(float(arr.mean()), 2),
                "Test_Std": round(float(arr.std(ddof=1)), 2),
                "Test_CI95": round(float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr))), 2),
                "N_Episodes": len(arr),
            })
        print(f"  [Unseen 3-way] {bb_label:9s}: " + " | ".join(
            f"K={k}:{np.mean(k_scores[k])*100:.2f}±{np.std(k_scores[k], ddof=1)*100:.2f}%"
            for k in [1, 5, 10, 20]
        ) + f" ({time.time()-t0:.0f}s)", flush=True)
    save_csv(rows, out_csv_name)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== OPTION B: UNIFIED MULTI-SEED MEAN±STD SUITE (TABLES 4 & 6 + FIG. 6) ON {device} ===", flush=True)
    store84 = exp_mod.FastImageStore(target_size=84)
    store224 = exp_mod.FastImageStore(target_size=224)
    seeds = [42, 100, 2024]

    print("\n[PART 1] Unseen-Class ProtoNet (4 CNN Backbones x 2/3-way x 4 K-shots, 105 eps/cell)...", flush=True)
    run_protonet_grid("unseen_class", store84, device, "jei_table6_protonet_unseen_multiseed.csv", seeds, eval_eps=35)

    print("\n[PART 2] Cross-Signer ProtoNet, FOMAML, ProtoMAML (4 CNN Backbones x 2/3-way x 4 K-shots, 105 eps/cell)...", flush=True)
    run_protonet_grid("cross_signer", store84, device, "jei_table4_cross_signer_multiseed.csv", seeds, eval_eps=35)
    run_cross_signer_maml_grid(store84, device, "jei_table4_cross_signer_multiseed.csv", seeds, eval_eps=35)

    run_vit_vs_resnet12_kshot(store84, store224, device, seeds)
    print(f"\nALL OPTION B UNIFIED MULTI-SEED EXPERIMENTS FINISHED in {(time.time()-t0)/60:.2f} min!", flush=True)


if __name__ == "__main__":
    main()
