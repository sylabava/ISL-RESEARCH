r"""
Single Source-of-Truth GPU Reconciliation Suite (RTX 5050 GPU, BF16, Per-Model Auto-Resuming):
  PART A: Unified 200-Iteration (k_train=5, Q_train=15, Q_eval=15) 5-Variant Paired Run
          (3-way 10-shot, seeds {42, 100, 2024} x 50 identical episodes = 150 episodes/protocol)
          for both Unseen-Class and Cross-Signer protocols.
          Creates ONE immutable Source-of-Truth record per protocol:
            - source_of_truth_metadata.json   (exact splits, held-out signer/classes, preprocessing,
                                               normalization, hyperparams, episode sampler seeds,
                                               checkpoint paths, model SHA-256 weight fingerprints)
            - source_of_truth_episodes.csv    (all 1,500 episode-level records with exact IDs & accuracies)
            - source_of_truth_predictions_unseen.csv (every query prediction for Dual_Score_Fusion with signer ID)
          Saves after EVERY individual (protocol, seed, model) so zero work is ever lost.
  PART B: MAML Inner-Loop Sensitivity Mini-Ablation (ResNet-12 FOMAML, Unseen 3-way 5-shot).
  PART C: MD5-Sanitized LOSO Folds 5 & 6 Verification (excluding upstream duplicated 'doctor' frames).
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


def model_weight_sha256(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k in sorted(model.state_dict().keys()):
        t = model.state_dict()[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


@torch.no_grad()
def eval_episodes_detailed(model, samples, class_to_idx, store84, store224, episodes, n_way, device, record_queries=False):
    model.eval()
    idx_to_name = {v: k for k, v in class_to_idx.items()}
    ep_records = []
    query_records = []
    use_amp = device.type == "cuda"

    for ep_local_idx, (s_idx, s_y, q_idx, q_y) in enumerate(episodes):
        sx, sy, qx, qy = exp_mod.materialize_episode_fast(
            samples, store84, store224, s_idx, s_y, q_idx, q_y, device, dual=True, training=False
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(sx, sy, qx, n_way)
        preds = logits.argmax(dim=-1)
        correct = int((preds == qy).sum().item())
        total = int(qy.numel())
        acc_pct = round(100.0 * correct / total, 4)

        local_to_global = {int(loc): class_to_idx[samples[idx][1]] for idx, loc in zip(s_idx, s_y)}
        ep_class_names = [idx_to_name[local_to_global[c]] for c in range(n_way)]
        ep_records.append({
            "Local_Episode_ID": ep_local_idx,
            "Episode_Classes": "|".join(ep_class_names),
            "Correct": correct,
            "Total_Queries": total,
            "Accuracy_Pct": acc_pct,
        })

        if record_queries:
            for q_sidx, t_loc, p_loc in zip(q_idx, qy.cpu().tolist(), preds.cpu().tolist()):
                fpath, true_cls_str, signer = samples[q_sidx]
                pred_cls_str = idx_to_name[local_to_global[p_loc]]
                assert true_cls_str == idx_to_name[local_to_global[t_loc]]
                query_records.append({
                    "Local_Episode_ID": ep_local_idx,
                    "Signer": signer,
                    "True_Class": true_cls_str,
                    "Pred_Class": pred_cls_str,
                    "Correct": int(true_cls_str == pred_cls_str),
                    "Image_Name": Path(fpath).name,
                })
    return ep_records, query_records


def build_sanitized_loso_split(data_root: Path, test_signer: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    class_dirs = sorted(
        [p for p in data_root.iterdir() if p.is_dir() and any(d.is_dir() for d in p.iterdir())]
    )
    class_to_idx = {c.name: i for i, c in enumerate(class_dirs)}
    te_samples, te_hashes = [], set()
    for cdir in class_dirs:
        sdir = cdir / test_signer
        if sdir.is_dir():
            for f in sorted([x for x in sdir.iterdir() if x.is_file() and x.suffix.lower() in exts]):
                te_hashes.add(hashlib.md5(f.read_bytes()).hexdigest())
                te_samples.append((str(f), cdir.name, test_signer))

    tr_samples, dropped = [], 0
    for cdir in class_dirs:
        for sdir in sorted([d for d in cdir.iterdir() if d.is_dir() and d.name != test_signer]):
            for f in sorted([x for x in sdir.iterdir() if x.is_file() and x.suffix.lower() in exts]):
                h = hashlib.md5(f.read_bytes()).hexdigest()
                if h in te_hashes:
                    dropped += 1
                    continue
                tr_samples.append((str(f), cdir.name, sdir.name))
    assert {s[2] for s in tr_samples}.isdisjoint({s[2] for s in te_samples})
    return (tr_samples, class_to_idx), (te_samples, class_to_idx), dropped


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== SINGLE SOURCE-OF-TRUTH GPU RECONCILIATION SUITE ON {device} ===", flush=True)

    store84 = exp_mod.FastImageStore(target_size=84)
    store224 = exp_mod.FastImageStore(target_size=224)
    seeds = [42, 100, 2024]

    # =====================================================================
    # PART A: Source-of-Truth 200-Iter (k_train=5, Q=15) 5-Variant Paired Run
    # =====================================================================
    sot_ep_csv = OUT_DIR / "source_of_truth_episodes.csv"
    sot_q_csv = OUT_DIR / "source_of_truth_predictions_unseen.csv"
    sot_meta_json = OUT_DIR / "source_of_truth_metadata.json"

    ep_rows = pd.read_csv(sot_ep_csv).to_dict("records") if sot_ep_csv.exists() else []
    q_rows = pd.read_csv(sot_q_csv).to_dict("records") if sot_q_csv.exists() else []
    meta = json.loads(sot_meta_json.read_text(encoding="utf-8")) if sot_meta_json.exists() else {
        "protocol_config": {
            "n_way": 3,
            "k_train": 5,
            "k_eval": 10,
            "q_train": 15,
            "q_eval": 15,
            "meta_iters": 200,
            "episodes_per_seed": 50,
            "seeds": seeds,
            "total_paired_episodes_per_protocol": 150,
            "optimizer": "Adam (weight_decay=1e-4, no LR scheduler, fixed iteration budget)",
            "lr_cnn_and_fusion": 1e-3,
            "lr_vit": 1e-5,
            "grad_clip_norm": 2.0,
            "mixed_precision": "torch.bfloat16 autocast on CUDA",
            "preprocessing": "ResNet-12: 84x84 RGB; ViT-B/16: 224x224 RGB; ImageNet mean/std norm; train brightness/contrast jitter (+/-15-20%), no horizontal flip",
            "embedding_normalization": "L2 unit hypersphere normalization (p=2, dim=-1) with learnable temperature tau (init=10.0)",
        },
        "splits": {},
        "runs": {},
    }
    done_runs = {(r["Protocol"], int(r["Seed"]), r["Model"]) for r in ep_rows}

    print("\n[PART A] Running Single Source-of-Truth 200-iter (k_train=5, Q=15) 5-Variant Paired Evaluation...", flush=True)
    for protocol in ["unseen_class", "cross_signer"]:
        (tr_s, tr_map), (va_s, va_map), (te_s, te_map) = exp_mod.build_protocol_splits(
            DATA_ROOT, protocol, seed=42, n_test_classes=3
        )
        tr_cm = exp_mod.build_classmap(tr_s, tr_map)
        te_cm = exp_mod.build_classmap(te_s, te_map)

        meta["splits"][protocol] = {
            "train_samples": len(tr_s),
            "val_samples": len(va_s),
            "test_samples": len(te_s),
            "train_classes": sorted(tr_map.keys()),
            "val_classes": sorted(va_map.keys()),
            "test_classes": sorted(te_map.keys()),
            "train_signers": sorted({s[2] for s in tr_s}),
            "val_signers": sorted({s[2] for s in va_s}),
            "test_signers": sorted({s[2] for s in te_s}),
        }

        for seed_idx, seed in enumerate(seeds):
            ep_rng_seed = 2000 + seed + 3 * 17
            ep_rng = random.Random(ep_rng_seed)
            test_eps = [exp_mod.sample_episode_indices(te_cm, 3, 10, 15, ep_rng) for _ in range(50)]

            specs = [
                ("ResNet-12_Only", lambda: exp_mod.UnifiedProtoNet(exp_mod.get_cnn_backbone("resnet12", False), use_vit_input=False)),
                ("ViT-B16_Only", lambda: exp_mod.UnifiedProtoNet(exp_mod.get_vit_backbone(True), use_vit_input=True)),
                ("Dual_Concat_Fusion", lambda: exp_mod.DualBackboneFusionProtoNet(exp_mod.get_cnn_backbone("resnet12", False), exp_mod.get_vit_backbone(True), fusion_type="concat")),
                ("Dual_Gated_Fusion", lambda: exp_mod.DualBackboneFusionProtoNet(exp_mod.get_cnn_backbone("resnet12", False), exp_mod.get_vit_backbone(True), fusion_type="gated")),
                ("Dual_Score_Fusion", lambda: exp_mod.DualBackboneFusionProtoNet(exp_mod.get_cnn_backbone("resnet12", False), exp_mod.get_vit_backbone(True), fusion_type="score")),
            ]

            for vname, builder in specs:
                if (protocol, seed, vname) in done_runs:
                    print(f"  [SKIP DONE] {protocol:12s} s={seed:4d} {vname}", flush=True)
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
                    ckpt_name = f"{protocol}_s{seed}_{vname}_fp16.pt"
                    ckpt_path = CKPT_DIR / ckpt_name
                    torch.save({k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu() for k, v in model.state_dict().items()}, ckpt_path)
                    ckpt_rel = f"checkpoints/{ckpt_name}"
                    w_val = round(float(torch.sigmoid(model.score_logit_weight).item()), 4)
                    tau_val = round(float(model.tau.item()), 3)
                else:
                    w_val = None
                    tau_val = round(float(model.tau.item()), 3) if hasattr(model, "tau") else None

                rec_queries = (protocol == "unseen_class" and vname == "Dual_Score_Fusion")
                ep_recs, q_recs = eval_episodes_detailed(
                    model, te_s, te_map, store84, store224, test_eps, 3, device, record_queries=rec_queries
                )

                for er in ep_recs:
                    ep_rows.append({
                        "Protocol": protocol,
                        "Model": vname,
                        "Seed": seed,
                        "Global_Episode_ID": seed_idx * 50 + er["Local_Episode_ID"],
                        "Local_Episode_ID": er["Local_Episode_ID"],
                        "Episode_Sampler_Seed": ep_rng_seed,
                        "Episode_Classes": er["Episode_Classes"],
                        "Correct": er["Correct"],
                        "Total_Queries": er["Total_Queries"],
                        "Accuracy_Pct": er["Accuracy_Pct"],
                        "Learned_w": w_val if w_val is not None else "",
                        "Learned_tau": tau_val if tau_val is not None else "",
                        "Weight_SHA256": sha,
                        "Checkpoint_File": ckpt_rel,
                    })

                if rec_queries:
                    for qr in q_recs:
                        qr["Seed"] = seed
                        qr["Global_Episode_ID"] = seed_idx * 50 + qr["Local_Episode_ID"]
                        q_rows.append(qr)

                mean_acc = float(np.mean([er["Accuracy_Pct"] for er in ep_recs]))
                std_acc = float(np.std([er["Accuracy_Pct"] for er in ep_recs], ddof=1))
                meta["runs"][f"{protocol}__s{seed}__{vname}"] = {
                    "mean_acc_pct": round(mean_acc, 2),
                    "std_acc_pct": round(std_acc, 2),
                    "learned_w": w_val,
                    "learned_tau": tau_val,
                    "weight_sha256": sha,
                    "checkpoint_file": ckpt_rel,
                    "elapsed_sec": round(time.time() - tm, 1),
                }

                # Flush immediately after EVERY single model so zero work can be lost!
                pd.DataFrame(ep_rows).to_csv(sot_ep_csv, index=False)
                pd.DataFrame(ep_rows).to_csv(GIT_RES / "source_of_truth_episodes.csv", index=False)
                if q_rows:
                    pd.DataFrame(q_rows).to_csv(sot_q_csv, index=False)
                    pd.DataFrame(q_rows).to_csv(GIT_RES / "source_of_truth_predictions_unseen.csv", index=False)
                sot_meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                (GIT_RES / "source_of_truth_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

                extra = f" | w={w_val:.4f}, tau={tau_val:.2f}" if w_val is not None else ""
                print(
                    f"  [SAVED] {protocol:12s} s={seed:4d} {vname:18s} -> {mean_acc:5.2f}% ± {std_acc:4.2f}% "
                    f"(sha={sha}{extra}) [{time.time()-tm:.0f}s]",
                    flush=True,
                )

                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    # =====================================================================
    # PART B: MAML Inner-Loop Hyperparameter Sensitivity Mini-Ablation
    # =====================================================================
    maml_sens_csv = OUT_DIR / "exp1b_maml_sensitivity_resnet12.csv"
    if not maml_sens_csv.exists():
        print("\n[PART B] MAML Inner-Loop Sensitivity Mini-Ablation (ResNet-12 FOMAML, Unseen 3w-5s)...", flush=True)
        (tr_s, tr_map), (va_s, va_map), (te_s, te_map) = exp_mod.build_protocol_splits(
            DATA_ROOT, "unseen_class", seed=42, n_test_classes=3
        )
        tr_cm = exp_mod.build_classmap(tr_s, tr_map)
        va_cm = exp_mod.build_classmap(va_s, va_map)
        te_cm = exp_mod.build_classmap(te_s, te_map)

        configs = [
            (0.001, 0.01, 5),
            (0.01,  0.10, 1),
            (0.01,  0.10, 5),
            (0.01,  0.10, 10),
            (0.05,  0.20, 5),
        ]
        sens_rows = []
        for lr_enc, lr_clf, steps in configs:
            all_accs, sm = [], []
            tc = time.time()
            for seed in seeds:
                exp_mod.set_seed(seed)
                enc, mclf, fenc, fclf, _ = exp_mod.train_fomaml_verified(
                    "resnet12", False, tr_s, tr_cm, va_s, va_cm, store84,
                    n_way=3, k_train=5, q_query=15, device=device,
                    meta_iters=100, inner_steps=steps,
                    inner_lr_enc=lr_enc, inner_lr_clf=lr_clf, meta_lr=1e-3,
                    grad_clip=1.0, proto_init=False, seed=seed, track_curve=False,
                )
                accs = exp_mod.eval_fomaml_verified(
                    enc, mclf, fenc, fclf, te_s, te_cm, store84,
                    3, 5, 15, device, inner_steps=steps,
                    inner_lr_enc=lr_enc, inner_lr_clf=lr_clf, grad_clip=1.0,
                    proto_init=False, episodes=35, seed=1000 + seed * 31 + 5,
                )
                all_accs.extend(accs)
                sm.append(round(float(np.mean(accs) * 100.0), 2))
                del enc, mclf, fenc, fclf
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            a = np.asarray(all_accs) * 100.0
            row = {
                "Inner_LR_Enc": lr_enc, "Inner_LR_Clf": lr_clf, "Inner_Steps": steps,
                "Mean_Acc": round(float(a.mean()), 2),
                "Std_Acc": round(float(a.std(ddof=1)), 2),
                "Seed_Means": str(sm),
            }
            sens_rows.append(row)
            print(f"  MAML (lr_enc={lr_enc}, steps={steps:2d}): {row['Mean_Acc']:5.2f}% ± {row['Std_Acc']:4.2f}% | seeds={sm} ({time.time()-tc:.0f}s)", flush=True)
        pd.DataFrame(sens_rows).to_csv(maml_sens_csv, index=False)
        pd.DataFrame(sens_rows).to_csv(GIT_RES / "exp1b_maml_sensitivity_resnet12.csv", index=False)
    else:
        print("  [SKIP DONE] Part B MAML sensitivity", flush=True)

    # =====================================================================
    # PART C: MD5-Sanitized LOSO Folds 5 & 6 Verification
    # =====================================================================
    san_loso_csv = OUT_DIR / "exp5_loso_sanitized_folds56.csv"
    if not san_loso_csv.exists():
        print("\n[PART C] MD5-Sanitized LOSO Folds 5 & 6 (excluding 150 duplicated User_5/User_6 'doctor' frames)...", flush=True)
        san_rows = []
        for fold_idx, test_signer in [(5, "User_5"), (6, "User_6")]:
            (tr_s, tr_map), (te_s, te_map), dropped = build_sanitized_loso_split(DATA_ROOT, test_signer)
            print(f"  [Sanitized Fold {fold_idx}: {test_signer}] Dropped {dropped} cross-signer MD5-duplicate training frames.", flush=True)
            tr_cm = exp_mod.build_classmap(tr_s, tr_map)
            te_cm = exp_mod.build_classmap(te_s, te_map)
            exp_mod.set_seed(42 + fold_idx)
            ep_rng = random.Random(7000 + fold_idx * 13)
            test_eps = [exp_mod.sample_episode_indices(te_cm, 3, 10, 15, ep_rng) for _ in range(50)]

            fres = {"Fold": fold_idx, "Test_Signer": test_signer, "Dropped_Train_Dups": dropped}
            for vname, builder in [
                ("ResNet-12_Only", lambda: exp_mod.UnifiedProtoNet(exp_mod.get_cnn_backbone("resnet12", False), use_vit_input=False)),
                ("ViT-B16_Only", lambda: exp_mod.UnifiedProtoNet(exp_mod.get_vit_backbone(True), use_vit_input=True)),
                ("Dual_Score_Fusion", lambda: exp_mod.DualBackboneFusionProtoNet(exp_mod.get_cnn_backbone("resnet12", False), exp_mod.get_vit_backbone(True), fusion_type="score")),
            ]:
                model = builder()
                model = exp_mod.train_protonet_model(
                    model, tr_s, tr_cm, store84, store224, 3, 10, 5, device,
                    meta_iters=80, lr_cnn=1e-3, lr_vit=1e-5, dual=True, seed=42 + fold_idx,
                )
                accs = exp_mod.eval_protonet_paired(model, te_s, store84, store224, test_eps, 3, device, dual=True)
                fres[f"{vname}_Mean"] = round(float(np.mean(accs) * 100.0), 2)
                fres[f"{vname}_Std"] = round(float(np.std(accs, ddof=1) * 100.0), 2)
                if vname == "Dual_Score_Fusion":
                    fres["Learned_w"] = round(float(torch.sigmoid(model.score_logit_weight).item()), 4)
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            san_rows.append(fres)
            print(
                f"  ==> [Sanitized Fold {fold_idx} {test_signer}] "
                f"ResNet-12={fres['ResNet-12_Only_Mean']:.2f}% | ViT={fres['ViT-B16_Only_Mean']:.2f}% | "
                f"Score={fres['Dual_Score_Fusion_Mean']:.2f}% (w={fres['Learned_w']:.4f})", flush=True
            )
        pd.DataFrame(san_rows).to_csv(san_loso_csv, index=False)
        pd.DataFrame(san_rows).to_csv(GIT_RES / "exp5_loso_sanitized_folds56.csv", index=False)
    else:
        print("  [SKIP DONE] Part C sanitized LOSO", flush=True)

    print(f"\nALL SOURCE-OF-TRUTH RECONCILIATION EXPERIMENTS COMPLETED in {(time.time()-t0)/60:.2f} min!", flush=True)


if __name__ == "__main__":
    main()
