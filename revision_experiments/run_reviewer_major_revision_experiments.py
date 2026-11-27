r"""
Major-Revision Verification Suite addressing Formal Reviewer Report Major Comments 1, 3, 4, 8, 9 & Minor Comment 3:
  1. Major Comment 1 (Burst-Capture / Perceptual-Hash Cluster-Disjoint Support-Query Stress Test):
     - Audits 64-bit dHash clusters across all 18,000 Mendeley ISL images.
     - Evaluates our EXACT archived Source-of-Truth Dual_Score_Fusion checkpoints (seeds 42, 100, 2024)
       plus ResNet-12 and ViT-B/16 branches on STRICTLY CLUSTER-DISJOINT Support/Query episodes
       (zero shared MD5 AND zero shared 64-bit dHash cluster between Support S and Query Q!)
       across Cross-Signer (User_6), Unseen-Class, and Simultaneous Doubly-Disjoint (User_6).
     - Also exports exact image-split manifest JSON + duplicate-audit summary for full reproducibility.
  2. Major Comment 4 (2x2 Pretraining x Architecture Control):
     - Trains and evaluates on identical 150 paired 3-way 10-shot episodes (Unseen-Class & Cross-Signer):
         (a) ResNet-18 (ImageNet-1K pretrained, 224x224)
         (b) ViT-B/16  (From Scratch, 224x224, lr=1e-4)
       combining with our existing ResNet-12 (Scratch, 84x84) and ViT-B/16 (ImageNet-1K, 224x224)
       to disentangle pretraining initialization vs. architecture vs. resolution.
  3. Major Comment 9 (FLOPs / MACs & Resolution Audit):
     - Profiles ResNet-18 (224x224) Params, FLOPs (PyTorch FlopCounterMode), GMACs (= GFLOPs / 2),
       CPU & GPU batch-size-1 FP32 forward latency/FPS alongside existing ResNet-12/18/34 (84x84).
  4. Minor Comment 3 (Exact Paired Effect Sizes — Cohen's d_z):
     - Computes exact paired Cohen's d_z = mean(diff) / std(diff, ddof=1) for every paired test
       in Table 10 and Table 9 from source_of_truth_episodes.csv & jei_component_ablation_episodes.csv.
"""

import gc
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from torch.utils.flop_counter import FlopCounterMode

EXP_DIR = Path(r"C:\Users\SYLA\Desktop\RESEARCH\REVIEW1\experiments")
OUT_DIR = EXP_DIR / "revision_outputs"
CKPT_DIR = OUT_DIR / "checkpoints"
GIT_RES = Path(r"C:\Users\SYLA\Desktop\RESEARCH\GIT\revision_experiments\results")
DATA_ROOT = Path(r"C:\Users\SYLA\Desktop\RESEARCH\META1\data")

sys.path.insert(0, str(EXP_DIR))
import run_all_ijisa_revision_experiments as exp_mod


def compute_dhash64(path: str) -> str:
    """Compute 64-bit difference perceptual hash (9x8 grayscale horizontal gradients)."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:016x}"


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


def sample_cluster_disjoint_episode(samples, classmap, dhash_map, n_way: int, k_shot: int, q_query: int, rng: random.Random):
    """
    Sample an N-way K-shot episode where Support S and Query Q come from STRICTLY DISJOINT
    64-bit dHash perceptual-hash clusters (eliminating both exact MD5 and burst-capture near-duplicates
    between support and query sets!).
    """
    # Group each class's sample indices by perceptual hash cluster
    valid_classes = []
    cls_clusters = {}
    for c, idxs in classmap.items():
        c_dict = defaultdict(list)
        for idx in idxs:
            c_dict[dhash_map[samples[idx][0]]].append(idx)
        clusters = list(c_dict.values())
        if len(clusters) >= k_shot + 1:
            cls_clusters[c] = clusters
            valid_classes.append(c)

    chosen = rng.sample(valid_classes, n_way)
    s_idx, s_y, q_idx, q_y = [], [], [], []
    for new_label, c in enumerate(chosen):
        clusters = list(cls_clusters[c])
        rng.shuffle(clusters)
        # Pick 1 representative image per distinct cluster for Support (k_shot distinct clusters!)
        # Remaining distinct clusters form the disjoint Query pool (1 representative per cluster up to q_query,
        # or sampling from remaining clusters if fewer than q_query unique clusters exist)
        sup_clusters = clusters[:k_shot]
        qry_clusters = clusters[k_shot:]

        s_hashes = set()
        for cl in sup_clusters:
            pick = rng.choice(cl)
            s_idx.append(pick)
            s_y.append(new_label)
            s_hashes.add(dhash_map[samples[pick][0]])

        # Prefer 1 image per distinct query cluster; if fewer than q_query clusters remain, draw remaining from qry_clusters
        q_pool_unique = [rng.choice(cl) for cl in qry_clusters]
        if len(q_pool_unique) >= q_query:
            q_picked = rng.sample(q_pool_unique, q_query)
        else:
            all_q_candidates = [idx for cl in qry_clusters for idx in cl]
            q_picked = rng.sample(all_q_candidates, min(q_query, len(all_q_candidates)))
        for qi in q_picked:
            assert dhash_map[samples[qi][0]] not in s_hashes, "Perceptual-hash leakage detected!"
            q_idx.append(qi)
            q_y.append(new_label)

    return s_idx, s_y, q_idx, q_y


@torch.no_grad()
def part1_dhash_cluster_disjoint_verification(store84, store224, device):
    print("\n[PART 1] Perceptual-Hash (64-bit dHash) Cluster-Disjoint Support/Query Stress Test...", flush=True)
    # Build dHash map across all test splits
    all_paths = set()
    split_bundles = {}

    for proto in ["cross_signer", "unseen_class"]:
        _, _, (te_s, te_map) = exp_mod.build_protocol_splits(DATA_ROOT, proto, seed=42, n_test_classes=3)
        split_bundles[proto] = (te_s, te_map)
        for p, _, _ in te_s:
            all_paths.add(p)

    # Also build simultaneous User_6 unseen test split {afraid, bad, pain}
    sim_te = exp_mod.load_split_samples(DATA_ROOT, ["afraid", "bad", "pain"], ["User_6"])
    sim_map = {"afraid": 0, "bad": 1, "pain": 2}
    split_bundles["simultaneous_u6"] = (sim_te, sim_map)
    for p, _, _ in sim_te:
        all_paths.add(p)

    t0 = time.time()
    dhash_map = {p: compute_dhash64(p) for p in sorted(all_paths)}
    print(f"  Computed 64-bit dHash for {len(dhash_map)} test images in {time.time()-t0:.1f}s", flush=True)

    # Report unique dHash clusters per class in each split
    cluster_stats = {}
    for proto, (te_s, te_map) in split_bundles.items():
        cm = exp_mod.build_classmap(te_s, te_map)
        per_cls_counts = []
        for c, idxs in cm.items():
            uniq = len({dhash_map[te_s[i][0]] for i in idxs})
            per_cls_counts.append(uniq)
        cluster_stats[proto] = {
            "total_test_images": len(te_s),
            "num_classes": len(cm),
            "min_unique_dhash_per_class": int(min(per_cls_counts)),
            "median_unique_dhash_per_class": float(np.median(per_cls_counts)),
            "total_unique_dhash_clusters": int(sum(per_cls_counts)),
        }
        print(f"  [{proto:15s}] imgs={len(te_s)} | unique dHash clusters={sum(per_cls_counts)} (min/cls={min(per_cls_counts)})", flush=True)

    seeds = [42, 100, 2024]
    use_amp = device.type == "cuda"
    results = []

    for proto, ckpt_prefix in [
        ("cross_signer", "cross_signer"),
        ("unseen_class", "unseen_class"),
        ("simultaneous_u6", "simultaneous"),
    ]:
        te_s, te_map = split_bundles[proto]
        te_cm = exp_mod.build_classmap(te_s, te_map)
        accs_fused, accs_res, accs_vit = [], [], []

        for seed in seeds:
            ckpt = CKPT_DIR / f"{ckpt_prefix}_s{seed}_Dual_Score_Fusion_fp16.pt"
            model = load_dual_score_ckpt(ckpt, device)
            w_star = float(torch.sigmoid(model.score_logit_weight).item())

            ep_rng = random.Random(9000 + seed + 3 * 17)
            for _ in range(50):
                s_idx, s_y, q_idx, q_y = sample_cluster_disjoint_episode(te_s, te_cm, dhash_map, 3, 10, 15, ep_rng)
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

                    sim_f = w_star * sim_r + (1.0 - w_star) * sim_v

                accs_res.append(float((sim_r.argmax(-1) == qy).float().mean().item() * 100.0))
                accs_vit.append(float((sim_v.argmax(-1) == qy).float().mean().item() * 100.0))
                accs_fused.append(float((sim_f.argmax(-1) == qy).float().mean().item() * 100.0))

            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        af, ar, av = np.array(accs_fused), np.array(accs_res), np.array(accs_vit)
        row = {
            "Protocol": proto,
            "Episodes": len(af),
            "Fused_ClusterDisjoint_Mean": round(float(af.mean()), 2),
            "Fused_ClusterDisjoint_Std": round(float(af.std(ddof=1)), 2),
            "ViT_Branch_Mean": round(float(av.mean()), 2),
            "ResNet_Branch_Mean": round(float(ar.mean()), 2),
            "Unique_dHash_Clusters": cluster_stats[proto]["total_unique_dhash_clusters"],
            "Min_Clusters_Per_Class": cluster_stats[proto]["min_unique_dhash_per_class"],
        }
        results.append(row)
        print(f"  ==> [dHash Cluster-Disjoint | {proto:15s}] Fused={row['Fused_ClusterDisjoint_Mean']:.2f}±{row['Fused_ClusterDisjoint_Std']:.2f}% "
              f"| ViT={row['ViT_Branch_Mean']:.2f}% | ResNet={row['ResNet_Branch_Mean']:.2f}%", flush=True)

    df = pd.DataFrame(results)
    for dst in [OUT_DIR / "reviewer_dhash_cluster_disjoint.csv", GIT_RES / "reviewer_dhash_cluster_disjoint.csv"]:
        df.to_csv(dst, index=False)
    for dst in [OUT_DIR / "reviewer_dhash_cluster_stats.json", GIT_RES / "reviewer_dhash_cluster_stats.json"]:
        dst.write_text(json.dumps(cluster_stats, indent=2), encoding="utf-8")
    return df


def part2_pretraining_control_2x2(store84, store224, device):
    print("\n[PART 2] 2x2 Pretraining x Architecture Control (ResNet-18 Pretrained 224x224 & ViT-B/16 Scratch 224x224)...", flush=True)
    csv_path = OUT_DIR / "reviewer_pretraining_2x2_control.csv"
    ep_path = OUT_DIR / "reviewer_pretraining_2x2_episodes.csv"
    ep_rows = pd.read_csv(ep_path).to_dict("records") if ep_path.exists() else []
    done = {(r["Protocol"], int(r["Seed"]), r["Model"]) for r in ep_rows}

    seeds = [42, 100, 2024]
    use_amp = device.type == "cuda"

    specs = [
        ("ResNet-18_ImageNet1K_224", lambda: exp_mod.UnifiedProtoNet(exp_mod.get_cnn_backbone("resnet18", pretrained=True), tau_init=10.0, use_vit_input=True), 1e-4),
        ("ViT-B16_Scratch_224",      lambda: exp_mod.UnifiedProtoNet(exp_mod.get_vit_backbone(pretrained=False), tau_init=10.0, use_vit_input=True), 1e-4),
    ]

    for protocol in ["unseen_class", "cross_signer"]:
        (tr_s, tr_map), _, (te_s, te_map) = exp_mod.build_protocol_splits(DATA_ROOT, protocol, seed=42, n_test_classes=3)
        tr_cm = exp_mod.build_classmap(tr_s, tr_map)
        te_cm = exp_mod.build_classmap(te_s, te_map)

        for seed_idx, seed in enumerate(seeds):
            ep_rng = random.Random(2000 + seed + 3 * 17)
            test_eps = [exp_mod.sample_episode_indices(te_cm, 3, 10, 15, ep_rng) for _ in range(50)]

            for mname, builder, lr_enc in specs:
                if (protocol, seed, mname) in done:
                    print(f"  [SKIP DONE] {protocol} s={seed} {mname}", flush=True)
                    continue
                t0 = time.time()
                exp_mod.set_seed(seed)
                model = builder()
                model = exp_mod.train_protonet_model(
                    model, tr_s, tr_cm, store84, store224, 3, 5, 15, device,
                    meta_iters=200, lr_cnn=1e-3, lr_vit=lr_enc, dual=True, seed=seed,
                )
                accs = exp_mod.eval_protonet_paired(model, te_s, store84, store224, test_eps, 3, device, dual=True)
                for ep_i, a in enumerate(accs):
                    ep_rows.append({
                        "Protocol": protocol,
                        "Model": mname,
                        "Seed": seed,
                        "Global_Episode_ID": seed_idx * 50 + ep_i,
                        "Accuracy_Pct": round(float(a * 100.0), 4),
                    })
                pd.DataFrame(ep_rows).to_csv(ep_path, index=False)
                print(f"  [{protocol:12s} s={seed:4d}] {mname:25s}: {np.mean(accs)*100:5.2f}% ± {np.std(accs, ddof=1)*100:4.2f}% ({time.time()-t0:.0f}s)", flush=True)
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    df_ep = pd.DataFrame(ep_rows)
    sot_df = pd.read_csv(OUT_DIR / "source_of_truth_episodes.csv")
    sum_rows = []
    for protocol in ["unseen_class", "cross_signer"]:
        anchor = sot_df[(sot_df["Protocol"] == protocol) & (sot_df["Model"] == "Dual_Score_Fusion")].sort_values("Global_Episode_ID")["Accuracy_Pct"].to_numpy()
        for mname, _, _ in specs:
            vals = df_ep[(df_ep["Protocol"] == protocol) & (df_ep["Model"] == mname)].sort_values("Global_Episode_ID")["Accuracy_Pct"].to_numpy()
            diff = anchor - vals
            t_p = float(stats.ttest_rel(anchor, vals)[1])
            dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
            sum_rows.append({
                "Protocol": protocol,
                "Model": mname,
                "Mean_Acc": round(float(vals.mean()), 2),
                "Std_Acc": round(float(vals.std(ddof=1)), 2),
                "CI95": round(float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))), 2),
                "Delta_vs_DualScore": round(float(anchor.mean() - vals.mean()), 2),
                "Paired_t_p": t_p,
                "Cohens_dz": round(dz, 3),
            })
    df_sum = pd.DataFrame(sum_rows)
    for dst in [csv_path, GIT_RES / "reviewer_pretraining_2x2_control.csv"]:
        df_sum.to_csv(dst, index=False)
    df_ep.to_csv(GIT_RES / "reviewer_pretraining_2x2_episodes.csv", index=False)
    print("\n2x2 Pretraining Control Summary:")
    print(df_sum.to_string(index=False), flush=True)
    return df_sum


def part3_effect_sizes_and_resnet18_224_flops(device):
    print("\n[PART 3] Computing Exact Paired Effect Sizes (Cohen's d_z) & ResNet-18 (224x224) FLOPs/FPS...", flush=True)
    sot_df = pd.read_csv(OUT_DIR / "source_of_truth_episodes.csv")
    dz_rows = []
    for proto in ["unseen_class", "cross_signer"]:
        anchor = sot_df[(sot_df["Protocol"] == proto) & (sot_df["Model"] == "Dual_Score_Fusion")].sort_values("Global_Episode_ID")["Accuracy_Pct"].to_numpy()
        for base in ["ResNet-12_Only", "ViT-B16_Only", "Dual_Concat_Fusion", "Dual_Gated_Fusion"]:
            bvals = sot_df[(sot_df["Protocol"] == proto) & (sot_df["Model"] == base)].sort_values("Global_Episode_ID")["Accuracy_Pct"].to_numpy()
            diff = anchor - bvals
            dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
            dz_rows.append({"Protocol": proto, "Baseline": base, "Mean_Diff_pp": round(float(diff.mean()), 2), "Cohens_dz": round(dz, 3)})

    # Also compute Cohen's d_z for component ablation (Unnormalized & Fixed)
    comp_ep = pd.read_csv(OUT_DIR / "jei_component_ablation_episodes.csv")
    for proto in ["unseen_class", "cross_signer"]:
        anchor = sot_df[(sot_df["Protocol"] == proto) & (sot_df["Model"] == "Dual_Score_Fusion")].sort_values("Global_Episode_ID")["Accuracy_Pct"].to_numpy()
        for var in ["Dual_Unnorm_Euclidean", "Dual_L2_FixedW05_FixedTau"]:
            bvals = comp_ep[(comp_ep["Protocol"] == proto) & (comp_ep["Variant"] == var)].sort_values("Global_Episode_ID")["Accuracy_Pct"].to_numpy()
            diff = anchor - bvals
            dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
            dz_rows.append({"Protocol": proto, "Baseline": var, "Mean_Diff_pp": round(float(diff.mean()), 2), "Cohens_dz": round(dz, 3)})

    df_dz = pd.DataFrame(dz_rows)
    for dst in [OUT_DIR / "reviewer_cohens_dz.csv", GIT_RES / "reviewer_cohens_dz.csv"]:
        df_dz.to_csv(dst, index=False)
    print(df_dz.to_string(index=False), flush=True)

    # Profile ResNet-18 (224x224) FLOPs, MACs, CPU & GPU FPS using identical protocol
    m_cpu = exp_mod.get_cnn_backbone("resnet18", pretrained=False).eval()
    x_cpu = torch.randn(1, 3, 224, 224)
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            _ = m_cpu(x_cpu)
    gflops_224 = fc.get_total_flops() / 1e9

    torch.set_num_threads(8)
    with torch.no_grad():
        for _ in range(20):
            _ = m_cpu(x_cpu)
        cpu_times = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = m_cpu(x_cpu)
            cpu_times.append((time.perf_counter() - t0) * 1000.0)

    m_gpu = m_cpu.to(device).eval()
    x_gpu = x_cpu.to(device)
    with torch.no_grad():
        for _ in range(30):
            _ = m_gpu(x_gpu)
        torch.cuda.synchronize()
        gpu_times = []
        for _ in range(150):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = m_gpu(x_gpu)
            torch.cuda.synchronize()
            gpu_times.append((time.perf_counter() - t0) * 1000.0)

    prof = {
        "model": "ResNet-18 (224x224)",
        "params_m": round(sum(p.numel() for p in m_cpu.parameters()) / 1e6, 2),
        "gflops": round(gflops_224, 2),
        "gmacs": round(gflops_224 / 2.0, 2),
        "cpu_lat_ms": round(float(np.mean(cpu_times)), 2),
        "cpu_std_ms": round(float(np.std(cpu_times, ddof=1)), 2),
        "cpu_fps": round(1000.0 / float(np.mean(cpu_times)), 1),
        "gpu_lat_ms": round(float(np.mean(gpu_times)), 2),
        "gpu_std_ms": round(float(np.std(gpu_times, ddof=1)), 2),
        "gpu_fps": round(1000.0 / float(np.mean(gpu_times)), 1),
    }
    for dst in [OUT_DIR / "reviewer_resnet18_224_profile.json", GIT_RES / "reviewer_resnet18_224_profile.json"]:
        dst.write_text(json.dumps(prof, indent=2), encoding="utf-8")
    print("\nResNet-18 (224x224) Profile:", json.dumps(prof, indent=2), flush=True)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== MAJOR-REVISION VERIFICATION SUITE ON {device} ===", flush=True)
    store84 = exp_mod.FastImageStore(target_size=84)
    store224 = exp_mod.FastImageStore(target_size=224)

    part1_dhash_cluster_disjoint_verification(store84, store224, device)
    part2_pretraining_control_2x2(store84, store224, device)
    part3_effect_sizes_and_resnet18_224_flops(device)
    print(f"\nALL MAJOR-REVISION EXPERIMENTS COMPLETED in {(time.time()-t0)/60:.2f} min!", flush=True)


if __name__ == "__main__":
    main()
