# Hyperspherical Dual-Backbone Score Fusion for Cross-Signer and Unseen-Class Few-Shot Indian Sign Language Recognition

[![DOI: Dataset](https://img.shields.io/badge/DOI-10.17632%2Fs6kgb6r3ss.2-blue)](https://data.mendeley.com/datasets/s6kgb6r3ss/2)
[![DOI: Code](https://zenodo.org/badge/DOI/10.5281/zenodo.19103031.svg)](https://doi.org/10.5281/zenodo.19103031)

This repository contains the official PyTorch source code, multi-seed GPU benchmark experiments, verification logs, complexity profiling scripts, and immutable source-of-truth episode records accompanying our manuscript:

> **Baby Syla L., Salim A., and Piyoosh Purushothaman Nair**, *"Hyperspherical Dual-Backbone Score Fusion for Cross-Signer and Unseen-Class Few-Shot Indian Sign Language Recognition."* (Submitted to *Journal of Electronic Imaging*, SPIE / IS&T; full citation and DOI will be updated upon publication.)

---

## 1. Overview & Key Contributions

Static Indian Sign Language (ISL) recognition under limited labeled examples faces three orthogonal generalization challenges evaluated on the 18,000-image, 20-class, 6-signer Mendeley ISL Real-Life Words benchmark:

1. **Cross-Signer Generalization (Signer-Disjoint Split):** Adapting to unseen signers (`Users 1–4` train, `User 5` val, `User 6` test; plus MD5-sanitized 6-fold Leave-One-Signer-Out across all 6 signers).
2. **Unseen-Class Recognition (Class-Disjoint Split):** Recognizing novel gesture categories from $K \in \{1, 5, 10, 20\}$ support examples (`14` train / `3` val / `3` test classes; plus `12 / 3 / 5` split for 5-way matched SOTA benchmarks).
3. **Simultaneous Doubly-Disjoint Shift (Unseen Signers $\cap$ Unseen Classes):** Strictly zero-overlap evaluation across both unseen signers and unseen classes under both same-signer and cross-signer enrollment directions.

Key methodological and verification components included in this release (`v3.0-jei-submission`):
- **Hyperspherical Score-Level Fusion:** Independent L2-normalized ResNet-12 ($84\times 84$) and ImageNet-pretrained ViT-B/16 ($224\times 224$) embeddings and unit-norm prototypes fused via a shared temperature $\tau$ and single scalar weight $w = \sigma(\alpha_w)$ frozen during meta-testing.
- **Controlled Component Ablation & $w$-Sensitivity Sweep:** Separates L2 hyperspherical normalization (`+1.64` to `+1.73` percentage points [pp] over unnormalized dual fusion, $p = 2.0\times 10^{-6}$, Cohen's $d_z = 0.408$) from scalar weight learning (`+0.09` pp over fixed $w=0.5$, $p=0.6912$, $d_z = 0.032$).
- **$2\times 2$ Pretraining & Resolution Control:** Evaluates from-scratch ViT-B/16 ($224\times 224$), from-scratch ResNet-12 ($84\times 84$), ImageNet-1K pretrained ResNet-18 ($224\times 224$), and ImageNet-1K pretrained ViT-B/16 ($224\times 224$) across 150 identical paired episodes.
- **64-Bit Perceptual dHash Cluster-Disjoint Verification:** Enforces zero shared MD5 hash and zero shared 64-bit dHash cluster between support $\mathcal{S}$ and query $\mathcal{Q}$ across Cross-Signer (`921` clusters), Unseen-Class (`790` clusters), and Simultaneous `User_6` (`152` clusters).
- **Uniform Multi-Seed Meta-Learning Grid & Paired Statistics:** 3-seed (`{42, 100, 2024}`) evaluation across ProtoNet, First-Order MAML, and ProtoMAML on Conv4, ResNet-12, ResNet-18, and ResNet-34; Holm–Bonferroni corrected paired $t$-tests, Wilcoxon signed-rank tests, and paired Cohen's $d_z$; exact CPU/GPU forward-pass profiling (`FlopCounterMode` GFLOPs/GMACs, latency, FPS); and Grad-CAM + ViT attention-rollout failure-case analysis.

---

## 2. Repository Structure

```text
ISL-RESEARCH/
├── README.md
├── requirements.txt
│
├── # Base Protocol Scripts
├── signer_split_baseline.py
├── signer_split_protonet.py
├── signer_split_maml.py
├── signer_split_protomaml.py
├── unseen_protonet.py
├── unseen_maml.py
├── unseen_protomaml.py
├── vit_baseline.py
│
└── revision_experiments/                              # Full Multi-Seed Reproducibility & Verification Suite
    ├── run_all_ijisa_revision_experiments.py          # Core multi-seed GPU suite (EXP 0–4)
    ├── run_score_fusion_sota.py                       # Matched N-way K-shot SOTA suite
    ├── run_scie_review_upgrades.py                    # 6-fold LOSO, score-anchored tests, confusion analysis
    ├── run_unified_reconciliation.py                  # Source-of-truth 1,500-episode logger + MD5 audit
    ├── recompute_from_source_of_truth.py              # Deterministic table/test recomputation from CSV logs
    ├── run_simultaneous_shift_experiment.py           # Simultaneous unseen-signer & unseen-class suite
    ├── eval_simultaneous_cross_enrollment.py          # Cross-signer enrollment direction evaluation
    ├── run_simultaneous_cross_enrollment_all5.py      # All-5-model cross-enrollment paired evaluation
    ├── run_jei_standard_upgrades.py                   # Component ablation, w-sweep, F1, Grad-CAM/Rollout, t-SNE
    ├── run_uniform_multiseed_tables46.py              # Uniform 3-seed ProtoNet/FOMAML/ProtoMAML tables & K-shot
    ├── run_reviewer_major_revision_experiments.py     # dHash cluster-disjoint, 2x2 pretraining control, R18-224, d_z
    ├── profile_complexity_latency.py                  # Params, GFLOPs/GMACs, CPU/GPU forward-pass latency & FPS
    ├── generate_jei_vol35_figures.py                  # 300-DPI grayscale-safe publication figure generator
    └── results/                                       # Immutable CSV/JSON logs & 300-DPI publication figures
        ├── source_of_truth_episodes.csv               # 1,500 paired episode records (CS & UC x 5 variants x 150 eps)
        ├── source_of_truth_metadata.json              # Seeds, split manifests, and archived weight SHA-256 hashes
        ├── source_of_truth_predictions_unseen.csv     # 6,750 unseen-class query predictions (2,250/class)
        ├── dataset_audit.json                         # 18,000 images, 20 classes, 6 signers audit
        ├── dataset_leakage_and_hw_audit.json          # Cross-signer MD5 overlap & hardware audit
        ├── hyperparameters_table.json                 # Complete training & evaluation hyperparameters
        ├── exp1_fixed_maml_protomaml_unseen.csv       # Corrected FOMAML & ProtoMAML unseen-class grid
        ├── exp1b_maml_sensitivity_resnet12.csv        # Inner-loop step & LR sensitivity ablation
        ├── exp2_fusion_ablation.csv                   # Dual-backbone fusion ablation summary
        ├── exp2_score_anchored_significance.csv       # Score-fusion-anchored paired tests
        ├── exp3_fair_sota_matched.csv                 # Matched 2w2s, 5w5s, 5w10s SOTA results
        ├── exp5_loso_6fold_cross_signer.csv           # Duplicate-sanitized 6-fold LOSO results
        ├── exp5_loso_sanitized_folds56.csv            # Sanitized Folds 5–6 verification
        ├── reconciled_table9_summary.csv              # Reconciled fusion table (from source_of_truth)
        ├── reconciled_table10_paired_tests.csv        # Holm–Bonferroni paired tests
        ├── reconciled_table13_confusion.json          # Exact per-class & per-signer error matrix (234/6,750)
        ├── simultaneous_source_of_truth_episodes.csv  # Simultaneous doubly-disjoint episode log
        ├── simultaneous_summary.csv                   # Simultaneous doubly-disjoint summary
        ├── simultaneous_paired_tests.csv              # Simultaneous paired significance tests
        ├── simultaneous_cross_enrollment_all5_*.csv   # Cross-signer enrollment paired logs & summary
        ├── jei_component_ablation_summary.csv         # Unnormalized vs. L2 Fixed w=0.5 vs. Learned (w, tau)
        ├── jei_component_ablation_episodes.csv        # 150 paired episodes per component ablation variant
        ├── jei_weight_sensitivity_sweep.csv           # Dense w in [0, 1] sensitivity sweep
        ├── jei_per_class_f1.json                      # Per-class Precision, Recall, F1 & Macro-F1 (96.52%)
        ├── jei_table4_cross_signer_multiseed.csv      # Uniform 3-seed Cross-Signer meta-learner grid
        ├── jei_table6_protonet_unseen_multiseed.csv   # Uniform 3-seed Unseen-Class ProtoNet grid
        ├── jei_unified_vit_vs_resnet12_3way_kshot.csv # Unified L2-normalized K-shot scaling curve
        ├── reviewer_dhash_cluster_disjoint.csv        # 64-bit dHash cluster-disjoint stress test results
        ├── reviewer_dhash_cluster_stats.json          # Unique dHash cluster counts per protocol/class
        ├── reviewer_pretraining_2x2_control.csv       # 2x2 Pretraining x Architecture control summary
        ├── reviewer_pretraining_2x2_episodes.csv      # 150 paired episodes for 2x2 pretraining control
        ├── reviewer_resnet18_224_profile.json         # ResNet-18 (224x224) GFLOPs/GMACs & CPU/GPU FPS
        ├── reviewer_cohens_dz.csv                     # Exact paired Cohen's d_z across all comparisons
        ├── complexity_latency_cpu.json                # CPU forward-pass latency & FPS profile
        ├── complexity_latency_gpu.json                # RTX 5050 GPU forward-pass latency & FPS profile
        └── fig_*.png                                  # 300-DPI RGB publication figures
```

---

## 3. Environment Setup & Dataset Preparation

### Dependencies
- Python 3.10+
- PyTorch 2.0+ with CUDA support (verified with `torch 2.11.0+cu128` on AMD Ryzen AI 7 350 CPU and NVIDIA GeForce RTX 5050 Laptop GPU 8 GB VRAM, BF16 autocast)

```bash
pip install -r requirements.txt
```

### Dataset
1. Download the **Indian Sign Language - Real-life Words** dataset (Tyagi & Bansal, 2022, Mendeley Data V2) from:
   **https://doi.org/10.17632/s6kgb6r3ss.2**
2. Extract it into `./data/` (expected layout: `data/<class_name>/User_<1..6>/*.jpg`, totaling **18,000 RGB images** across **20 classes** and **6 signers**).

---

## 4. Reproducing All Manuscript Results

```bash
# 1. Core multi-seed benchmark suite (Seeds {42, 100, 2024})
python revision_experiments/run_all_ijisa_revision_experiments.py --data_root ./data --seeds 42 100 2024

# 2. Deterministic recomputation of Tables & Holm-Bonferroni tests from immutable CSV logs
python revision_experiments/recompute_from_source_of_truth.py

# 3. Simultaneous doubly-disjoint & cross-signer enrollment suite
python revision_experiments/run_simultaneous_shift_experiment.py
python revision_experiments/run_simultaneous_cross_enrollment_all5.py

# 4. Component ablation, w-sensitivity sweep, F1, Grad-CAM/Attention Rollout & t-SNE
python revision_experiments/run_jei_standard_upgrades.py

# 5. Uniform 3-seed Meta-Learner grids (Tables 4 & 6) and unified K-shot curve
python revision_experiments/run_uniform_multiseed_tables46.py

# 6. Reviewer verification suite: 64-bit dHash cluster-disjoint, 2x2 pretraining control, R18-224 & Cohen's d_z
python revision_experiments/run_reviewer_major_revision_experiments.py

# 7. Standalone CPU/GPU forward-pass complexity & latency profiler
python revision_experiments/profile_complexity_latency.py
```

---

## 5. Verified Benchmark Summary (150 Paired Episodes: 3 Seeds × 50 Episodes)

### A. Dual-Backbone Fusion Ablation (3-Way 10-Shot)

| Protocol | Architecture / Fusion Variant | Accuracy (% ± ep. std) | Across-Seed (% ± std) | Paired Δ vs. Proposed (pp) | Paired Cohen's $d_z$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Cross-Signer** | ResNet-12 + ProtoNet | 92.83 ± 9.00 | 92.83 ± 1.19 | −7.07 pp | 0.785 |
| | ViT-B/16 + ProtoNet | 99.84 ± 0.90 | 99.84 ± 0.12 | −0.06 pp ($p=0.452$) | 0.062 |
| | Dual-Backbone (Concatenation) | 98.15 ± 3.58 | 98.15 ± 0.81 | −1.75 pp | 0.487 |
| | Dual-Backbone (Gated Feature Fusion) | 99.19 ± 2.31 | 99.19 ± 0.11 | −0.71 pp | 0.325 |
| | **Dual-Backbone (Weighted Score Fusion)** | **99.90 ± 0.54** | **99.90 ± 0.15** | — | — |
| **Unseen-Class** | ResNet-12 + ProtoNet | 84.01 ± 8.40 | 84.02 ± 6.68 | −12.52 pp | 1.723 |
| | ViT-B/16 + ProtoNet | 94.49 ± 4.03 | 94.49 ± 0.58 | −2.04 pp (Holm $p<10^{-9}$) | 0.541 |
| | Dual-Backbone (Concatenation) | 83.84 ± 6.97 | 83.84 ± 5.22 | −12.70 pp | 1.792 |
| | Dual-Backbone (Gated Feature Fusion) | 91.30 ± 4.96 | 91.30 ± 2.12 | −5.23 pp | 1.054 |
| | **Dual-Backbone (Weighted Score Fusion)** | **96.53 ± 3.72** | **96.53 ± 2.47** | — | — |

### B. Component Ablation, $2\times 2$ Pretraining Control & 64-Bit dHash Cluster-Disjoint Verification

| Verification Experiment | Variant / Regime | Cross-Signer (%) | Unseen-Class (%) | Key Finding |
| :--- | :--- | :---: | :---: | :--- |
| **Component Ablation** | Unnormalized Dual (Neg. Euclidean) | 99.84 ± 0.76 | 94.80 ± 3.79 | Baseline without L2 alignment |
| | L2 Hyperspherical Dual (Fixed $w=0.50$) | 99.80 ± 0.79 | 96.44 ± 3.53 | **+1.64 pp** from L2 normalization |
| | L2 Hyperspherical Dual (Learned $w,\tau$) | 99.90 ± 0.54 | 96.53 ± 3.72 | +0.09 pp vs. fixed ($p=0.691$, n.s.) |
| **$2\times 2$ Pretraining Control** | ViT-B/16 (Scratch, $224\times 224$) | 33.66 ± 2.77 | 39.96 ± 8.62 | Collapses without ImageNet-1K pretraining |
| | ResNet-18 (ImageNet-1K, $224\times 224$) | 99.51 ± 1.15 | 86.71 ± 5.95 | Trails Dual Score by −9.82 pp unseen ($d_z=1.796$) |
| **64-Bit dHash Disjoint** | Dual Weighted Score (0 dHash overlap) | **99.59 ± 1.30** | **96.43 ± 3.29** | Only −0.31 / −0.10 pp shift vs. standard |
| **6-Fold Sanitized LOSO** | Dual Weighted Score / ViT-B/16 | 99.76 ± 0.38 / 99.86 ± 0.18 | — | ViT numerically edges fusion by 0.10 pp |

### C. Forward-Pass Computational Complexity & Inference Throughput *(excludes camera capture/decoding)*

| Model | Input Res. | Params (M) | GFLOPs (GMACs) | CPU Latency (ms) | CPU FPS | RTX 5050 Latency (ms) | RTX 5050 FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv4 | $84\times 84$ | 0.11 | 0.19 (0.10) | 1.96 ± 0.41 | 509.2 | 0.73 ± 0.25 | 1375.5 |
| ResNet-12 | $84\times 84$ | 12.42 | 7.02 (3.51) | 24.30 ± 1.05 | 41.1 | 2.16 ± 0.53 | 463.4 |
| ResNet-18 | $84\times 84$ | 11.18 | 0.59 (0.30) | 13.20 ± 1.20 | 75.8 | 2.79 ± 0.71 | 357.8 |
| ResNet-18 | $224\times 224$ | 11.18 | 3.63 (1.81) | 17.04 ± 0.90 | 58.7 | 2.37 ± 0.34 | 421.4 |
| ResNet-34 | $84\times 84$ | 21.28 | 1.22 (0.61) | 22.67 ± 1.73 | 44.1 | 5.47 ± 1.09 | 182.8 |
| ViT-B/16 | $224\times 224$ | 85.80 | 22.54 (11.27) | 119.40 ± 1.84 | 8.4 | 10.07 ± 0.40 | 99.3 |
| **Dual-Backbone (Score / Gated)** | $84^2 + 224^2$ | **98.22 / 98.72** | **29.56 (14.78)** | **144.47 ± 1.72** | **6.9** | **12.93 ± 0.74** | **77.4** |

---

## 6. Contact & Citation
For questions regarding the codebase, dataset splits, or manuscript reproduction, please open an Issue on this repository or contact the corresponding author (`syla@cet.ac.in`).
