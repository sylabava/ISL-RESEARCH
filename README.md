# Dual-Backbone Meta-Learning Framework for Cross-Signer and Unseen Class Indian Sign Language Recognition

[![DOI: Dataset](https://img.shields.io/badge/DOI-10.17632%2Fs6kgb6r3ss.2-blue)](https://data.mendeley.com/datasets/s6kgb6r3ss/2)
[![DOI: Code](https://zenodo.org/badge/DOI/10.5281/zenodo.19103031.svg)](https://doi.org/10.5281/zenodo.19103031)

This repository contains the official PyTorch source code, multi-seed GPU benchmark experiments, complexity profiling scripts, and verified experimental result tables accompanying our manuscript:

> **Baby Syla L., Salim A., and Piyoosh Purushothaman Nair**, *"Dual-Backbone Meta-Learning Framework for Cross-Signer and Unseen Class Indian Sign Language Recognition."* (Full journal citation and DOI will be added upon publication.)

---

## 1. Overview & Key Contributions

Static Indian Sign Language (ISL) recognition in real-world deployments faces two orthogonal generalization bottlenecks:
1. **Cross-Signer Generalization (Signer-Disjoint Split):** Adapting to completely unseen users (Users 1–4 train, User 5 validation, User 6 test; 3,000 images/signer).
2. **Unseen-Class Recognition (Class-Disjoint Split):** Recognizing novel gesture categories from only $K \in \{1, 5, 10, 20\}$ support examples (14 train / 3 validation / 3 test classes; plus 12 / 3 / 5 split for 5-way matched SOTA benchmarks).

Key components implemented in this repository:
- **Unified Hyperspherical Prototypical Metric Space:** L2-normalizes ResNet-12 and ViT-B/16 embeddings onto the unit hypersphere so squared Euclidean distance equals $2 - 2\cos(z_q, c_k)$ with learnable temperature scaling.
- **Dual-Backbone Fusion Variants:** Systematic ablation comparing standalone **ResNet-12**, standalone **ViT-B/16**, **Concatenation Fusion**, **Gated Feature Fusion**, and **Learnable Weighted Score Fusion** across 150 identical paired test episodes with paired $t$-tests and Wilcoxon signed-rank tests.
- **Corrected First-Order MAML & ProtoMAML:** Fixes gradient transfer and episodic BatchNorm synchronization across Conv4, ResNet-12, ResNet-18, and ResNet-34 on unseen gesture classes.
- **Matched $N$-Way $K$-Shot SOTA Benchmarks & CPU/GPU Profiling:** Evaluates 2-way 2-shot, 5-way 5-shot, and 5-way 10-shot protocols, and measures exact parameters, GFLOPs (`FlopCounterMode`), CPU/GPU latency, and FPS.

---

## 2. Repository Structure

```text
ISL-RESEARCH/
├── README.md
├── requirements.txt
│
├── # Original Protocol Scripts
├── signer_split_baseline.py          # Supervised CNN baselines (Cross-Signer)
├── signer_split_protonet.py          # Prototypical Networks (Cross-Signer)
├── signer_split_maml.py              # MAML (Cross-Signer)
├── signer_split_protomaml.py         # ProtoMAML (Cross-Signer)
├── unseen_protonet.py                # Prototypical Networks (Unseen-Class)
├── unseen_maml.py                    # MAML (Unseen-Class)
├── unseen_protomaml.py               # ProtoMAML (Unseen-Class)
├── vit_baseline.py                   # Vision Transformer (ViT-B/16) ProtoNet
│
└── revision_experiments/             # Multi-Seed Reproducibility & Ablation Suite
    ├── run_all_ijisa_revision_experiments.py  # Full multi-seed GPU suite (EXP 0–4)
    ├── run_score_fusion_sota.py               # Matched SOTA evaluation for Weighted Score Fusion
    ├── run_scie_review_upgrades.py            # Score-anchored paired stats, learned w, confusion & 6-fold LOSO
    ├── profile_complexity_latency.py          # Params, GFLOPs, CPU/GPU latency & FPS
    ├── regenerate_revised_figures.py          # 300-DPI manuscript figure generator
    └── results/                               # Exact verified experimental outputs
        ├── dataset_audit.json                 # 18,000 images, 20 classes, 6 signers, 0 duplicates
        ├── hyperparameters_table.json         # Complete training & evaluation hyperparameters
        ├── exp1_fixed_maml_protomaml_unseen.csv
        ├── exp2_fusion_ablation.csv
        ├── exp2_statistical_significance.csv
        ├── exp2_score_anchored_significance.csv  # Direct Weighted Score Fusion paired tests
        ├── exp3_fair_sota_matched.csv
        ├── exp5_loso_6fold_cross_signer.csv      # Full 6-fold Leave-One-Signer-Out validation
        ├── score_fusion_analysis.json            # Learned weights w, tau & 3-class unseen confusion matrix
        ├── complexity_latency_cpu.json
        ├── complexity_latency_gpu.json
        ├── fig_maml_loss_curves.png           # 300-DPI 2-panel MAML/ProtoMAML convergence plot
        └── fig_fusion_ablation.png            # 300-DPI 2-panel Dual-Backbone Fusion ablation plot
```

---

## 3. Environment Setup & Dataset Preparation

### Dependencies
- Python 3.10+
- PyTorch 2.0+ with CUDA support (tested with `torch 2.11.0+cu128` on NVIDIA GeForce RTX 5050 Laptop GPU 8 GB BF16)

```bash
pip install -r requirements.txt
```

### Dataset
1. Download the **Indian Sign Language - Real-life Words** dataset (Tyagi & Bansal, 2022, Mendeley Data V2) from:
   **https://doi.org/10.17632/s6kgb6r3ss.2**
2. Extract it into `./data/` (expected layout: `data/<class_name>/User_<1..6>/*.jpg`, totaling **18,000 RGB images** across **20 classes** and **6 signers**).

---

## 4. Reproducing All Manuscript Results

### A. Full Multi-Seed Reproducibility Suite (Seeds `{42, 100, 2024}`)
Runs Dataset Audit (`EXP 0`), Corrected FOMAML & ProtoMAML (`EXP 1`), Dual-Backbone Fusion Ablation + Paired Statistical Tests (`EXP 2`), Fair Matched SOTA Benchmarks (`EXP 3`), and GPU Latency Profiling (`EXP 4`):

```bash
python revision_experiments/run_all_ijisa_revision_experiments.py --data_root ./data --seeds 42 100 2024
```
All tables and figures are automatically written to `revision_experiments/results/` (supports automatic per-block resume if interrupted).

### B. Score-Fusion-Anchored Paired Tests, Learned Branch Weights & 6-Fold LOSO Validation
Runs direct Weighted-Score-anchored paired $t$/Wilcoxon tests against standalone ResNet-12 and ViT-B/16, logs exact learned fusion weights $w=\sigma(\alpha_w)$ and 3-class unseen confusion matrix, and executes full 6-fold Leave-One-Signer-Out cross-signer validation (`User_1`–`User_6`, 300 episodes):
```bash
python revision_experiments/run_scie_review_upgrades.py --data_root ./data
```

### C. Standalone Complexity & Latency Profiler (CPU or GPU)
```bash
python revision_experiments/profile_complexity_latency.py
```

### D. Individual Protocol Scripts
```bash
python signer_split_protonet.py
python unseen_protonet.py
python vit_baseline.py
```

---

## 5. Verified Benchmark Summary (Multi-Seed GPU Results)

### Dual-Backbone Fusion Ablation — Single Source-of-Truth (150 Paired Episodes: 3 Seeds × 50 Episodes)

| Protocol | Architecture / Fusion Variant | 2-Way 10-Shot (%) | 3-Way 10-Shot (%) | 3-Way Seed Mean ± Std (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Cross-Signer** | ResNet-12 + ProtoNet | 95.64 ± 8.86 | 92.83 ± 9.00 | 92.83 ± 1.19 |
| | ViT-B/16 + ProtoNet | 99.98 ± 0.27 | 99.84 ± 0.90 | 99.84 ± 0.12 |
| | Dual-Backbone (Concatenation) | 98.24 ± 4.10 | 98.15 ± 3.58 | 98.15 ± 0.81 |
| | Dual-Backbone (Gated Feature Fusion) | 99.53 ± 1.85 | 99.19 ± 2.31 | 99.19 ± 0.11 |
| | **Dual-Backbone (Weighted Score Fusion)** | **99.93 ± 0.47** | **99.90 ± 0.54** | **99.90 ± 0.15** |
| **Unseen-Class** | ResNet-12 + ProtoNet | 86.78 ± 15.67 | 84.01 ± 8.40 | 84.02 ± 6.68 |
| | ViT-B/16 + ProtoNet | 95.42 ± 5.15 | 94.49 ± 4.03 | 94.49 ± 0.58 |
| | Dual-Backbone (Concatenation) | 87.36 ± 11.14 | 83.84 ± 6.97 | 83.84 ± 5.22 |
| | Dual-Backbone (Gated Feature Fusion) | 91.93 ± 7.31 | 91.30 ± 4.96 | 91.30 ± 2.12 |
| | **Dual-Backbone (Weighted Score Fusion)** | **95.69 ± 5.34** | **96.53 ± 3.72** | **96.53 ± 2.47** |

### MD5-Sanitized 6-Fold Leave-One-Signer-Out (LOSO) Cross-Signer Validation (3-Way 10-Shot, 300 Episodes)

| Model | 6-Fold Macro Mean ± Std (%) | Hardest Signer (`User_2`) | Learned Branch Weight $w$ |
| :--- | :---: | :---: | :---: |
| ResNet-12 + ProtoNet (Single) | 92.90 ± 4.62 | 84.00 ± 12.39 | — |
| ViT-B/16 + ProtoNet (Single) | 99.86 ± 0.18 | 99.69 ± 1.35 | — |
| **Dual-Backbone (Weighted Score Fusion)** | **99.76 ± 0.38** | **99.56 ± 1.27** | **0.4810 ± 0.0008** |

*All 3-way 10-shot results, Holm–Bonferroni adjusted paired tests (`Score Fusion 96.53%` vs. `ViT-B/16 94.49%`, $+2.04\%$, Holm $t$-$p=5.94\times 10^{-10}$, Holm Wilcoxon $p=1.51\times 10^{-9}$), balanced learned weights ($w = 0.4585 \pm 0.0010$ unseen, $w = 0.4579 \pm 0.0023$ cross-signer), and 6,750-query per-signer confusion matrices are deterministically recomputed from `revision_experiments/results/source_of_truth_episodes.csv` via `python revision_experiments/recompute_from_source_of_truth.py`.*

### Computational Complexity & Inference Efficiency

| Model | Params (M) | GFLOPs | CPU Latency (ms) | CPU FPS | RTX 5050 Latency (ms) | RTX 5050 FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv4 ($84\times 84$) | 0.11 | 0.19 | 1.96 ± 0.41 | 509.2 | 0.73 ± 0.25 | 1375.5 |
| ResNet-12 ($84\times 84$) | 12.42 | 7.02 | 24.30 ± 1.05 | 41.1 | 2.16 ± 0.53 | 463.4 |
| ResNet-18 ($84\times 84$) | 11.18 | 0.59 | 13.20 ± 1.20 | 75.8 | 2.79 ± 0.71 | 357.8 |
| ResNet-34 ($84\times 84$) | 21.28 | 1.22 | 22.67 ± 1.73 | 44.1 | 5.47 ± 1.09 | 182.8 |
| ViT-B/16 ($224\times 224$) | 85.80 | 22.54 | 119.40 ± 1.84 | 8.4 | 10.07 ± 0.40 | 99.3 |
| **Dual-Backbone (Score / Gated)** | **98.22 / 98.72** | **29.56** | **144.47 ± 1.72** | **6.9** | **12.93 ± 0.74** | **77.4** |

---

## 6. Contact & Citation
For questions regarding the codebase, dataset splits, or manuscript reproduction, please open an Issue on this repository or contact the corresponding author.
