# Dual-Backbone Meta-Learning Framework for Cross-Signer and Unseen Class Indian Sign Language Recognition

[![DOI: Dataset](https://img.shields.io/badge/DOI-10.17632%2Fs6kgb6r3ss.2-blue)](https://data.mendeley.com/datasets/s6kgb6r3ss/2)
[![DOI: Code](https://zenodo.org/badge/DOI/10.5281/zenodo.19103031.svg)](https://doi.org/10.5281/zenodo.19103031)

This repository contains the official PyTorch source code, multi-seed GPU revision experiments, complexity profiling scripts, and verified experimental result tables for our manuscript:

> **Baby Syla L., Salim A., and Piyoosh Purushothaman Nair**, *"Dual-Backbone Meta-Learning Framework for Cross-Signer and Unseen Class Indian Sign Language Recognition,"* submitted to the **International Journal of Intelligent Systems and Applications (IJISA)**, MECS Press.

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
└── revision_experiments/             # IJISA First Revision Reproducibility Suite
    ├── run_all_ijisa_revision_experiments.py  # Full multi-seed GPU suite (EXP 0–4)
    ├── profile_complexity_latency.py          # Params, GFLOPs, CPU/GPU latency & FPS
    └── results/                               # Exact verified experimental outputs
        ├── dataset_audit.json                 # 18,000 images, 20 classes, 6 signers, 0 duplicates
        ├── hyperparameters_table.json         # Complete training & evaluation hyperparameters
        ├── exp1_fixed_maml_protomaml_unseen.csv
        ├── exp2_fusion_ablation.csv
        ├── exp2_statistical_significance.csv
        ├── exp3_fair_sota_matched.csv
        ├── complexity_latency_cpu.json
        ├── complexity_latency_gpu.json
        └── fig_maml_loss_curves.png           # 300-DPI MAML/ProtoMAML convergence plot
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

### A. Full Multi-Seed IJISA Revision Suite (Seeds `{42, 100, 2024}`)
Runs Dataset Audit (`EXP 0`), Corrected FOMAML & ProtoMAML (`EXP 1`), Dual-Backbone Fusion Ablation + Paired Statistical Tests (`EXP 2`), Fair Matched SOTA Benchmarks (`EXP 3`), and GPU Latency Profiling (`EXP 4`):

```bash
python revision_experiments/run_all_ijisa_revision_experiments.py --data_root ./data --seeds 42 100 2024
```
All tables and figures are automatically written to `revision_experiments/results/` (supports automatic per-block resume if interrupted).

### B. Standalone Complexity & Latency Profiler (CPU or GPU)
```bash
python revision_experiments/profile_complexity_latency.py
```

### C. Individual Protocol Scripts
```bash
python signer_split_protonet.py
python unseen_protonet.py
python vit_baseline.py
```

---

## 5. Verified Benchmark Summary (Multi-Seed GPU Results)

### Dual-Backbone Fusion Ablation (150 Paired Episodes: 3 Seeds × 50 Episodes)

| Protocol | Architecture / Fusion Variant | 2-Way 10-Shot (%) | 3-Way 10-Shot (%) |
| :--- | :--- | :---: | :---: |
| **Cross-Signer** | ResNet-12 + ProtoNet | 95.64 ± 8.86 | 92.83 ± 9.00 |
| | ViT-B/16 + ProtoNet | 99.98 ± 0.27 | 99.84 ± 0.90 |
| | Dual-Backbone (Concatenation) | 98.24 ± 4.10 | 98.76 ± 3.53 |
| | Dual-Backbone (Gated Feature Fusion) | 99.53 ± 1.85 | 99.16 ± 2.40 |
| | **Dual-Backbone (Weighted Score Fusion)** | **99.93 ± 0.47** | **99.67 ± 1.62** |
| **Unseen-Class** | ResNet-12 + ProtoNet | 86.78 ± 15.67 | 84.01 ± 8.40 |
| | ViT-B/16 + ProtoNet | 95.42 ± 5.15 | 94.83 ± 4.08 |
| | Dual-Backbone (Concatenation) | 87.36 ± 11.14 | 83.19 ± 6.42 |
| | Dual-Backbone (Gated Feature Fusion) | 91.93 ± 7.31 | 90.55 ± 4.94 |
| | **Dual-Backbone (Weighted Score Fusion)** | **95.69 ± 5.34** | **96.04 ± 3.36** |

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
