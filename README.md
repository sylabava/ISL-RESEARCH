# Meta-Learning Framework with Vision Transformers for Enhanced Indian Sign Language Recognition

[![DOI: Dataset](https://img.shields.io/badge/DOI-10.17632%2Fs6kgb6r3ss.2-blue)](https://data.mendeley.com/datasets/s6kgb6r3ss/2)
[![DOI: Code](https://zenodo.org/badge/DOI/10.5281/zenodo.19103031.svg)](https://doi.org/10.5281/zenodo.19103031)

This repository contains the official PyTorch implementation of the open-source code for our research on **Meta-Learning Framework with Vision Transformers for Enhanced Indian Sign Language Recognition**. The models are rigorously evaluated on Indian Sign Language (ISL) datasets under Few-Shot Learning configurations.

> **Important Notice:** This codebase is directly related to our manuscript currently submitted to **The Visual Computer**. If you use this code, algorithms, or our open-source datasets in your research, we urge you to cite our relevant manuscript. (Full citation details and Journal DOI will be updated upon publication).

## 1. Project Overview & Key Algorithms
The primary objective of this research is to tackle two significant challenges in static sign language recognition: **cross-signer generalization** (adapting to unseen individuals) and **unseen class recognition** (identifying novel signs with minimal data).

We designed a dual-generalization framework utilizing the following key algorithms:
- **Prototypical Networks (ProtoNet):** A metric-based meta-learning approach. Our implementation uses a ResNet-12 backbone. It computes distances to class prototypes to handle high cross-signer variance stably.
- **Model-Agnostic Meta-Learning (MAML):** An optimization-based algorithm that injects high-velocity localized gradient mutations for rapid adaptation.
- **ProtoMAML:** A hybrid architecture combining ProtoNet's metric-based prototype initialization with MAML's gradient updates.
- **Vision Transformer (ViT-B/16):** Integrated into the ProtoNet architecture to replace traditional CNN spatial encoders. It captures global dependencies via self-attention, excelling in separating subtle variations in unrepresented classes.

## 2. Dependencies and Requirements

### System Requirements
- OS: Windows / Linux / macOS
- GPU: NVIDIA GPU with CUDA support recommended (e.g., A100, RTX series) for accelerated training.
- Python: 3.8+

### Python Dependencies
All required libraries are listed in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```
Key packages include: `torch>=1.13.0`, `torchvision`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`.

## 3. Repository Structure

The repository is modularized based on the evaluation protocol (Signer-Split vs. Unseen-Class).

**Cross-Signer Generalization (User-Disjoint Split):**
- `signer_split_baseline.py` : Standard supervised baseline training/evaluation.
- `signer_split_protonet.py` : ProtoNet implementation (ResNet architectures).
- `signer_split_maml.py` : MAML implementation.
- `signer_split_protomaml.py` : Hybrid ProtoMAML implementation.

**Novel-Class Recognition (Class-Disjoint Split):**
- `unseen_protonet.py` : ProtoNet tailored for unseen gestures.
- `unseen_maml.py` : MAML optimized for novel classes.
- `unseen_protomaml.py` : ProtoMAML on unseen classes.
- `vit_baseline.py` : Vision Transformer (ViT-B/16) integration for superior unseen-class adaptation.

## 4. Replicating the Experiments

To facilitate readers and reviewers in effortlessly replicating our experiments and evaluating the results, please follow these guidelines:

### Step 1: Dataset Preparation
1. Download the ISL Dataset from Mendeley Data: [https://data.mendeley.com/datasets/s6kgb6r3ss/2](https://data.mendeley.com/datasets/s6kgb6r3ss/2)
2. Extract the dataset into your local directory.
3. Open the corresponding Python script you wish to run (e.g., `signer_split_protonet.py`) and update the `data_dir` variable to point to your extracted dataset path.

### Step 2: Running Evaluation Baselines
To evaluate standard deep learning architectures without episodic training:
```bash
python signer_split_baseline.py
```

### Step 3: Meta-Learning Training & Inference
You can train and test specific meta-learning methods. The scripts automatically handle episodic $N$-way $K$-shot task generation.
For example, to run the **Cross-Signer generalization** using ProtoNet:
```bash
python signer_split_protonet.py
```
To run the **Unseen-Class recognition** using the Vision Transformer:
```bash
python vit_baseline.py
```

*Note: You can tweak hyperparameters like `n_way`, `k_shot`, and `num_epochs` directly inside the `if __name__ == '__main__':` block of each script.*

## 5. Contact
For any questions regarding the source code, dataset, or manuscript, please feel free to open an Issue or contact the authors.
