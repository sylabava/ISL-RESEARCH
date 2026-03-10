# ISL-RESEARCH: Few-Shot Learning for Indian Sign Language

This repository contains the codebase for our research on **Meta-Learning Framework with Vision Transformers for Enhanced Indian Sign Language Recognitions** using Meta-Learning and Few-Shot Learning approaches. The models are evaluated on Indian Sign Language (ISL) datasets.

## Project Overview
The main goal of this research is to tackle the challenges of cross-signer generalization and unseen class recognition in static sign language recognition. We explored several state-of-the-art meta-learning algorithms, including:
- **ProtoNet (Prototypical Networks)** with ResNet-12 backbone for Cross-Signer validation.
- **MAML (Model-Agnostic Meta-Learning)**
- **ProtoMAML**
- **Vision Transformer (ViT)** for base and unseen class recognition.

## Repository Structure
- `signer_split_baseline.py` : Base model architecture for Signer-Split protocol.
- `signer_split_protonet.py` : Prototypical Networks implementation for Cross-Signer validation.
- `signer_split_maml.py` : MAML implementation on Signer-Split dataset.
- `signer_split_protomaml.py` : ProtoMAML implementation on Signer-Split dataset.
- `unseen_protonet.py` : ProtoNet for Unseen-Class validation.
- `unseen_maml.py` : MAML setup for Unseen-Class protocol.
- `unseen_protomaml.py` : ProtoMAML implementation on Unseen-Class protocol.
- `vit_baseline.py` : Vision Transformer (ViT) model for Unseen-class recognition.

## Requirements
To run this project, ensure you have Python 3.8+ installed. Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Getting Started
1. **Dataset Overview**: Place the ISL dataset in the appropriate directory. Update the directory paths inside each configuration script to match your local machine. Dataset is available in the link: https://data.mendeley.com/datasets/s6kgb6r3ss/2
2. **Training Models**: Run the desired script for training the models. For example, to train the ProtoNet model on the cross-signer split, execute:
    ```bash
    python signer_split_protonet.py
    ```

## Citation and Manuscript Reference
**Important Notice:** This codebase is directly related to our manuscript titled *"Meta-Learning Framework with Vision Transformers for Enhanced Indian Sign Language Recognition"* which is currently submitted to **The Visual Computer**.

If you use this code or our open-source datasets in your research, we strongly encourage you to cite our relevant manuscript.
*(Full citation details and DOI will be updated upon publication)*

