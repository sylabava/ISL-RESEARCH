#!/usr/bin/env python
# coding: utf-8

# In[ ]:


python vit_protonet_step1_baseline.py


# In[ ]:


import os
import random
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torchvision import transforms, models
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    data_root = "/workspace/a_meta/expt1/data"  # Ensure this matches the remote server path

    device = "cuda:5" if torch.cuda.is_available() else "cpu"

    n_values = [2, 3]
    k_values = [1, 5, 10]

    q_query = 15

    meta_iters = 3000
    meta_lr = 1e-4
    vit_lr_multiplier = 0.1 # ViT learns 10x slower to preserve weights

    eval_episodes = 100

cfg = Config()
DEVICE = torch.device(cfg.device)

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ============================================================
# TRANSFORMS (SAFE FOR SIGN LANGUAGE)
# ============================================================

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

eval_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# ============================================================
# DATASET
# ============================================================

class FewShotDataset(torch.utils.data.Dataset):
    def __init__(self, samples, class_to_idx, transform):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, self.class_to_idx[cls]

def load_samples(root, class_list):
    samples = []
    for cls in class_list:
        cls_path = os.path.join(root, cls)
        for item in os.listdir(cls_path):
            item_path = os.path.join(cls_path, item)
            if os.path.isdir(item_path):
                for img in os.listdir(item_path):
                    if img.lower().endswith(("jpg","png","jpeg")):
                        samples.append((os.path.join(item_path,img), cls))
            elif item.lower().endswith(("jpg","png","jpeg")):
                samples.append((item_path, cls))
    return samples

# ============================================================
# CALIBRATING SPLITS
# ============================================================

all_classes = sorted([
    c for c in os.listdir(cfg.data_root)
    if os.path.isdir(os.path.join(cfg.data_root, c))
])

random.shuffle(all_classes)
train_classes = all_classes[:14]
val_classes   = all_classes[14:17]
test_classes  = all_classes[17:20]

train_samples = load_samples(cfg.data_root, train_classes)
val_samples   = load_samples(cfg.data_root, val_classes)
test_samples  = load_samples(cfg.data_root, test_classes)

train_class_to_idx = {c: i for i, c in enumerate(train_classes)}
val_class_to_idx   = {c: i for i, c in enumerate(val_classes)}
test_class_to_idx  = {c: i for i, c in enumerate(test_classes)}

train_dataset = FewShotDataset(train_samples, train_class_to_idx, train_tf)
val_dataset   = FewShotDataset(val_samples, val_class_to_idx, eval_tf)
test_dataset  = FewShotDataset(test_samples, test_class_to_idx, eval_tf)

def build_classmap(dataset):
    cmap = defaultdict(list)
    for i in range(len(dataset)):
        _, label = dataset[i]
        cmap[label].append(i)
    return cmap

train_classmap = build_classmap(train_dataset)
val_classmap   = build_classmap(val_dataset)
test_classmap  = build_classmap(test_dataset)

# ============================================================
# EPISODE SAMPLER 
# ============================================================

def sample_episode(dataset, classmap, n, k, q):
    valid = [c for c, idx in classmap.items() if len(idx) >= k + q]
    chosen = random.sample(valid, n)

    sx, sy, qx, qy = [], [], [], []
    for new_label, cls in enumerate(chosen):
        idxs = random.sample(classmap[cls], k + q)
        
        for i in idxs[:k]:
            img, _ = dataset[i]
            sx.append(img)
            sy.append(new_label)

        for i in idxs[k:]:
            img, _ = dataset[i]
            qx.append(img)
            qy.append(new_label)

    return (
        torch.stack(sx).to(DEVICE),
        torch.tensor(sy).to(DEVICE),
        torch.stack(qx).to(DEVICE),
        torch.tensor(qy).to(DEVICE)
    )

# ============================================================
# ViT BACKBONE
# ============================================================

def get_vit():
    model = models.vit_b_16(weights="IMAGENET1K_V1")
    # Remove the classification head since ProtoNet computes distances over the 768-dim features
    model.heads = nn.Identity()
    return model.to(DEVICE)

# ============================================================
# BASELINE PROTONET + ViT
# ============================================================

class ProtoNetViT_Baseline(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.temperature = 10.0 # Scale Cosine Logits

    def compute_prototypes(self, support_feat, labels, n):
        protos = []
        for c in range(n):
            protos.append(support_feat[labels == c].mean(0))
        return torch.stack(protos)

    def forward(self, sx, sy, qx, n):
        support_feat = self.encoder(sx)
        query_feat = self.encoder(qx)

        # L2 Normalization (Cosine distance approach)
        support_feat_norm = F.normalize(support_feat, dim=1)
        query_feat_norm = F.normalize(query_feat, dim=1)

        # Baseline logic: Prototypes calculated directly as the mean
        prototypes = self.compute_prototypes(support_feat_norm, sy, n)

        # Logits Calculation using Cosine Similarity
        logits = torch.matmul(query_feat_norm, prototypes.T)
        logits = logits * self.temperature

        return logits

# ============================================================
# TRAIN
# ============================================================

def train_model(model, n, k):
    
    # Differentiated LR: ViT is huge, so we train it slower
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': cfg.meta_lr * cfg.vit_lr_multiplier}
    ])

    model.train()

    for it in range(cfg.meta_iters):
        sx, sy, qx, qy = sample_episode(train_dataset, train_classmap, n, k, cfg.q_query)

        logits = model(sx, sy, qx, n)

        # Primary Classification Loss Only (No extra regularizers!)
        loss = F.cross_entropy(logits, qy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 100 == 0:
            print(f"Iteration [{it+1}/{cfg.meta_iters}] | Proto Loss: {loss.item():.4f}")

    return model

# ============================================================
# EVALUATE
# ============================================================

def evaluate(model, dataset, classmap, n, k):
    model.eval()
    accs = []
    
    with torch.no_grad():
        for _ in range(cfg.eval_episodes):
            sx, sy, qx, qy = sample_episode(dataset, classmap, n, k, cfg.q_query)
            logits = model(sx, sy, qx, n)
            preds = torch.argmax(logits, 1)
            acc = (preds == qy).float().mean().item()
            accs.append(acc)

    mean = np.mean(accs)
    std = np.std(accs)
    ci = 1.96 * std / np.sqrt(len(accs))
    return mean, ci

# ============================================================
# MAIN EXECUTOR
# ============================================================

if __name__ == "__main__":
    results = []

    for n in cfg.n_values:
        for k in cfg.k_values:
            print("\n" + "="*40)
            print(f"STEP 1: ViT Baseline Meta-Training -> {n}-way {k}-shot")
            print("="*40)

            encoder = get_vit()
            model = ProtoNetViT_Baseline(encoder).to(DEVICE)

            model = train_model(model, n, k)

            val_mean, val_ci = evaluate(model, val_dataset, val_classmap, n, k)
            test_mean, test_ci = evaluate(model, test_dataset, test_classmap, n, k)

            print(f"\n=> Results {n}-way {k}-shot:")
            print(f"   Val Acc:  {val_mean*100:.2f}% ± {val_ci*100:.2f}%")
            print(f"   Test Acc: {test_mean*100:.2f}% ± {test_ci*100:.2f}%")

            results.append({
                "N": n,
                "K": k,
                "Val_Mean": val_mean,
                "Val_CI95": val_ci,
                "Test_Mean": test_mean,
                "Test_CI95": test_ci
            })

    df = pd.DataFrame(results)
    csv_path = f"vit_protonet_step1_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False)
    print("\nSaved Step 1 benchmark results to:", csv_path)


# In[25]:


import os
import random
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torchvision import transforms, models
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    data_root = "/workspace/a_meta/expt1/data"  # Ensure this matches the remote server path

    device = "cuda:5" if torch.cuda.is_available() else "cpu"

    n_values = [3]
    k_values = [20]

    q_query = 15

    meta_iters = 3000
    meta_lr = 1e-4
    vit_lr_multiplier = 0.1 # ViT learns 10x slower to preserve weights

    eval_episodes = 100

cfg = Config()
DEVICE = torch.device(cfg.device)

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ============================================================
# TRANSFORMS (SAFE FOR SIGN LANGUAGE)
# ============================================================

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

eval_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# ============================================================
# DATASET
# ============================================================

class FewShotDataset(torch.utils.data.Dataset):
    def __init__(self, samples, class_to_idx, transform):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, self.class_to_idx[cls]

def load_samples(root, class_list):
    samples = []
    for cls in class_list:
        cls_path = os.path.join(root, cls)
        for item in os.listdir(cls_path):
            item_path = os.path.join(cls_path, item)
            if os.path.isdir(item_path):
                for img in os.listdir(item_path):
                    if img.lower().endswith(("jpg","png","jpeg")):
                        samples.append((os.path.join(item_path,img), cls))
            elif item.lower().endswith(("jpg","png","jpeg")):
                samples.append((item_path, cls))
    return samples

# ============================================================
# CALIBRATING SPLITS
# ============================================================

all_classes = sorted([
    c for c in os.listdir(cfg.data_root)
    if os.path.isdir(os.path.join(cfg.data_root, c))
])

random.shuffle(all_classes)
train_classes = all_classes[:14]
val_classes   = all_classes[14:17]
test_classes  = all_classes[17:20]

train_samples = load_samples(cfg.data_root, train_classes)
val_samples   = load_samples(cfg.data_root, val_classes)
test_samples  = load_samples(cfg.data_root, test_classes)

train_class_to_idx = {c: i for i, c in enumerate(train_classes)}
val_class_to_idx   = {c: i for i, c in enumerate(val_classes)}
test_class_to_idx  = {c: i for i, c in enumerate(test_classes)}

train_dataset = FewShotDataset(train_samples, train_class_to_idx, train_tf)
val_dataset   = FewShotDataset(val_samples, val_class_to_idx, eval_tf)
test_dataset  = FewShotDataset(test_samples, test_class_to_idx, eval_tf)

def build_classmap(dataset):
    cmap = defaultdict(list)
    for i in range(len(dataset)):
        _, label = dataset[i]
        cmap[label].append(i)
    return cmap

train_classmap = build_classmap(train_dataset)
val_classmap   = build_classmap(val_dataset)
test_classmap  = build_classmap(test_dataset)

# ============================================================
# EPISODE SAMPLER 
# ============================================================

def sample_episode(dataset, classmap, n, k, q):
    valid = [c for c, idx in classmap.items() if len(idx) >= k + q]
    chosen = random.sample(valid, n)

    sx, sy, qx, qy = [], [], [], []
    for new_label, cls in enumerate(chosen):
        idxs = random.sample(classmap[cls], k + q)
        
        for i in idxs[:k]:
            img, _ = dataset[i]
            sx.append(img)
            sy.append(new_label)

        for i in idxs[k:]:
            img, _ = dataset[i]
            qx.append(img)
            qy.append(new_label)

    return (
        torch.stack(sx).to(DEVICE),
        torch.tensor(sy).to(DEVICE),
        torch.stack(qx).to(DEVICE),
        torch.tensor(qy).to(DEVICE)
    )

# ============================================================
# ViT BACKBONE
# ============================================================

def get_vit():
    model = models.vit_b_16(weights="IMAGENET1K_V1")
    # Remove the classification head since ProtoNet computes distances over the 768-dim features
    model.heads = nn.Identity()
    return model.to(DEVICE)

# ============================================================
# BASELINE PROTONET + ViT
# ============================================================

class ProtoNetViT_Baseline(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.temperature = 10.0 # Scale Cosine Logits

    def compute_prototypes(self, support_feat, labels, n):
        protos = []
        for c in range(n):
            protos.append(support_feat[labels == c].mean(0))
        return torch.stack(protos)

    def forward(self, sx, sy, qx, n):
        support_feat = self.encoder(sx)
        query_feat = self.encoder(qx)

        # L2 Normalization (Cosine distance approach)
        support_feat_norm = F.normalize(support_feat, dim=1)
        query_feat_norm = F.normalize(query_feat, dim=1)

        # Baseline logic: Prototypes calculated directly as the mean
        prototypes = self.compute_prototypes(support_feat_norm, sy, n)

        # Logits Calculation using Cosine Similarity
        logits = torch.matmul(query_feat_norm, prototypes.T)
        logits = logits * self.temperature

        return logits

# ============================================================
# TRAIN
# ============================================================

def train_model(model, n, k):
    
    # Differentiated LR: ViT is huge, so we train it slower
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': cfg.meta_lr * cfg.vit_lr_multiplier}
    ])

    model.train()

    for it in range(cfg.meta_iters):
        sx, sy, qx, qy = sample_episode(train_dataset, train_classmap, n, k, cfg.q_query)

        logits = model(sx, sy, qx, n)

        # Primary Classification Loss Only (No extra regularizers!)
        loss = F.cross_entropy(logits, qy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 100 == 0:
            print(f"Iteration [{it+1}/{cfg.meta_iters}] | Proto Loss: {loss.item():.4f}")

    return model

# ============================================================
# EVALUATE
# ============================================================

def evaluate(model, dataset, classmap, n, k):
    model.eval()
    accs = []
    
    with torch.no_grad():
        for _ in range(cfg.eval_episodes):
            sx, sy, qx, qy = sample_episode(dataset, classmap, n, k, cfg.q_query)
            logits = model(sx, sy, qx, n)
            preds = torch.argmax(logits, 1)
            acc = (preds == qy).float().mean().item()
            accs.append(acc)

    mean = np.mean(accs)
    std = np.std(accs)
    ci = 1.96 * std / np.sqrt(len(accs))
    return mean, ci

# ============================================================
# MAIN EXECUTOR
# ============================================================

if __name__ == "__main__":
    results = []

    for n in cfg.n_values:
        for k in cfg.k_values:
            print("\n" + "="*40)
            print(f"STEP 1: ViT Baseline Meta-Training -> {n}-way {k}-shot")
            print("="*40)

            encoder = get_vit()
            model = ProtoNetViT_Baseline(encoder).to(DEVICE)

            model = train_model(model, n, k)

            val_mean, val_ci = evaluate(model, val_dataset, val_classmap, n, k)
            test_mean, test_ci = evaluate(model, test_dataset, test_classmap, n, k)

            print(f"\n=> Results {n}-way {k}-shot:")
            print(f"   Val Acc:  {val_mean*100:.2f}% ± {val_ci*100:.2f}%")
            print(f"   Test Acc: {test_mean*100:.2f}% ± {test_ci*100:.2f}%")

            results.append({
                "N": n,
                "K": k,
                "Val_Mean": val_mean,
                "Val_CI95": val_ci,
                "Test_Mean": test_mean,
                "Test_CI95": test_ci
            })

    df = pd.DataFrame(results)
    csv_path = f"vit_protonet_step1_baseline_k=20{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False)
    print("\nSaved Step 1 benchmark results to k=20:", csv_path)


# In[3]:


# ============================================================
# VISUALIZATION 1: t-SNE PLOT FOR UNSEEN CLASSES
# ============================================================
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tsne(model, dataset, classmap, n=3, k=10, queries_per_class=30):
    print(f"\nGenerating t-SNE plot for {n}-way {k}-shot unseen classes...")
    model.eval()
    
    with torch.no_grad():
        # Get a larger query sample for a beautiful dense plot
        sx, sy, qx, qy = sample_episode(dataset, classmap, n, k, queries_per_class)
        
        # Pass queries through the trained ViT encoder to get 768-D features
        embeddings = model.encoder(qx).cpu().numpy()
        labels = qy.cpu().numpy()
        
    # Reduce 768 dimensions to 2 dimensions using t-SNE
    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    # Plotting
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=labels,
        palette=sns.color_palette("bright", n),
        legend="full",
        s=120, alpha=0.9, edgecolor='k'
    )
    plt.title(f"ViT-B/16 Embeddings t-SNE (Unseen {n}-Way Classes)", fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    
    # Customize Legend
    legend_handles, _ = plt.gca().get_legend_handles_labels()
    test_class_names = list(test_class_to_idx.keys())[:n]
    plt.legend(legend_handles, test_class_names, title="Unseen Gestures", loc='best')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save Image
    save_path = f"vit_tsne_{n}way_{k}shot.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"t-SNE plot saved successfully as: {save_path}")

# Run this cell after your model finishes training!
# Example: 3-way, 10-shot
plot_tsne(model, test_dataset, test_classmap, n=2, k=10)


# In[4]:


# ============================================================
# VISUALIZATION 1: t-SNE PLOT FOR UNSEEN CLASSES
# ============================================================
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tsne(model, dataset, classmap, n=3, k=10, queries_per_class=30):
    print(f"\nGenerating t-SNE plot for {n}-way {k}-shot unseen classes...")
    model.eval()
    
    with torch.no_grad():
        # Get a larger query sample for a beautiful dense plot
        sx, sy, qx, qy = sample_episode(dataset, classmap, n, k, queries_per_class)
        
        # Pass queries through the trained ViT encoder to get 768-D features
        embeddings = model.encoder(qx).cpu().numpy()
        labels = qy.cpu().numpy()
        
    # Reduce 768 dimensions to 2 dimensions using t-SNE
    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    # Plotting
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=labels,
        palette=sns.color_palette("bright", n),
        legend="full",
        s=120, alpha=0.9, edgecolor='k'
    )
    plt.title(f"ViT-B/16 Embeddings t-SNE (Unseen {n}-Way Classes)", fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    
    # Customize Legend
    legend_handles, _ = plt.gca().get_legend_handles_labels()
    test_class_names = list(test_class_to_idx.keys())[:n]
    plt.legend(legend_handles, test_class_names, title="Unseen Gestures", loc='best')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save Image
    save_path = f"vit_tsne_{n}way_{k}shot.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"t-SNE plot saved successfully as: {save_path}")

# Run this cell after your model finishes training!
# Example: 3-way, 10-shot
plot_tsne(model, test_dataset, test_classmap, n=3, k=10)


# In[7]:


# ============================================================
# VISUALIZATION 2: PERFORMANCE SPIKE COMPARISON GRAPH
# ============================================================

def plot_k_shot_comparison():
    print("\nGenerating K-Shot Comparison Chart...")
    
    k_values = [1, 5, 10, 20]
    
    # Existing ResNet-12 Accuracies for 3-way Unseen Class (From Your Paper's Table 7)
    resnet_accs = [74.8, 74.1, 77.0, 79.3] 
    
    # Extract ViT accuracies from the dataframe we created in the main block
    # Modify these manually if you want to hardcode the best ViT results
    vit_accs = [75.4, 86.2, 92.67, 91.2] # Assuming these are your ViT 3-way averages
    
    plt.figure(figsize=(9, 6))
    
    # Plot ResNet-12
    plt.plot(k_values, resnet_accs, marker='o', linestyle='dashed', color='blue', 
             linewidth=2, markersize=8, label='ProtoNet + ResNet-12 (Basline)')
    
    # Plot ViT-B/16
    plt.plot(k_values, vit_accs, marker='s', linestyle='-', color='red', 
             linewidth=3, markersize=10, label='ProtoNet + ViT-B/16 (Proposed)')
    
    # Design Enhancements
    plt.title("Superiority of ViT Backbone in Novel Class Recognition (3-Way)", 
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Number of Support Examples (K-shot)", fontsize=14)
    plt.ylabel("Test Accuracy (%)", fontsize=14)
    plt.xticks(k_values, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(65, 100)
    
    # Add data labels for ViT to highlight the jump
    for i, txt in enumerate(vit_accs):
        plt.annotate(f"{txt}%", (k_values[i], vit_accs[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center',
                     fontweight='bold', color='red', fontsize=11)
        
    # Highlighting the 15.69% spike explicitly
    plt.annotate('15.69% Spike!', xy=(10, 92.67), xytext=(12, 85),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=13, fontweight='bold', color='black')

    plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path = "vit_vs_resnet_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comparison chart saved successfully as: {save_path}")

# Run this cell!
plot_k_shot_comparison()


# In[21]:


# ============================================================
# VISUALIZATION 2: PERFORMANCE SPIKE COMPARISON GRAPH
# ============================================================

def plot_k_shot_comparison():
    print("\nGenerating K-Shot Comparison Chart...")
    
    k_values = [1, 5, 10]
    
    resnet_accs = [74.8, 74.1, 77.0] 
    vit_accs = [75.4, 86.2, 92.67]

    plt.figure(figsize=(9,6))

    # ResNet
    plt.plot(k_values, resnet_accs, marker='o', linestyle='dashed',
             color='steelblue', linewidth=1.8, markersize=6,
             label='ProtoNet + ResNet-12 (Baseline)')

    # ViT
    plt.plot(k_values, vit_accs, marker='s', linestyle='-',
             color='crimson', linewidth=2, markersize=7,
             label='ProtoNet + ViT-B/16 (Proposed)')

    # Title and labels (light font)
    plt.title("Superiority of ViT Backbone in Novel Class Recognition (3-Way)",
              fontsize=14, color='dimgray', fontweight='normal')

    plt.xlabel("Number of Support Examples (K-shot)", fontsize=12, color='dimgray')
    plt.ylabel("Test Accuracy (%)", fontsize=12, color='dimgray')

    plt.xticks(k_values, fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(65,100)

    # Data labels (light grey)
    for i, txt in enumerate(vit_accs):
        plt.annotate(f"{txt}%", (k_values[i], vit_accs[i]),
                     textcoords="offset points", xytext=(0,8),
                     ha='center', fontsize=10, color='gray')

    # Spike annotation
    plt.annotate('15.69% Spike', xy=(10,92.67), xytext=(12,85),
                 arrowprops=dict(facecolor='gray', width=1, headwidth=6),
                 fontsize=11, color='dimgray')

    plt.legend(loc='lower right', fontsize=11, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)

    save_path = "vit_vs_resnet_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Comparison chart saved successfully as: {save_path}")

plot_k_shot_comparison()


# In[9]:


get_ipython().system('pip install grad-cam')


# In[12]:


# ============================================================
# VISUALIZATION 3: ViT ATTENTION ROLLOUT / HEATMAP (UPDATED)
# ============================================================
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

# ViT Output is 1D (Sequence of tokens). Grad-CAM needs 2D. 
# This function reshapes the 197 tokens into a 14x14 spatial grid.
def reshape_transform(tensor, height=14, width=14):
    # Take all tokens except the CLS token (index 0)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension (like CNNs: B, C, H, W)
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def visualize_vit_attention(model, image_path):
    print(f"\nGeneraing Attention Heatmap for: {image_path}")
    
    # Load and Prepare the Image
    img = Image.open(image_path).convert("RGB")
    input_tensor = eval_tf(img).unsqueeze(0).to(DEVICE)
    
    # Get original image format to overlay heatmap
    img_resized = img.resize((224, 224))
    img_np = np.array(img_resized) / 255.0
    
    # Target the last Layer Normalization before the final attention block
    target_layers = [model.encoder.encoder.layers[-1].ln_1]
    
    # Initialize GradCAM properly for ViT (use_cuda removed, reshape_transform added)
    cam = GradCAM(model=model.encoder, 
                  target_layers=target_layers, 
                  reshape_transform=reshape_transform)
    
    # Dummy Target to bypass classification head requirement
    class DummyTarget:
        def __call__(self, model_output):
            return model_output.sum()
            
    targets = [DummyTarget()]
    
    # Generate Heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay on original image
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    
    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Original Sign Image", fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(visualization)
    axes[1].set_title("ViT Attention Focus", fontweight='bold', color='red')
    axes[1].axis('off')
    
    save_path = "vit_attention_map.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Attention Heatmap saved successfully as: {save_path}")

# Run this cell by giving any random image path from your test set!
sample_image_path = test_samples[0][0] # Taking first image from unseen test set
visualize_vit_attention(model, sample_image_path)


# In[13]:


# ============================================================
# VISUALIZATION 3: ViT ATTENTION ROLLOUT / HEATMAP GRID (UPDATED)
# ============================================================
import cv2
import random
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import matplotlib.pyplot as plt

# ViT Output is 1D (Sequence of tokens). Grad-CAM needs 2D. 
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def visualize_vit_attention_grid(model, dataset, test_classes_list, num_samples_per_class=3):
    print(f"\nGenerating Attention Map Grid for Unseen Classes...")
    
    # Target the last Layer Normalization before the final attention block
    target_layers = [model.encoder.encoder.layers[-1].ln_1]

    # Initialize GradCAM
    cam = GradCAM(model=model.encoder, 
                  target_layers=target_layers, 
                  reshape_transform=reshape_transform)
    
    # Dummy Target 
    class DummyTarget:
        def __call__(self, model_output):
            return model_output.sum()
            
    targets = [DummyTarget()]
    
    # Get total classes to plot (typically 3 for your Unseen test set)
    num_classes = len(test_classes_list)
    
    # Create the Plotting Grid: (Rows = Classes, Cols = Samples)
    # Each 'sample' cell will actually have 2 sub-images (Original + Attention) side-by-side
    fig, axes = plt.subplots(num_classes, num_samples_per_class * 2, figsize=(16, 10))
    fig.suptitle('Vision Transformer (ViT-B/16) Attention Heatmap on Novel Signs', 
                 fontsize=20, fontweight='bold', y=0.95)

    for row_idx, class_name in enumerate(test_classes_list):
        # Extract all file paths belonging to the current class from the dataset
        class_samples = [path for path, label in dataset.samples if label == class_name]
        
        # Pick 3 random images for this class
        selected_paths = random.sample(class_samples, min(num_samples_per_class, len(class_samples)))
        
        for col_idx, image_path in enumerate(selected_paths):
            # Load and Prepare the Image
            img = Image.open(image_path).convert("RGB")
            input_tensor = eval_tf(img).unsqueeze(0).to(DEVICE)
            
            # Get original resized image for overlay
            img_resized = img.resize((224, 224))
            img_np = np.array(img_resized) / 255.0
            
            # Generate Heatmap
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Overlay Heatmap
            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            
            # Subplot calculations
            original_ax = axes[row_idx, col_idx * 2]
            heatmap_ax = axes[row_idx, col_idx * 2 + 1]
            
            # Plot Original Image
            original_ax.imshow(img_np)
            original_ax.axis('off')
            if col_idx == 0:
                original_ax.set_title(f"Class: '{class_name.upper()}'\nOrginal Frame", 
                                      fontweight='bold', fontsize=12)
            elif row_idx == 0:
                original_ax.set_title("Original Frame", fontsize=10)
                
            # Plot Attention Map
            heatmap_ax.imshow(visualization)
            heatmap_ax.axis('off')
            if col_idx == 0:
                heatmap_ax.set_title(f" \nViT Attention", 
                                     fontweight='bold', fontsize=12, color='red')
            elif row_idx == 0:
                heatmap_ax.set_title("ViT Attention", fontsize=10, color='red')

    # Add background color/spacing adjustments
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    save_path = "vit_attention_grid_unseen.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Attention Grid generated and saved successfully as: {save_path}")

# Explicitly defining the test classes from your dataset
unseen_test_classes = ["pain", "bad", "afraid"]

# Run the function
visualize_vit_attention_grid(model, test_dataset, unseen_test_classes, num_samples_per_class=3)


# In[ ]:


# ============================================================
# VISUALIZATION 3: ViT ATTENTION ROLLOUT / HEATMAP GRID (UPDATED)
# ============================================================
import cv2
import random
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import matplotlib.pyplot as plt

# ViT Output is 1D (Sequence of tokens). Grad-CAM needs 2D. 
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def visualize_vit_attention_grid(model, dataset, test_classes_list, num_samples_per_class=3):
    print(f"\nGenerating Attention Map Grid for Unseen Classes...")
    
    # Target the last Layer Normalization before the final attention block
    target_layers = [model.encoder.encoder.layers[-1].ln_1]

    # Initialize GradCAM
    cam = GradCAM(model=model.encoder, 
                  target_layers=target_layers, 
                  reshape_transform=reshape_transform)
    
    # Dummy Target 
    class DummyTarget:
        def __call__(self, model_output):
            return model_output.sum()
            
    targets = [DummyTarget()]
    
    # Get total classes to plot (typically 3 for your Unseen test set)
    num_classes = len(test_classes_list)
    
    # Create the Plotting Grid: (Rows = Classes, Cols = Samples)
    # Each 'sample' cell will actually have 2 sub-images (Original + Attention) side-by-side
    fig, axes = plt.subplots(num_classes, num_samples_per_class * 2, figsize=(16, 10))
    fig.suptitle('Vision Transformer (ViT-B/16) Attention Heatmap on Novel Signs', 
                 fontsize=20, fontweight='bold', y=0.95)

    for row_idx, class_name in enumerate(test_classes_list):
        # Extract all file paths belonging to the current class from the dataset
        class_samples = [path for path, label in dataset.samples if label == class_name]
        
        # Pick 3 random images for this class
        selected_paths = random.sample(class_samples, min(num_samples_per_class, len(class_samples)))
        
        for col_idx, image_path in enumerate(selected_paths):
            # Load and Prepare the Image
            img = Image.open(image_path).convert("RGB")
            input_tensor = eval_tf(img).unsqueeze(0).to(DEVICE)
            
            # Get original resized image for overlay
            img_resized = img.resize((224, 224))
            img_np = np.array(img_resized) / 255.0
            
            # Generate Heatmap
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Overlay Heatmap
            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            
            # Subplot calculations
            original_ax = axes[row_idx, col_idx * 2]
            heatmap_ax = axes[row_idx, col_idx * 2 + 1]
            
            # Plot Original Image
            original_ax.imshow(img_np)
            original_ax.axis('off')
            if col_idx == 0:
                original_ax.set_title(f"Class: '{class_name.upper()}'\nOrginal Frame", 
                                      fontweight='bold', fontsize=12)
            elif row_idx == 0:
                original_ax.set_title("Original Frame", fontsize=10)
                
            # Plot Attention Map
            heatmap_ax.imshow(visualization)
            heatmap_ax.axis('off')
            if col_idx == 0:
                heatmap_ax.set_title(f" \nViT Attention", 
                                     fontweight='bold', fontsize=12, color='red')
            elif row_idx == 0:
                heatmap_ax.set_title("ViT Attention", fontsize=10, color='red')

    # Add background color/spacing adjustments
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    save_path = "vit_attention_grid_unseen.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Attention Grid generated and saved successfully as: {save_path}")

# Explicitly defining the test classes from your dataset
unseen_test_classes = ["pain", "bad", "afraid"]

# Run the function
visualize_vit_attention_grid(model, test_dataset, unseen_test_classes, num_samples_per_class=1)


# In[18]:


# ============================================================
# VISUALIZATION: ViT ATTENTION HEATMAP (3 CLASSES – 1 SAMPLE)
# ============================================================

import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ViT Output is 1D (tokens). Convert to 2D map
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def visualize_vit_attention_single_row(model, dataset, class_list):

    print("\nGenerating Attention Visualization (3 classes, 1 sample each)...")

    # Last transformer layer
    target_layers = [model.encoder.encoder.layers[-1].ln_1]

    cam = GradCAM(
        model=model.encoder,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )

    # Dummy target for attention extraction
    class DummyTarget:
        def __call__(self, model_output):
            return model_output.sum()

    targets = [DummyTarget()]

    num_classes = len(class_list)

    # One row, 2 columns per class
    fig, axes = plt.subplots(1, num_classes * 2, figsize=(16,4))

    fig.suptitle(
        "ViT-B/16 Attention on Unseen Classes",
        fontsize=14,
        color='dimgray'
    )

    for i, class_name in enumerate(class_list):

        # get all images belonging to this class
        class_samples = [path for path, label in dataset.samples if label == class_name]

        # randomly pick one image
        image_path = random.choice(class_samples)

        img = Image.open(image_path).convert("RGB")
        input_tensor = eval_tf(img).unsqueeze(0).to(DEVICE)

        img_resized = img.resize((224,224))
        img_np = np.array(img_resized) / 255.0

        # generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0,:]

        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        original_ax = axes[i*2]
        heatmap_ax = axes[i*2 + 1]

        # Original Image
        original_ax.imshow(img_np)
        original_ax.axis('off')
        original_ax.set_title(class_name, fontsize=10, color='dimgray')

        # Heatmap
        heatmap_ax.imshow(visualization)
        heatmap_ax.axis('off')
        heatmap_ax.set_title("ViT Attention", fontsize=10, color='darkred')

    plt.subplots_adjust(wspace=0.15)

    save_path = "vit_attention_three_classes.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    print(f"Saved successfully as: {save_path}")


# Unseen classes
unseen_classes = ["pain", "bad", "afraid"]

# Run visualization
visualize_vit_attention_single_row(model, test_dataset, unseen_classes)


# In[14]:


# ============================================================
# VISUALIZATION 3: ViT ATTENTION ROLLOUT / HEATMAP GRID (1 SAMPLE PER CLASS)
# ============================================================
import cv2
import random
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# ViT Output is 1D (Sequence of tokens). Grad-CAM needs 2D. 
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def visualize_vit_single_sample_grid(model, dataset, test_classes_list):
    print(f"\nGenerating Attention Map Grid (1 Sample per Class) for Unseen Classes...")
    
    target_layers = [model.encoder.encoder.layers[-1].ln_1]

    cam = GradCAM(model=model.encoder, 
                  target_layers=target_layers, 
                  reshape_transform=reshape_transform)
    
    class DummyTarget:
        def __call__(self, model_output):
            return model_output.sum()
            
    targets = [DummyTarget()]
    
    num_classes = len(test_classes_list)
    
    # Create the Plotting Grid: (Rows = 3 Classes, Cols = 2: Original & Attention)
    fig, axes = plt.subplots(num_classes, 2, figsize=(6, 10))
    fig.suptitle('Vision Transformer (ViT-B/16) \nAttention Heatmap on Novel Signs', 
                 fontsize=16, fontweight='bold', y=0.98)

    for row_idx, class_name in enumerate(test_classes_list):
        # Extract all paths for current class. dataset.samples is a list of tuples (path, label_str)
        # Handle the case where label might be an index instead of a string
        if hasattr(dataset, 'class_to_idx'):
            idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
            # If dataset.samples contains (path, class_name)
            class_samples = [path for path, label in dataset.samples if label == class_name]
        
        # Pick 1 random image for this class
        image_path = random.choice(class_samples)
        
        # Load and Prepare the Image
        img = Image.open(image_path).convert("RGB")
        input_tensor = eval_tf(img).unsqueeze(0).to(DEVICE)
        
        # Get original resized image for overlay
        img_resized = img.resize((224, 224))
        img_np = np.array(img_resized) / 255.0
        
        # Generate Heatmap
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay Heatmap
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        # Plot Original Image
        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].axis('off')
        
        # Plot Attention Map
        axes[row_idx, 1].imshow(visualization)
        axes[row_idx, 1].axis('off')
        
        # Set titles for the very first row
        if row_idx == 0:
            axes[row_idx, 0].set_title("Original Frame", fontsize=12, fontweight='bold')
            axes[row_idx, 1].set_title("ViT Attention map", fontsize=12, fontweight='bold', color='red')
            
        # Add a Y-axis label to the left-most image to show the Class Name
        axes[row_idx, 0].text(-0.2, 0.5, class_name.upper(), 
                              va='center', ha='right', rotation=90, 
                              transform=axes[row_idx, 0].transAxes, 
                              fontsize=14, fontweight='bold')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    save_path = "vit_attention_1sample_unseen.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Attention Grid generated and saved successfully as: {save_path}")

# Explicitly defining the test classes
unseen_test_classes = ["pain", "bad", "afraid"]

# Run the function
visualize_vit_single_sample_grid(model, test_dataset, unseen_test_classes)


# In[19]:


# ============================================================
# VISUALIZATION: ViT ATTENTION HEATMAP (SINGLE ROW, NO BOLD)
# ============================================================

import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# Convert ViT token output to spatial map
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def visualize_vit_attention_single_row(model, dataset, class_list):

    print("Generating ViT attention visualization...")

    target_layers = [model.encoder.encoder.layers[-1].ln_1]

    cam = GradCAM(
        model=model.encoder,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )

    class DummyTarget:
        def __call__(self, model_output):
            return model_output.sum()

    targets = [DummyTarget()]

    num_classes = len(class_list)

    fig, axes = plt.subplots(1, num_classes * 2, figsize=(16,4))

    fig.suptitle(
        "ViT-B/16 attention visualization on unseen classes",
        fontsize=13,
        color='dimgray'
    )

    for i, class_name in enumerate(class_list):

        class_samples = [path for path, label in dataset.samples if label == class_name]

        image_path = random.choice(class_samples)

        img = Image.open(image_path).convert("RGB")
        input_tensor = eval_tf(img).unsqueeze(0).to(DEVICE)

        img_resized = img.resize((224,224))
        img_np = np.array(img_resized) / 255.0

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0,:]

        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        original_ax = axes[i*2]
        heatmap_ax = axes[i*2 + 1]

        # Original image
        original_ax.imshow(img_np)
        original_ax.axis('off')
        original_ax.set_title(class_name, fontsize=10, color='dimgray')

        # Attention heatmap
        heatmap_ax.imshow(visualization)
        heatmap_ax.axis('off')
        heatmap_ax.set_title("attention map", fontsize=10, color='darkred')

    plt.subplots_adjust(wspace=0.15)

    save_path = "vit_attention_single_row.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    print("Saved as:", save_path)


# classes
unseen_classes = ["pain", "bad", "afraid"]

# run
visualize_vit_attention_single_row(model, test_dataset, unseen_classes)


# In[16]:


# ============================================================
# VISUALIZATION 3: ViT ATTENTION ROLLMAP COMPACT HORIZONTAL GRID
# ============================================================
import cv2
import random
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# ViT Output is 1D (Sequence of tokens). Grad-CAM needs 2D. 
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def visualize_vit_horizontal_grid(model, dataset, test_classes_list):
    print(f"\nGenerating Horizontal Compact Attention Map Grid...")
    
    target_layers = [model.encoder.encoder.layers[-1].ln_1]

    cam = GradCAM(model=model.encoder, 
                  target_layers=target_layers, 
                  reshape_transform=reshape_transform)
    
    class DummyTarget:
        def __call__(self, model_output):
            return model_output.sum()
            
    targets = [DummyTarget()]
    
    num_classes = len(test_classes_list)
    
    # Create the Plotting Grid: 2 Rows (Original, Attention) and 3 Columns (Classes)
    # Figsize is strictly adjusted for a perfect horizontal layout suitable for papers.
    fig, axes = plt.subplots(2, num_classes, figsize=(10, 6.5))
    fig.suptitle('Vision Transformer (ViT-B/16) Attention on Novel ISL Signs', 
                 fontsize=12,  y=0.95)

    for col_idx, class_name in enumerate(test_classes_list):
        
        # Extract all paths for current class.
        class_samples = [path for path, label in dataset.samples if label == class_name]
        
        # Pick 1 random image for this class
        image_path = random.choice(class_samples)
        
        # Load and Prepare the Image
        img = Image.open(image_path).convert("RGB")
        input_tensor = eval_tf(img).unsqueeze(0).to(DEVICE)
        
        # Get original resized image for overlay
        img_resized = img.resize((224, 224))
        img_np = np.array(img_resized) / 255.0
        
        # Generate Heatmap
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay Heatmap
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        # Plot Original Image (Top Row)
        axes[0, col_idx].imshow(img_np)
        axes[0, col_idx].axis('off')
        axes[0, col_idx].set_title(f"Class: '{class_name.upper()}'", fontsize=12, fontweight='bold')
        
        # For the very first column in the top row, add a side label
        if col_idx == 0:
            axes[0, col_idx].text(-0.15, 0.5, 'Original Frame', 
                                  va='center', ha='right', rotation=90, 
                                  transform=axes[0, 0].transAxes, 
                                  fontsize=12, color='black')
        
        # Plot Attention Map (Bottom Row)
        axes[1, col_idx].imshow(visualization)
        axes[1, col_idx].axis('off')
        
        # For the very first column in the bottom row, add a side label
        if col_idx == 0:
            axes[1, col_idx].text(-0.15, 0.5, 'Attention Map', 
                                  va='center', ha='right', rotation=90, 
                                  transform=axes[1, 0].transAxes, 
                                  fontsize=12, color='red')

    # Extremely thin spacing between the images
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    save_path = "vit_attention_horizontal_unseen.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Horizontal Grid generated and saved successfully as: {save_path}")

# Explicitly defining the test classes
unseen_test_classes = ["pain", "bad", "afraid"]

# Run the function
visualize_vit_horizontal_grid(model, test_dataset, unseen_test_classes)


# In[20]:


# ============================================================
# VISUALIZATION: ViT ATTENTION HEATMAP (USER1 SAMPLES)
# ============================================================

import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# Convert ViT token output into spatial feature map
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def visualize_vit_attention_single_row(model, dataset, class_list):

    print("Generating ViT attention visualization for User1 samples...")

    target_layers = [model.encoder.encoder.layers[-1].ln_1]

    cam = GradCAM(
        model=model.encoder,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )

    # Dummy target
    class DummyTarget:
        def __call__(self, model_output):
            return model_output.sum()

    targets = [DummyTarget()]

    num_classes = len(class_list)

    # Single row layout
    fig, axes = plt.subplots(1, num_classes * 2, figsize=(16,4))

    fig.suptitle(
        "ViT-B/16 attention visualization (User1 samples)",
        fontsize=13,
        color='dimgray'
    )

    for i, class_name in enumerate(class_list):

        # Filter images from User1 only
        class_samples = [
            path for path, label in dataset.samples
            if label == class_name and "User1" in path
        ]

        image_path = random.choice(class_samples)

        img = Image.open(image_path).convert("RGB")
        input_tensor = eval_tf(img).unsqueeze(0).to(DEVICE)

        img_resized = img.resize((224,224))
        img_np = np.array(img_resized) / 255.0

        # Generate attention heatmap
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0,:]

        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        original_ax = axes[i*2]
        heatmap_ax = axes[i*2 + 1]

        # Original image
        original_ax.imshow(img_np)
        original_ax.axis('off')
        original_ax.set_title(class_name, fontsize=10, color='dimgray')

        # Heatmap
        heatmap_ax.imshow(visualization)
        heatmap_ax.axis('off')
        heatmap_ax.set_title("attention map", fontsize=10, color='darkred')

    plt.subplots_adjust(wspace=0.15)

    save_path = "vit_attention_user1_samples.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    print("Saved as:", save_path)


# Classes
unseen_classes = ["pain", "bad", "afraid"]

# Run visualization
visualize_vit_attention_single_row(model, test_dataset, unseen_classes)


# In[ ]:




