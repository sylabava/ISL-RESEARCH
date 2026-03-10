#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# ============================================================
# PROTONET FULL BENCHMARK
# N = {2,3}
# K = {1,5,10,20}
# Backbones = conv4, resnet12, resnet18, resnet34
# Includes:
#   - Training
#   - Evaluation
#   - Heatmaps
#   - Confusion matrix
#   - Per-class accuracy
#   - t-SNE visualization
# ============================================================

import os
import random
import csv
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/workspace/a_meta/expt1/data"

N_VALUES = [2, 3]
K_VALUES = [1, 5, 10, 20]
BACKBONES = ["conv4", "resnet12", "resnet18", "resnet34"]

Q_QUERY = 15
META_ITERS = 200
EVAL_EPISODES = 50

random.seed(42)
torch.manual_seed(42)

# ============================================================
# DATASET
# ============================================================

train_tf = transforms.Compose([
    transforms.Resize(92),
    transforms.RandomCrop(84),
    transforms.ToTensor()
])

eval_tf = transforms.Compose([
    transforms.Resize(92),
    transforms.CenterCrop(84),
    transforms.ToTensor()
])

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


def load_samples(class_list):
    samples = []
    for cls in class_list:
        cls_path = os.path.join(DATA_ROOT, cls)
        for user in os.listdir(cls_path):
            user_path = os.path.join(cls_path, user)
            if not os.path.isdir(user_path):
                continue
            for img in os.listdir(user_path):
                if img.lower().endswith((".jpg", ".png", ".jpeg")):
                    samples.append((os.path.join(user_path, img), cls))
    return samples


all_classes = sorted([
    c for c in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, c))
])

random.shuffle(all_classes)

train_classes = all_classes[:14]
val_classes = all_classes[14:17]
test_classes = all_classes[17:20]

train_samples = load_samples(train_classes)
val_samples = load_samples(val_classes)
test_samples = load_samples(test_classes)

train_class_to_idx = {c: i for i, c in enumerate(train_classes)}
val_class_to_idx = {c: i for i, c in enumerate(val_classes)}
test_class_to_idx = {c: i for i, c in enumerate(test_classes)}

train_dataset = FewShotDataset(train_samples, train_class_to_idx, train_tf)
val_dataset = FewShotDataset(val_samples, val_class_to_idx, eval_tf)
test_dataset = FewShotDataset(test_samples, test_class_to_idx, eval_tf)

def build_classmap(dataset):
    cmap = defaultdict(list)
    for i in range(len(dataset)):
        _, label = dataset[i]
        cmap[label].append(i)
    return cmap

train_classmap = build_classmap(train_dataset)
val_classmap = build_classmap(val_dataset)
test_classmap = build_classmap(test_dataset)

# ============================================================
# BACKBONES
# ============================================================

class Conv4(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        self.encoder = nn.Sequential(
            block(3, 64),
            block(64, 64),
            block(64, 64),
            block(64, 64)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.view(x.size(0), -1)

    @property
    def output_dim(self):
        return 64


# -----------------------------
# RESNET12 (Few-Shot Standard)
# -----------------------------

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.downsample = None
        if in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c)
            )

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(out + identity)
        out = self.maxpool(out)
        return out


class ResNet12(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BasicBlock(3, 64)
        self.layer2 = BasicBlock(64, 160)
        self.layer3 = BasicBlock(160, 320)
        self.layer4 = BasicBlock(320, 640)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.view(x.size(0), -1)

    @property
    def output_dim(self):
        return 640


def get_resnet(name):
    if name == "resnet18":
        model = models.resnet18(weights=None)
    elif name == "resnet34":
        model = models.resnet34(weights=None)
    else:
        raise ValueError

    layers = list(model.children())[:-1]
    encoder = nn.Sequential(*layers)

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = encoder
        def forward(self, x):
            x = self.enc(x)
            return x.view(x.size(0), -1)
        @property
        def output_dim(self):
            return 512

    return Wrapper().to(DEVICE)


def get_backbone(name):
    if name == "conv4":
        return Conv4().to(DEVICE)
    if name == "resnet12":
        return ResNet12().to(DEVICE)
    if name in ["resnet18", "resnet34"]:
        return get_resnet(name)
    raise ValueError("Unknown backbone")

# ============================================================
# PROTONET
# ============================================================

class ProtoNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, support_x, support_y, query_x, n_way):
        z_support = self.encoder(support_x)
        z_query = self.encoder(query_x)

        prototypes = []
        for c in range(n_way):
            idx = (support_y == c).nonzero(as_tuple=True)[0]
            prototypes.append(z_support[idx].mean(0))
        prototypes = torch.stack(prototypes)

        dists = torch.cdist(z_query, prototypes)
        return -dists

# ============================================================
# EPISODE SAMPLER
# ============================================================

def sample_episode(dataset, classmap, n_way, k, q):
    valid = [c for c, idxs in classmap.items() if len(idxs) >= k+q]
    chosen = random.sample(valid, n_way)

    sx, sy, qx, qy = [], [], [], []

    for new_label, cls in enumerate(chosen):
        idxs = random.sample(classmap[cls], k+q)
        for i in idxs[:k]:
            img,_ = dataset[i]
            sx.append(img)
            sy.append(new_label)
        for i in idxs[k:]:
            img,_ = dataset[i]
            qx.append(img)
            qy.append(new_label)

    return (
        torch.stack(sx).to(DEVICE),
        torch.tensor(sy).to(DEVICE),
        torch.stack(qx).to(DEVICE),
        torch.tensor(qy).to(DEVICE)
    )

# ============================================================
# TRAIN & EVAL
# ============================================================

def train_protonet(model, N, K):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(META_ITERS):
        sx, sy, qx, qy = sample_episode(train_dataset, train_classmap, N, K, Q_QUERY)
        logits = model(sx, sy, qx, N)
        loss = F.cross_entropy(logits, qy)

        opt.zero_grad()
        loss.backward()
        opt.step()

    return model


def evaluate(model, dataset, classmap, N, K):
    model.eval()
    accs = []
    for _ in range(EVAL_EPISODES):
        sx, sy, qx, qy = sample_episode(dataset, classmap, N, K, Q_QUERY)
        logits = model(sx, sy, qx, N)
        preds = torch.argmax(logits, 1)
        accs.append((preds == qy).float().mean().item())
    return np.mean(accs)

# ============================================================
# BENCHMARK LOOP
# ============================================================

results = []

for backbone in BACKBONES:
    for N in N_VALUES:
        for K in K_VALUES:

            print(f"\n{backbone} | {N}-way | {K}-shot")

            encoder = get_backbone(backbone)
            model = ProtoNet(encoder).to(DEVICE)

            model = train_protonet(model, N, K)

            val_acc = evaluate(model, val_dataset, val_classmap, N, K)
            test_acc = evaluate(model, test_dataset, test_classmap, N, K)

            results.append({
                "Backbone": backbone,
                "N": N,
                "K": K,
                "Val_Acc": val_acc,
                "Test_Acc": test_acc
            })

df = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df.to_csv(f"protonet_results_{timestamp}.csv", index=False)

print("\nBenchmark finished.")


# In[2]:


# ============================================================
# CELL — HEATMAPS (VAL + TEST)
# ============================================================

import seaborn as sns
import matplotlib.pyplot as plt

for backbone in df["Backbone"].unique():

    print(f"\nGenerating heatmaps for: {backbone}")

    df_b = df[df["Backbone"] == backbone]

    # ---- Validation Heatmap ----
    val_table = df_b.pivot(index="N", columns="K", values="Val_Acc")

    plt.figure(figsize=(6,5))
    sns.heatmap(val_table, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f"{backbone} — Validation Accuracy")
    plt.xlabel("K-shot")
    plt.ylabel("N-way")
    plt.show()

    # ---- Test Heatmap ----
    test_table = df_b.pivot(index="N", columns="K", values="Test_Acc")

    plt.figure(figsize=(6,5))
    sns.heatmap(test_table, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f"{backbone} — Test Accuracy")
    plt.xlabel("K-shot")
    plt.ylabel("N-way")
    plt.show()


# In[3]:


# ============================================================
# CELL — CONFUSION MATRIX (3-WAY)
# ============================================================

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Sample one 3-way episode
sx, sy, qx, qy = sample_episode(test_dataset, test_classmap, 3, 5, Q_QUERY)

model.eval()
logits = model(sx, sy, qx, 3)
preds = torch.argmax(logits, dim=1)

cm = confusion_matrix(qy.cpu(), preds.cpu())

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (3-Way Few-Shot)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# In[4]:


# ============================================================
# CELL — PER-CLASS ACCURACY
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

class_acc = []

for cls in range(3):
    mask = (qy == cls)
    acc = (preds[mask] == cls).float().mean().item()
    class_acc.append(acc)

plt.figure(figsize=(6,4))
plt.bar(range(3), class_acc)
plt.ylim(0,1)
plt.title("Per-Class Accuracy (3-Way)")
plt.xlabel("Class")
plt.ylabel("Accuracy")

for i,v in enumerate(class_acc):
    plt.text(i, v+0.02, f"{v:.2f}", ha='center')

plt.show()


# In[5]:


# ============================================================
# CELL — ROC CURVES (3-WAY OVR)
# ============================================================

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
y_true = label_binarize(qy.cpu().numpy(), classes=[0,1,2])

plt.figure(figsize=(6,5))

for i in range(3):
    fpr, tpr, _ = roc_curve(y_true[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (3-Way Few-Shot)")
plt.legend()
plt.show()


# In[6]:


# ============================================================
# CELL — t-SNE VISUALIZATION (TEST SET)
# ============================================================

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

encoder = model.encoder
encoder.eval()

features = []
labels = []

for cls, idxs in test_classmap.items():
    for i in idxs[:20]:
        img, _ = test_dataset[i]
        img = img.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat = encoder(img).cpu().numpy()

        features.append(feat)
        labels.append(cls)

features = np.concatenate(features)
emb = TSNE(n_components=2, random_state=42).fit_transform(features)

plt.figure(figsize=(6,6))
for cls in set(labels):
    idx = np.array(labels) == cls
    plt.scatter(emb[idx,0], emb[idx,1], label=str(cls))

plt.legend()
plt.title("t-SNE Embeddings (Test Classes)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()


# In[ ]:




