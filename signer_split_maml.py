#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# In[2]:


import torch.nn as nn
from torchvision import models

# ---------- Conv4 ----------
class Conv4(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.encoder(x).view(x.size(0), -1)


# ---------- Backbone Loader ----------
def get_backbone(name):
    if name == "conv4":
        return Conv4().to(DEVICE)

    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    else:
        raise ValueError("Unknown backbone")

    encoder = nn.Sequential(*list(model.children())[:-1])
    return encoder.to(DEVICE)


# In[5]:


import random

def sample_episode(dataset, classmap, N, K, Q=15):
    valid_classes = [c for c, idxs in classmap.items() if len(idxs) >= K + Q]
    N_eff = min(N, len(valid_classes))
    classes = random.sample(valid_classes, N_eff)

    s_x, s_y, q_x, q_y = [], [], [], []

    for i, cls in enumerate(classes):
        idxs = random.sample(classmap[cls], K + Q)

        for j in idxs[:K]:
            x, _ = dataset[j]
            s_x.append(x)
            s_y.append(i)

        for j in idxs[K:]:
            x, _ = dataset[j]
            q_x.append(x)
            q_y.append(i)

    return (
        torch.stack(s_x).to(DEVICE),
        torch.tensor(s_y).to(DEVICE),
        torch.stack(q_x).to(DEVICE),
        torch.tensor(q_y).to(DEVICE),
        N_eff
    )


# In[6]:


import torch.nn.functional as F

class MAML(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        z = self.encoder(x)
        if z.dim() == 4:
            z = z.view(z.size(0), -1)
        return z


# In[7]:


def inner_loop(model, s_x, s_y, n_way, inner_lr=0.4, steps=1):
    z = model(s_x)

    feat_dim = z.size(1)
    clf = nn.Linear(feat_dim, n_way).to(DEVICE)
    opt = torch.optim.SGD(clf.parameters(), lr=inner_lr)

    for _ in range(steps):
        logits = clf(z)
        loss = F.cross_entropy(logits, s_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return clf


# In[8]:


def inner_loop_eval(model, s_x, s_y, n_way, inner_lr=0.4, steps=5):
    with torch.no_grad():
        z = model(s_x).detach()

    feat_dim = z.size(1)
    clf = nn.Linear(feat_dim, n_way).to(DEVICE)
    opt = torch.optim.SGD(clf.parameters(), lr=inner_lr)

    for _ in range(steps):
        logits = clf(z)
        loss = F.cross_entropy(logits, s_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return clf


# In[9]:


def train_maml(model, dataset, classmap,
               N=5, K=5, Q=15,
               meta_iters=300, meta_lr=1e-3):

    meta_opt = torch.optim.Adam(model.parameters(), lr=meta_lr)
    model.train()

    for it in range(1, meta_iters + 1):

        s_x, s_y, q_x, q_y, N_eff = sample_episode(
            dataset, classmap, N, K, Q
        )

        clf = inner_loop(model, s_x, s_y, N_eff)

        logits_q = clf(model(q_x))
        loss = F.cross_entropy(logits_q, q_y)

        meta_opt.zero_grad()
        loss.backward()
        meta_opt.step()

        if it % 50 == 0:
            acc = (logits_q.argmax(1) == q_y).float().mean().item()
            print(f"[MAML] Iter {it} | Loss={loss:.3f} | Acc={acc:.3f}")


# In[10]:


def eval_maml(model, dataset, classmap,
              N, K, Q=15, episodes=100):

    model.eval()
    accs = []

    for _ in range(episodes):
        s_x, s_y, q_x, q_y, N_eff = sample_episode(
            dataset, classmap, N, K, Q
        )

        clf = inner_loop_eval(model, s_x, s_y, N_eff)

        with torch.no_grad():
            logits = clf(model(q_x))
            acc = (logits.argmax(1) == q_y).float().mean().item()
            accs.append(acc)

    return sum(accs) / len(accs)


# In[12]:


import os
from PIL import Image
from torch.utils.data import Dataset

class ISLSupervisedDataset(Dataset):

    SPLITS = {
        "train": ["User_1", "User_2", "User_3", "User_4"],
        "val":   ["User_5"],
        "test":  ["User_6"],
    }

    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.allowed_users = self.SPLITS[split]

        self.samples = []
        self.class_to_idx = {}

        self._build_index()

    def _build_index(self):
        class_names = sorted(os.listdir(self.root))

        for class_idx, cls in enumerate(class_names):
            class_path = os.path.join(self.root, cls)
            if not os.path.isdir(class_path):
                continue

            self.class_to_idx[cls] = class_idx

            for user in os.listdir(class_path):
                if user not in self.allowed_users:
                    continue

                user_path = os.path.join(class_path, user)

                for fname in os.listdir(user_path):
                    if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                        self.samples.append(
                            (os.path.join(user_path, fname), class_idx)
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# In[13]:


from torchvision import transforms

IMG_SIZE = 84

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# In[14]:


DATA_ROOT = "/workspace/a_meta/expt1/data"

train_ds = ISLSupervisedDataset(DATA_ROOT, split="train", transform=train_transform)
val_ds   = ISLSupervisedDataset(DATA_ROOT, split="val",   transform=eval_transform)
test_ds  = ISLSupervisedDataset(DATA_ROOT, split="test",  transform=eval_transform)

print("Train:", len(train_ds))
print("Val:", len(val_ds))
print("Test:", len(test_ds))


# In[15]:


from collections import defaultdict

def build_classmap(dataset):
    classmap = defaultdict(list)
    for i in range(len(dataset)):
        _, y = dataset[i]
        classmap[y].append(i)
    return classmap

train_classmap = build_classmap(train_ds)
val_classmap   = build_classmap(val_ds)
test_classmap  = build_classmap(test_ds)

print("Classes:", len(train_classmap))


# In[ ]:





# In[16]:


backbones = ["conv4", "resnet18", "resnet34"]
N_list = [2, 3]
K_list = [1, 5, 10, 20]

results = []

for bb in backbones:
    print(f"\n===== MAML BACKBONE: {bb.upper()} =====")

    encoder = get_backbone(bb)
    model = MAML(encoder).to(DEVICE)

    # Meta-train
    train_maml(
        model,
        train_ds,
        train_classmap,
        N=5,
        K=5
    )

    # Evaluate
    for N in N_list:
        for K in K_list:
            val_acc = eval_maml(model, val_ds, val_classmap, N, K)
            test_acc = eval_maml(model, test_ds, test_classmap, N, K)

            results.append({
                "Method": "MAML",
                "Backbone": bb,
                "N_way": N,
                "K_shot": K,
                "Val_Acc": val_acc,
                "Test_Acc": test_acc
            })

            print(f"N={N}, K={K} | VAL={val_acc:.4f} | TEST={test_acc:.4f}")


# In[17]:


import pandas as pd

# Convert to DataFrame
df_maml = pd.DataFrame(results)

# Save to CSV
csv_name = "signer_maml_NK_backbone_results.csv"
df_maml.to_csv(csv_name, index=False)

print(f"Results saved to {csv_name}")
print(df_maml.head())


# In[19]:


import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Basic Residual Block --------
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.LeakyReLU(0.1)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        out = self.pool(out)

        return out


# -------- ResNet12 --------
class ResNet12(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = BasicBlock(3, 64)
        self.layer2 = BasicBlock(64, 160)
        self.layer3 = BasicBlock(160, 320)
        self.layer4 = BasicBlock(320, 640)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        return x.view(x.size(0), -1)


# In[20]:


# =====================================
# Run MAML with ResNet12 Backbone Only
# =====================================

import pandas as pd

print("\n===== MAML BACKBONE: RESNET12 =====")

# Create encoder and model
encoder = ResNet12().to(DEVICE)
model = MAML(encoder).to(DEVICE)

# Meta-train
train_maml(
    model,
    train_ds,
    train_classmap,
    N=5,
    K=5
)

# Evaluate
N_list = [2, 3]
K_list = [1, 5, 10, 20]

results_resnet12_maml = []

for N in N_list:
    for K in K_list:

        val_acc = eval_maml(model, val_ds, val_classmap, N, K)
        test_acc = eval_maml(model, test_ds, test_classmap, N, K)

        results_resnet12_maml.append({
            "Method": "MAML",
            "Backbone": "resnet12",
            "N_way": N,
            "K_shot": K,
            "Val_Acc": val_acc,
            "Test_Acc": test_acc
        })

        print(f"N={N}, K={K} | VAL={val_acc:.4f} | TEST={test_acc:.4f}")

# Save results
df_resnet12_maml = pd.DataFrame(results_resnet12_maml)
df_resnet12_maml.to_csv("resnet12_maml_results.csv", index=False)

print("Saved resnet12_maml_results.csv")


# In[21]:


import pandas as pd

# Load existing MAML results
df_main = pd.read_csv("signer_maml_NK_backbone_results.csv")

# Load ResNet12 MAML results
df_resnet12 = pd.read_csv("resnet12_maml_results.csv")

print("Main shape:", df_main.shape)
print("ResNet12 shape:", df_resnet12.shape)

# Combine both
df_combined = pd.concat([df_main, df_resnet12], ignore_index=True)

# Remove duplicates (safe practice)
df_combined = df_combined.drop_duplicates()

print("Combined shape:", df_combined.shape)

# Save merged file
output_name = "signer_maml_with_resnet12.csv"
df_combined.to_csv(output_name, index=False)

print(f"Saved combined file as: {output_name}")
df_combined.head()


# In[ ]:


macroavg roc


# In[25]:


import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize

def eval_maml_with_scores(model, dataset, classmap, N, K, Q=15, episodes=100):

    model.eval()

    all_true = []
    all_scores = []

    for _ in range(episodes):

        s_x, s_y, q_x, q_y, N_eff = sample_episode(
            dataset, classmap, N, K, Q
        )

        if N_eff < N:
            continue

        clf = inner_loop_eval(model, s_x, s_y, N_eff)

        with torch.no_grad():
            z_q = model(q_x)
            logits = clf(z_q)
            probs = torch.softmax(logits, dim=1)

        if probs.shape[1] < 2:
            continue

        all_true.append(q_y.cpu().numpy())
        all_scores.append(probs.cpu().numpy())

    return np.concatenate(all_true), np.concatenate(all_scores)


# In[ ]:


macro avg roc with ci


# In[26]:


import matplotlib.pyplot as plt

N_list = [2, 3]
K_list = [1, 5, 10, 20]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))

for row, N in enumerate(N_list):
    for col, K in enumerate(K_list):

        print(f"Processing MAML: N={N}, K={K}")

        y_true, y_score = eval_maml_with_scores(
            model,
            test_ds,
            test_classmap,
            N,
            K
        )

        ax = axes[row, col]

        # -------- Binary Case --------
        if y_score.shape[1] == 2:

            fpr, tpr, _ = roc_curve(
                y_true,
                y_score[:, 1]
            )

            auc_val = roc_auc_score(
                y_true,
                y_score[:, 1]
            )

            ax.plot(fpr, tpr, linewidth=2)

        # -------- Multi-class Case --------
        else:

            n_classes = y_score.shape[1]
            y_true_bin = label_binarize(
                y_true,
                classes=range(n_classes)
            )

            auc_val = roc_auc_score(
                y_true_bin,
                y_score,
                multi_class="ovr"
            )

            # Macro-average ROC curve
            fpr_grid = np.linspace(0, 1, 200)
            mean_tpr = np.zeros_like(fpr_grid)

            for i in range(n_classes):
                fpr_i, tpr_i, _ = roc_curve(
                    y_true_bin[:, i],
                    y_score[:, i]
                )
                mean_tpr += np.interp(fpr_grid, fpr_i, tpr_i)

            mean_tpr /= n_classes

            ax.plot(fpr_grid, mean_tpr, linewidth=2)

        ax.plot([0,1],[0,1],'k--', alpha=0.3)

        ax.set_title(f"N={N}, K={K}\nAUC={auc_val:.3f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True)

plt.tight_layout()
plt.show()

fig.savefig("MAML_Macro_ROC.png", dpi=300, bbox_inches="tight")


# In[22]:


import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize

def eval_maml_with_scores(model, dataset, classmap, N, K, Q=15, episodes=100):

    model.eval()

    all_true = []
    all_scores = []

    for _ in range(episodes):

        s_x, s_y, q_x, q_y, N_eff = sample_episode(
            dataset, classmap, N, K, Q
        )

        if N_eff < N:
            continue

        clf = inner_loop_eval(model, s_x, s_y, N_eff)

        with torch.no_grad():
            z_q = model(q_x)
            logits = clf(z_q)
            probs = torch.softmax(logits, dim=1)

        if probs.shape[1] < 2:
            continue

        all_true.append(q_y.cpu().numpy())
        all_scores.append(probs.cpu().numpy())

    return np.concatenate(all_true), np.concatenate(all_scores)


# In[23]:


def bootstrap_macro_roc(y_true, y_score, n_bootstraps=500):

    n_classes = y_score.shape[1]
    fpr_grid = np.linspace(0, 1, 200)

    tprs = []
    aucs = []

    for _ in range(n_bootstraps):

        indices = np.random.choice(
            len(y_true),
            len(y_true),
            replace=True
        )

        if n_classes == 2:

            fpr, tpr, _ = roc_curve(
                y_true[indices],
                y_score[indices, 1]
            )

            auc_val = roc_auc_score(
                y_true[indices],
                y_score[indices, 1]
            )

            interp_tpr = np.interp(fpr_grid, fpr, tpr)

        else:

            y_true_bin = label_binarize(
                y_true[indices],
                classes=range(n_classes)
            )

            auc_val = roc_auc_score(
                y_true_bin,
                y_score[indices],
                multi_class="ovr"
            )

            interp_tpr = np.zeros_like(fpr_grid)

            for i in range(n_classes):
                fpr_i, tpr_i, _ = roc_curve(
                    y_true_bin[:, i],
                    y_score[indices, i]
                )
                interp_tpr += np.interp(fpr_grid, fpr_i, tpr_i)

            interp_tpr /= n_classes

        tprs.append(interp_tpr)
        aucs.append(auc_val)

    tprs = np.array(tprs)
    aucs = np.array(aucs)

    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)

    auc_mean = aucs.mean()
    auc_ci_lower = np.percentile(aucs, 2.5)
    auc_ci_upper = np.percentile(aucs, 97.5)

    return fpr_grid, mean_tpr, std_tpr, auc_mean, auc_ci_lower, auc_ci_upper


# In[24]:


import matplotlib.pyplot as plt

N_list = [2, 3]
K_list = [1, 5, 10, 20]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))

for row, N in enumerate(N_list):
    for col, K in enumerate(K_list):

        print(f"Processing MAML: N={N}, K={K}")

        y_true, y_score = eval_maml_with_scores(
            model,
            test_ds,
            test_classmap,
            N,
            K
        )

        ax = axes[row, col]

        fpr, mean_tpr, std_tpr, auc_mean, ci_low, ci_high = \
            bootstrap_macro_roc(y_true, y_score)

        ax.plot(fpr, mean_tpr, linewidth=2)

        ax.fill_between(
            fpr,
            mean_tpr - std_tpr,
            mean_tpr + std_tpr,
            alpha=0.3
        )

        ax.plot([0,1],[0,1],'k--', alpha=0.3)

        ax.set_title(
            f"N={N}, K={K}\nAUC={auc_mean:.3f} "
            f"[{ci_low:.3f}, {ci_high:.3f}]"
        )

        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.grid(True)

plt.tight_layout()
plt.show()

fig.savefig("MAML_Macro_ROC_With_CI.png", dpi=300, bbox_inches="tight")


# In[ ]:




