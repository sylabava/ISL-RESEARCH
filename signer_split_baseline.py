#!/usr/bin/env python
# coding: utf-8

# In[16]:


# ============================================================
# CELL 1 — IMPORTS + CONFIG + TRANSFORMS
# ============================================================

import os
import random
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, models

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
class Config:
    data_root = "/workspace/a_meta/expt1/data"   # your dataset path
    image_size = 224                             # use 224 for CNN backbones
    ways = 5                                      # default N-way
    shots = 5                                     # default K-shot
    queries = 10                                  # per-class queries
    inner_steps = 1                               # MAML inner-loop steps
    inner_lr = 0.01                               # MAML inner-loop LR
    meta_lr = 1e-3                                # meta LR

cfg = Config()

# ------------------------------------------------------------
# IMAGE TRANSFORMS
# ------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((cfg.image_size, cfg.image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((cfg.image_size, cfg.image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

print("Config + transforms loaded successfully.")


# In[17]:


# ============================================================
# CELL 2 — CLASS-FIRST DATASET LOADER
# ============================================================

class ISLDatasetClassFirst:
    """
    Dataset Layout:
        data_root/class_name/signer_name/*.jpg

    Produces a list of:
        (image_path, class_name, signer_name)
    """

    def __init__(self, data_root):
        self.data_root = Path(data_root)

        self.samples = []            # (path, class_name, signer)
        self.class_to_idx = {}
        signer_set = set()

        # Walk through dataset
        class_dirs = sorted([p for p in self.data_root.iterdir() if p.is_dir()])

        for cls_dir in class_dirs:
            class_name = cls_dir.name
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)

            for signer_dir in sorted(cls_dir.iterdir()):
                if not signer_dir.is_dir():
                    continue

                signer_name = signer_dir.name
                signer_set.add(signer_name)

                for img in signer_dir.glob("*"):
                    if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        self.samples.append((str(img), class_name, signer_name))

        self.signers = sorted(list(signer_set))
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        print("Dataset loaded.")
        print("Classes found:", len(self.class_to_idx))
        print("Signers found:", self.signers)
        print("Total images:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]  # (path, class_name, signer)


# Instantiate dataset
ds = ISLDatasetClassFirst(cfg.data_root)


# In[18]:


# ============================================================
# CELL 3C — FIXED SIGNER VISUALIZATION
# ============================================================

import matplotlib.pyplot as plt

def show_signer_images(dataset, num_images=4):
    """
    Visualize random images for:
        - Train signers (User_1 ... User_4)
        - Validation signer (User_5)
        - Test signer (User_6)
    """

    train_signers = dataset.signers[:4]
    val_signer    = dataset.signers[4]
    test_signer   = dataset.signers[5]

    # pick one random class
    cls = random.choice(list(dataset.class_to_idx.keys()))
    print(f"\n📌 Showing samples for CLASS = '{cls}'")

    # gather all samples for this class
    cls_samples = [(p, s) for (p, c, s) in dataset.samples if c == cls]

    def pick_images(signer):
        imgs = [p for (p, s) in cls_samples if s == signer]
        if len(imgs) == 0:
            return []
        return random.sample(imgs, min(num_images, len(imgs)))

    # total rows:
    # 4 (train signers) + 1 (val) + 1 (test) = 6
    total_rows = len(train_signers) + 2
    total_cols = num_images

    plt.figure(figsize=(5 * total_cols, 3 * total_rows))

    row = 1
    col = 1
    idx = 1

    # -------------------
    # TRAIN SIGNERS
    # -------------------
    for signer in train_signers:
        paths = pick_images(signer)
        for p in paths:
            img = Image.open(p).convert("RGB")

            plt.subplot(total_rows, total_cols, idx)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"TRAIN: {signer}")
            idx += 1

    # -------------------
    # VALIDATION SIGNER
    # -------------------
    paths = pick_images(val_signer)
    for p in paths:
        img = Image.open(p).convert("RGB")

        plt.subplot(total_rows, total_cols, idx)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"VAL: {val_signer}")
        idx += 1

    # -------------------
    # TEST SIGNER
    # -------------------
    paths = pick_images(test_signer)
    for p in paths:
        img = Image.open(p).convert("RGB")

        plt.subplot(total_rows, total_cols, idx)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"TEST: {test_signer}")
        idx += 1

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# RUN fix visualization
# ------------------------------------------------------------
show_signer_images(ds, num_images=4)


# In[19]:


# ============================================================
# CELL S1 — SUPERVISED DATASET (TRAIN/VAL/TEST)
# ============================================================

from torch.utils.data import Dataset, DataLoader

class ISLSupervisedDataset(Dataset):
    """
    Converts your CLASS-FIRST dataset into a supervised dataset:
        (image, class_index)
    And allows signer-based splitting:
        train: User_1..User_4
        val:   User_5
        test:  User_6
    """
    def __init__(self, dataset, split="train", transform=None):
        self.transform = transform
        self.items = []

        if split == "train":
            allowed_signers = dataset.signers[:4]
        elif split == "val":
            allowed_signers = [dataset.signers[4]]
        elif split == "test":
            allowed_signers = [dataset.signers[5]]
        else:
            raise ValueError("split must be train/val/test")

        for path, cls, signer in dataset.samples:
            if signer in allowed_signers:
                cls_idx = dataset.class_to_idx[cls]
                self.items.append((path, cls_idx))

        print(f"{split.upper()} dataset created: {len(self.items)} samples | allowed signers = {allowed_signers}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# Create splits
train_ds = ISLSupervisedDataset(ds, split="train", transform=train_transform)
val_ds   = ISLSupervisedDataset(ds, split="val",   transform=eval_transform)
test_ds  = ISLSupervisedDataset(ds, split="test",  transform=eval_transform)

# Build dataloaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

print("Supervised dataloaders ready.")


# In[20]:


def build_classmap(samples):
    classmap = defaultdict(list)
    for i, (_, cls) in enumerate(samples):
        classmap[cls].append(i)
    return classmap


# In[21]:


from collections import defaultdict

def build_classmap_from_dataset(dataset):
    classmap = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        classmap[label].append(idx)
    return classmap


# In[24]:


# Create splits
train_ds = ISLSupervisedDataset(ds, split="train", transform=train_transform)
val_ds   = ISLSupervisedDataset(ds, split="val",   transform=eval_transform)
test_ds  = ISLSupervisedDataset(ds, split="test",  transform=eval_transform)

# Dataloaders (already correct)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

print("Supervised dataloaders ready.")

# # ✅ Build classmaps for ProtoNet / ProtoMAML
# train_classmap = build_classmap_from_dataset(train_ds)
# val_classmap   = build_classmap_from_dataset(val_ds)
# test_classmap  = build_classmap_from_dataset(test_ds)


# In[25]:


def load_user_split(root, allowed_users):
    samples = []
    for cls in os.listdir(root):
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path):
            continue
        for user in os.listdir(cls_path):
            if user not in allowed_users:
                continue
            user_path = os.path.join(cls_path, user)
            for img in os.listdir(user_path):
                if img.lower().endswith((".jpg", ".png", ".jpeg")):
                    samples.append((os.path.join(user_path, img), cls))
    return samples


# In[26]:


# ============================================================
# CELL S_NEW — RESNET12 SUPERVISED BASELINE
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity
        out = F.relu(out)
        out = self.pool(out)

        return out


class ResNet12Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = nn.Sequential(
            ResBlock(3, 64),
            ResBlock(64, 128),
            ResBlock(128, 256),
            ResBlock(256, 512)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# In[27]:


def train_resnet12(epochs=10, lr=1e-3):

    print("\n🚀 Training ResNet12 Baseline...")

    model = ResNet12Classifier(
        num_classes=len(ds.class_to_idx)
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):

        # ---------- TRAIN ----------
        model.train()
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # ---------- VAL ----------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(f"Epoch {epoch}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    print("\n🎉 ResNet12 Training Complete!")

    return model


# In[29]:


device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# In[30]:


resnet12_model = train_resnet12(
    epochs=10,
    lr=1e-3
)


# In[33]:


# ============================================================
# DEFINE EVALUATION FUNCTION (SAFE RE-COPY)
# ============================================================

def evaluate_supervised(model, loader):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            preds = logits.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"\n📊 TEST ACCURACY: {acc:.4f}")
    return acc


# In[34]:


print("\n🚀 Evaluating ResNet12 on TEST signer (User_6)...")

resnet12_test_acc = evaluate_supervised(
    resnet12_model,
    test_loader
)

print(f"\n📌 FINAL TEST ACCURACY (ResNet12, User 6): {resnet12_test_acc:.4f}")


# In[1]:


import matplotlib.pyplot as plt

# Backbone names
backbones = ['SimpleCNN', 'ResNet12', 'ResNet18', 'ResNet34']

# Test accuracies
accuracies = [0.22, 0.385, 0.47, 0.25]

# Convert to percentage (optional)
accuracies_percent = [a * 100 for a in accuracies]

# Create bar chart
plt.figure(figsize=(8,5))
bars = plt.bar(backbones, accuracies_percent)

# Labels and title
plt.xlabel("Backbone Architecture")
plt.ylabel("Test Accuracy (%)")
plt.title("Backbone Performance on Test Signer (User_6)")

# Show values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height + 1,
             f'{height:.1f}%',
             ha='center')

plt.ylim(0, 55)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# In[38]:


import torch

def test_few_samples(model, dataset, class_names, num_samples=5):

    model.eval()

    for i in range(num_samples):

        img, true_label = dataset[i]

        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img)
            pred = logits.argmax(1).item()

        print(f"True Label: {class_names[true_label]}")
        print(f"Predicted : {class_names[pred]}")
        print("-" * 40)


# In[35]:


# ============================================================
# CELL S2 — SIMPLE CNN BASELINE (CONV4) FOR SUPERVISED TRAINING
# ============================================================

class SimpleConvNet(nn.Module):
    """
    A small CNN similar to conv4 but ending in a classifier.
    Good baseline for image classification.
    """

    def __init__(self, num_classes=20):
        super().__init__()
        self.encoder = nn.Sequential(
            self.block(3, 64),
            self.block(64, 64),
            self.block(64, 64),
            self.block(64, 64)
        )
        self.classifier = nn.Linear(64 * (cfg.image_size // 16) * (cfg.image_size // 16), num_classes)

    def block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ------------------------------------------------------------
# TRAINING FUNCTION
# ------------------------------------------------------------

def train_supervised(model, train_loader, val_loader, epochs=10, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):

        # -------------------------
        # TRAIN
        # -------------------------
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(f"Epoch {epoch}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return model


# In[36]:


# ============================================================
# CELL S3 — TRAIN SIMPLE CNN BASELINE
# ============================================================

num_classes = len(ds.class_to_idx)

simple_cnn = SimpleConvNet(num_classes=num_classes)

print("\n🚀 Training Simple CNN Baseline...")
simple_cnn = train_supervised(
    model=simple_cnn,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,        # you can increase to 20–30 for better accuracy
    lr=1e-3
)

print("\n🎉 Training complete!")


# In[37]:


# ============================================================
# CELL S4 — EVALUATE SIMPLE CNN ON TEST SIGNER (USER 6)
# ============================================================

def evaluate_supervised(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"\n📊 TEST ACCURACY (Unseen signer - User 6): {acc:.4f}")
    return acc


print("\n🚀 Evaluating Simple CNN baseline on TEST set...")
test_accuracy_cnn = evaluate_supervised(simple_cnn, test_loader)


# In[ ]:


# ============================================================
# CELL S5 — RESNET-18 SUPERVISED BASELINE
# ============================================================

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.backbone = models.resnet18(weights=None)  # training from scratch
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)


def train_resnet18(epochs=10, lr=1e-3):
    print("\n🚀 Training ResNet-18 Baseline...")

    model = ResNet18Classifier(num_classes=len(ds.class_to_idx)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
        # -------------------
        # TRAIN
        # -------------------
        model.train()
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # -------------------
        # VALIDATION
        # -------------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(f"Epoch {epoch}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    print("\n🎉 ResNet-18 Training Complete!")

    return model


# In[23]:


# ============================================================
# CELL S6 — RUN TRAINING FOR RESNET-18
# ============================================================

resnet18_model = train_resnet18(epochs=10, lr=1e-3)


# In[25]:


# ============================================================
# CELL S7 — TEST EVALUATION FOR RESNET-18
# ============================================================

print("\n🚀 Evaluating ResNet-18 on TEST signer (User_6)...")
resnet18_test_acc = evaluate_supervised(resnet18_model, test_loader)

print(f"\n📌 FINAL TEST ACCURACY (ResNet-18, User 6): {resnet18_test_acc:.4f}")


# In[26]:


# ============================================================
# CELL S8 — RESNET-50 SUPERVISED BASELINE
# ============================================================

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.backbone(x)


def train_resnet50(epochs=10, lr=1e-3):
    print("\n🚀 Training ResNet-50 Baseline...")

    model = ResNet50Classifier(num_classes=len(ds.class_to_idx)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
        # --------------- TRAIN ---------------
        model.train()
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # --------------- VALIDATION ---------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(f"Epoch {epoch}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    print("\n🎉 ResNet-50 Training Complete!")
    return model


# In[27]:


# ============================================================
# CELL S9 — RUN RESNET-50 TRAINING
# ============================================================

resnet50_model = train_resnet50(epochs=10, lr=1e-3)


# In[28]:


# ============================================================
# CELL S10 — TEST EVALUATION FOR RESNET-50
# ============================================================

print("\n🚀 Evaluating ResNet-50 on TEST signer (User_6)...")
resnet50_test_acc = evaluate_supervised(resnet50_model, test_loader)

print(f"\n📌 FINAL TEST ACCURACY (ResNet-50, User 6): {resnet50_test_acc:.4f}")


# In[29]:


# ============================================================
# CELL S11 — MOBILENETV2 SUPERVISED BASELINE
# ============================================================

class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        self.backbone.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.backbone(x)


def train_mobilenetv2(epochs=10, lr=1e-3):
    print("\n🚀 Training MobileNetV2 Baseline...")

    model = MobileNetV2Classifier(num_classes=len(ds.class_to_idx)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
        # Training
        model.train()
        total = 0
        correct = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(f"Epoch {epoch}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    print("\n🎉 MobileNetV2 Training Complete!")
    return model


# In[30]:


# ============================================================
# CELL S12 — TRAIN MOBILENETV2
# ============================================================

mobilenet_model = train_mobilenetv2(
    epochs=10,
    lr=1e-3
)


# In[31]:


# ============================================================
# CELL S13 — TEST MOBILENETV2 ON USER 6
# ============================================================

print("\n🚀 Evaluating MobileNetV2 on TEST signer (User_6)...")
mobilenet_test_acc = evaluate_supervised(mobilenet_model, test_loader)

print(f"\n📌 FINAL TEST ACCURACY (MobileNetV2, User 6): {mobilenet_test_acc:.4f}")


# In[32]:


# ============================================================
# CELL S14 — EFFICIENTNET-B0 SUPERVISED BASELINE
# ============================================================

class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def train_efficientnet_b0(epochs=10, lr=1e-3):
    print("\n🚀 Training EfficientNet-B0 Baseline...")

    model = EfficientNetB0Classifier(num_classes=len(ds.class_to_idx)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
        # ---------------- TRAIN ----------------
        model.train()
        total = 0
        correct = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # ---------------- VAL ----------------
        model.eval()
        total = 0
        correct = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(f"Epoch {epoch}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    print("\n🎉 EfficientNet-B0 Training Complete!")
    return model


# In[33]:


# ============================================================
# CELL S15 — RUN EFFICIENTNET-B0 TRAINING
# ============================================================

efficientnet_model = train_efficientnet_b0(
    epochs=10,
    lr=1e-3
)


# In[34]:


# ============================================================
# CELL S16 — TEST EfficientNet-B0 ON USER 6
# ============================================================

print("\n🚀 Evaluating EfficientNet-B0 on TEST signer (User_6)...")
efficientnet_test_acc = evaluate_supervised(efficientnet_model, test_loader)

print(f"\n📌 FINAL TEST ACCURACY (EfficientNet-B0, User 6): {efficientnet_test_acc:.4f}")


# In[ ]:


# ============================================================
# CELL P1 — TRAINING FUNCTION THAT STORES ACCURACY CURVES
# ============================================================

def train_supervised_with_curves(model, train_loader, val_loader, epochs=10, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_acc_curve = []
    val_acc_curve = []

    for epoch in range(1, epochs+1):

        # --------------------------- TRAIN ------------------------------
        model.train()
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_acc_curve.append(train_acc)

        # --------------------------- VALIDATION -------------------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        val_acc_curve.append(val_acc)

        print(f"Epoch {epoch}/{epochs} | Train={train_acc:.4f} | Val={val_acc:.4f}")

    return model, train_acc_curve, val_acc_curve


# In[ ]:


# ============================================================
# CELL P2 — PLOT TRAINING & VALIDATION ACCURACY CURVES
# ============================================================

import matplotlib.pyplot as plt

def plot_accuracy_curves(train_curve, val_curve, title="Accuracy Curve"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_curve, label="Train Accuracy", marker='o')
    plt.plot(val_curve, label="Validation Accuracy", marker='s')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.show()


# In[ ]:


# Train with curve storage
resnet18_model = ResNet18Classifier(num_classes=len(ds.class_to_idx))
resnet18_model, train_curve_18, val_curve_18 = train_supervised_with_curves(
    resnet18_model,
    train_loader,
    val_loader,
    epochs=10,
    lr=1e-3
)

# Plot
plot_accuracy_curves(train_curve_18, val_curve_18, title="ResNet-18 Accuracy Curve")


# In[ ]:


# ============================================================
# CELL P4 — BAR PLOT OF TEST ACCURACIES FOR ALL BACKBONES
# ============================================================

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# ENTER YOUR TEST ACCURACIES HERE
# (Use the values printed earlier)
# ------------------------------------------------------------
acc_simplecnn     = 0.22      # <-- replace with your actual simple CNN test accuracy
acc_resnet18      = 0.4683    # good performer
acc_resnet34      = 0.2527
#acc_mobilenetv2   = 0.3450
#acc_efficientnet  = 0.3507

# Labels & accuracy list
models = [
    "SimpleCNN",
    "ResNet18",
    "ResNet34",

]

test_accs = [
    acc_simplecnn,
    acc_resnet18,
    acc_resnet34,
    acc_mobilenetv2,
    acc_efficientnet
]

# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------
plt.figure(figsize=(10,6))
bars = plt.bar(models, test_accs)

# Annotate bar values
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
             f"{height:.2f}", ha='center', fontsize=12)

plt.ylim(0, 1.0)
plt.ylabel("Test Accuracy (User 6)")
plt.title("Supervised Baselines — Test Accuracy Comparison")
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.show()


# In[ ]:


print("Testing safe backbones...")

for b in ["conv4", "resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0"]:
    enc = get_backbone(b).to(device)
    x = torch.randn(2, 3, cfg.image_size, cfg.image_size).to(device)
    out = enc(x)
    print(f"{b:15s} → output shape {tuple(out.shape)}")


# In[ ]:


from sklearn.manifold import TSNE

def plot_tsne(embeddings, labels, title):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    z2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10,10))
    plt.scatter(z2d[:,0], z2d[:,1], c=labels, cmap="tab20", alpha=0.7)
    plt.title(title)
    plt.colorbar()
    plt.show()


# In[126]:


emb_train, lbl_train = extract_protonet_embeddings(
    protonet_model, sampler, split="train"
)
plot_tsne(emb_train, lbl_train, "ProtoNet Train Latent Space")

emb_val, lbl_val = extract_protonet_embeddings(
    protonet_model, sampler, split="val"
)
plot_tsne(emb_val, lbl_val, "ProtoNet Val Latent Space")

emb_test, lbl_test = extract_protonet_embeddings(
    protonet_model, sampler, split="test"
)
plot_tsne(emb_test, lbl_test, "ProtoNet Test Latent Space")


# In[ ]:




