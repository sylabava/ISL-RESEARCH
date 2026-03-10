#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# ============================================================
# FULL MAML PIPELINE (First-Order)
# N = {2,3}
# K = {1,5,10,20}
# Backbones = conv4, resnet12, resnet18, resnet34
# Includes:
#   - Training
#   - Evaluation
#   - Heatmaps
#   - Confusion Matrix
#   - ROC Curves
#   - t-SNE
#   - CSV export
# ============================================================

import os
import random
from datetime import datetime
from collections import defaultdict
import copy

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
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/workspace/a_meta/expt1/data"

N_VALUES = [2, 3]
K_VALUES = [1, 5, 10, 20]
BACKBONES = ["conv4", "resnet12", "resnet18", "resnet34"]

Q_QUERY = 15
META_ITERS = 200
INNER_STEPS = 5
INNER_LR = 0.01
META_LR = 1e-3
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
                if img.lower().endswith((".jpg",".png",".jpeg")):
                    samples.append((os.path.join(user_path,img), cls))
    return samples


all_classes = sorted([c for c in os.listdir(DATA_ROOT)
                      if os.path.isdir(os.path.join(DATA_ROOT,c))])

random.shuffle(all_classes)

train_classes = all_classes[:14]
val_classes = all_classes[14:17]
test_classes = all_classes[17:20]

train_samples = load_samples(train_classes)
val_samples = load_samples(val_classes)
test_samples = load_samples(test_classes)

train_class_to_idx = {c:i for i,c in enumerate(train_classes)}
val_class_to_idx = {c:i for i,c in enumerate(val_classes)}
test_class_to_idx = {c:i for i,c in enumerate(test_classes)}

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
        def block(i,o):
            return nn.Sequential(
                nn.Conv2d(i,o,3,padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        self.encoder = nn.Sequential(
            block(3,64),
            block(64,64),
            block(64,64),
            block(64,64)
        )
    def forward(self,x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x,1)
        return x.view(x.size(0),-1)

class BasicBlock(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c,out_c,3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c,out_c,3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(out_c,out_c,3,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.relu = nn.LeakyReLU(0.1,inplace=True)
        self.downsample = None
        if in_c!=out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c,out_c,1,bias=False),
                nn.BatchNorm2d(out_c)
            )
        self.pool = nn.MaxPool2d(2)
    def forward(self,x):
        identity=x
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.relu(self.bn2(self.conv2(out)))
        out=self.bn3(self.conv3(out))
        if self.downsample:
            identity=self.downsample(identity)
        out=self.relu(out+identity)
        return self.pool(out)

class ResNet12(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=BasicBlock(3,64)
        self.layer2=BasicBlock(64,160)
        self.layer3=BasicBlock(160,320)
        self.layer4=BasicBlock(320,640)
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=F.adaptive_avg_pool2d(x,1)
        return x.view(x.size(0),-1)

def get_resnet(name):
    model=models.resnet18(weights=None) if name=="resnet18" else models.resnet34(weights=None)
    layers=list(model.children())[:-1]
    enc=nn.Sequential(*layers)
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc=enc
        def forward(self,x):
            x=self.enc(x)
            return x.view(x.size(0),-1)
    return Wrapper()

def get_backbone(name):
    if name=="conv4": return Conv4().to(DEVICE)
    if name=="resnet12": return ResNet12().to(DEVICE)
    if name in ["resnet18","resnet34"]:
        return get_resnet(name).to(DEVICE)

# ============================================================
# EPISODE SAMPLER
# ============================================================

def sample_episode(dataset,classmap,n,k,q):
    valid=[c for c,idx in classmap.items() if len(idx)>=k+q]
    chosen=random.sample(valid,n)
    sx,sy,qx,qy=[],[],[],[]
    for new_label,cls in enumerate(chosen):
        idxs=random.sample(classmap[cls],k+q)
        for i in idxs[:k]:
            img,_=dataset[i]; sx.append(img); sy.append(new_label)
        for i in idxs[k:]:
            img,_=dataset[i]; qx.append(img); qy.append(new_label)
    return (torch.stack(sx).to(DEVICE),
            torch.tensor(sy).to(DEVICE),
            torch.stack(qx).to(DEVICE),
            torch.tensor(qy).to(DEVICE))

# ============================================================
# MAML TRAINING (First-Order)
# ============================================================

def train_maml(backbone,N,K):

    encoder = backbone
    meta_opt = torch.optim.Adam(encoder.parameters(), lr=META_LR)

    for _ in range(META_ITERS):

        sx,sy,qx,qy = sample_episode(train_dataset,train_classmap,N,K,Q_QUERY)

        # Clone encoder for task
        fast_encoder = copy.deepcopy(encoder)

        # Task-specific classifier
        feat_s = fast_encoder(sx)
        clf = nn.Linear(feat_s.size(1), N).to(DEVICE)
        inner_opt = torch.optim.SGD(list(fast_encoder.parameters())+list(clf.parameters()),
                                    lr=INNER_LR)

        # Inner updates
        for _ in range(INNER_STEPS):
            logits_s = clf(fast_encoder(sx))
            loss_s = F.cross_entropy(logits_s, sy)
            inner_opt.zero_grad()
            loss_s.backward()
            inner_opt.step()

        # Meta-loss
        logits_q = clf(fast_encoder(qx))
        loss_q = F.cross_entropy(logits_q, qy)

        meta_opt.zero_grad()
        loss_q.backward()
        meta_opt.step()

    return encoder


def evaluate_maml(encoder,N,K,dataset,classmap):
    encoder.eval()
    accs=[]
    for _ in range(EVAL_EPISODES):
        sx,sy,qx,qy = sample_episode(dataset,classmap,N,K,Q_QUERY)

        fast_encoder = copy.deepcopy(encoder)
        feat_s = fast_encoder(sx)
        clf = nn.Linear(feat_s.size(1), N).to(DEVICE)
        inner_opt = torch.optim.SGD(list(fast_encoder.parameters())+list(clf.parameters()),
                                    lr=INNER_LR)

        for _ in range(INNER_STEPS):
            logits_s = clf(fast_encoder(sx))
            loss_s = F.cross_entropy(logits_s, sy)
            inner_opt.zero_grad()
            loss_s.backward()
            inner_opt.step()

        logits_q = clf(fast_encoder(qx))
        preds = torch.argmax(logits_q,1)
        accs.append((preds==qy).float().mean().item())

    return np.mean(accs)

# ============================================================
# BENCHMARK
# ============================================================

results=[]
for backbone_name in BACKBONES:
    for N in N_VALUES:
        for K in K_VALUES:

            print(backbone_name,N,K)

            backbone = get_backbone(backbone_name)
            encoder = train_maml(backbone,N,K)

            val_acc = evaluate_maml(encoder,N,K,val_dataset,val_classmap)
            test_acc = evaluate_maml(encoder,N,K,test_dataset,test_classmap)

            results.append({"Backbone":backbone_name,"N":N,"K":K,
                            "Val_Acc":val_acc,"Test_Acc":test_acc})

df=pd.DataFrame(results)
csv_path=f"maml_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df.to_csv(csv_path,index=False)
print("Saved:",csv_path)

# ============================================================
# HEATMAP
# ============================================================

for b in df["Backbone"].unique():
    table=df[df["Backbone"]==b].pivot(index="N",columns="K",values="Test_Acc")
    plt.figure(figsize=(6,5))
    sns.heatmap(table,annot=True,fmt=".3f",cmap="viridis")
    plt.title(f"{b} Test Accuracy")
    plt.show()


# In[2]:


from sklearn.metrics import classification_report

def classification_report_episode(model, dataset, classmap, n, k):

    model.eval()

    sx, sy, qx, qy = sample_episode(dataset, classmap, n, k, cfg.q_query)

    logits = model(sx, sy, qx, n, k, cfg.inner_steps, cfg.inner_lr)
    preds = torch.argmax(logits, dim=1)

    y_true = qy.cpu().numpy()
    y_pred = preds.cpu().numpy()

    report = classification_report(
        y_true,
        y_pred,
        digits=4,
        output_dict=False
    )

    print("\n=== Classification Report ===")
    print(report)

    return report


# In[3]:


print("\nRunning Classification Report (3-way, 5-shot on Test set)\n")

# pick best configuration if desired
best_row = df.sort_values("Test_Mean", ascending=False).iloc[0]

best_backbone = best_row["Backbone"]
best_n = int(best_row["N"])
best_k = int(best_row["K"])

print(f"Best Config → Backbone={best_backbone}, N={best_n}, K={best_k}")

encoder = get_backbone(best_backbone)
model = ProtoMAML(encoder).to(DEVICE)
model = train_protomaml(model, best_n, best_k)

classification_report_episode(
    model,
    test_dataset,
    test_classmap,
    best_n,
    best_k
)


# In[ ]:




