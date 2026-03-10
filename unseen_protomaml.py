#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# ============================================================
# PROTOMAML — FULL GRID EVALUATION
# ============================================================

import os
import random
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torchvision import transforms, models
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    data_root: str = "/workspace/a_meta/expt1/data"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_values = [2, 3]
    k_values = [1, 5, 10, 20]
    backbones = ["conv4", "resnet12", "resnet18", "resnet34"]
    q_query: int = 15
    meta_iters: int = 200
    inner_steps: int = 5
    inner_lr: float = 0.01
    meta_lr: float = 1e-3
    eval_episodes: int = 100

cfg = Config()
DEVICE = torch.device(cfg.device)

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

def load_samples(root, class_list):
    samples = []
    for cls in class_list:
        cls_path = os.path.join(root, cls)
        for user in os.listdir(cls_path):
            user_path = os.path.join(cls_path, user)
            if not os.path.isdir(user_path):
                continue
            for img in os.listdir(user_path):
                if img.lower().endswith((".jpg",".png",".jpeg")):
                    samples.append((os.path.join(user_path,img), cls))
    return samples

all_classes = sorted([c for c in os.listdir(cfg.data_root)
                      if os.path.isdir(os.path.join(cfg.data_root,c))])

random.shuffle(all_classes)

train_classes = all_classes[:14]
val_classes = all_classes[14:17]
test_classes = all_classes[17:20]

train_samples = load_samples(cfg.data_root, train_classes)
val_samples = load_samples(cfg.data_root, val_classes)
test_samples = load_samples(cfg.data_root, test_classes)

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
        self.conv1=nn.Conv2d(in_c,out_c,3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_c)
        self.conv2=nn.Conv2d(out_c,out_c,3,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_c)
        self.conv3=nn.Conv2d(out_c,out_c,3,padding=1,bias=False)
        self.bn3=nn.BatchNorm2d(out_c)
        self.relu=nn.LeakyReLU(0.1,inplace=True)
        self.downsample=None
        if in_c!=out_c:
            self.downsample=nn.Sequential(
                nn.Conv2d(in_c,out_c,1,bias=False),
                nn.BatchNorm2d(out_c))
        self.pool=nn.MaxPool2d(2)
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
# PROTOMAML
# ============================================================

class ProtoMAML(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder=encoder

    def forward(self,sx,sy,qx,n,k,inner_steps,inner_lr):

        support_feat=self.encoder(sx).detach()
        query_feat=self.encoder(qx)

        dim=support_feat.size(1)
        clf=nn.Linear(dim,n).to(DEVICE)

        # prototype init
        for c in range(n):
            proto=support_feat[sy==c].mean(0)
            clf.weight.data[c]=proto
        clf.bias.data.zero_()

        inner_opt=torch.optim.SGD(clf.parameters(),lr=inner_lr)

        for _ in range(inner_steps):
            logits_s=clf(support_feat)
            loss_s=F.cross_entropy(logits_s,sy)
            inner_opt.zero_grad()
            loss_s.backward()
            inner_opt.step()

        return clf(query_feat)

# ============================================================
# TRAIN / EVAL
# ============================================================

def train_protomaml(model,n,k):
    meta_opt=torch.optim.Adam(model.encoder.parameters(),lr=cfg.meta_lr)

    for _ in range(cfg.meta_iters):
        sx,sy,qx,qy=sample_episode(train_dataset,train_classmap,n,k,cfg.q_query)
        logits=model(sx,sy,qx,n,k,cfg.inner_steps,cfg.inner_lr)
        loss=F.cross_entropy(logits,qy)
        meta_opt.zero_grad()
        loss.backward()
        meta_opt.step()

    return model

def evaluate(model,dataset,classmap,n,k):
    model.eval()
    accs=[]
    for _ in range(cfg.eval_episodes):
        sx,sy,qx,qy=sample_episode(dataset,classmap,n,k,cfg.q_query)
        logits=model(sx,sy,qx,n,k,cfg.inner_steps,cfg.inner_lr)
        preds=torch.argmax(logits,1)
        accs.append((preds==qy).float().mean().item())

    mean=np.mean(accs)
    std=np.std(accs)
    ci=1.96*std/np.sqrt(len(accs))
    return mean,ci

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
# GRID EVALUATION
# ============================================================

results=[]

for backbone in cfg.backbones:
    for n in cfg.n_values:
        for k in cfg.k_values:

            print(backbone,n,k)

            encoder=get_backbone(backbone)
            model=ProtoMAML(encoder).to(DEVICE)

            model=train_protomaml(model,n,k)

            val_mean,val_ci=evaluate(model,val_dataset,val_classmap,n,k)
            test_mean,test_ci=evaluate(model,test_dataset,test_classmap,n,k)

            results.append({
                "Backbone":backbone,
                "N":n,
                "K":k,
                "Val_Mean":val_mean,
                "Val_CI95":val_ci,
                "Test_Mean":test_mean,
                "Test_CI95":test_ci
            })

df=pd.DataFrame(results)
csv_path=f"protomaml_fullgrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df.to_csv(csv_path,index=False)
print("Saved:",csv_path)

# ============================================================
# HEATMAPS
# ============================================================

for b in df["Backbone"].unique():
    table=df[df["Backbone"]==b].pivot(index="N",columns="K",values="Test_Mean")
    plt.figure(figsize=(6,5))
    sns.heatmap(table,annot=True,fmt=".3f",cmap="viridis")
    plt.title(f"{b} Test Accuracy")
    plt.show()


# In[4]:


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


# In[5]:


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




