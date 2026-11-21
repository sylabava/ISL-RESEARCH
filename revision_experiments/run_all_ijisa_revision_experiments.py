#!/usr/bin/env python
"""
GPU-Optimized Full Revision Experiment Suite for IJISA PAPER021185.
Runs cleanly on NVIDIA GeForce RTX 5050 (8 GB VRAM, BF16 Tensor Cores) or A100.

Experiments:
  - EXP 0: Dataset audit & zero-leakage assertions (18,000 imgs, 20 classes, 6 signers)
  - EXP 4: GPU inference latency & FPS profiling (skips if already saved)
  - EXP 1: Corrected First-Order MAML & ProtoMAML on Unseen-Class split
           (4 backbones x N in {2,3} x K in {1,5,10,20} x 3 seeds, + 300-DPI loss curves)
  - EXP 2: Dual-Backbone Gated Fusion Ablation + Paired Statistical Significance
           (5 variants x 2 protocols x N in {2,3} x K=10 x 3 seeds x 100 paired episodes)
  - EXP 3: Fair Matched N-way K-shot SOTA Benchmarks
           (2-way 2-shot, 5-way 5-shot, 5-way 10-shot x 3 models x 3 seeds x 100 episodes)
"""

import argparse
import gc
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from torchvision import models


# =====================================================================
# 1. REPRODUCIBILITY & FAST IN-MEMORY IMAGE CACHE
# =====================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_detect_data_root(cli_root: str | None) -> Path:
    candidates = []
    if cli_root:
        candidates.append(Path(cli_root))
    candidates.extend([
        Path("data"),
        Path(__file__).resolve().parent.parent / "data",
        Path("./datasets/isl_words"),
    ])
    for c in candidates:
        if c.exists() and (c / "afraid").exists():
            return c
    raise FileNotFoundError(
        f"ISL dataset not found. Download from https://doi.org/10.17632/s6kgb6r3ss.2 "
        f"and pass --data_root <path>. Checked: {[str(c) for c in candidates]}"
    )


def audit_dataset(data_root: Path, out_dir: Path) -> dict:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    class_dirs = sorted(
        [p for p in data_root.iterdir() if p.is_dir() and any(d.is_dir() for d in p.iterdir())]
    )
    signer_counts = Counter()
    class_counts = {}
    all_paths = set()
    dup_count = 0

    for cdir in class_dirs:
        c_total = 0
        for sdir in sorted([d for d in cdir.iterdir() if d.is_dir()]):
            files = [f for f in sdir.iterdir() if f.is_file() and f.suffix.lower() in exts]
            for f in files:
                key = str(f.resolve())
                if key in all_paths:
                    dup_count += 1
                all_paths.add(key)
            cnt = len(files)
            signer_counts[sdir.name] += cnt
            c_total += cnt
        class_counts[cdir.name] = c_total

    assert dup_count == 0, f"Duplicate files detected: {dup_count}"
    summary = {
        "dataset_name": "Indian Sign Language - Real-life Words (Tyagi & Bansal, 2022, Mendeley Data V2)",
        "doi": "10.17632/s6kgb6r3ss.2",
        "num_classes": len(class_dirs),
        "classes": [c.name for c in class_dirs],
        "num_signers": len(signer_counts),
        "signers": sorted(signer_counts.keys()),
        "total_images": sum(class_counts.values()),
        "per_signer_counts": dict(sorted(signer_counts.items())),
        "per_class_counts": class_counts,
        "duplicates": dup_count,
    }
    (out_dir / "dataset_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class FastImageStore:
    def __init__(self, target_size: int = 224):
        self.target_size = target_size
        self.cache: dict[str, torch.Tensor] = {}

    def get(self, path: str) -> torch.Tensor:
        t = self.cache.get(path)
        if t is None:
            with Image.open(path) as im:
                im = im.convert("RGB").resize((self.target_size, self.target_size), Image.Resampling.BILINEAR)
                arr = np.asarray(im, dtype=np.uint8)
            t = torch.from_numpy(arr.copy()).permute(2, 0, 1)
            self.cache[path] = t
        return t


def load_split_samples(data_root: Path, allowed_classes, allowed_signers):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    samples = []
    for cls in sorted(allowed_classes):
        cdir = data_root / cls
        for signer in sorted(allowed_signers):
            sdir = cdir / signer
            if not sdir.is_dir():
                continue
            for f in sorted(sdir.iterdir()):
                if f.is_file() and f.suffix.lower() in exts:
                    samples.append((str(f), cls, signer))
    return samples


def build_protocol_splits(data_root: Path, protocol: str, seed: int = 42, n_test_classes: int = 3):
    all_classes = sorted(
        [p.name for p in data_root.iterdir() if p.is_dir() and any(d.is_dir() for d in p.iterdir())]
    )
    all_signers = ["User_1", "User_2", "User_3", "User_4", "User_5", "User_6"]

    if protocol == "cross_signer":
        tr_c, va_c, te_c = all_classes, all_classes, all_classes
        tr_u, va_u, te_u = all_signers[:4], [all_signers[4]], [all_signers[5]]
        assert set(tr_u).isdisjoint(va_u) and set(tr_u).isdisjoint(te_u) and set(va_u).isdisjoint(te_u)
    elif protocol == "unseen_class":
        rng = random.Random(seed)
        shuffled = list(all_classes)
        rng.shuffle(shuffled)
        n_val = 3
        n_train = len(shuffled) - n_val - n_test_classes
        tr_c = shuffled[:n_train]
        va_c = shuffled[n_train : n_train + n_val]
        te_c = shuffled[n_train + n_val :]
        tr_u, va_u, te_u = all_signers, all_signers, all_signers
        assert set(tr_c).isdisjoint(va_c) and set(tr_c).isdisjoint(te_c) and set(va_c).isdisjoint(te_c)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    tr_s = load_split_samples(data_root, tr_c, tr_u)
    va_s = load_split_samples(data_root, va_c, va_u)
    te_s = load_split_samples(data_root, te_c, te_u)

    tr_paths = {s[0] for s in tr_s}
    va_paths = {s[0] for s in va_s}
    te_paths = {s[0] for s in te_s}
    assert tr_paths.isdisjoint(va_paths) and tr_paths.isdisjoint(te_paths) and va_paths.isdisjoint(te_paths)

    return (
        (tr_s, {c: i for i, c in enumerate(tr_c)}),
        (va_s, {c: i for i, c in enumerate(va_c)}),
        (te_s, {c: i for i, c in enumerate(te_c)}),
    )


def build_classmap(samples, class_to_idx):
    cmap = defaultdict(list)
    for i, (_, cls, _) in enumerate(samples):
        cmap[class_to_idx[cls]].append(i)
    return cmap


def sample_episode_indices(classmap, n_way: int, k_shot: int, q_query: int, rng: random.Random):
    valid = [c for c, idxs in classmap.items() if len(idxs) >= k_shot + q_query]
    assert len(valid) >= n_way, f"Only {len(valid)} valid classes for {n_way}-way {k_shot}-shot"
    chosen = rng.sample(valid, n_way)
    s_idx, s_y, q_idx, q_y = [], [], [], []
    for new_label, cls in enumerate(chosen):
        picked = rng.sample(classmap[cls], k_shot + q_query)
        s_idx.extend(picked[:k_shot])
        s_y.extend([new_label] * k_shot)
        q_idx.extend(picked[k_shot:])
        q_y.extend([new_label] * q_query)
    return s_idx, s_y, q_idx, q_y


def gpu_augment_batch(x: torch.Tensor, training: bool) -> torch.Tensor:
    if not training:
        return x
    B = x.size(0)
    bright = 1.0 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.3
    mean = x.mean(dim=(2, 3), keepdim=True)
    contrast = 1.0 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.3
    x = (x - mean) * contrast + mean
    return torch.clamp(x * bright, 0.0, 1.0)


def materialize_episode_fast(samples, store84: FastImageStore, store224: FastImageStore | None,
                             s_idx, s_y, q_idx, q_y, device, dual: bool = False, training: bool = False):
    s84 = torch.stack([store84.get(samples[i][0]) for i in s_idx]).to(device, non_blocking=True).float() / 255.0
    q84 = torch.stack([store84.get(samples[i][0]) for i in q_idx]).to(device, non_blocking=True).float() / 255.0
    s84 = gpu_augment_batch(s84, training)
    q84 = gpu_augment_batch(q84, training)
    sy = torch.tensor(s_y, dtype=torch.long, device=device)
    qy = torch.tensor(q_y, dtype=torch.long, device=device)

    if dual:
        assert store224 is not None
        s224 = torch.stack([store224.get(samples[i][0]) for i in s_idx]).to(device, non_blocking=True).float() / 255.0
        q224 = torch.stack([store224.get(samples[i][0]) for i in q_idx]).to(device, non_blocking=True).float() / 255.0
        s224 = gpu_augment_batch(s224, training)
        q224 = gpu_augment_batch(q224, training)
        return (s84, s224), sy, (q84, q224), qy

    return s84, sy, q84, qy


# =====================================================================
# 2. BACKBONE ARCHITECTURES & DUAL-BACKBONE FUSION
# =====================================================================
class Conv4(nn.Module):
    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.encoder = nn.Sequential(
            block(3, 64), block(64, 64), block(64, 64), block(64, 64)
        )
        self.output_dim = 64

    def forward(self, x):
        return F.adaptive_avg_pool2d(self.encoder(x), 1).flatten(1)


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
                nn.BatchNorm2d(out_c),
            )
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.maxpool(self.relu(out + identity))


class ResNet12(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BasicBlock(3, 64)
        self.layer2 = BasicBlock(64, 160)
        self.layer3 = BasicBlock(160, 320)
        self.layer4 = BasicBlock(320, 640)
        self.output_dim = 640

    def forward(self, x):
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


def get_cnn_backbone(name: str, pretrained: bool = True):
    if name == "conv4":
        return Conv4()
    if name == "resnet12":
        return ResNet12()
    if name == "resnet18":
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=w)
    elif name == "resnet34":
        w = models.ResNet34_Weights.DEFAULT if pretrained else None
        base = models.resnet34(weights=w)
    else:
        raise ValueError(f"Unknown CNN backbone: {name}")
    enc = nn.Sequential(*list(base.children())[:-1])

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc
            self.output_dim = 512

        def forward(self, x):
            return self.enc(x).flatten(1)

    return Wrapper()


def get_vit_backbone(pretrained: bool = True):
    w = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.vit_b_16(weights=w)
    m.heads = nn.Identity()
    m.output_dim = 768
    return m


def encode_in_chunks(encoder: nn.Module, x: torch.Tensor, chunk_size: int = 32) -> torch.Tensor:
    if x.size(0) <= chunk_size:
        return encoder(x)
    return torch.cat([encoder(x[i : i + chunk_size]) for i in range(0, x.size(0), chunk_size)], dim=0)


class UnifiedProtoNet(nn.Module):
    def __init__(self, encoder: nn.Module, tau_init: float = 10.0, use_vit_input: bool = False):
        super().__init__()
        self.encoder = encoder
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        self.use_vit_input = use_vit_input

    def encode(self, x):
        if isinstance(x, tuple):
            x = x[1] if self.use_vit_input else x[0]
        feat = encode_in_chunks(self.encoder, x, chunk_size=32 if self.use_vit_input else 64)
        return F.normalize(feat.float(), p=2, dim=-1)

    def forward(self, sx, sy, qx, n_way: int):
        zs = self.encode(sx)
        zq = self.encode(qx)
        protos = F.normalize(
            torch.stack([zs[sy == c].mean(0) for c in range(n_way)]), p=2, dim=-1
        )
        return self.tau * torch.matmul(zq, protos.T)


class DualBackboneFusionProtoNet(nn.Module):
    def __init__(
        self,
        resnet: nn.Module,
        vit: nn.Module,
        fusion_type: str = "gated",
        shared_dim: int = 256,
        tau_init: float = 10.0,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.resnet = resnet
        self.vit = vit
        self.fusion_type = fusion_type
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))

        if fusion_type in ("gated", "concat"):
            self.proj_res = nn.Sequential(
                nn.Linear(640, shared_dim),
                nn.LayerNorm(shared_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.proj_vit = nn.Sequential(
                nn.Linear(768, shared_dim),
                nn.LayerNorm(shared_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        if fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(shared_dim * 2, shared_dim),
                nn.Sigmoid(),
            )
        elif fusion_type == "concat":
            self.fuse_proj = nn.Linear(shared_dim * 2, shared_dim)
        elif fusion_type == "score":
            self.score_logit_weight = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def encode_fused(self, x_pair):
        x84, x224 = x_pair
        f_res = encode_in_chunks(self.resnet, x84, chunk_size=64).float()
        f_vit = encode_in_chunks(self.vit, x224, chunk_size=32).float()
        z_res = F.normalize(self.proj_res(f_res), p=2, dim=-1)
        z_vit = F.normalize(self.proj_vit(f_vit), p=2, dim=-1)
        if self.fusion_type == "gated":
            alpha = self.gate(torch.cat([z_res, z_vit], dim=-1))
            z = alpha * z_res + (1.0 - alpha) * z_vit
            return F.normalize(z, p=2, dim=-1)
        else:
            z = self.fuse_proj(torch.cat([z_res, z_vit], dim=-1))
            return F.normalize(z, p=2, dim=-1)

    def forward(self, sx_pair, sy, qx_pair, n_way: int):
        if self.fusion_type == "score":
            sr = F.normalize(encode_in_chunks(self.resnet, sx_pair[0], 64).float(), p=2, dim=-1)
            qr = F.normalize(encode_in_chunks(self.resnet, qx_pair[0], 64).float(), p=2, dim=-1)
            pr = F.normalize(torch.stack([sr[sy == c].mean(0) for c in range(n_way)]), p=2, dim=-1)
            logits_r = self.tau * torch.matmul(qr, pr.T)

            sv = F.normalize(encode_in_chunks(self.vit, sx_pair[1], 32).float(), p=2, dim=-1)
            qv = F.normalize(encode_in_chunks(self.vit, qx_pair[1], 32).float(), p=2, dim=-1)
            pv = F.normalize(torch.stack([sv[sy == c].mean(0) for c in range(n_way)]), p=2, dim=-1)
            logits_v = self.tau * torch.matmul(qv, pv.T)

            w = torch.sigmoid(self.score_logit_weight)
            return w * logits_r + (1.0 - w) * logits_v

        zs = self.encode_fused(sx_pair)
        zq = self.encode_fused(qx_pair)
        protos = F.normalize(
            torch.stack([zs[sy == c].mean(0) for c in range(n_way)]), p=2, dim=-1
        )
        return self.tau * torch.matmul(zq, protos.T)


# =====================================================================
# 3. VERIFIED FIXED FOMAML & PROTOMAML (Fast persistent fast_enc + Episodic BN)
# =====================================================================
def train_fomaml_verified(
    bb_name: str,
    pretrained: bool,
    tr_s,
    tr_cm,
    va_s,
    va_cm,
    store84: FastImageStore,
    n_way: int,
    k_train: int,
    q_query: int,
    device,
    meta_iters: int = 200,
    inner_steps: int = 5,
    inner_lr_enc: float = 0.01,
    inner_lr_clf: float = 0.1,
    meta_lr: float = 1e-3,
    grad_clip: float = 1.0,
    proto_init: bool = False,
    seed: int = 42,
    track_curve: bool = False,
):
    rng = random.Random(seed)
    encoder = get_cnn_backbone(bb_name, pretrained=pretrained).to(device)
    fast_enc = get_cnn_backbone(bb_name, pretrained=False).to(device)
    meta_clf = nn.Linear(encoder.output_dim, n_way).to(device)
    fast_clf = nn.Linear(encoder.output_dim, n_way).to(device)

    meta_params = list(encoder.parameters()) + ([] if proto_init else list(meta_clf.parameters()))
    meta_opt = torch.optim.Adam(meta_params, lr=meta_lr, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    history = {"iter": [], "train_loss": [], "train_acc": [], "val_acc": []}

    for it in range(1, meta_iters + 1):
        s_idx, s_y, q_idx, q_y = sample_episode_indices(tr_cm, n_way, k_train, q_query, rng)
        sx, sy, qx, qy = materialize_episode_fast(tr_s, store84, None, s_idx, s_y, q_idx, q_y, device, dual=False, training=True)

        # Copy weights into persistent fast_enc (avoids slow copy.deepcopy allocation!)
        fast_enc.load_state_dict(encoder.state_dict())
        fast_enc.train()

        if proto_init:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                zs0 = F.normalize(fast_enc(sx).float(), p=2, dim=-1)
                for c in range(n_way):
                    p = zs0[sy == c].mean(0)
                    fast_clf.weight.data[c] = 2.0 * p
                    fast_clf.bias.data[c] = -(p.norm() ** 2)
        else:
            fast_clf.load_state_dict(meta_clf.state_dict())

        inner_opt = torch.optim.SGD([
            {"params": fast_enc.parameters(), "lr": inner_lr_enc},
            {"params": fast_clf.parameters(), "lr": inner_lr_clf},
        ])
        for _ in range(inner_steps):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                zs = F.normalize(fast_enc(sx).float(), p=2, dim=-1)
                loss_s = F.cross_entropy(10.0 * fast_clf(zs), sy)
            inner_opt.zero_grad()
            loss_s.backward()
            torch.nn.utils.clip_grad_norm_(fast_enc.parameters(), grad_clip)
            inner_opt.step()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            zq = F.normalize(fast_enc(qx).float(), p=2, dim=-1)
            logits_q = 10.0 * fast_clf(zq)
            loss_q = F.cross_entropy(logits_q, qy)

        meta_opt.zero_grad()
        fast_enc.zero_grad()
        fast_clf.zero_grad()
        loss_q.backward()

        # Verified FOMAML meta-gradient transfer + BN buffer sync
        for pm, pf in zip(encoder.parameters(), fast_enc.parameters()):
            if pf.grad is not None:
                pm.grad = pf.grad.detach().clone()
        for bm, bf in zip(encoder.buffers(), fast_enc.buffers()):
            bm.copy_(bf)
        if not proto_init:
            for pm, pf in zip(meta_clf.parameters(), fast_clf.parameters()):
                if pf.grad is not None:
                    pm.grad = pf.grad.detach().clone()

        torch.nn.utils.clip_grad_norm_(meta_params, grad_clip)
        meta_opt.step()

        if track_curve and (it % max(1, meta_iters // 20) == 0 or it == 1):
            tr_acc = (logits_q.argmax(1) == qy).float().mean().item()
            va_accs = eval_fomaml_verified(
                encoder, meta_clf, fast_enc, fast_clf, va_s, va_cm, store84,
                n_way, k_train, q_query, device, inner_steps, inner_lr_enc, inner_lr_clf,
                grad_clip, proto_init, episodes=15, seed=seed + it
            )
            history["iter"].append(it)
            history["train_loss"].append(float(loss_q.item()))
            history["train_acc"].append(float(tr_acc))
            history["val_acc"].append(float(np.mean(va_accs)))

    return encoder, meta_clf, fast_enc, fast_clf, history


def eval_fomaml_verified(
    encoder: nn.Module,
    meta_clf: nn.Module,
    fast_enc: nn.Module,
    fast_clf: nn.Module,
    samples,
    classmap,
    store84: FastImageStore,
    n_way: int,
    k_shot: int,
    q_query: int,
    device,
    inner_steps: int = 5,
    inner_lr_enc: float = 0.01,
    inner_lr_clf: float = 0.1,
    grad_clip: float = 1.0,
    proto_init: bool = False,
    episodes: int = 60,
    seed: int = 999,
):
    rng = random.Random(seed)
    use_amp = device.type == "cuda"
    accs = []
    for _ in range(episodes):
        s_idx, s_y, q_idx, q_y = sample_episode_indices(classmap, n_way, k_shot, q_query, rng)
        sx, sy, qx, qy = materialize_episode_fast(samples, store84, None, s_idx, s_y, q_idx, q_y, device, dual=False, training=False)

        fast_enc.load_state_dict(encoder.state_dict())
        fast_enc.train()  # Episodic BN stats per Finn et al. (2017) MAML protocol

        if proto_init:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                zs0 = F.normalize(fast_enc(sx).float(), p=2, dim=-1)
                for c in range(n_way):
                    p = zs0[sy == c].mean(0)
                    fast_clf.weight.data[c] = 2.0 * p
                    fast_clf.bias.data[c] = -(p.norm() ** 2)
        else:
            fast_clf.load_state_dict(meta_clf.state_dict())

        inner_opt = torch.optim.SGD([
            {"params": fast_enc.parameters(), "lr": inner_lr_enc},
            {"params": fast_clf.parameters(), "lr": inner_lr_clf},
        ])
        for _ in range(inner_steps):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                zs = F.normalize(fast_enc(sx).float(), p=2, dim=-1)
                loss_s = F.cross_entropy(10.0 * fast_clf(zs), sy)
            inner_opt.zero_grad()
            loss_s.backward()
            torch.nn.utils.clip_grad_norm_(fast_enc.parameters(), grad_clip)
            inner_opt.step()

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            zq = F.normalize(fast_enc(qx).float(), p=2, dim=-1)
            logits_q = 10.0 * fast_clf(zq)
            accs.append((logits_q.argmax(1) == qy).float().mean().item())
    return accs


# =====================================================================
# 4. PROTONET / FUSION TRAINING & PAIRED EVALUATION
# =====================================================================
def train_protonet_model(
    model: nn.Module,
    tr_s,
    tr_cm,
    store84: FastImageStore,
    store224: FastImageStore | None,
    n_way: int,
    k_shot: int,
    q_query: int,
    device,
    meta_iters: int = 250,
    lr_cnn: float = 1e-3,
    lr_vit: float = 1e-5,
    dual: bool = False,
    seed: int = 42,
):
    rng = random.Random(seed)
    model = model.to(device)
    use_amp = device.type == "cuda"

    param_groups = []
    if isinstance(model, DualBackboneFusionProtoNet):
        param_groups.append({"params": model.resnet.parameters(), "lr": lr_cnn})
        param_groups.append({"params": model.vit.parameters(), "lr": lr_vit})
        other_params = [
            p for n, p in model.named_parameters()
            if not n.startswith("resnet.") and not n.startswith("vit.")
        ]
        param_groups.append({"params": other_params, "lr": lr_cnn})
    elif isinstance(model, UnifiedProtoNet) and model.use_vit_input:
        param_groups.append({"params": model.encoder.parameters(), "lr": lr_vit})
        param_groups.append({"params": [model.tau], "lr": lr_cnn})
    else:
        param_groups.append({"params": model.parameters(), "lr": lr_cnn})

    opt = torch.optim.Adam(param_groups, weight_decay=1e-4)
    model.train()

    for _ in range(1, meta_iters + 1):
        s_idx, s_y, q_idx, q_y = sample_episode_indices(tr_cm, n_way, k_shot, q_query, rng)
        sx, sy, qx, qy = materialize_episode_fast(tr_s, store84, store224, s_idx, s_y, q_idx, q_y, device, dual=dual, training=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(sx, sy, qx, n_way)
            loss = F.cross_entropy(logits, qy)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

    return model


@torch.no_grad()
def eval_protonet_paired(model, samples, store84, store224, pre_sampled_episodes, n_way: int, device, dual: bool = False):
    model.eval()
    use_amp = device.type == "cuda"
    accs = []
    for s_idx, s_y, q_idx, q_y in pre_sampled_episodes:
        sx, sy, qx, qy = materialize_episode_fast(samples, store84, store224, s_idx, s_y, q_idx, q_y, device, dual=dual, training=False)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(sx, sy, qx, n_way)
        accs.append((logits.argmax(1) == qy).float().mean().item())
    return accs


def compute_stats_and_sig(accs_a, accs_b):
    a = np.asarray(accs_a, dtype=float)
    b = np.asarray(accs_b, dtype=float)
    t_stat, t_p = stats.ttest_rel(a, b)
    diff = a - b
    if np.allclose(diff, 0.0):
        w_stat, w_p = 0.0, 1.0
    else:
        w_stat, w_p = stats.wilcoxon(a, b)
    return {
        "mean_a": float(np.mean(a) * 100),
        "std_a": float(np.std(a, ddof=1) * 100) if len(a) > 1 else 0.0,
        "mean_b": float(np.mean(b) * 100),
        "std_b": float(np.std(b, ddof=1) * 100) if len(b) > 1 else 0.0,
        "delta": float((np.mean(a) - np.mean(b)) * 100),
        "t_stat": float(t_stat) if np.isfinite(t_stat) else 0.0,
        "t_pvalue": float(t_p) if np.isfinite(t_p) else 1.0,
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pvalue": float(w_p),
    }


def summarize_scores(accs):
    arr = np.asarray(accs, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci95, "n_episodes": len(arr)}


def plot_maml_loss_curves(histories: dict, out_path: Path):
    plt.figure(figsize=(8.5, 4.8), dpi=300)
    for label, hist in histories.items():
        plt.plot(hist["iter"], hist["train_loss"], marker="o", markersize=3, linewidth=1.8, label=label)
    plt.xlabel("Meta-Training Iteration", fontsize=10)
    plt.ylabel("Query Cross-Entropy Loss", fontsize=10)
    plt.title("Corrected First-Order MAML & ProtoMAML Convergence Curves (Unseen-Class Split)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# =====================================================================
# 5. MAIN ORCHESTRATOR
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).parent / "results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2024])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = auto_detect_data_root(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_start = time.time()
    print("=== IJISA PAPER021185 FULL GPU REVISION SUITE ===", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}) | Data: {data_root}", flush=True)

    # EXP 0: Dataset Audit
    audit = audit_dataset(data_root, out_dir)
    print(f"[EXP 0] Dataset verified: {audit['total_images']} images, {audit['num_classes']} classes, {audit['num_signers']} signers.", flush=True)

    store84 = FastImageStore(target_size=84)
    store224 = FastImageStore(target_size=224)

    seeds = args.seeds
    maml_iters = 150
    proto_iters = 200
    maml_eval_eps = 35   # 35 x 3 seeds = 105 episodes/cell
    proto_eval_eps = 50  # 50 x 3 seeds = 150 paired episodes/cell
    q_query = 15

    # Remove stale smoke-test CSVs if they have < 6 rows or old 37.5 smoke values
    for stale_name in ["exp2_fusion_ablation.csv", "exp2_statistical_significance.csv", "exp3_fair_sota_matched.csv"]:
        p = out_dir / stale_name
        if p.exists():
            txt = p.read_text(encoding="utf-8")
            if "37.5" in txt or "62.5,17.68,24.5" in txt:
                p.unlink()
                print(f"  Cleared stale smoke-test file: {stale_name}", flush=True)

    # -----------------------------------------------------------------
    # EXP 1: Corrected MAML & ProtoMAML on Unseen-Class Split (Auto-Resume)
    # -----------------------------------------------------------------
    print("\n[EXP 1] Running Corrected FOMAML & ProtoMAML (Unseen-Class Split, Auto-Resume)...", flush=True)
    exp1_path = out_dir / "exp1_fixed_maml_protomaml_unseen.csv"
    maml_rows = pd.read_csv(exp1_path).to_dict("records") if exp1_path.exists() else []
    done_exp1 = {(r["Method"], r["Backbone"], int(r["N"])) for r in maml_rows}

    loss_histories = {}
    (tr_s, tr_map), (va_s, va_map), (te_s, te_map) = build_protocol_splits(
        data_root, "unseen_class", seed=42, n_test_classes=3
    )
    tr_cm = build_classmap(tr_s, tr_map)
    va_cm = build_classmap(va_s, va_map)
    te_cm = build_classmap(te_s, te_map)

    for method, proto_init in [("MAML", False), ("ProtoMAML", True)]:
        for bb in ["conv4", "resnet12", "resnet18", "resnet34"]:
            for n_way in [2, 3]:
                if (method, bb, n_way) in done_exp1:
                    print(f"  [SKIP DONE] {method:9s} | {bb:8s} | {n_way}-way", flush=True)
                    continue
                k_scores = {k: [] for k in [1, 5, 10, 20]}
                for seed in seeds:
                    set_seed(seed)
                    enc, mclf, fenc, fclf, hist = train_fomaml_verified(
                        bb, True, tr_s, tr_cm, va_s, va_cm, store84,
                        n_way=n_way, k_train=5, q_query=q_query, device=device,
                        meta_iters=maml_iters, inner_steps=5,
                        inner_lr_enc=0.01, inner_lr_clf=0.1, meta_lr=1e-3,
                        grad_clip=1.0, proto_init=proto_init, seed=seed,
                        track_curve=(seed == seeds[0] and n_way == 3),
                    )
                    if seed == seeds[0] and n_way == 3:
                        loss_histories[f"{method}-{bb} (3-way)"] = hist
                    for k_shot in [1, 5, 10, 20]:
                        accs = eval_fomaml_verified(
                            enc, mclf, fenc, fclf, te_s, te_cm, store84,
                            n_way, k_shot, q_query, device,
                            inner_steps=5, inner_lr_enc=0.01, inner_lr_clf=0.1,
                            grad_clip=1.0, proto_init=proto_init,
                            episodes=maml_eval_eps, seed=1000 + seed * 31 + k_shot,
                        )
                        k_scores[k_shot].extend(accs)
                    del enc, mclf, fenc, fclf
                    gc.collect()
                    torch.cuda.empty_cache()

                for k_shot in [1, 5, 10, 20]:
                    s = summarize_scores(k_scores[k_shot])
                    maml_rows.append({
                        "Method": method, "Backbone": bb, "N": n_way, "K": k_shot,
                        "Test_Mean": round(s["mean"] * 100, 2),
                        "Test_Std": round(s["std"] * 100, 2),
                        "Test_CI95": round(s["ci95"] * 100, 2),
                    })
                    print(f"  {method:9s} | {bb:8s} | {n_way}w-{k_shot:2d}s -> {s['mean']*100:5.2f}% ± {s['std']*100:4.2f}%", flush=True)
                pd.DataFrame(maml_rows).to_csv(exp1_path, index=False)

    if loss_histories:
        plot_maml_loss_curves(loss_histories, out_dir / "fig_maml_loss_curves.png")
    print(f"[EXP 1 DONE] Elapsed: {(time.time()-t_start)/60:.1f} min", flush=True)

    # -----------------------------------------------------------------
    # EXP 2: Dual-Backbone Fusion Ablation + Paired Significance (Auto-Resume)
    # -----------------------------------------------------------------
    print("\n[EXP 2] Running Dual-Backbone Fusion Ablation & Paired Statistical Tests...", flush=True)
    exp2_path = out_dir / "exp2_fusion_ablation.csv"
    sig2_path = out_dir / "exp2_statistical_significance.csv"
    exp3_path = out_dir / "exp3_fair_sota_matched.csv"
    fusion_rows = pd.read_csv(exp2_path).to_dict("records") if exp2_path.exists() else []
    sig_tests = pd.read_csv(sig2_path).to_dict("records") if sig2_path.exists() else []
    done_exp2 = {(r["Protocol"], int(r["N"])) for r in fusion_rows}
    sota_2w2s_scores = defaultdict(list)

    for protocol in ["cross_signer", "unseen_class"]:
        (tr_s, tr_map), (va_s, va_map), (te_s, te_map) = build_protocol_splits(
            data_root, protocol, seed=42, n_test_classes=3
        )
        tr_cm = build_classmap(tr_s, tr_map)
        te_cm = build_classmap(te_s, te_map)

        for n_way in [2, 3]:
            if (protocol, n_way) in done_exp2:
                print(f"  [SKIP DONE] EXP 2 {protocol} {n_way}-way", flush=True)
                continue
            paired_scores_k10 = defaultdict(list)
            for seed in seeds:
                set_seed(seed)
                ep_rng = random.Random(2000 + seed + n_way * 17)
                test_eps_k10 = [
                    sample_episode_indices(te_cm, n_way, 10, q_query, ep_rng)
                    for _ in range(proto_eval_eps)
                ]
                test_eps_2w2s = None
                if protocol == "unseen_class" and n_way == 2:
                    ep_rng2 = random.Random(3000 + seed)
                    test_eps_2w2s = [
                        sample_episode_indices(te_cm, 2, 2, q_query, ep_rng2)
                        for _ in range(proto_eval_eps)
                    ]

                variant_specs = [
                    ("ResNet-12_Only", lambda: UnifiedProtoNet(get_cnn_backbone("resnet12", False), use_vit_input=False)),
                    ("ViT-B16_Only", lambda: UnifiedProtoNet(get_vit_backbone(True), use_vit_input=True)),
                    ("Dual_Concat_Fusion", lambda: DualBackboneFusionProtoNet(get_cnn_backbone("resnet12", False), get_vit_backbone(True), fusion_type="concat")),
                    ("Dual_Score_Fusion", lambda: DualBackboneFusionProtoNet(get_cnn_backbone("resnet12", False), get_vit_backbone(True), fusion_type="score")),
                    ("Dual_Gated_Fusion_Proposed", lambda: DualBackboneFusionProtoNet(get_cnn_backbone("resnet12", False), get_vit_backbone(True), fusion_type="gated")),
                ]

                for vname, builder in variant_specs:
                    model = builder()
                    model = train_protonet_model(
                        model, tr_s, tr_cm, store84, store224, n_way, 5, q_query, device,
                        meta_iters=proto_iters, lr_cnn=1e-3, lr_vit=1e-5, dual=True, seed=seed,
                    )
                    accs = eval_protonet_paired(model, te_s, store84, store224, test_eps_k10, n_way, device, dual=True)
                    paired_scores_k10[vname].extend(accs)

                    if test_eps_2w2s is not None and vname in ("ResNet-12_Only", "ViT-B16_Only", "Dual_Gated_Fusion_Proposed"):
                        accs2 = eval_protonet_paired(model, te_s, store84, store224, test_eps_2w2s, 2, device, dual=True)
                        sota_2w2s_scores[vname].extend(accs2)

                    del model
                    gc.collect()
                    torch.cuda.empty_cache()

            for vname, accs in paired_scores_k10.items():
                s = summarize_scores(accs)
                fusion_rows.append({
                    "Protocol": protocol, "Model": vname, "N": n_way, "K": 10,
                    "Test_Mean": round(s["mean"] * 100, 2),
                    "Test_Std": round(s["std"] * 100, 2),
                    "Test_CI95": round(s["ci95"] * 100, 2),
                })
                print(f"  [{protocol:12s}] {vname:27s} | {n_way}w-10s -> {s['mean']*100:5.2f}% ± {s['std']*100:4.2f}%", flush=True)

            for baseline in ["ResNet-12_Only", "ViT-B16_Only", "Dual_Concat_Fusion", "Dual_Score_Fusion"]:
                sig = compute_stats_and_sig(paired_scores_k10["Dual_Gated_Fusion_Proposed"], paired_scores_k10[baseline])
                sig.update({"Protocol": protocol, "N": n_way, "K": 10, "Comparison": f"Gated_Fusion vs {baseline}"})
                sig_tests.append(sig)
                print(f"    SIG ({protocol} {n_way}w-10s) vs {baseline:18s}: Δ={sig['delta']:+.2f}%, t-p={sig['t_pvalue']:.4g}, W-p={sig['wilcoxon_pvalue']:.4g}", flush=True)

            pd.DataFrame(fusion_rows).to_csv(exp2_path, index=False)
            pd.DataFrame(sig_tests).to_csv(sig2_path, index=False)

            if protocol == "unseen_class" and n_way == 2 and sota_2w2s_scores:
                sota_existing = pd.read_csv(exp3_path).to_dict("records") if exp3_path.exists() else []
                name_map = {
                    "ResNet-12_Only": "ResNet-12_ProtoNet",
                    "ViT-B16_Only": "ViT-B16_ProtoNet",
                    "Dual_Gated_Fusion_Proposed": "Dual_Gated_Fusion_Proposed",
                }
                for vname, accs in sota_2w2s_scores.items():
                    s = summarize_scores(accs)
                    sota_existing.append({
                        "Setting": "2-way 2-shot", "Test_Classes": 3, "Model": name_map[vname],
                        "Test_Mean": round(s["mean"] * 100, 2),
                        "Test_Std": round(s["std"] * 100, 2),
                        "Test_CI95": round(s["ci95"] * 100, 2),
                    })
                    print(f"  [Fair SOTA 2w- 2s] {name_map[vname]:27s} -> {s['mean']*100:5.2f}% ± {s['std']*100:4.2f}%", flush=True)
                pd.DataFrame(sota_existing).to_csv(exp3_path, index=False)

    print(f"[EXP 2 DONE] Elapsed: {(time.time()-t_start)/60:.1f} min", flush=True)

    # -----------------------------------------------------------------
    # EXP 3: Fair Matched 5-Way SOTA Benchmarks (5w-5s & 5w-10s)
    # -----------------------------------------------------------------
    print("\n[EXP 3] Running Fair Matched 5-Way SOTA Benchmarks (5w-5s & 5w-10s)...", flush=True)
    sota_rows = pd.read_csv(exp3_path).to_dict("records") if exp3_path.exists() else []
    done_sota = {r["Setting"] for r in sota_rows}

    if "5-way 5-shot" not in done_sota:
        (tr_s, tr_map), (va_s, va_map), (te_s, te_map) = build_protocol_splits(
            data_root, "unseen_class", seed=42, n_test_classes=5
        )
        tr_cm = build_classmap(tr_s, tr_map)
        te_cm = build_classmap(te_s, te_map)

        sota_5w5s = defaultdict(list)
        sota_5w10s = defaultdict(list)
        for seed in seeds:
            set_seed(seed)
            ep_rng5 = random.Random(4000 + seed)
            ep_rng10 = random.Random(5000 + seed)
            eps_5w5s = [sample_episode_indices(te_cm, 5, 5, q_query, ep_rng5) for _ in range(proto_eval_eps)]
            eps_5w10s = [sample_episode_indices(te_cm, 5, 10, q_query, ep_rng10) for _ in range(proto_eval_eps)]

            for vname, builder in [
                ("ResNet-12_ProtoNet", lambda: UnifiedProtoNet(get_cnn_backbone("resnet12", False), use_vit_input=False)),
                ("ViT-B16_ProtoNet", lambda: UnifiedProtoNet(get_vit_backbone(True), use_vit_input=True)),
                ("Dual_Gated_Fusion_Proposed", lambda: DualBackboneFusionProtoNet(get_cnn_backbone("resnet12", False), get_vit_backbone(True), fusion_type="gated")),
            ]:
                model = builder()
                model = train_protonet_model(
                    model, tr_s, tr_cm, store84, store224, 5, 5, q_query, device,
                    meta_iters=proto_iters, lr_cnn=1e-3, lr_vit=1e-5, dual=True, seed=seed,
                )
                sota_5w5s[vname].extend(eval_protonet_paired(model, te_s, store84, store224, eps_5w5s, 5, device, dual=True))
                sota_5w10s[vname].extend(eval_protonet_paired(model, te_s, store84, store224, eps_5w10s, 5, device, dual=True))
                del model
                gc.collect()
                torch.cuda.empty_cache()

        for setting, bucket in [("5-way 5-shot", sota_5w5s), ("5-way 10-shot", sota_5w10s)]:
            for vname, accs in bucket.items():
                s = summarize_scores(accs)
                sota_rows.append({
                    "Setting": setting, "Test_Classes": 5, "Model": vname,
                    "Test_Mean": round(s["mean"] * 100, 2),
                    "Test_Std": round(s["std"] * 100, 2),
                    "Test_CI95": round(s["ci95"] * 100, 2),
                })
                print(f"  [Fair SOTA {setting}] {vname:27s} -> {s['mean']*100:5.2f}% ± {s['std']*100:4.2f}%", flush=True)

        pd.DataFrame(sota_rows).to_csv(exp3_path, index=False)
    print(f"\nALL FULL GPU REVISION EXPERIMENTS COMPLETED in {(time.time()-t_start)/60:.1f} min!", flush=True)


if __name__ == "__main__":
    main()
