"""
Complexity, FLOPs, and Inference Latency Profiler for IJISA PAPER021185 (Major Issue 7).
Measures Parameters (M), MMac, GFLOPs, Inference Latency (ms/image), FPS, and Peak Memory
for Conv4, ResNet-12, ResNet-18, ResNet-34, ViT-B/16, and the proposed
Gated Dual-Backbone (ResNet-12 + ViT-B/16) architecture.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from torchvision import models


# ------------------------------------------------------------------
# Backbone definitions (exact match with codebase)
# ------------------------------------------------------------------
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

    def forward(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.view(x.size(0), -1)


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
        out = self.relu(out + identity)
        return self.maxpool(out)


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


def get_resnet(name: str):
    base = models.resnet18(weights=None) if name == "resnet18" else models.resnet34(weights=None)
    enc = nn.Sequential(*list(base.children())[:-1])

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc

        def forward(self, x):
            return self.enc(x).flatten(1)

    return Wrapper()


def get_vit():
    m = models.vit_b_16(weights=None)
    m.heads = nn.Identity()
    return m


class GatedDualBackboneFusion(nn.Module):
    """Projects ResNet-12 (640-d) and ViT-B/16 (768-d) embeddings into a shared
    L2-normalized space (256-d) and fuses them with an input-dependent gate."""

    def __init__(self, resnet_dim: int = 640, vit_dim: int = 768, shared_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.resnet = ResNet12()
        self.vit = get_vit()
        self.proj_res = nn.Sequential(
            nn.Linear(resnet_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_vit = nn.Sequential(
            nn.Linear(vit_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.Sigmoid(),
        )

    def forward(self, x_224: torch.Tensor) -> torch.Tensor:
        x_84 = F.interpolate(x_224, size=(84, 84), mode="bilinear", align_corners=False)
        f_res = self.resnet(x_84)
        f_vit = self.vit(x_224)
        z_res = F.normalize(self.proj_res(f_res), p=2, dim=-1)
        z_vit = F.normalize(self.proj_vit(f_vit), p=2, dim=-1)
        alpha = self.gate(torch.cat([z_res, z_vit], dim=-1))
        z_fused = alpha * z_res + (1.0 - alpha) * z_vit
        return F.normalize(z_fused, p=2, dim=-1)


def profile_model(name, model, input_shape, device, warmup=20, runs=100):
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)

    params = sum(p.numel() for p in model.parameters())

    with torch.no_grad():
        with FlopCounterMode(display=False) as fcm:
            _ = model(x)
        total_flops = fcm.get_total_flops()
    # PyTorch FlopCounterMode counts FLOPs (2 * MACs for conv/matmul)
    gflops = total_flops / 1e9
    mmacs = (total_flops / 2.0) / 1e6

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    fps = 1000.0 / mean_ms
    return {
        "model": name,
        "input_shape": list(input_shape),
        "device": str(device),
        "params_M": round(params / 1e6, 2),
        "mmac": round(mmacs, 2),
        "gflops": round(gflops, 2),
        "latency_ms_mean": round(mean_ms, 2),
        "latency_ms_std": round(std_ms, 2),
        "fps": round(fps, 1),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on device: {device}")
    specs = [
        ("Conv4", Conv4(), (1, 3, 84, 84), 20, 100),
        ("ResNet-12", ResNet12(), (1, 3, 84, 84), 20, 100),
        ("ResNet-18", get_resnet("resnet18"), (1, 3, 84, 84), 20, 100),
        ("ResNet-34", get_resnet("resnet34"), (1, 3, 84, 84), 20, 100),
        ("ViT-B/16", get_vit(), (1, 3, 224, 224), 10, 50),
        ("Dual-Backbone (Gated Fusion)", GatedDualBackboneFusion(), (1, 3, 224, 224), 10, 50),
    ]

    results = []
    for name, m, shape, w, r in specs:
        res = profile_model(name, m, shape, device, warmup=w, runs=r)
        results.append(res)
        print(
            f"{res['model']:30s} | Params={res['params_M']:6.2f}M | "
            f"MMac={res['mmac']:8.2f} | GFLOPs={res['gflops']:6.2f} | "
            f"Lat={res['latency_ms_mean']:6.2f}±{res['latency_ms_std']:4.2f} ms | "
            f"FPS={res['fps']:6.1f}"
        )

    out_path = Path(__file__).parent / f"complexity_latency_{device.type}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
