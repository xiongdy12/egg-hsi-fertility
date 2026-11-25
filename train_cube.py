#!/usr/bin/env python3
"""
Tahap 3b — Klasifikasi fertilitas dari kubus HSI dengan seleksi band yang bisa dilatih + backbone YOLOv11-cls.

Arsitektur:  kubus (300 band) -> SELECTOR (k kanal) -> normalisasi -> BACKBONE -> fertil / infertil
  SELECTOR
    concrete : pilih k band diskret (Concrete/Gumbel-softmax, suhu dianneal; evaluasi pakai one-hot)
    led      : k "LED" Gaussian (FWHM tetap, mis. 20 nm) dengan pusat panjang gelombang yang dipelajari
    fixed    : band tetap dari --bands (nm), mis. hasil VIP dari baseline CSV
    rgb      : simulasi kamera RGB (CIE 1931 -> sRGB)
    all      : konv 1x1 300->k (batas atas tanpa kendala sensor; input SNV per piksel)
  BACKBONE
    yolo11{n,s,m}-cls (Ultralytics, pretrained ImageNet, konv pertama diganti k kanal)
    mobilenet_v3_small | efficientnet_b0 | resnet18 (torchvision)
    specpool : KONTROL tanpa informasi spasial (rata-rata dalam mask -> MLP)
Validasi: Leave-One-Date-Out (25 tanggal), hyperparameter tetap (tidak dituning pada fold uji).
Multi-GPU: tiap proses (rank) memuat data sekali ke GPU-nya, lalu mengerjakan semua konfigurasi
untuk fold dengan indeks % world == rank.

Pemakaian:
  python train_cube.py --make_configs configs.jsonl            # buat daftar konfigurasi standar
  python train_cube.py --data /raid/.../cubes --configs configs.jsonl --out runs --rank 0 --world 8
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================ selector


def cie_rgb_weights(wl):
    def g(x, mu, s1, s2):
        s = np.where(x < mu, s1, s2)
        return np.exp(-0.5 * ((x - mu) / s) ** 2)
    xb = 1.056 * g(wl, 599.8, 37.9, 31.0) + 0.362 * g(wl, 442.0, 16.0, 26.7) - 0.065 * g(wl, 501.1, 20.4, 26.2)
    yb = 0.821 * g(wl, 568.8, 46.9, 40.5) + 0.286 * g(wl, 530.9, 16.3, 31.1)
    zb = 1.217 * g(wl, 437.0, 11.8, 36.0) + 0.681 * g(wl, 459.0, 26.0, 13.8)
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    W = M @ np.vstack([xb, yb, zb])
    W = np.clip(W, 0, None)  # sensor fisik tidak bisa negatif
    return W / W.sum(1, keepdims=True)


class Selector(nn.Module):
    def __init__(self, mode, wl, k, bands=None, fwhm=20.0):
        super().__init__()
        self.mode, self.k = mode, k
        self.register_buffer("wl", torch.tensor(wl, dtype=torch.float32))
        B = len(wl)
        self.T = 1.0
        self.hard = False  # True: pakai band final (one-hot) juga saat training
        if mode == "concrete":
            self.logits = nn.Parameter(0.01 * torch.randn(k, B))
        elif mode == "led":
            init = np.linspace(420, 950, k) if not bands else np.array(bands[:k], float)
            lo, hi = float(wl.min()), float(wl.max())
            p = (init - lo) / (hi - lo)
            self.u = nn.Parameter(torch.tensor(np.log(p / (1 - p)), dtype=torch.float32))
            self.lo, self.hi, self.sigma = lo, hi, fwhm / 2.3548
        elif mode == "fixed":
            idx = [int(np.argmin(np.abs(wl - b))) for b in bands]
            W = np.zeros((len(idx), B), np.float32)
            W[np.arange(len(idx)), idx] = 1
            self.register_buffer("W", torch.tensor(W))
            self.k = len(idx)
        elif mode == "rgb":
            self.register_buffer("W", torch.tensor(cie_rgb_weights(np.asarray(wl)), dtype=torch.float32))
            self.k = 3
        elif mode == "all":
            self.conv = nn.Conv2d(B, k, 1)
        elif mode == "bin":   # rata-rata k kelompok band berurutan (untuk model 3D, HSI penuh)
            edges = np.linspace(0, B, k + 1).round().astype(int)
            W = np.zeros((k, B), np.float32)
            for i in range(k):
                W[i, edges[i]:edges[i + 1]] = 1.0 / (edges[i + 1] - edges[i])
            self.register_buffer("W", torch.tensor(W))
        else:
            raise ValueError(mode)

    def centers(self):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(self.u)

    def weights(self):
        if self.mode == "concrete":
            if self.training and not self.hard:
                u = torch.rand_like(self.logits).clamp(1e-9, 1 - 1e-9)
                g = -torch.log(-torch.log(u))  # noise Gumbel(0,1)
                return F.softmax((self.logits + g) / self.T, dim=1)
            return F.one_hot(self.logits.argmax(1), self.logits.shape[1]).float()
        if self.mode == "led":
            c = self.centers()[:, None]
            w = torch.exp(-0.5 * ((self.wl[None] - c) / self.sigma) ** 2)
            return w / w.sum(1, keepdim=True)
        return self.W

    def selected_nm(self):
        if self.mode == "concrete":
            return self.wl[self.logits.argmax(1)].tolist()
        if self.mode == "led":
            return self.centers().tolist()
        if self.mode == "fixed":
            return self.wl[self.W.argmax(1)].tolist()
        return None

    def forward(self, x):
        if self.mode == "all":
            return self.conv(x)
        return torch.einsum("kb,nbhw->nkhw", self.weights().to(x.dtype), x)

# ============================================================================ backbone


def adapt_first_conv(conv, k):
    new = nn.Conv2d(k, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    conv.dilation, conv.groups, conv.bias is not None)
    with torch.no_grad():
        new.weight.copy_(conv.weight.mean(1, keepdim=True).repeat(1, k, 1, 1) * 3.0 / k)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


class YoloCls(nn.Module):
    def __init__(self, name, k):
        super().__init__()
        from ultralytics import YOLO
        net = YOLO(f"{name}.pt").model
        layers = list(net.model)
        layers[0].conv = adapt_first_conv(layers[0].conv, k)
        head = layers[-1]
        head.linear = nn.Linear(head.linear.in_features, 2)
        self.body = nn.Sequential(*layers[:-1])
        self.head = head
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(self, x, mask=None):
        h = self.head
        x = self.body(x)
        return h.linear(h.drop(h.pool(h.conv(x)).flatten(1)))  # logits (stabil lintas versi ultralytics)


class TVModel(nn.Module):
    def __init__(self, name, k):
        super().__init__()
        import torchvision.models as tvm
        weights = {"mobilenet_v3_small": "IMAGENET1K_V1", "efficientnet_b0": "IMAGENET1K_V1",
                   "resnet18": "IMAGENET1K_V1"}[name]
        try:
            m = getattr(tvm, name)(weights=weights)
        except Exception as e:
            print(f"  (bobot pretrained {name} gagal diunduh: {e}; pakai inisialisasi acak)", flush=True)
            m = getattr(tvm, name)(weights=None)
        if name == "resnet18":
            m.conv1 = adapt_first_conv(m.conv1, k)
            m.fc = nn.Linear(m.fc.in_features, 2)
        else:
            m.features[0][0] = adapt_first_conv(m.features[0][0], k)
            last = m.classifier[-1]
            m.classifier[-1] = nn.Linear(last.in_features, 2)
        self.m = m

    def forward(self, x, mask=None):
        return self.m(x)


class SpecPool(nn.Module):
    """Kontrol: buang informasi spasial — rata-rata kanal di dalam mask -> MLP."""
    def __init__(self, k):
        super().__init__()
        self.mlp = nn.Sequential(nn.BatchNorm1d(k), nn.Linear(k, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(self, x, mask):
        m = mask[:, None].to(x.dtype)
        v = (x * m).sum((2, 3)) / m.sum((2, 3)).clamp_min(1)
        return self.mlp(v)


class HybridSN(nn.Module):
    """Pembanding khusus HSI: CNN 3D->2D (gaya HybridSN, Roy et al. 2020), versi klasifikasi tingkat-citra, dari nol."""
    def __init__(self, k):
        super().__init__()

        def c3(i, o, ks):
            return nn.Sequential(nn.Conv3d(i, o, ks, 2, padding=(ks[0] // 2, 1, 1), bias=False),
                                 nn.BatchNorm3d(o), nn.ReLU(inplace=True))
        self.f3 = nn.Sequential(c3(1, 8, (7, 3, 3)), c3(8, 16, (5, 3, 3)), c3(16, 32, (3, 3, 3)))
        with torch.no_grad():
            d = self.f3(torch.zeros(1, 1, k, 32, 32)).shape[2]
        self.f2 = nn.Sequential(nn.Conv2d(32 * d, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                nn.Conv2d(128, 256, 3, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, x, mask=None):
        z = self.f3(x.unsqueeze(1))
        n, c, d, h, w = z.shape
        return self.head(self.f2(z.reshape(n, c * d, h, w)))


def make_backbone(name, k):
    if name == "hybridsn":
        return HybridSN(k)
    if name.startswith("yolo"):
        return YoloCls(name, k)
    if name == "specpool":
        return SpecPool(k)
    return TVModel(name, k)


class Net(nn.Module):
    def __init__(self, cfg, wl):
        super().__init__()
        self.cfg = cfg
        self.sel = Selector(cfg["selector"], wl, cfg.get("k", 3), cfg.get("bands"), cfg.get("fwhm", 20.0))
        self.backbone = make_backbone(cfg["backbone"], self.sel.k)

    def normalize(self, x, mask):
        m = mask[:, None].to(x.dtype)
        pn = self.cfg.get("pixnorm", "log")
        if pn == "log":    # log-intensitas (kamera terkalibrasi dark/white); tanpa normalisasi antar-kanal
            x = torch.log(x.clamp_min(1e-3))
        elif pn == "img":   # bagi dengan rata-rata intensitas telur (hilangkan iluminasi/ketebalan global)
            mu = (x * m).sum((1, 2, 3), keepdim=True) / (m.sum((1, 2, 3), keepdim=True) * x.shape[1]).clamp_min(1)
            x = torch.log(x.clamp_min(0) / mu.clamp_min(1e-6) + 1e-3)
        elif pn == "snv":  # SNV per piksel antar kanal
            x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
        return x * m

    def forward(self, cube, mask, chan_jitter=0.0):
        if self.cfg["selector"] in ("all", "bin"):   # SNV per piksel pada 300 band, lalu proyeksi linear
            m = mask[:, None].to(cube.dtype)
            cube = (cube - cube.mean(1, keepdim=True)) / (cube.std(1, keepdim=True) + 1e-6) * m
            x = self.sel(cube) * m
        else:
            x = self.sel(cube)
            if chan_jitter > 0:  # simulasi drift intensitas LED per kanal
                x = x * (1 + chan_jitter * (2 * torch.rand(x.shape[0], x.shape[1], 1, 1, device=x.device) - 1))
            x = self.normalize(x, mask)
        return self.backbone(x, mask)

# ============================================================================ training


def augment(x, m):
    if torch.rand(1).item() < 0.5:
        x, m = x.flip(-1), m.flip(-1)
    if torch.rand(1).item() < 0.5:
        x, m = x.flip(-2), m.flip(-2)
    r = int(torch.randint(0, 4, (1,)).item())
    if r:
        x, m = torch.rot90(x, r, (-2, -1)), torch.rot90(m, r, (-2, -1))
    s = 1 + 0.1 * (2 * torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 1)
    return x * s, m


@torch.no_grad()
def init_concrete_fisher(sel, data, masks, y, tr, scale=3.0, gap=5):
    """Titik awal logits concrete: k puncak skor Fisher (spektrum rata-rata SNV, DATA LATIH saja), berjarak >= gap band."""
    spec = []
    for i in range(0, len(tr), 64):
        b = torch.as_tensor(tr[i:i + 64], device=data.device)
        m = masks[b][:, None].float()
        spec.append(((data[b].float() * m).sum((2, 3)) / m.sum((2, 3)).clamp_min(1)).cpu())
    X = torch.cat(spec).numpy()
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
    yy = np.asarray(y)[tr]
    a, b = X[yy == 1], X[yy == 0]
    f = (a.mean(0) - b.mean(0)) ** 2 / (a.var(0) + b.var(0) + 1e-12)
    chosen = []
    for j in np.argsort(-f):
        if all(abs(j - c) >= gap for c in chosen):
            chosen.append(int(j))
        if len(chosen) == sel.k:
            break
    B = sel.logits.shape[1]
    grid = np.arange(B)
    L = np.stack([scale * np.exp(-((grid - c) / 3.0) ** 2) for c in chosen])
    sel.logits.data.copy_(torch.tensor(L, dtype=torch.float32, device=sel.logits.device))


@torch.no_grad()
def recalibrate_bn(net, data, masks, idx, dev, bs=64):
    """Hitung ulang statistik BatchNorm dari seluruh data latih (tanpa augmentasi) sebelum evaluasi."""
    bns = [m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # rata-rata kumulatif
    net.train()
    net.sel.hard = True
    for mod in net.modules():
        if isinstance(mod, nn.Dropout):
            mod.eval()
    use_amp = dev.type == "cuda"
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
        for i in range(0, len(idx), bs):
            b = torch.as_tensor(idx[i:i + bs], device=dev)
            if len(b) < 2:
                continue
            net(data[b].float() / 4096.0, masks[b])


def run_fold(cfg, data, masks, y, tr, te, wl, dev, log_every=10, return_net=False):
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))
    net = Net(cfg, wl).to(dev)
    if cfg["selector"] == "concrete" and cfg.get("init", "fisher") == "fisher":
        # catatan: skor Fisher dihitung pada spektrum SNV-300 dari fold latih
        init_concrete_fisher(net.sel, data, masks, y, tr, cfg.get("init_scale", 3.0))
    ep, bs = cfg.get("epochs", 40), cfg.get("batch", 32)
    off = None
    if cfg.get("anchor") == "pls":
        o_tr, o_te, _ = pls_anchor_offsets(ANCHOR["X"], np.asarray(y), ANCHOR["g"], np.asarray(tr), np.asarray(te))
        off = torch.zeros(len(y), device=dev)
        off[torch.as_tensor(tr, device=dev)] = torch.tensor(o_tr, dtype=torch.float32, device=dev)
        off[torch.as_tensor(te, device=dev)] = torch.tensor(o_te, dtype=torch.float32, device=dev)

    def add_off(logit, idx):
        if off is None:
            return logit
        o = off[idx]
        return logit.float() + torch.stack([torch.zeros_like(o), o], 1)
    sel_params = [p for n, p in net.named_parameters() if n.startswith("sel.")]
    other = [p for n, p in net.named_parameters() if not n.startswith("sel.")]
    groups = [{"params": other, "lr": cfg.get("lr", 1e-3)}]
    if sel_params:
        groups.append({"params": sel_params, "weight_decay": 0.0,
                       "lr": cfg.get("sel_lr", {"concrete": 5e-2, "led": 1e-2}.get(cfg["selector"], 1e-3))})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.get("wd", 1e-4))
    T0, T1 = cfg.get("T0", 10.0), cfg.get("T1", 0.1)
    use_amp = dev.type == "cuda"
    tr_t = torch.as_tensor(tr, device=dev)
    yt = torch.as_tensor(np.array(y), device=dev)
    spe = math.ceil(len(tr) / bs)
    # concrete: tahap 1 = cari band (Gumbel dianneal), tahap 2 = band dikunci + jadwal LR baru untuk backbone
    hard_frac = cfg.get("hard_frac", 0.4) if cfg["selector"] == "concrete" else 0.0
    ep_fix = int(round(ep * hard_frac))
    stages = [("cari", ep - ep_fix), ("kunci", ep_fix)] if ep_fix else [("latih", ep)]
    hist = []
    for name, n_ep in stages:
        if n_ep <= 0:
            continue
        net.sel.hard = name == "kunci"
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                    total_steps=n_ep * spe, pct_start=0.1)
        for e in range(n_ep):
            net.train()
            net.sel.T = T0 * (T1 / T0) ** (e / max(n_ep - 1, 1))
            perm = tr_t[torch.randperm(len(tr_t), device=dev)]
            tot = 0.0
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                if len(b) < 2:
                    sched.step()
                    continue
                x = data[b].float() / 4096.0
                m = masks[b]
                x, m = augment(x, m)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                    logit = add_off(net(x, m, chan_jitter=cfg.get("chan_jitter", 0.03)), b)
                    loss = F.cross_entropy(logit.float(), yt[b], label_smoothing=0.05)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                sched.step()
                tot += loss.item() * len(b)
            hist.append(tot / len(tr))
    recalibrate_bn(net, data, masks, tr, dev)
    net.eval()
    scores = []
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
        for i in range(0, len(te), 64):
            b = torch.as_tensor(te[i:i + 64], device=dev)
            x = data[b].float() / 4096.0
            p = 0
            for flip in (False, True):  # TTA ringan
                xx, mm = (x.flip(-1), masks[b].flip(-1)) if flip else (x, masks[b])
                p = p + F.softmax(add_off(net(xx, mm).float(), b), 1)[:, 1]
            scores.append((p / 2).cpu().numpy())
    sel_nm = net.sel.selected_nm()
    if return_net:
        return np.concatenate(scores), hist, sel_nm, net
    return np.concatenate(scores), hist, sel_nm


ANCHOR = {}   # diisi main(): spektrum rata-rata CSV (X) & tanggal (g), urut sesuai manifest


def pls_anchor_offsets(X, y, g, tr, te):
    """Logit PLS-DA (SNV) sebagai offset tetap. Telur latih: skor cross-fitted (GroupKFold per tanggal di dalam
    data latih, tanpa kebocoran); telur uji: PLS dilatih pada semua tanggal latih. Skor -> logit via Platt."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    Xs = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)
    ytr, gtr = y[tr], g[tr]
    splits = list(GroupKFold(min(5, len(np.unique(gtr)))).split(tr, ytr, gtr))

    def fp(a_idx, b_idx, nc):
        m = PLSRegression(n_components=min(nc, len(a_idx) - 1), scale=True).fit(Xs[a_idx], y[a_idx].astype(float))
        return m.predict(Xs[b_idx]).ravel()
    best = (-1, None, None)
    for nc in (2, 5, 10, 15, 20, 27, 30):
        cf = np.zeros(len(tr))
        for a_, b_ in splits:
            cf[b_] = fp(tr[a_], tr[b_], nc)
        sc = balanced_accuracy_score(ytr, cf > 0.5)
        if sc > best[0]:
            best = (sc, nc, cf)
    _, nc, cf = best
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(cf[:, None], ytr)
    return lr.decision_function(cf[:, None]), lr.decision_function(fp(tr, te, nc)[:, None]), nc


def make_configs(path):
    vip5 = [499, 510, 520, 659, 672]
    C = []
    for k in (3, 5, 8):
        C.append({"tag": f"yolo11s_concrete_k{k}", "backbone": "yolo11s-cls", "selector": "concrete", "k": k})
        C.append({"tag": f"yolo11s_led_k{k}", "backbone": "yolo11s-cls", "selector": "led", "k": k})
        C.append({"tag": f"specpool_concrete_k{k}", "backbone": "specpool", "selector": "concrete", "k": k,
                  "epochs": 150, "lr": 3e-3})
    C.append({"tag": "yolo11s_rgb", "backbone": "yolo11s-cls", "selector": "rgb"})
    C.append({"tag": "yolo11s_vip5", "backbone": "yolo11s-cls", "selector": "fixed", "bands": vip5})
    C.append({"tag": "yolo11s_all16", "backbone": "yolo11s-cls", "selector": "all", "k": 16})
    C.append({"tag": "specpool_all16", "backbone": "specpool", "selector": "all", "k": 16, "epochs": 150, "lr": 3e-3})
    C.append({"tag": "yolo11s_concrete_k5_imgnorm", "backbone": "yolo11s-cls", "selector": "concrete", "k": 5,
              "pixnorm": "img"})
    C.append({"tag": "yolo11s_concrete_k5_randinit", "backbone": "yolo11s-cls", "selector": "concrete", "k": 5,
              "init": "random"})
    for bb in ("yolo11n-cls", "yolo11m-cls", "mobilenet_v3_small", "efficientnet_b0", "resnet18"):
        C.append({"tag": f"{bb.replace('-cls','')}_concrete_k5", "backbone": bb, "selector": "concrete", "k": 5})
    with open(path, "w") as f:
        for c in C:
            f.write(json.dumps(c) + "\n")
    print(f"{len(C)} konfigurasi ditulis ke {path}")


def make_configs_nca(path, path_temporal):
    """Eksperimen tambahan untuk NCA (ditetapkan sebelum melihat hasilnya)."""
    C = []
    for bb in ("resnet18", "mobilenet_v3_small", "efficientnet_b0", "yolo11n-cls", "yolo11m-cls"):
        C.append({"tag": f"{bb.replace('-cls', '')}_all16", "backbone": bb, "selector": "all", "k": 16})
    for k in (4, 8, 32, 64):
        C.append({"tag": f"yolo11s_all{k}", "backbone": "yolo11s-cls", "selector": "all", "k": k})
        C.append({"tag": f"specpool_all{k}", "backbone": "specpool", "selector": "all", "k": k, "epochs": 150, "lr": 3e-3})
    C.append({"tag": "hybridsn_bin75", "backbone": "hybridsn", "selector": "bin", "k": 75, "epochs": 80})
    yolo = {"backbone": "yolo11s-cls", "selector": "all", "k": 16}
    spec = {"backbone": "specpool", "selector": "all", "k": 16, "epochs": 150, "lr": 3e-3}
    for s in range(5):
        C.append({"tag": "pls_anchor_yolo11s_all16" + (f"_s{s}" if s else ""), **yolo, "anchor": "pls", "seed": s})
    for s in range(3):
        C.append({"tag": "pls_anchor_specpool_all16" + (f"_s{s}" if s else ""), **spec, "anchor": "pls", "seed": s})
    with open(path, "w") as f:
        for c in C:
            f.write(json.dumps(c) + "\n")
    T = []
    for s in range(5):
        T.append({"tag": f"yolo11s_all16_s{s}", **yolo, "seed": s})
        T.append({"tag": f"pls_anchor_yolo11s_all16_s{s}", **yolo, "anchor": "pls", "seed": s})
    for s in range(3):
        T.append({"tag": f"specpool_all16_s{s}", **spec, "seed": s})
    with open(path_temporal, "w") as f:
        for c in T:
            f.write(json.dumps(c) + "\n")
    print(f"{len(C)} konfigurasi -> {path}; {len(T)} konfigurasi temporal -> {path_temporal}")


def load_data(d, man, dev, res):
    """Muat data_f16.npy ke perangkat, dikecilkan ke resolusi `res` (avg-pool)."""
    d = Path(d)
    arr = np.load(d / "data_f16.npy", mmap_mode="r")
    rows = man["row"].to_numpy()
    f = max(1, arr.shape[-1] // res)   # faktor pengecilan, mis. 256 -> 128 = 2
    S = arr.shape[-1] // f
    data = torch.empty((len(rows), arr.shape[1], S, S), dtype=torch.float16, device=dev)
    for i in range(0, len(rows), 16):    # muat bertahap supaya memori GPU sementara kecil
        ch = torch.from_numpy(np.ascontiguousarray(arr[rows[i:i + 16]])).to(dev).float()
        data[i:i + 16] = (F.avg_pool2d(ch, f) if f > 1 else ch).half()
        del ch
    mk = torch.from_numpy(np.load(d / "masks.npy")[rows]).float().to(dev)[:, None]
    masks = ((F.avg_pool2d(mk, f) if f > 1 else mk)[:, 0] > 0.5)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return data, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_configs")
    ap.add_argument("--make_configs_nca", nargs=2, metavar=("NCA", "TEMPORAL"))
    ap.add_argument("--anchor_csv", default="", help="spectra_reference.csv (wajib untuk konfigurasi anchor=pls)")
    ap.add_argument("--split", default="lodo", choices=["lodo", "temporal"])
    ap.add_argument("--cut", default="18Dec", help="split temporal: latih s.d. tanggal ini, uji sesudahnya")
    ap.add_argument("--data")
    ap.add_argument("--configs")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--only", default="", help="jalankan hanya tag ini (dipisah koma)")
    ap.add_argument("--epochs", type=int, default=0, help="override epoch (uji cepat)")
    ap.add_argument("--max_folds", type=int, default=0, help="batasi jumlah fold (uji cepat)")
    ap.add_argument("--res", type=int, default=256, help="resolusi input (128 = data ~11 GB di GPU)")
    ap.add_argument("--exclude", default="", help="CSV berkolom 'Sample ID': telur yang dikeluarkan (analisis sensitivitas)")
    a = ap.parse_args()
    if a.make_configs:
        return make_configs(a.make_configs)
    if a.make_configs_nca:
        return make_configs_nca(*a.make_configs_nca)

    d = Path(a.data)
    man = pd.read_csv(d / "manifest.csv")
    if a.exclude:
        ex = set(pd.read_csv(a.exclude)["Sample ID"].astype(str))
        n0 = len(man)
        man = man[~man["Sample ID"].isin(ex)].reset_index(drop=True)
        print(f"[rank {a.rank}] dikeluarkan {n0 - len(man)} telur (sisa {len(man)})", flush=True)
    wl = np.array(json.load(open(d / "meta.json"))["wavelengths"])
    ngpu = torch.cuda.device_count()
    dev = torch.device(f"cuda:{a.rank % ngpu}" if ngpu else "cpu")
    t0 = time.time()
    data, masks = load_data(d, man, dev, a.res)
    y = man["y"].to_numpy()
    dates = man["date"].to_numpy()
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info(dev)
        print(f"[rank {a.rank}] memori GPU bebas setelah muat data: {free/2**30:.1f} / {total/2**30:.0f} GiB", flush=True)
    print(f"[rank {a.rank}] data {tuple(data.shape)} di {dev} ({(time.time()-t0)/60:.1f} mnt muat)", flush=True)

    cfgs = [json.loads(l) for l in open(a.configs) if l.strip()]
    if a.only:
        keep = set(a.only.split(","))
        cfgs = [c for c in cfgs if c["tag"] in keep]
    if any(c.get("anchor") for c in cfgs):
        import baseline as BL
        df, X, _, _ = BL.load(a.anchor_csv)
        pos = {sid.upper(): i for i, sid in enumerate(df["Sample ID"])}
        idx = [pos[str(s).upper()] for s in man["Sample ID"]]
        ANCHOR["X"], ANCHOR["g"] = X[idx], dates
        assert (df["y"].to_numpy()[idx] == y).all(), "label CSV != manifest"
        print(f"[rank {a.rank}] spektrum CSV untuk anchor PLS: {X[idx].shape}", flush=True)
    if a.split == "temporal":
        mon = {"Nov": 11, "Dec": 12, "Jan": 13, "Feb": 14}
        key = lambda s: (mon[s[2:]], int(s[:2]))   # noqa: E731
        is_tr = np.array([key(d) <= key(a.cut) for d in dates])
        splits = [("temporal", np.where(is_tr)[0], np.where(~is_tr)[0])]
        cfgs = cfgs[a.rank::a.world]       # 1 split: bagi konfigurasi antar-GPU
        mine = splits
        print(f"[rank {a.rank}] split temporal: latih {is_tr.sum()} telur ({len(set(dates[is_tr]))} tanggal), "
              f"uji {(~is_tr).sum()} telur ({len(set(dates[~is_tr]))} tanggal)", flush=True)
    else:
        folds = sorted(np.unique(dates))
        if a.max_folds:
            folds = folds[:a.max_folds]
        mine = [(f, np.where(dates != f)[0], np.where(dates == f)[0])
                for i, f in enumerate(folds) if i % a.world == a.rank]
    for cfg in cfgs:
        if a.epochs:
            cfg["epochs"] = a.epochs
        od = Path(a.out) / cfg["tag"]
        od.mkdir(parents=True, exist_ok=True)
        json.dump(cfg, open(od / "config.json", "w"))
        for fd, tr, te in mine:
            fp = od / f"fold_{fd}.json"
            if fp.exists():
                continue
            t1 = time.time()
            s, hist, nm = run_fold(cfg, data, masks, y, tr, te, wl, dev)
            acc = float(((s > 0.5).astype(int) == y[te]).mean())
            json.dump({"fold": fd, "ids": man["Sample ID"].iloc[te].tolist(), "y": y[te].tolist(),
                       "score": s.tolist(), "loss": hist, "selected_nm": nm,
                       "sec": time.time() - t1}, open(fp, "w"))
            print(f"[rank {a.rank}] {cfg['tag']:28s} fold {fd}: acc={acc:.3f} loss_akhir={hist[-1]:.3f} "
                  f"band={None if nm is None else [round(v) for v in nm]} ({time.time()-t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
