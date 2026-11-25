#!/usr/bin/env python3
"""
Bukti penggunaan informasi spasial oleh YOLOv11s (300 band, konfigurasi yolo11s_all16).

Untuk setiap fold LODO: latih ulang model (seed 0, protokol identik), lalu pada telur uji:
  1. Grad-CAM (kelas yang diprediksi) pada lapisan backbone stride 16  -> peta 128x128.
  2. Uji kausal spasial (spektrum tiap piksel dipertahankan):
       shuffle : posisi piksel di dalam mask diacak (n kali, rata-rata skor)
       flat    : setiap piksel diganti spektrum rata-rata telurnya (variasi spasial hilang total)
     Jika akurasi/AUC turun -> model memakai struktur spasial, bukan hanya spektrum rata-rata.

Tahap latih (paralel per GPU):
  CUDA_VISIBLE_DEVICES=4 python gradcam_spatial.py --data .../cubes --rank 0 --world 4 &
  ... (rank 1..3 di GPU 5..7)
Tahap laporan + gambar:
  python gradcam_spatial.py --data .../cubes --report --out results_spatial
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import train_cube as T  # noqa: E402


def gradcam(net, x, m, layer):
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda mod, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda mod, gi, go: grads.__setitem__("g", go[0]))
    net.zero_grad(set_to_none=True)
    logit = net(x, m).float()
    pred = logit.argmax(1)
    logit.gather(1, pred[:, None]).sum().backward()
    h1.remove()
    h2.remove()
    a, g = acts["a"].float(), grads["g"].float()
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam * m
    cam = cam / cam.flatten(1).amax(1).clamp_min(1e-9)[:, None, None]
    return cam.detach(), F.softmax(logit, 1)[:, 1].detach()


@torch.no_grad()
def score(net, x, m):
    return F.softmax(net(x, m).float(), 1)[:, 1]


@torch.no_grad()
def shuffle_within_mask(x, m, gen):
    out = x.clone()
    for i in range(x.shape[0]):
        idx = torch.nonzero(m[i].flatten(), as_tuple=False)[:, 0]
        perm = idx[torch.randperm(len(idx), generator=gen, device="cpu").to(idx.device)]
        flat = out[i].flatten(1)
        flat[:, idx] = x[i].flatten(1)[:, perm]
    return out


@torch.no_grad()
def flatten_spatial(x, m):
    mm = m[:, None].to(x.dtype)
    mean = (x * mm).sum((2, 3), keepdim=True) / mm.sum((2, 3), keepdim=True).clamp_min(1)
    return mean * mm


def train_mode(a):
    d = Path(a.data)
    man = pd.read_csv(d / "manifest.csv")
    wl = np.array(json.load(open(d / "meta.json"))["wavelengths"])
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data, masks = T.load_data(d, man, dev, a.res)
    y = man["y"].to_numpy()
    dates = man["date"].to_numpy()
    cfg = next(json.loads(l) for l in open(a.configs) if json.loads(l)["tag"] == a.tag)
    if a.epochs:
        cfg["epochs"] = a.epochs
    out = Path(a.out) / "folds"
    out.mkdir(parents=True, exist_ok=True)
    folds = sorted(np.unique(dates))[: a.max_folds or None]
    gen = torch.Generator().manual_seed(0)
    for i, fd in enumerate(folds):
        if i % a.world != a.rank or (out / f"{fd}.npz").exists():
            continue
        t0 = time.time()
        tr, te = np.where(dates != fd)[0], np.where(dates == fd)[0]
        _, _, _, net = T.run_fold(cfg, data, masks, y, tr, te, wl, dev, return_net=True)
        net.eval()
        body = net.backbone.body if hasattr(net.backbone, "body") else None
        layer = body[a.cam_layer] if body is not None else None
        cams, s0, ssh, sfl = [], [], [], []
        for j in range(0, len(te), 16):
            b = torch.as_tensor(te[j:j + 16], device=dev)
            x = data[b].float() / 4096.0
            m = masks[b]
            if layer is not None:
                with torch.enable_grad():
                    c, p = gradcam(net, x.requires_grad_(False), m, layer)
                cams.append(c.half().cpu().numpy())
            else:
                p = score(net, x, m)
            s0.append(p.cpu().numpy())
            sh = torch.zeros(len(b), device=dev)
            for _ in range(a.n_shuffle):
                sh += score(net, shuffle_within_mask(x, m, gen), m)
            ssh.append((sh / a.n_shuffle).cpu().numpy())
            sfl.append(score(net, flatten_spatial(x, m), m).cpu().numpy())
        np.savez_compressed(out / f"{fd}.npz", ids=man["Sample ID"].iloc[te].to_numpy(), y=y[te],
                            rows=man["row"].iloc[te].to_numpy(), s_orig=np.concatenate(s0),
                            s_shuffle=np.concatenate(ssh), s_flat=np.concatenate(sfl),
                            cam=np.concatenate(cams) if cams else np.zeros(0))
        acc = lambda s: float(((s > 0.5) == y[te]).mean())
        print(f"[rank {a.rank}] fold {fd}: acc asli={acc(np.concatenate(s0)):.3f} "
              f"acak={acc(np.concatenate(ssh)):.3f} rata={acc(np.concatenate(sfl)):.3f} ({time.time()-t0:.0f}s)",
              flush=True)


def report_mode(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score as bacc
    sys.path.insert(0, str(Path(__file__).parent))
    import baseline as Bl
    import make_figures as MF  # gaya & palet yang sama
    out = Path(a.out)
    fs = sorted((out / "folds").glob("*.npz"))
    if not fs:
        sys.exit("Belum ada hasil fold.")
    Z = [np.load(f, allow_pickle=True) for f in fs]
    cat = lambda k: np.concatenate([z[k] for z in Z])
    y, s0, ssh, sfl, rows = cat("y"), cat("s_orig"), cat("s_shuffle"), cat("s_flat"), cat("rows")
    cam = cat("cam").astype(np.float32) if Z[0]["cam"].size else None
    L = [f"# Uji spasial YOLOv11s ({len(fs)} fold, n={len(y)})\n"]
    if len(fs) < 25:
        L.append(f"PERINGATAN: baru {len(fs)}/25 fold. Cek logs/spatial_rank*.log lalu jalankan ulang rank yang gagal.\n")

    def paired(sa, sb, n=5000):
        rng = np.random.default_rng(0)
        dA, dB = [], []
        for _ in range(n):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) < 2:
                continue
            dA.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
            dB.append(bacc(y[i], sa[i] > .5) - bacc(y[i], sb[i] > .5))
        dA, dB = np.array(dA), np.array(dB)
        return dA.mean(), *np.percentile(dA, [2.5, 97.5]), dB.mean(), *np.percentile(dB, [2.5, 97.5])

    for nm, s in [("asli", s0), ("piksel diacak", ssh), ("diratakan (spektrum rata-rata)", sfl)]:
        mt = Bl.metrics(y, s)
        L.append(f"{nm:32s} bal_acc={mt['bal_acc']:.3f} AUC={mt['auc']:.3f}")
    for nm, s in [("asli - acak", ssh), ("asli - rata", sfl)]:
        r = paired(s0, s)
        L.append(f"{nm:14s} dAUC={r[0]:+.4f} [{r[1]:+.4f},{r[2]:+.4f}]  dBalAcc={r[3]:+.4f} [{r[4]:+.4f},{r[5]:+.4f}]")

    if cam is not None:
        d = Path(a.data)
        masks = np.load(d / "masks.npy", mmap_mode="r")
        H = cam.shape[-1]
        f = masks.shape[-1] // H
        mk = np.array([masks[r][::f, ::f][:H, :H] for r in rows]).astype(bool)
        yy, xx = np.mgrid[:H, :H]
        prof = {0: [], 1: []}
        nb = 10
        for i in range(len(y)):
            if not mk[i].any() or (s0[i] > .5) != y[i]:
                continue
            cy, cx = yy[mk[i]].mean(), xx[mk[i]].mean()
            rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            rr = rr / rr[mk[i]].max()
            h = [cam[i][mk[i] & (rr >= k / nb) & (rr < (k + 1) / nb)].mean() if
                 (mk[i] & (rr >= k / nb) & (rr < (k + 1) / nb)).any() else np.nan for k in range(nb)]
            prof[int(y[i])].append(h)
        P = {k: np.array(v) for k, v in prof.items()}
        L.append("\nProfil radial Grad-CAM (telur terklasifikasi benar; 0 = pusat, 1 = tepi):")
        for k, nm in [(1, "fertil"), (0, "infertil")]:
            if len(P[k]):
                L.append(f"  {nm:8s} " + " ".join(f"{v:.2f}" for v in np.nanmean(P[k], 0)))

        data = np.load(d / "data_f16.npy", mmap_mode="r")
        wl = np.array(json.load(open(d / "meta.json"))["wavelengths"])
        bi = [int(np.argmin(np.abs(wl - t))) for t in (640, 550, 460)]

        def rgb(r):
            c = np.asarray(data[r][bi], np.float32)[:, ::f, ::f][:, :H, :H].transpose(1, 2, 0)
            lo, hi = np.percentile(c[c > 0], [1, 99]) if (c > 0).any() else (0, 1)
            return np.clip((c - lo) / (hi - lo + 1e-9), 0, 1)
        conf = np.abs(s0 - 0.5)
        ex = []
        for k in (1, 0):
            cand = np.where((y == k) & ((s0 > .5) == y))[0]
            ex += list(cand[np.argsort(-conf[cand])[:3]])
        fig = plt.figure(figsize=(MF.W2, MF.W2 * 0.62))
        gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.55, 1.6], hspace=0.28, wspace=0.08)
        img_axes = []
        for j, i in enumerate(ex):
            ax = fig.add_subplot(gs[j // 3, j % 3])
            ax.imshow(rgb(rows[i]))
            im = ax.imshow(np.ma.masked_where(~mk[i], cam[i]), cmap="magma", alpha=0.55, vmin=0, vmax=1)
            img_axes.append(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"{'Fertile' if y[i] else 'Infertile'}, p={s0[i]:.2f}", fontsize=6.5)
            if j % 3 == 0:
                ax.text(-0.08, 1.12, "(a)" if j == 0 else "", transform=ax.transAxes, fontweight="bold")
        cb = fig.colorbar(im, ax=img_axes, orientation="horizontal", fraction=0.04, pad=0.04, aspect=40)
        cb.set_label("Grad-CAM (normalised per egg)", fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
        ax = fig.add_subplot(gs[:, 4])
        rc = (np.arange(nb) + 0.5) / nb
        for k, nm, col, mkr in [(1, "Fertile", MF.C[0], "o"), (0, "Infertile", MF.C[1], "s")]:
            if len(P[k]):
                mu = np.nanmean(P[k], 0)
                se = np.nanstd(P[k], 0) / np.sqrt(np.sum(~np.isnan(P[k]), 0))
                ax.plot(rc, mu, color=col, marker=mkr, ms=3.5, label=f"{nm} (n={len(P[k])})")
                ax.fill_between(rc, mu - 1.96 * se, mu + 1.96 * se, color=col, alpha=0.18, lw=0)
        ax.set_xlabel("Normalised radius\n(0 = centre, 1 = rim)")
        ax.set_ylabel("Mean Grad-CAM (correct predictions)")
        ax.legend(loc="upper right")
        ax.text(-0.3, 1.02, "(b)", transform=ax.transAxes, fontweight="bold")
        MF.save(fig, out, "fig_gradcam")

        fig, ax = plt.subplots(figsize=(MF.W1, MF.W1 * 0.7))
        vals = [Bl.metrics(y, s)["bal_acc"] for s in (s0, ssh, sfl)]
        cis = [Bl.boot_ci(y, s) for s in (s0, ssh, sfl)]
        labs = ["Intact cube", "Pixels shuffled\nwithin egg", "Mean spectrum\nat every pixel"]
        xs = np.arange(3)
        ax.errorbar(xs, vals, yerr=[[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]],
                    fmt="o", ms=5, color=MF.C[0], elinewidth=1.2, capsize=0, zorder=3)
        for xi, v, c in zip(xs, vals, cis):
            ax.text(xi, c[1] + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=6.3, color=MF.INK2)
        ax.axhline(0.5, color=MF.INK2, lw=0.8, ls=(0, (2, 2)))
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, fontsize=6.5)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylabel("Balanced accuracy, YOLOv11s (LODO)")
        ax.set_ylim(0.4, 1.0)
        ax.grid(axis="x", visible=False)
        MF.save(fig, out, "fig_spatial_ablation")

    (out / "summary.md").write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--configs", default="configs.jsonl")
    ap.add_argument("--tag", default="yolo11s_all16")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--cam_layer", type=int, default=6, help="indeks lapisan backbone YOLO (6 = stride 16)")
    ap.add_argument("--n_shuffle", type=int, default=5)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--max_folds", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default="results_spatial")
    a = ap.parse_args()
    report_mode(a) if a.report else train_mode(a)


if __name__ == "__main__":
    main()
