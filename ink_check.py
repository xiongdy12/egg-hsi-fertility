#!/usr/bin/env python3
"""
Cek potensi jalan pintas: coretan/tulisan tangan di cangkang (tinta gelap, garis tipis).

Per telur (kubus 256x256 hasil prepare_cubes):
  1. Citra intensitas = rata-rata band 550-700 nm (transmisi tinggi, tinta tampak gelap).
  2. Black top-hat 5x5: hanya struktur gelap yang lebih sempit dari 5 px (garis tinta), relatif thd sekitar.
  3. Piksel 'tinta' = >= 15% lebih gelap dari sekitarnya (dalam mask dikikis 6 px), komponen >= 15 px,
     garis tegak panjang (artefak kamera line-scan) dikecualikan.
  4. Skor tinta = fraksi piksel telur yang tertandai.
Keluaran:
  - ink_scores.csv, summary.md: prevalensi per kelas & per tanggal, Mann-Whitney, AUC skor tinta -> label
  - fig_ink_examples.png: 8 skor tertinggi per kelas dengan piksel tertandai (untuk dicek mata)
  - (opsional --oof) akurasi model pada telur bertinta vs tidak bertinta

Pemakaian:
  python ink_check.py --cubes /raid/home/tegar/egg_hsi/cubes --out results_ink --workers 16 \
      --oof results_final/seedavg_oof_scores.csv
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def score_one(args):
    path, row, b0, b1, k, min_px = args
    from scipy import ndimage as ndi
    data = np.load(path / "data_f16.npy", mmap_mode="r")
    m = np.load(path / "masks.npy", mmap_mode="r")[row].astype(bool)
    img = np.asarray(data[row, b0:b1], np.float32).mean(0)
    inner = ndi.binary_erosion(m, iterations=6)
    if inner.sum() < 100:
        return row, np.nan, 0
    # black top-hat: hanya garis/titik gelap yang lebih sempit dari 5 px (coretan tinta), relatif thd sekitar
    clo = ndi.grey_closing(img, size=(5, 5))
    rel = (clo - img) / np.maximum(clo, 1e-6)
    # buang pola per kolom (garis vertikal kamera line-scan)
    colmed = np.array([np.median(rel[inner[:, c], c]) if inner[:, c].any() else 0.0 for c in range(rel.shape[1])])
    rel = rel - colmed[None, :]
    ink = (rel > k) & inner
    lab, n = ndi.label(ink)
    keep = np.zeros_like(ink)
    if n:
        H = np.ptp(np.where(inner)[0]) + 1
        for j, sl in enumerate(ndi.find_objects(lab), 1):
            comp = lab[sl] == j
            h, w = comp.shape
            if comp.sum() < min_px:
                continue
            if h > 0.4 * H and w <= 4:   # garis tegak panjang = artefak kamera, bukan coretan
                continue
            keep[sl] |= comp
    return row, float(keep.sum() / inner.sum()), int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes", required=True)
    ap.add_argument("--out", default="results_ink")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--k", type=float, default=0.15, help="piksel >= k (fraksi) lebih gelap dari sekitarnya")
    ap.add_argument("--min_px", type=int, default=15)
    ap.add_argument("--thr", type=float, default=0.002, help="skor > thr dianggap 'bertinta'")
    ap.add_argument("--oof", default="")
    a = ap.parse_args()
    d = Path(a.cubes)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(d / "manifest.csv")
    wl = np.array(json.load(open(d / "meta.json"))["wavelengths"])
    b0, b1 = int(np.argmin(np.abs(wl - 550))), int(np.argmin(np.abs(wl - 700))) + 1
    jobs = [(d, int(r), b0, b1, a.k, a.min_px) for r in man["row"]]
    with ProcessPoolExecutor(a.workers) as ex:
        res = {r: (s, n) for r, s, n in ex.map(score_one, jobs, chunksize=8)}
    man["ink_score"] = [res[r][0] for r in man["row"]]
    man["ink_px"] = [res[r][1] for r in man["row"]]
    man["inked"] = man["ink_score"] > a.thr
    man.to_csv(out / "ink_scores.csv", index=False)

    from scipy.stats import mannwhitneyu, fisher_exact
    from sklearn.metrics import roc_auc_score
    y = man["y"].to_numpy()
    s = man["ink_score"].fillna(0).to_numpy()
    L = ["# Cek coretan/tinta di cangkang\n",
         f"Parameter: band {wl[b0]:.0f}-{wl[b1-1]:.0f} nm, top-hat 5 px, >= {100*a.k:.0f}% lebih gelap, komponen >= {a.min_px} px, "
         f"ambang 'bertinta' skor > {a.thr}\n"]
    for lab, nm in [(1, "fertil"), (0, "infertil")]:
        v = man[man.y == lab]
        L.append(f"{nm:8s}: bertinta {int(v.inked.sum())}/{len(v)} ({100*v.inked.mean():.1f}%), "
                 f"skor median {v.ink_score.median():.4f}, p90 {v.ink_score.quantile(.9):.4f}")
    tab = pd.crosstab(man.y, man.inked)
    if tab.shape == (2, 2):
        L.append(f"Fisher exact (bertinta vs kelas): p={fisher_exact(tab.values)[1]:.3g}")
    L.append(f"Mann-Whitney skor tinta F vs IF: p={mannwhitneyu(s[y == 1], s[y == 0]).pvalue:.3g}")
    L.append(f"AUC skor tinta untuk menebak 'fertil': {roc_auc_score(y, s):.3f} (0,5 = tidak informatif)")
    bd = man.groupby(["date", "y"]).inked.mean().unstack().rename(columns={0: "IF", 1: "F"}).round(2)
    L.append("\nProporsi bertinta per tanggal:\n" + bd.to_string())

    if a.oof and Path(a.oof).exists():
        o = pd.read_csv(a.oof, index_col=0)
        mm = man.set_index("Sample ID").join(o, how="inner", rsuffix="_o")
        L.append("\nAkurasi model pada telur bertinta vs tidak (prediksi OOF):")
        for col in [c for c in ["pls", "yolo11s_all16", "specpool_all16"] if c in mm.columns]:
            for flag, nm in [(True, "bertinta"), (False, "tanpa tinta")]:
                g = mm[mm.inked == flag]
                if len(g):
                    acc = ((g[col] > .5).astype(int) == g["y"]).mean()
                    L.append(f"  {col:16s} {nm:12s} n={len(g):4d} acc={acc:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi
    data = np.load(d / "data_f16.npy", mmap_mode="r")
    masks = np.load(d / "masks.npy", mmap_mode="r")
    fig, axs = plt.subplots(2, 8, figsize=(13, 3.6))
    for i, (lab, nm) in enumerate([(1, "Fertile"), (0, "Infertile")]):
        top = man[man.y == lab].sort_values("ink_score", ascending=False).head(8)
        for j, (_, r) in enumerate(top.iterrows()):
            img = np.asarray(data[int(r.row), b0:b1], np.float32).mean(0)
            mk = masks[int(r.row)].astype(bool)
            lo, hi = np.percentile(img[mk], [1, 99])
            axs[i, j].imshow(np.clip((img - lo) / (hi - lo + 1e-9), 0, 1) * mk, cmap="gray")
            _, sc, _ = score_one((d, int(r.row), b0, b1, a.k, a.min_px))
            axs[i, j].set_title(f"{nm} {r['Sample ID'][-9:]}\nink={r.ink_score:.4f}", fontsize=6)
            axs[i, j].axis("off")
    fig.suptitle("8 skor tinta tertinggi per kelas (cek mata: apakah benar coretan?)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_ink_examples.png", dpi=150)
    (out / "summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
