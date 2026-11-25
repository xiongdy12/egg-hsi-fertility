#!/usr/bin/env python3
"""
Lembar kontak untuk audit visual tulisan/coretan di cangkang: semua telur, 96 per halaman (12 x 8).
Citra = rata-rata band 550-700 nm, kontras 1-99 persentil, label = nomor urut + ID singkat.
Kelas TIDAK ditampilkan (audit buta); urutan acak tetap (seed 0) supaya kelas & tanggal bercampur.

Pemakaian:
  python contact_sheets.py --cubes /raid/home/tegar/egg_hsi/cubes --out results_ink/sheets
  -> sheet_01.png ... sheet_13.png + sheet_index.csv (nomor -> Sample ID)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes", required=True)
    ap.add_argument("--out", default="results_ink/sheets")
    ap.add_argument("--per_page", type=int, default=96)
    ap.add_argument("--cols", type=int, default=12)
    ap.add_argument("--thumb", type=int, default=160)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = Path(a.cubes)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(d / "manifest.csv")
    wl = np.array(json.load(open(d / "meta.json"))["wavelengths"])
    b0, b1 = int(np.argmin(np.abs(wl - 550))), int(np.argmin(np.abs(wl - 700))) + 1
    data = np.load(d / "data_f16.npy", mmap_mode="r")
    masks = np.load(d / "masks.npy", mmap_mode="r")
    order = np.random.default_rng(0).permutation(len(man))
    idx = pd.DataFrame({"no": np.arange(1, len(man) + 1), "Sample ID": man["Sample ID"].to_numpy()[order],
                        "row": man["row"].to_numpy()[order]})
    idx.to_csv(out / "sheet_index.csv", index=False)
    S = data.shape[-1]
    step = max(1, S // a.thumb)
    rows_pp = a.per_page // a.cols
    for p in range(0, len(idx), a.per_page):
        chunk = idx.iloc[p:p + a.per_page]
        fig, axs = plt.subplots(rows_pp, a.cols, figsize=(a.cols * 1.25, rows_pp * 1.35))
        for ax in axs.ravel():
            ax.axis("off")
        for ax, (_, r) in zip(axs.ravel(), chunk.iterrows()):
            img = np.asarray(data[int(r.row), b0:b1, ::step, ::step], np.float32).mean(0)
            mk = masks[int(r.row)][::step, ::step].astype(bool)
            lo, hi = np.percentile(img[mk], [1, 99]) if mk.any() else (0, 1)
            ax.imshow(np.clip((img - lo) / (hi - lo + 1e-9), 0, 1) * mk, cmap="gray", interpolation="lanczos")
            ax.set_title(f"#{r.no}", fontsize=7, pad=1.5)
        fig.subplots_adjust(left=0.005, right=0.995, top=0.985, bottom=0.005, wspace=0.04, hspace=0.18)
        fn = out / f"sheet_{p // a.per_page + 1:02d}.png"
        fig.savefig(fn, dpi=110)
        plt.close(fig)
        print(fn, flush=True)


if __name__ == "__main__":
    main()
