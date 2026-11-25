#!/usr/bin/env python3
"""
Audit identitas kubus vs CSV: untuk setiap kubus (eggs/*.npy), cari baris CSV yang spektrumnya paling cocok.
Kubus lolos jika baris terbaik = ID-nya sendiri dan jauh lebih cocok daripada telur lain.
Juga mendeteksi kubus duplikat (dua file berisi telur yang sama).

Pemakaian:
  python audit_ids.py --cubes /raid/home/tegar/egg_hsi/cubes --csv .../spectra_reference.csv --workers 16
"""
import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def mean_spec(stem_dir):
    stem, d = stem_dir
    c = np.load(d / f"{stem}.npy", mmap_mode="r")
    m = np.load(d / f"{stem}_mask.npy").astype(bool)
    return stem, np.asarray(c[:, m], dtype=np.float32).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    d = Path(a.cubes) / "eggs"
    df = pd.read_csv(a.csv)
    df["Sample ID"] = df["Sample ID"].astype(str).str.strip().str.strip("'\"")
    cols = [c for c in df.columns if re.fullmatch(r"\s*\d+(\.\d+)?\s*", str(c))]
    X = df[cols].to_numpy(float)
    ids = df["Sample ID"].tolist()
    low = {i.lower(): i for i in ids}
    stems = sorted(p.stem for p in d.glob("*.npy") if not p.stem.endswith("_mask"))
    with ProcessPoolExecutor(a.workers) as ex:
        specs = dict(ex.map(mean_spec, [(s, d) for s in stems], chunksize=8))
    rows = []
    for s in stems:
        v = specs[s]
        k = (X @ v) / (X * X).sum(1)
        err = np.sqrt(((X * k[:, None] - v) ** 2).mean(1)) / v.mean()
        o = np.argsort(err)
        own = low.get(s.lower())
        rows.append({"file": s, "csv_id_by_name": own, "best_match": ids[o[0]], "err_best": err[o[0]],
                     "err_2nd": err[o[1]], "err_own": err[ids.index(own)] if own else np.nan,
                     "ok": own == ids[o[0]]})
    r = pd.DataFrame(rows)
    out = Path(a.cubes) / "audit_ids.csv"
    r.to_csv(out, index=False)
    print(f"Kubus: {len(r)} | cocok dengan ID sendiri: {int(r.ok.sum())} | bermasalah: {int((~r.ok).sum())}")
    print(f"Selisih ke ID sendiri (yang cocok): median={r[r.ok].err_best.median():.4f}, maks={r[r.ok].err_best.max():.4f}")
    print(f"Jarak ke telur terdekat berikutnya: median={r.err_2nd.median():.4f}, min={r.err_2nd.min():.4f}")
    bad = r[~r.ok]
    if len(bad):
        print("\nBermasalah:")
        print(bad.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    dup = r[r.duplicated("best_match", keep=False)].sort_values("best_match")
    if len(dup):
        print("\nKubus berbeda nama yang cocok ke telur CSV yang sama (kemungkinan duplikat):")
        print(dup[["file", "best_match", "err_best"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        for g, gg in dup.groupby("best_match"):
            a0, a1 = [np.load(d / f"{f}.npy", mmap_mode="r") for f in gg.file[:2]]
            print(f"  {gg.file.iloc[0]} vs {gg.file.iloc[1]}: identik byte-per-byte = {np.array_equal(a0, a1)}")
    no_cube = sorted(set(ids) - {low.get(s.lower()) for s in stems})
    print(f"\nID CSV tanpa kubus (dengan nama yang sesuai): {no_cube}")
    print(f"Detail: {out}")


if __name__ == "__main__":
    main()
