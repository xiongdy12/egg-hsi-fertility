#!/usr/bin/env python3
"""
Inspeksi dataset HSI telur UIUC (IDB-8141497) sebelum membangun pipeline YOLO.

Yang dicek:
  1. Struktur file .mat (v5 atau v7.3/HDF5): nama variabel, shape, dtype, sumbu panjang gelombang.
  2. Statistik intensitas dan ukuran telur di frame (mask Otsu): porsi area, bbox, jumlah objek.
  3. CSV referensi: kolom, kolom ID/label/spektrum, distribusi label, kecocokan ID dengan nama file .mat.
  4. Pseudo-RGB (band ~640/550/460 nm) + overlay mask -> PNG untuk dicek visual.
  5. Spektrum rata-rata per kelas (kalau label tersedia).

Pemakaian:
  python inspect_data.py --mat_dir /data/extracted --csv /data/raw/<file>.csv --out ./inspect_out
  python inspect_data.py --mat_dir /data/extracted --csv ... --n_samples 40 --headers_only_all

Kirim balik ke Claude: inspect_out/report.md (dan beberapa PNG di inspect_out/png).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

# ----------------------------------------------------------------------------- I/O .mat

def is_hdf5(path):
    with open(path, "rb") as f:
        return f.read(8) == b"\x89HDF\r\n\x1a\n"


def mat_headers(path):
    """Daftar (nama, shape, dtype) tanpa memuat data penuh."""
    out = []
    if is_hdf5(path):
        import h5py
        with h5py.File(path, "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # MATLAB v7.3 menyimpan array dalam urutan kolom -> shape terbalik
                    out.append((name, tuple(obj.shape[::-1]), str(obj.dtype), "v7.3"))
            f.visititems(visit)
    else:
        import scipy.io as sio
        for name, shape, cls in sio.whosmat(str(path)):
            out.append((name, tuple(shape), cls, "v5"))
    return out


def mat_load(path):
    """Return dict nama -> np.ndarray (hanya array numerik)."""
    data = {}
    if is_hdf5(path):
        import h5py
        with h5py.File(path, "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "fiub":
                    data[name] = np.asarray(obj).T  # balik ke urutan MATLAB
            f.visititems(visit)
    else:
        import scipy.io as sio
        m = sio.loadmat(str(path))
        for k, v in m.items():
            if not k.startswith("__") and isinstance(v, np.ndarray) and v.dtype.kind in "fiub":
                data[k] = v
    return data


def pick_cube(arrays, n_bands_hint=300):
    """Pilih array 3D terbesar sebagai kubus; tentukan sumbu band."""
    cands = [(k, v) for k, v in arrays.items() if v.ndim == 3]
    if not cands:
        return None, None, None
    k, v = max(cands, key=lambda kv: kv[1].size)
    shp = v.shape
    if n_bands_hint in shp:
        band_axis = shp.index(n_bands_hint)
    else:
        band_axis = int(np.argmin(shp))  # tebakan: sumbu terkecil = band
    cube = np.moveaxis(v, band_axis, -1)  # -> H x W x B
    return k, cube, band_axis


def pick_wavelengths(arrays, n_bands):
    """Cari vektor panjang gelombang (1D, panjang n_bands, nilai 300-1100)."""
    for k, v in arrays.items():
        vv = np.squeeze(v)
        if vv.ndim == 1 and vv.size == n_bands and np.nanmin(vv) > 250 and np.nanmax(vv) < 1200:
            return k, vv.astype(float)
    return None, None

# ----------------------------------------------------------------------------- analisis citra

def otsu(x):
    x = x[np.isfinite(x)].ravel()
    hist, edges = np.histogram(x, bins=256)
    p = hist.astype(float) / max(hist.sum(), 1)
    w = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    mu_t = mu[-1]
    sb = (mu_t * w - mu) ** 2 / (w * (1 - w) + 1e-12)
    return (edges[np.nanargmax(sb)] + edges[np.nanargmax(sb) + 1]) / 2


def egg_mask(cube):
    """Mask telur dari citra intensitas rata-rata. Transmisi: telur bisa lebih terang ATAU lebih gelap."""
    img = np.nanmean(cube.astype(np.float32), axis=-1)
    t = otsu(img)
    m = img > t
    # pilih sisi yang tidak menyentuh banyak tepi frame (latar belakang biasanya menyentuh tepi)
    def border_frac(mm):
        b = np.concatenate([mm[0], mm[-1], mm[:, 0], mm[:, -1]])
        return b.mean()
    if border_frac(m) > border_frac(~m):
        m = ~m
    try:
        from scipy import ndimage as ndi
        m = ndi.binary_opening(m, iterations=2)
        m = ndi.binary_fill_holes(m)
        lab, n = ndi.label(m)
        if n > 0:
            sizes = ndi.sum(m, lab, range(1, n + 1))
            n_big = int((sizes > 0.01 * m.size).sum())
            keep = int(np.argmax(sizes)) + 1
            m = lab == keep
        else:
            n_big = 0
    except ImportError:
        n_big = -1
    return img, m, n_big


def bbox(m):
    ys, xs = np.where(m)
    if ys.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def pseudo_rgb(cube, wl, targets=(640, 550, 460)):
    B = cube.shape[-1]
    if wl is None:
        # asumsi dari paper: 374-1015 nm, 300 band (tandai sebagai ASUMSI di laporan)
        wl = np.linspace(374, 1015, B)
    idx = [int(np.argmin(np.abs(wl - t))) for t in targets]
    rgb = cube[..., idx].astype(np.float32)
    lo, hi = np.nanpercentile(rgb, 1), np.nanpercentile(rgb, 99)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
    return rgb, idx

# ----------------------------------------------------------------------------- CSV

def norm_id(s):
    s = Path(str(s)).stem.lower().strip()
    s = re.sub(r"[^a-z0-9]", "", s)
    m = re.search(r"\d+", s)
    return (s, str(int(m.group())) if m else None)


def inspect_csv(csv_path, mat_stems):
    rep = {}
    if pd is None:
        rep["error"] = "pandas tidak terpasang"
        return rep, None
    df = pd.read_csv(csv_path)
    rep["n_rows"], rep["n_cols"] = df.shape
    cols = list(df.columns)
    num_name = [c for c in cols if re.fullmatch(r"\s*\d+(\.\d+)?\s*", str(c))]
    rep["spectral_cols"] = {"n": len(num_name),
                            "first": num_name[:3], "last": num_name[-3:]}
    non_spec = [c for c in cols if c not in num_name]
    rep["non_spectral_cols"] = non_spec
    rep["head_non_spectral"] = df[non_spec].head(5).astype(str).to_dict(orient="records")
    id_col = next((c for c in non_spec if re.search(r"id|image|name|file|sample", str(c), re.I)), None)
    lab_col = next((c for c in non_spec if re.search(r"fert", str(c), re.I)), None)
    rep["id_col_guess"], rep["label_col_guess"] = id_col, lab_col
    if lab_col is not None:
        rep["label_counts"] = df[lab_col].astype(str).value_counts().to_dict()
    for c in non_spec:
        if c not in (id_col, lab_col) and df[c].dtype.kind in "fi":
            rep.setdefault("numeric_ref_stats", {})[c] = {
                "mean": float(df[c].mean()), "std": float(df[c].std()),
                "min": float(df[c].min()), "max": float(df[c].max()),
                "n_missing": int(df[c].isna().sum())}
    if id_col is not None and mat_stems:
        csv_full = {norm_id(x)[0] for x in df[id_col]}
        csv_num = {norm_id(x)[1] for x in df[id_col]}
        mat_full = {norm_id(x)[0] for x in mat_stems}
        mat_num = {norm_id(x)[1] for x in mat_stems}
        rep["id_match"] = {
            "n_mat_files_seen": len(mat_stems),
            "exact_match": len(csv_full & mat_full),
            "numeric_match": len(csv_num & mat_num),
            "mat_examples": sorted(mat_stems)[:5],
            "csv_examples": df[id_col].astype(str).head(5).tolist(),
            "duplicated_ids_in_csv": int(df[id_col].duplicated().sum()),
        }
    return rep, (df, id_col, lab_col, num_name)

# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_dir", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default="inspect_out")
    ap.add_argument("--n_samples", type=int, default=40)
    ap.add_argument("--n_png", type=int, default=8)
    ap.add_argument("--headers_only_all", action="store_true",
                    help="baca header SEMUA .mat (cepat) untuk cek konsistensi shape")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out); (out / "png").mkdir(parents=True, exist_ok=True)
    mats = sorted(Path(a.mat_dir).rglob("*.mat"))
    if not mats:
        sys.exit(f"Tidak ada .mat di {a.mat_dir}")
    rng = np.random.default_rng(a.seed)
    sample = [mats[i] for i in sorted(rng.choice(len(mats), min(a.n_samples, len(mats)), replace=False))]
    report = {"n_mat_files": len(mats), "n_inspected": len(sample)}

    # 1. header
    report["headers_first_file"] = mat_headers(sample[0])
    if a.headers_only_all:
        shapes = {}
        t0 = time.time()
        for p in mats:
            try:
                h = mat_headers(p)
                big = max(h, key=lambda r: np.prod(r[1]) if r[1] else 0)
                shapes[str(big[1])] = shapes.get(str(big[1]), 0) + 1
            except Exception as e:
                shapes["ERROR:" + type(e).__name__] = shapes.get("ERROR:" + type(e).__name__, 0) + 1
        report["all_files_main_shape_counts"] = shapes
        report["headers_scan_seconds"] = round(time.time() - t0, 1)

    # 2. sampel penuh
    per_file, spectra = [], {}
    wl_global, wl_src = None, None
    for i, p in enumerate(sample):
        t0 = time.time()
        rec = {"file": p.name, "size_MB": round(p.stat().st_size / 2**20, 1)}
        try:
            arrs = mat_load(p)
            key, cube, bax = pick_cube(arrs)
            if cube is None:
                rec["error"] = "tidak ada array 3D"; per_file.append(rec); continue
            wk, wl = pick_wavelengths(arrs, cube.shape[-1])
            if wl is not None and wl_global is None:
                wl_global, wl_src = wl, f"variabel '{wk}' di {p.name}"
            img, m, n_obj = egg_mask(cube)
            bb = bbox(m)
            spec = np.nanmean(cube[m], axis=0) if m.any() else np.nanmean(cube.reshape(-1, cube.shape[-1]), 0)
            spectra[p.stem] = spec
            rec.update({
                "cube_var": key, "orig_band_axis": bax, "HxWxB": list(cube.shape),
                "dtype": str(cube.dtype),
                "min": float(np.nanmin(cube)), "max": float(np.nanmax(cube)),
                "mean": float(np.nanmean(cube)), "n_nan": int(np.isnan(cube).sum()) if cube.dtype.kind == "f" else 0,
                "egg_area_frac": round(float(m.mean()), 3),
                "egg_bbox_yxyx": bb, "n_large_objects": n_obj,
                "other_vars": {k: list(v.shape) for k, v in arrs.items() if k != key},
                "load_s": round(time.time() - t0, 2),
            })
            if i < a.n_png:
                import matplotlib; matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                rgb, idx = pseudo_rgb(cube, wl_global)
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(rgb); ax[0].set_title(f"pseudo-RGB bands {idx}")
                ax[1].imshow(img, cmap="gray"); ax[1].set_title("mean intensity")
                ax[2].imshow(img, cmap="gray"); ax[2].contour(m, colors="r", linewidths=1)
                ax[2].set_title(f"mask, area={m.mean():.2f}")
                for x in ax: x.axis("off")
                fig.suptitle(p.name); fig.tight_layout()
                fig.savefig(out / "png" / f"{p.stem}.png", dpi=90); plt.close(fig)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        per_file.append(rec)
        print(f"[{i+1}/{len(sample)}] {p.name} {rec.get('HxWxB', rec.get('error'))}", flush=True)
    report["per_file"] = per_file
    report["wavelengths"] = ({"source": wl_src, "min": float(wl_global.min()), "max": float(wl_global.max()),
                              "n": int(wl_global.size)} if wl_global is not None
                             else "TIDAK DITEMUKAN di .mat; pseudo-RGB memakai ASUMSI linspace(374,1015,B)")

    ok = [r for r in per_file if "HxWxB" in r]
    if ok:
        report["summary_sample"] = {
            "distinct_shapes": sorted({str(r["HxWxB"]) for r in ok}),
            "egg_area_frac_mean": round(float(np.mean([r["egg_area_frac"] for r in ok])), 3),
            "egg_area_frac_min": min(r["egg_area_frac"] for r in ok),
            "egg_area_frac_max": max(r["egg_area_frac"] for r in ok),
            "files_with_>1_large_object": sum(1 for r in ok if r["n_large_objects"] > 1),
            "value_range": [min(r["min"] for r in ok), max(r["max"] for r in ok)],
            "mean_file_MB": round(float(np.mean([r["size_MB"] for r in ok])), 1),
            "mean_load_s": round(float(np.mean([r["load_s"] for r in ok])), 2),
        }

    # 3. CSV
    if a.csv:
        csv_rep, ctx = inspect_csv(a.csv, [p.stem for p in mats])
        report["csv"] = csv_rep
        # 5. spektrum per kelas (dari CSV mean spectra + dari kubus sampel)
        if ctx is not None:
            df, id_col, lab_col, num_cols = ctx
            try:
                import matplotlib; matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                if lab_col and num_cols:
                    wl_csv = np.array([float(c) for c in num_cols])
                    fig, ax = plt.subplots(figsize=(7, 4))
                    for lab, g in df.groupby(lab_col):
                        mu = g[num_cols].to_numpy(float).mean(0)
                        sd = g[num_cols].to_numpy(float).std(0)
                        ax.plot(wl_csv, mu, label=f"{lab} (n={len(g)})")
                        ax.fill_between(wl_csv, mu - sd, mu + sd, alpha=0.2)
                    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("mean spectrum (CSV)")
                    ax.legend(); fig.tight_layout()
                    fig.savefig(out / "class_mean_spectra_csv.png", dpi=110); plt.close(fig)
                    # separabilitas kasar: Cohen's d per band
                    labs = df[lab_col].unique()
                    if len(labs) == 2:
                        A = df[df[lab_col] == labs[0]][num_cols].to_numpy(float)
                        Bm = df[df[lab_col] == labs[1]][num_cols].to_numpy(float)
                        d = (A.mean(0) - Bm.mean(0)) / np.sqrt((A.var(0) + Bm.var(0)) / 2 + 1e-12)
                        top = np.argsort(-np.abs(d))[:10]
                        report["csv"]["top10_bands_cohens_d"] = [
                            {"nm": float(wl_csv[j]), "d": round(float(d[j]), 3)} for j in top]
            except Exception as e:
                report["csv"]["plot_error"] = str(e)

    # tulis laporan
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    md = ["# Laporan inspeksi dataset HSI telur\n",
          f"- File .mat: **{report['n_mat_files']}**, diinspeksi penuh: {report['n_inspected']}",
          f"- Header file pertama: `{report['headers_first_file']}`",
          f"- Panjang gelombang: `{report['wavelengths']}`"]
    if "all_files_main_shape_counts" in report:
        md.append(f"- Shape utama semua file: `{report['all_files_main_shape_counts']}`")
    if "summary_sample" in report:
        md.append("\n## Ringkasan sampel\n```json\n" + json.dumps(report["summary_sample"], indent=2) + "\n```")
    errs = [r for r in per_file if "error" in r]
    if errs:
        md.append(f"\n## Error ({len(errs)})\n" + "\n".join(f"- {r['file']}: {r['error']}" for r in errs[:10]))
    if "csv" in report:
        md.append("\n## CSV\n```json\n" + json.dumps(report["csv"], indent=2, default=str) + "\n```")
    md.append("\n## 5 file pertama\n```json\n" + json.dumps(per_file[:5], indent=2, default=str) + "\n```")
    (out / "report.md").write_text("\n".join(md))
    print(f"\nSelesai. Kirim {out/'report.md'} + beberapa PNG di {out/'png'}")


if __name__ == "__main__":
    main()
