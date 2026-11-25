#!/usr/bin/env python3
"""
Tahap 3a — Ekstraksi & preprocessing kubus HSI (850x900x300) -> tensor ringkas (300x256x256 float16).

Per telur (langsung dari all_data.rar, satu per satu, tanpa mengekstrak 235 GB ke disk):
  1. Ambil file .mat via `7zz e -so` ke memori, baca variabel 'spec'.
  2. Destriping: koreksi gain per kolom detektor line-scan (garis vertikal), per band.
  3. Mask telur (Otsu pada citra intensitas rata-rata, komponen terbesar, isi lubang).
  4. Crop bujur sangkar di sekitar telur (+5% margin), resize ke 256x256 (area), latar = 0.
  5. Simpan <ID>.npy (float16, band x H x W) + <ID>_mask.npy (uint8).
  6. QC: korelasi spektrum rata-rata (dalam mask, sebelum destripe) dengan baris CSV ber-ID sama.
Bisa dijalankan saat unduhan belum selesai: file yang belum lengkap dicatat gagal dan dicoba lagi saat run ulang.

Lalu `--pack` menggabungkan semua ke satu memmap: data_f16.npy (N,300,256,256), masks.npy, manifest.csv.

Pemakaian:
  python prepare_cubes.py --rar /raid/.../all_data.rar --csv .../spectra_reference.csv \
      --out /raid/home/tegar/egg_hsi/cubes --workers 12
  python prepare_cubes.py --out /raid/home/tegar/egg_hsi/cubes --csv ... --pack
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SEVENZ = os.environ.get("SEVENZ", os.path.expanduser("~/bin/7zz"))


def list_rar(rar):
    out = subprocess.run([SEVENZ, "l", "-slt", rar], capture_output=True, text=True).stdout
    items, cur = [], {}
    for line in out.splitlines():
        if line.startswith("Path = "):
            cur = {"path": line[7:].strip()}
        elif line.startswith("Size = ") and cur:
            cur["size"] = int(line[7:].strip() or 0)
            if cur["path"].endswith(".mat"):
                items.append(cur)
            cur = {}
    return items


def egg_id(path):
    return Path(path).stem


def otsu(x):
    hist, edges = np.histogram(x.ravel(), bins=256)
    p = hist / max(hist.sum(), 1)
    w = np.cumsum(p)
    c = (edges[:-1] + edges[1:]) / 2
    mu = np.cumsum(p * c)
    sb = (mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12)
    return c[np.nanargmax(sb)]


def egg_mask(img):
    from scipy import ndimage as ndi
    m = img > otsu(img)
    border = np.r_[m[0], m[-1], m[:, 0], m[:, -1]].mean()
    if border > 0.5:
        m = ~m
    m = ndi.binary_opening(m, iterations=3)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = lab == (int(np.argmax(sizes)) + 1)
    return ndi.binary_fill_holes(m)


def destripe(cube, win=51, lo=0.8, hi=1.25):
    """cube: H x W x B. Gain per kolom = median kolom / median kolom yang dihaluskan."""
    from scipy.ndimage import uniform_filter1d
    med = np.median(cube, axis=0)                       # W x B
    smooth = uniform_filter1d(med, size=win, axis=0, mode="nearest")
    ratio = np.where(smooth > 1e-6 * max(smooth.max(), 1), med / np.maximum(smooth, 1e-9), 1.0)
    ratio = np.clip(ratio, lo, hi)
    return cube / ratio[None, :, :]


def process_one(rar, item, out_dir, size, csv_spec, do_destripe):
    import torch
    torch.set_num_threads(4)
    from scipy.io import loadmat
    eid = egg_id(item["path"])
    t0 = time.time()
    raw = subprocess.run([SEVENZ, "e", "-so", rar, item["path"]], capture_output=True).stdout
    if len(raw) != item["size"]:
        return {"id": eid, "ok": False, "err": f"bytes {len(raw)}/{item['size']} (arsip belum lengkap?)"}
    try:
        cube = loadmat(io.BytesIO(raw), variable_names=["spec"])["spec"]
    except Exception as e:
        return {"id": eid, "ok": False, "err": f"loadmat: {type(e).__name__}: {e}"}
    del raw
    cube = np.asarray(cube, dtype=np.float32)          # H x W x B
    if cube.ndim != 3:
        return {"id": eid, "ok": False, "err": f"shape {cube.shape}"}
    img = cube.mean(-1)
    m = egg_mask(img)
    ys, xs = np.where(m)
    if ys.size < 1000:
        return {"id": eid, "ok": False, "err": "mask kosong"}
    mean_spec = cube[m].mean(0)
    rec = {"id": eid, "ok": True, "H": cube.shape[0], "W": cube.shape[1], "B": cube.shape[2],
           "mask_frac": float(m.mean()), "dn_max": float(cube.max())}
    if csv_spec is not None and eid in csv_spec:
        ref = csv_spec[eid]
        rec["r_csv"] = float(np.corrcoef(ref, mean_spec)[0, 1])
        rec["ratio_csv"] = float(np.median(mean_spec / np.maximum(ref, 1e-9)))
    if do_destripe:
        cube = destripe(cube)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    side = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 1.05) + 2
    y0, x0 = int(round(cy - side / 2)), int(round(cx - side / 2))
    pad = max(0, -y0, -x0, y0 + side - cube.shape[0], x0 + side - cube.shape[1])
    if pad:
        cube = np.pad(cube, ((pad, pad), (pad, pad), (0, 0)))
        m = np.pad(m, pad)
        y0, x0 = y0 + pad, x0 + pad
    crop = cube[y0:y0 + side, x0:x0 + side] * m[y0:y0 + side, x0:x0 + side, None]
    mc = m[y0:y0 + side, x0:x0 + side].astype(np.float32)
    t = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1)))[None]   # 1 x B x S x S
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="area")[0]
    mk = torch.nn.functional.interpolate(torch.from_numpy(mc)[None, None], size=(size, size), mode="area")[0, 0]
    mk = (mk > 0.5)
    t = t * mk
    np.save(out_dir / f"{eid}.npy", t.numpy().astype(np.float16))
    np.save(out_dir / f"{eid}_mask.npy", mk.numpy().astype(np.uint8))
    rec["sec"] = round(time.time() - t0, 1)
    return rec


def load_csv_spectra(csv):
    import pandas as pd
    df = pd.read_csv(csv).copy()
    df["Sample ID"] = df["Sample ID"].astype(str).str.strip().str.strip("'\"")
    cols = [c for c in df.columns if re.fullmatch(r"\s*\d+(\.\d+)?\s*", str(c))]
    return df, cols, {r["Sample ID"]: r[cols].to_numpy(float) for _, r in df.iterrows()}


def pack(out, csv, alias_args=None, use_audit=False):
    df, cols, _ = load_csv_spectra(csv)
    cube_dir = out / "eggs"
    files = {p.stem: p.stem for p in cube_dir.glob("*.npy") if not p.stem.endswith("_mask")}
    lower = {k.lower(): k for k in files}
    alias = dict(x.split("=", 1) for x in (alias_args or []))
    stem_of, how = {}, {}
    if use_audit:
        au = pd.read_csv(out / "audit_ids.csv")
        acc = au[(au.err_best <= 0.005) & (au.err_2nd >= 1.5 * au.err_best)]
        rej = au[~au.index.isin(acc.index)]
        if len(rej):
            print(f"  PERINGATAN: {len(rej)} kubus identitasnya tidak meyakinkan, tidak dipakai: {rej.file.tolist()}")
        dup = acc[acc.best_match.duplicated(keep=False)]
        if len(dup):
            sys.exit(f"Audit tidak satu-satu (dua kubus -> satu ID): {dup[['file','best_match']].values.tolist()}")
        for f, b, own in zip(acc.file, acc.best_match, acc.csv_id_by_name):
            stem_of[b] = f
            if f != b:
                how[b] = "audit spektral" if own != b else "huruf besar/kecil"
    for i in df["Sample ID"]:
        if i in stem_of:
            continue
        if i in files:
            stem_of[i] = i
        elif i in alias and alias[i] in files:
            stem_of[i], how[i] = alias[i], "alias"
        elif i.lower() in lower:
            stem_of[i], how[i] = lower[i.lower()], "huruf besar/kecil"
    for i, h in sorted(how.items()):
        print(f"  pasangan: CSV {i}  <-  file {stem_of[i]}  ({h})")
    ids = [i for i in df["Sample ID"] if i in stem_of]
    missing = [i for i in df["Sample ID"] if i not in stem_of]
    if not ids:
        sys.exit("Tidak ada kubus untuk dipaket.")
    first = np.load(cube_dir / f"{stem_of[ids[0]]}.npy", mmap_mode="r")
    shape = (len(ids),) + first.shape
    print(f"Paket {len(ids)} telur, shape {shape}, ~{np.prod(shape)*2/2**30:.1f} GiB; belum ada: {len(missing)}")
    data = np.lib.format.open_memmap(out / "data_f16.npy", mode="w+", dtype=np.float16, shape=shape)
    masks = np.zeros((len(ids),) + first.shape[1:], np.uint8)
    for k, i in enumerate(ids):
        data[k] = np.load(cube_dir / f"{stem_of[i]}.npy")
        masks[k] = np.load(cube_dir / f"{stem_of[i]}_mask.npy")
        if k % 100 == 0:
            print(f"  {k}/{len(ids)}", flush=True)
    data.flush()
    np.save(out / "masks.npy", masks)
    m = df["Sample ID"].str.extract(r"Egg-([A-Za-z]+)-(\d{2}[A-Za-z]{3})-(\d+)")
    df["y"] = (m[0].str.upper() == "F").astype(int)
    df["date"] = m[1]
    man = df.set_index("Sample ID").loc[ids, ["y", "date"]].reset_index()
    man["row"] = np.arange(len(ids))
    man["file_stem"] = [stem_of[i] for i in ids]
    man.to_csv(out / "manifest.csv", index=False)
    json.dump({"wavelengths": [float(c) for c in cols], "missing": missing, "renamed": how}, open(out / "meta.json", "w"))
    print(f"Selesai: {out/'data_f16.npy'}, manifest.csv, masks.npy, meta.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rar")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--no_destripe", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="proses maksimal N file (uji)")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--use_audit", action="store_true", help="pasangkan kubus-ID berdasarkan audit_ids.csv (sidik spektral)")
    ap.add_argument("--alias", nargs="*", help="pasangan ID_CSV=nama_file, mis. Egg-F-02Dec-0072=Egg-IF-02Dec-0072")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "eggs").mkdir(parents=True, exist_ok=True)
    if a.pack:
        return pack(out, a.csv, a.alias, a.use_audit)

    items = list_rar(a.rar)
    todo = [it for it in items if not (out / "eggs" / f"{egg_id(it['path'])}.npy").exists()]
    if a.limit:
        todo = todo[:a.limit]
    print(f"Arsip: {len(items)} file .mat terbaca, sudah diproses: {len(items)-len(todo) if not a.limit else '-'}, "
          f"akan diproses: {len(todo)}", flush=True)
    _, _, csv_spec = load_csv_spectra(a.csv)
    log_path = out / "prepare_log.jsonl"
    ok = fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex, open(log_path, "a") as lg:
        futs = [ex.submit(process_one, a.rar, it, out / "eggs", a.size, csv_spec, not a.no_destripe) for it in todo]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as e:
                r = {"id": "?", "ok": False, "err": f"{type(e).__name__}: {e}"}
            lg.write(json.dumps(r) + "\n")
            lg.flush()
            ok += r["ok"]
            fail += not r["ok"]
            if k % 20 == 0 or k == len(futs):
                el = time.time() - t0
                print(f"[{time.strftime('%H:%M:%S')}] {k}/{len(futs)} ok={ok} gagal={fail} "
                      f"({el/k:.1f}s/telur efektif, sisa ~{(len(futs)-k)*el/k/60:.0f} mnt) "
                      f"contoh r_csv={r.get('r_csv', float('nan')):.4f}", flush=True)
    print(f"Selesai: ok={ok}, gagal={fail}. Log: {log_path}. Jalankan ulang untuk mencoba yang gagal.")


if __name__ == "__main__":
    main()
