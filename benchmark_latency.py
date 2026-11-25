#!/usr/bin/env python3
"""
Benchmark efisiensi dengan SATU protokol untuk semua model (menjawab kritik latency yang tidak konsisten).

Protokol (dilaporkan apa adanya di paper):
  - PyTorch eager, FP32, batch 1, input acak berukuran sama dengan input nyata.
  - CPU: torch.set_num_threads(T) untuk T in {1, 4}; 50 pemanasan + 300 pengukuran, laporkan median & IQR (ms).
  - GPU (opsional): CUDA sinkron, 50 pemanasan + 300 pengukuran.
  - Dua tingkat:
      (a) "sensor"     : input k kanal langsung (kamera k-band) -> backbone saja.
      (b) "HSI-penuh"  : input kubus 300 band -> selector + backbone (yang dipakai di eksperimen).
  - Parameter (juta) dan FLOPs (G, torch.utils.flop_counter; 1 MAC = 2 FLOPs).
  - PLS-DA (CSV) diukur dengan protokol yang sama (SNV + proyeksi linear 300 band).

Pemakaian:
  python benchmark_latency.py --configs configs.jsonl --res 128 --out results_latency [--gpu]
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import train_cube as T


def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def flops_of(fn, *inp):
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        fn(*inp)
    return fc.get_total_flops() / 1e9


def time_fn(fn, inp, warm=50, reps=300, cuda=False):
    with torch.no_grad():
        for _ in range(warm):
            fn(*inp)
        if cuda:
            torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(*inp)
            if cuda:
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    return float(np.median(ts)), float(np.percentile(ts, 25)), float(np.percentile(ts, 75))


class SensorModel(nn.Module):
    """Backbone dengan input k kanal yang sudah 'ditangkap sensor' (tanpa selector 300 band)."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, m):
        x = self.net.normalize(x, m) if self.net.cfg["selector"] not in ("all", "bin") else x
        return self.net.backbone(x, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="configs.jsonl")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--bands", type=int, default=300)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default="results_latency")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wl = np.linspace(374.14, 1015.32, a.bands)
    cfgs = [json.loads(l) for l in open(a.configs) if l.strip()]
    # satu baris per arsitektur unik (selector+backbone+k), varian ablasi (imgnorm/randinit) sama secara komputasi
    seen, uniq = set(), []
    for c in cfgs:
        key = (c["backbone"], c["selector"], c.get("k"), tuple(c.get("bands") or []))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    torch.manual_seed(0)
    rows = []
    cube = torch.rand(1, a.bands, a.res, a.res)
    mask = torch.ones(1, a.res, a.res, dtype=torch.bool)
    for c in uniq:
        net = T.Net(c, wl).eval()
        k = net.sel.k
        sensor = SensorModel(net).eval()
        xk = torch.rand(1, k, a.res, a.res)
        sensor_in = (cube if c["selector"] in ("all", "bin") else xk, mask)
        if c["selector"] in ("all", "bin"):  # 'all' butuh semua band: tingkat sensor = HSI penuh
            sensor_fn = lambda x, m, net=net: net(x, m)
        else:
            sensor_fn = sensor
        r = {"tag": c["tag"], "backbone": c["backbone"], "selector": c["selector"], "k": k,
             "params_M": count_params(net.backbone) + count_params(net.sel),
             "GFLOPs_sensor": flops_of(sensor_fn, *sensor_in),
             "GFLOPs_hsi": flops_of(net, cube, mask)}
        for th in a.threads:
            torch.set_num_threads(th)
            r[f"cpu{th}_sensor_ms"], r[f"cpu{th}_sensor_q1"], r[f"cpu{th}_sensor_q3"] = time_fn(
                sensor_fn, sensor_in, reps=a.reps)
            r[f"cpu{th}_hsi_ms"], r[f"cpu{th}_hsi_q1"], r[f"cpu{th}_hsi_q3"] = time_fn(net, (cube, mask), reps=a.reps)
        if a.gpu and torch.cuda.is_available():
            g = net.cuda()
            gi = (cube.cuda(), mask.cuda())
            r["gpu_hsi_ms"], r["gpu_hsi_q1"], r["gpu_hsi_q3"] = time_fn(g, gi, reps=a.reps, cuda=True)
            net.cpu()
        rows.append(r)
        print(f"{c['tag']:32s} k={k:3d} params={r['params_M']:.2f}M GFLOPs sensor={r['GFLOPs_sensor']:.3f} "
              f"hsi={r['GFLOPs_hsi']:.3f} | CPU1 sensor {r['cpu1_sensor_ms']:.2f} ms, hsi {r['cpu1_hsi_ms']:.2f} ms",
              flush=True)

    # PLS-DA: SNV + w·x + b pada spektrum 300 band (spektrum rata-rata dari mask)
    w, b = torch.randn(a.bands), torch.randn(1)

    def plsda(cube_, mask_):
        s = cube_[0][:, mask_[0]].mean(1)                 # spektrum rata-rata dalam mask
        s = (s - s.mean()) / (s.std() + 1e-9)             # SNV
        return s @ w + b
    r = {"tag": "PLS-DA (SNV, spektrum rata-rata)", "backbone": "linear", "selector": "-", "k": a.bands,
         "params_M": (a.bands + 1) / 1e6,
         # analitis: rata-rata dalam mask (B*P), SNV (~4B), proyeksi linear (2B)
         "GFLOPs_sensor": (a.bands * a.res * a.res + 6 * a.bands) / 1e9,
         "GFLOPs_hsi": (a.bands * a.res * a.res + 6 * a.bands) / 1e9}
    for th in a.threads:
        torch.set_num_threads(th)
        ms = time_fn(plsda, (cube, mask), reps=a.reps)
        r[f"cpu{th}_sensor_ms"], r[f"cpu{th}_sensor_q1"], r[f"cpu{th}_sensor_q3"] = ms
        r[f"cpu{th}_hsi_ms"], r[f"cpu{th}_hsi_q1"], r[f"cpu{th}_hsi_q3"] = ms
    rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(out / "latency.csv", index=False)
    cpu = platform.processor() or platform.machine()
    try:
        cpu = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:
        pass
    meta = {"cpu": cpu, "torch": torch.__version__, "res": a.res, "bands": a.bands, "threads": a.threads,
            "reps": a.reps, "warmup": 50, "batch": 1, "dtype": "fp32", "mode": "eager",
            "gpu": torch.cuda.get_device_name(0) if (a.gpu and torch.cuda.is_available()) else None}
    json.dump(meta, open(out / "protocol.json", "w"), indent=2)
    cols = ["tag", "k", "params_M", "GFLOPs_sensor", "GFLOPs_hsi"] + \
           [f"cpu{t}_{lv}_ms" for t in a.threads for lv in ("sensor", "hsi")] + (["gpu_hsi_ms"] if a.gpu else [])
    md = [f"# Benchmark efisiensi\n\nProtokol: {json.dumps(meta)}\n",
          df[[c for c in cols if c in df.columns]].to_string(index=False, float_format=lambda v: f"{v:.3f}")]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
