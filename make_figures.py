#!/usr/bin/env python3
"""
Gambar paper (PDF vektor + PNG 300 dpi), lebar kolom Springer: 1 kolom 84 mm, 2 kolom 174 mm.

  fig_roc.pdf             ROC: PLS-DA (CSV), YOLOv11s (5 seed), kontrol spektral (5 seed), fusi PLS+YOLO
  fig_per_date.pdf        Akurasi per tanggal (LODO): PLS-DA, YOLOv11s, fusi
  fig_bands.pdf           (a) spektrum rata-rata SNV per kelas, (b) skor Fisher, (c) frekuensi band concrete, (d) pusat LED
  fig_summary.pdf         bal_acc + CI 95% semua model utama (split acak vs LODO; HSI penuh vs k-band vs RGB)
  fig_confound.pdf        spektrum vs morfometri: data penuh vs subset matched
  fig_efficiency.pdf      GFLOPs vs bal_acc (jika results_latency/latency.csv ada)

Pemakaian (dari ~/egg_hsi):
  python make_figures.py --csv /raid/home/tegar/egg_hsi/raw/spectra_reference.csv --out figures
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import baseline as B  # noqa: E402

# Palet kategori tervalidasi (urutan tetap): biru, oranye, aqua, kuning
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MM = 1 / 25.4
W1, W2 = 84 * MM, 174 * MM

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.5, "legend.frameon": False,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.pdf/.png")


def panel(ax, letter):
    ax.text(-0.13, 1.04, f"({letter})", transform=ax.transAxes, fontweight="bold", va="bottom")


def load_runs(runs):
    out = {}
    for d in sorted(Path(runs).iterdir()):
        fs = sorted(d.glob("fold_*.json"))
        if fs and (d / "config.json").exists():
            out[d.name] = {"cfg": json.load(open(d / "config.json")), "folds": [json.load(open(f)) for f in fs]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--baseline_dir", default="results_baseline")
    ap.add_argument("--cube_dir", default="results_cube")
    ap.add_argument("--final_dir", default="results_final")
    ap.add_argument("--confound_dir", default="results_confound")
    ap.add_argument("--latency_dir", default="results_latency")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    bo = pd.read_csv(Path(a.baseline_dir) / "oof_scores.csv").set_index("Sample ID")
    fin = pd.read_csv(Path(a.final_dir) / "seedavg_oof_scores.csv", index_col=0)
    cube = pd.read_csv(Path(a.cube_dir) / "cube_oof_scores.csv", index_col=0)
    ids = fin.index
    y = fin["y"].to_numpy()
    S = {
        "PLS-DA (mean spectrum)": fin["pls"].to_numpy(),
        "Spectral control, 300 bands": fin["specpool_all16"].to_numpy(),
        "YOLOv11s, 300 bands": fin["yolo11s_all16"].to_numpy(),
        "Fusion: PLS-DA + YOLOv11s": (fin["pls"] + fin["yolo11s_all16"]).to_numpy() / 2,
    }
    order = ["PLS-DA (mean spectrum)", "YOLOv11s, 300 bands", "Fusion: PLS-DA + YOLOv11s",
             "Spectral control, 300 bands"]
    style = {order[0]: (C[0], "-"), order[1]: (C[1], "--"), order[2]: (C[2], "-"), order[3]: (C[3], ":")}
    print("Membuat gambar:")

    # ---------------------------------------------------------------- ROC
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.95))
    for name in order:
        fpr, tpr, _ = roc_curve(y, S[name])
        auc = B.metrics(y, S[name])["auc"]
        col, ls = style[name]
        ax.plot(fpr, tpr, color=col, ls=ls, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=(0, (1, 2)))
    ax.set_xlabel("False positive rate (infertile called fertile)")
    ax.set_ylabel("True positive rate (fertile detected)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", handlelength=2.2)
    save(fig, out, "fig_roc")

    # ---------------------------------------------------------------- per tanggal
    dates = bo.loc[ids, "date"].to_numpy()
    mon = {"Nov": 11, "Dec": 12, "Jan": 13, "Feb": 14}
    ud = sorted(np.unique(dates), key=lambda d: (mon[d[2:]], int(d[:2])))
    per = {}
    for name in ["PLS-DA (mean spectrum)", "YOLOv11s, 300 bands", "Fusion: PLS-DA + YOLOv11s"]:
        per[name] = [((S[name][dates == d] > .5).astype(int) == y[dates == d]).mean() for d in ud]
    nper = [int((dates == d).sum()) for d in ud]
    fig, ax = plt.subplots(figsize=(W2, W2 * 0.33))
    xs = np.arange(len(ud))
    mk = {"PLS-DA (mean spectrum)": "o", "YOLOv11s, 300 bands": "s", "Fusion: PLS-DA + YOLOv11s": "^"}
    off = {"PLS-DA (mean spectrum)": -0.22, "YOLOv11s, 300 bands": 0.0, "Fusion: PLS-DA + YOLOv11s": 0.22}
    for name, v in per.items():
        col = style[name][0]
        ax.scatter(xs + off[name], v, s=16, marker=mk[name], color=col, edgecolor="white", linewidth=0.6,
                   zorder=3, label=f"{name} (mean {np.mean(v):.3f})")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d}\n(n={n})" for d, n in zip(ud, nper)], rotation=0, fontsize=5.6)
    ax.set_ylabel("Accuracy on held-out date")
    ax.set_ylim(min(0.4, min(min(v) for v in per.values()) - 0.05), 1.02)
    ax.set_xlim(-0.6, len(ud) - 0.4)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, borderaxespad=0)
    save(fig, out, "fig_per_date")

    # ---------------------------------------------------------------- band
    df = pd.read_csv(a.csv)
    cols = [c for c in df.columns if re.fullmatch(r"\s*\d+(\.\d+)?\s*", str(c))]
    wl = np.array([float(c) for c in cols])
    X = df[cols].to_numpy(float)
    Z = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    yy = df["Sample ID"].astype(str).str.contains(r"Egg-F-", case=False, regex=True).to_numpy().astype(int)
    fis = B.fisher_scores(Z, yy)
    runs = load_runs(a.runs)
    conc, led = [], []
    for tag, r in runs.items():
        sel = r["cfg"].get("selector")
        for f in r["folds"]:
            if f.get("selected_nm"):
                (conc if sel == "concrete" else led if sel == "led" else []).extend(f["selected_nm"])
    hl = [(510, 530), (665, 680)]
    fig, axs = plt.subplots(2, 2, figsize=(W2, W2 * 0.55), sharex=True)
    ax = axs[0, 0]
    for lab, nm, col in [(1, "Fertile", C[0]), (0, "Infertile", C[1])]:
        mu, sd = Z[yy == lab].mean(0), Z[yy == lab].std(0)
        ax.plot(wl, mu, color=col, label=f"{nm} (n={int((yy == lab).sum())})")
        ax.fill_between(wl, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
    ax.set_ylabel("SNV transmittance")
    ax.legend(loc="upper left")
    panel(ax, "a")
    ax = axs[0, 1]
    ax.plot(wl, fis, color=C[0])
    ax.set_ylabel("Fisher score (SNV)")
    panel(ax, "b")
    bins = np.arange(370, 1020, 10)
    ax = axs[1, 0]
    if conc:
        ax.hist(conc, bins=bins, color=C[0], edgecolor="white", linewidth=0.6)
    ax.set_ylabel("Times selected\n(concrete, all folds)")
    ax.set_xlabel("Wavelength (nm)")
    panel(ax, "c")
    ax = axs[1, 1]
    if led:
        ax.hist(led, bins=bins, color=C[0], edgecolor="white", linewidth=0.6)
    ax.set_ylabel("Learned LED centres\n(all folds)")
    ax.set_xlabel("Wavelength (nm)")
    panel(ax, "d")
    for ax in axs.ravel():
        for lo, hi in hl:
            ax.axvspan(lo, hi, color=GRID, alpha=0.8, lw=0, zorder=0)
    axs[0, 1].text(520, axs[0, 1].get_ylim()[1] * 0.95, "~520", ha="center", va="top", color=INK2, fontsize=6.5)
    axs[0, 1].text(672, axs[0, 1].get_ylim()[1] * 0.95, "~675", ha="center", va="top", color=INK2, fontsize=6.5)
    save(fig, out, "fig_bands")

    # ---------------------------------------------------------------- ringkasan bal_acc
    rows = []
    ma = pd.read_csv(Path(a.baseline_dir) / "metrics_all.csv")
    A = ma[ma.block == "A_random_split"]
    if len(A):
        m, sd = float(A.acc.iloc[0]), float(A.acc_std.iloc[0])
        rows.append(("PLS-DA, random split (10×)", m, m - 1.96 * sd, m + 1.96 * sd, "Protocol"))

    def add(label, s, grp):
        s = np.asarray(s, float)
        ok = ~np.isnan(s)
        mt = B.metrics(y[ok], s[ok])
        lo, hi = B.boot_ci(y[ok], s[ok])
        rows.append((label, mt["bal_acc"], lo, hi, grp))
    add("PLS-DA, leave-one-date-out", S["PLS-DA (mean spectrum)"], "Protocol")
    add("Spectral control (no spatial)", S["Spectral control, 300 bands"], "300 bands (LODO)")
    add("YOLOv11s (5 seeds)", S["YOLOv11s, 300 bands"], "300 bands (LODO)")
    add("Fusion PLS-DA + YOLOv11s", S["Fusion: PLS-DA + YOLOv11s"], "300 bands (LODO)")
    kb = [("yolo11s_led_k8", "YOLOv11s, 8 learned LEDs"), ("yolo11s_concrete_k8", "YOLOv11s, 8 selected bands"),
          ("yolo11s_led_k5", "YOLOv11s, 5 learned LEDs"), ("yolo11s_concrete_k5", "YOLOv11s, 5 selected bands"),
          ("yolo11s_led_k3", "YOLOv11s, 3 learned LEDs"), ("yolo11s_concrete_k3", "YOLOv11s, 3 selected bands"),
          ("yolo11s_rgb", "YOLOv11s, simulated RGB camera")]
    for tag, lab in kb:
        if tag in cube.columns:
            add(lab, cube.reindex(ids)[tag].to_numpy(), "Band-limited sensor (LODO, 1 seed)")
    R = pd.DataFrame(rows, columns=["label", "bal", "lo", "hi", "grp"])
    R.to_csv(out / "fig_summary_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(W2 * 0.62, 0.17 * len(R) + 0.5))
    yp = np.arange(len(R))[::-1]
    ax.errorbar(R.bal, yp, xerr=[R.bal - R.lo, R.hi - R.bal], fmt="o", ms=4, color=C[0], ecolor=C[0],
                elinewidth=1.2, capsize=0, zorder=3)
    for yi, (_, r) in zip(yp, R.iterrows()):
        ax.text(r.hi + 0.006, yi, f"{r.bal:.3f}", va="center", fontsize=6.3, color=INK2)
    ax.set_yticks(yp)
    ax.set_yticklabels(R.label)
    for g in R.grp.unique()[1:]:
        i = R.index[R.grp == g][0]
        ax.axhline(yp[i] + 0.5, color=INK2, lw=0.6)
    ax.axvline(R.bal[R.label == "PLS-DA, leave-one-date-out"].iloc[0], color=INK2, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.set_xlabel("Balanced accuracy (95% CI)")
    ax.set_xlim(max(0.5, R.lo.min() - 0.03), min(1.0, R.hi.max() + 0.04))
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig_summary")

    # ---------------------------------------------------------------- confound
    cf = Path(a.confound_dir) / "confound_metrics.csv"
    if cf.exists():
        c = pd.read_csv(cf)
        c = c[(c.model == "PLS-DA") & c["mode"].isin(["spec", "morph"])]
        sets = list(dict.fromkeys(c["set"]))
        lab = {"full": "All eggs", **{s: f"Matched ({s.replace('matched_cal', 'caliper ')} SD)" for s in sets if s != "full"}}
        lab = {k: v.replace(" (", "\n(") for k, v in lab.items()}
        fig, ax = plt.subplots(figsize=(W1, W1 * 0.72))
        xs = np.arange(len(sets))
        for j, (mode, nm, mk) in enumerate([("spec", "Spectrum (SNV)", "o"), ("morph", "Morphometry only", "s")]):
            v = [c[(c["set"] == s) & (c["mode"] == mode)] for s in sets]
            val = np.array([float(t.bal_acc.iloc[0]) for t in v])
            lo = np.array([float(t.bal_acc_lo.iloc[0]) for t in v])
            hi = np.array([float(t.bal_acc_hi.iloc[0]) for t in v])
            xx = xs + (j - 0.5) * 0.28
            ax.errorbar(xx, val, yerr=[val - lo, hi - val], fmt=mk, ms=4.5, color=C[j], ecolor=C[j],
                        elinewidth=1.2, capsize=0, label=nm, zorder=3)
            for xi, vi, h in zip(xx, val, hi):
                ax.text(xi, h + 0.012, f"{vi:.2f}", ha="center", va="bottom", fontsize=6.2, color=INK2)
        ax.axhline(0.5, color=INK2, lw=0.8, ls=(0, (2, 2)))
        ax.text(-0.45, 0.505, "chance", ha="left", va="bottom", fontsize=6, color=INK2)
        ax.set_xticks(xs)
        ax.set_xticklabels([lab[s] for s in sets], fontsize=6.5)
        ax.set_xlim(-0.5, len(sets) - 0.5)
        ax.set_ylabel("Balanced accuracy (PLS-DA, LODO)")
        ax.set_ylim(0.4, 1.02)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, borderaxespad=0)
        ax.grid(axis="x", visible=False)
        save(fig, out, "fig_confound")

    # ---------------------------------------------------------------- efisiensi
    lf = Path(a.latency_dir) / "latency.csv"
    cm = Path(a.cube_dir) / "cube_metrics.csv"
    if lf.exists() and cm.exists():
        L = pd.read_csv(lf).set_index("tag")
        M = pd.read_csv(cm).set_index("tag")
        common = [t for t in M.index if t in L.index and not t.endswith(("_imgnorm", "_randinit"))]
        fig, ax = plt.subplots(figsize=(W1, W1 * 0.8))
        gf = L.loc[common, "GFLOPs_sensor"].clip(lower=1e-4)
        acc = M.loc[common, "bal_acc"]
        fam = {t: ("Spectral control" if t.startswith("specpool") else
                   "YOLOv11" if t.startswith("yolo11") else "Other lightweight CNN") for t in common}
        for j, (f, mk) in enumerate([("YOLOv11", "o"), ("Other lightweight CNN", "s"), ("Spectral control", "D")]):
            tt = [t for t in common if fam[t] == f]
            if tt:
                ax.scatter(gf[tt], acc[tt], s=18, marker=mk, color=C[j], edgecolor="white", linewidth=0.6,
                           zorder=3, label=f)
        best, pareto = -1, []
        for t in sorted(common, key=lambda t: gf[t]):
            if acc[t] > best:
                best = acc[t]
                pareto.append(t)
        nice = lambda t: (t.replace("yolo11", "YOLOv11").replace("specpool", "spectral").replace("_concrete_k", ", k=")
                          .replace("_led_k", ", LED k=").replace("_all16", ", 300 bands").replace("_rgb", ", RGB")
                          .replace("_vip5", ", VIP k=5").replace("mobilenet_v3_small", "MobileNetV3-S")
                          .replace("efficientnet_b0", "EfficientNet-B0").replace("resnet18", "ResNet18"))
        for t in pareto:
            ax.annotate(nice(t), (gf[t], acc[t]), xytext=(4, -9), textcoords="offset points", fontsize=5.6,
                        color=INK2)
        ax.plot([gf[t] for t in pareto], [acc[t] for t in pareto], color=INK2, lw=0.7, ls=(0, (1, 1.5)), zorder=2)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, borderaxespad=0, fontsize=6.2)
        pls = B.metrics(y, S["PLS-DA (mean spectrum)"])["bal_acc"]
        ax.axhline(pls, color=INK2, lw=0.8, ls=(0, (2, 2)))
        ax.text(gf.max() * 1.6, pls + 0.004, "PLS-DA (mean spectrum)", fontsize=6, color=INK2, va="bottom", ha="right")
        ax.set_xscale("log")
        ax.set_xlabel("GFLOPs per egg (sensor-level input)")
        ax.set_ylabel("Balanced accuracy (LODO)")
        save(fig, out, "fig_efficiency")
    print(f"Selesai: {out}/")


if __name__ == "__main__":
    main()
