#!/usr/bin/env python3
"""
Analisis eksperimen tambahan untuk Neural Computing and Applications.

  A. Pembanding backbone (front end all16 sama) + pembanding khusus HSI (HybridSN 3D-2D)
  B. Ablasi lebar proyeksi (k = 4, 8, 16, 32, 64): YOLO11s vs kontrol spektral
  C. Model usulan: YOLO11s ber-anchor PLS-DA (residual learning), 5 seed + kontrol anchor non-spasial
  D. Fusi seed tunggal (biaya deployment realistis)
  E. Kalibrasi (Brier, ECE) dan titik operasi penyortiran
  F. Uji temporal maju (latih Nov-18 Des, uji 19 Des-Feb)

Pemakaian (CPU, beberapa menit):
  python nca_analysis.py --runs runs --runs_temporal runs_temporal \
      --baseline_oof results_baseline/oof_scores.csv \
      --csv /raid/home/tegar/egg_hsi/raw/spectra_reference.csv --out results_nca
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
import baseline as B  # noqa: E402
from final_analysis import fmt_paired, load_oof, paired  # noqa: E402

L = []


def say(m=""):
    print(m, flush=True)
    L.append(m)


def seeds_of(runs, base):
    """Semua run untuk satu model: base (seed 0), base_s1.., atau base_s0.."""
    out = {}
    for d in sorted(Path(runs).glob(base + "*")):
        suf = d.name[len(base):]
        if suf == "" or (suf.startswith("_s") and suf[2:].isdigit()):
            s = load_oof(d)
            if s is not None:
                out[d.name] = s
    return out


def ens(S, ids):
    return pd.concat([s.loc[ids] for s in S.values()], axis=1).mean(1).to_numpy()


def line(name, y, s, ref=None, refname=""):
    m = B.metrics(y, s)
    lo, hi = B.boot_ci(y, s)
    t = f"{name:34s} bal_acc={m['bal_acc']:.3f} [{lo:.3f},{hi:.3f}] AUC={m['auc']:.3f} MCC={m['mcc']:.3f}"
    if ref is not None:
        p, n01, n10 = B.mcnemar(y, (s > .5).astype(int), (ref > .5).astype(int))
        t += f" | McNemar vs {refname} p={p:.3g} ({n01}/{n10})"
    say(t)
    return {"model": name, "bal_acc": m["bal_acc"], "lo": lo, "hi": hi, "auc": m["auc"], "mcc": m["mcc"]}


def ece(y, s, nb=10):
    bins = np.clip((s * nb).astype(int), 0, nb - 1)
    return sum(abs(s[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(nb) if (bins == b).any())


def op_point(y, s, tpr_target):
    """Fertil = positif. Pada ambang yang mempertahankan >= tpr_target telur fertil, berapa % infertil tersortir keluar?"""
    fpr, tpr, thr = roc_curve(y, s)
    i = np.argmax(tpr >= tpr_target)
    return 1 - fpr[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--runs_temporal", default="runs_temporal")
    ap.add_argument("--baseline_oof", default="results_baseline/oof_scores.csv")
    ap.add_argument("--baseline_key", default="B|SNV|PLS-DA")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cut", default="18Dec")
    ap.add_argument("--out", default="results_nca")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    b = pd.read_csv(a.baseline_oof).set_index("Sample ID")
    R = Path(a.runs)
    yolo = seeds_of(R, "yolo11s_all16")
    spec = seeds_of(R, "specpool_all16")
    anc = seeds_of(R, "pls_anchor_yolo11s_all16")
    ancs = seeds_of(R, "pls_anchor_specpool_all16")
    ids = sorted(set(b.index).intersection(*[set(s.index) for s in list(yolo.values()) + list(spec.values())]))
    y = b.loc[ids, "y"].to_numpy()
    pls = np.clip(b.loc[ids, a.baseline_key].to_numpy(), 0, 1)
    Y = {"yolo": ens(yolo, ids), "spec": ens(spec, ids)}
    y0 = yolo["yolo11s_all16"].loc[ids].to_numpy()
    say(f"# Eksperimen tambahan (NCA)\n\nn={len(ids)}; seed YOLO={len(yolo)}, kontrol={len(spec)}, "
        f"anchor={len(anc)}, anchor-kontrol={len(ancs)}")

    # ---------------------------------------------------------------- A
    say("\n## A. Backbone dengan front end all16 yang sama (seed 0) + HybridSN 3D-2D")
    rows = [line("YOLO11s (all16)", y, y0)]
    for t, nm in [("yolo11n_all16", "YOLO11n"), ("yolo11m_all16", "YOLO11m"), ("resnet18_all16", "ResNet-18"),
                  ("mobilenet_v3_small_all16", "MobileNetV3-small"), ("efficientnet_b0_all16", "EfficientNet-B0"),
                  ("hybridsn_bin75", "HybridSN 3D-2D (75 bin)")]:
        s = load_oof(R / t)
        if s is None:
            say(f"{nm}: belum ada")
            continue
        rows.append(line(nm, y, s.loc[ids].to_numpy(), y0, "YOLO11s"))
    pd.DataFrame(rows).to_csv(out / "A_backbones.csv", index=False)

    # ---------------------------------------------------------------- B
    say("\n## B. Ablasi lebar proyeksi k (seed 0): spasial (YOLO11s) vs non-spasial (kontrol)")
    rows = []
    for k in (4, 8, 16, 32, 64):
        sy, ss = load_oof(R / f"yolo11s_all{k}"), load_oof(R / f"specpool_all{k}")
        if sy is None or ss is None:
            say(f"k={k}: belum lengkap")
            continue
        vy, vs = sy.loc[ids].to_numpy(), ss.loc[ids].to_numpy()
        o = paired(y, vy, vs)
        my, ms = B.metrics(y, vy), B.metrics(y, vs)
        say(f"k={k:2d}  YOLO11s {my['bal_acc']:.3f}/{my['auc']:.3f}  kontrol {ms['bal_acc']:.3f}/{ms['auc']:.3f}  | "
            + fmt_paired(o))
        rows.append({"k": k, "yolo_bal": my["bal_acc"], "yolo_auc": my["auc"], "spec_bal": ms["bal_acc"],
                     "spec_auc": ms["auc"], "dAUC": o["dAUC"][0], "dAUC_lo": o["dAUC"][1], "dAUC_hi": o["dAUC"][2],
                     "dBal": o["dBalAcc"][0], "dBal_lo": o["dBalAcc"][1], "dBal_hi": o["dBalAcc"][2],
                     "mcnemar_p": o["mcnemar"][0]})
    pd.DataFrame(rows).to_csv(out / "B_width.csv", index=False)

    # ---------------------------------------------------------------- C
    say("\n## C. Model usulan: YOLO11s ber-anchor PLS-DA (residual), 5 seed")
    fus = (pls + Y["yolo"]) / 2
    if anc:
        per = np.array([[bacc(y, s.loc[ids] > .5), roc_auc_score(y, s.loc[ids])] for s in anc.values()])
        say(f"per seed (n={len(anc)}): bal_acc {per[:,0].mean():.3f} ± {per[:,0].std(ddof=1) if len(per)>1 else 0:.3f}"
            f" | AUC {per[:,1].mean():.3f} ± {per[:,1].std(ddof=1) if len(per)>1 else 0:.3f}")
        Y["anchor"] = ens(anc, ids)
        line("Anchor-YOLO11s (ensemble)", y, Y["anchor"])
        for nm, ref in [("PLS-DA", pls), ("YOLO11s ensemble", Y["yolo"]), ("fusi rata-rata", fus)]:
            say(f"  vs {nm}: " + fmt_paired(paired(y, Y["anchor"], ref)))
    if ancs:
        Y["anchor_spec"] = ens(ancs, ids)
        line(f"Anchor-kontrol ({len(ancs)} seed)", y, Y["anchor_spec"])
        say("  vs PLS-DA: " + fmt_paired(paired(y, Y["anchor_spec"], pls)))
        if "anchor" in Y:
            say("  Anchor-YOLO vs Anchor-kontrol (nilai tambah spasial di atas PLS): "
                + fmt_paired(paired(y, Y["anchor"], Y["anchor_spec"])))

    # ---------------------------------------------------------------- D
    say("\n## D. Fusi dengan SATU model YOLO11s (tanpa ensemble)")
    fs = np.array([[bacc(y, (pls + s.loc[ids].to_numpy()) / 2 > .5), roc_auc_score(y, (pls + s.loc[ids].to_numpy()) / 2)]
                   for s in yolo.values()])
    say(f"fusi PLS + 1 seed (n={len(fs)}): bal_acc {fs[:,0].mean():.3f} ± {fs[:,0].std(ddof=1):.3f} "
        f"(min {fs[:,0].min():.3f}) | AUC {fs[:,1].mean():.3f} ± {fs[:,1].std(ddof=1):.3f}")
    if anc:
        say(f"anchor 1 seed: bal_acc {per[:,0].mean():.3f} ± {per[:,0].std(ddof=1):.3f} (min {per[:,0].min():.3f})")

    # ---------------------------------------------------------------- E
    say("\n## E. Kalibrasi dan titik operasi (fertil = positif; % infertil tersortir saat X% fertil dipertahankan)")
    rows = []
    M = {"PLS-DA": pls, "YOLO11s ens": Y["yolo"], "Kontrol ens": Y["spec"], "Fusi": fus}
    if "anchor" in Y:
        M["Anchor-YOLO11s ens"] = Y["anchor"]
    for nm, s in M.items():
        r = {"model": nm, "brier": float(np.mean((s - y) ** 2)), "ece": float(ece(y, s)),
             "spec_at_tpr95": op_point(y, s, .95), "spec_at_tpr99": op_point(y, s, .99)}
        rows.append(r)
        say(f"{nm:22s} Brier={r['brier']:.3f} ECE={r['ece']:.3f} | infertil tersortir @fertil95%={r['spec_at_tpr95']:.3f}"
            f" @fertil99%={r['spec_at_tpr99']:.3f}")
    pd.DataFrame(rows).to_csv(out / "E_calibration.csv", index=False)

    # ---------------------------------------------------------------- F
    say(f"\n## F. Uji temporal maju (latih s.d. {a.cut}, uji sesudahnya)")
    RT = Path(a.runs_temporal)
    ty, ts, ta = seeds_of(RT, "yolo11s_all16"), seeds_of(RT, "specpool_all16"), seeds_of(RT, "pls_anchor_yolo11s_all16")
    if not ty:
        say("belum ada hasil temporal")
    else:
        df, X, _, _ = B.load(a.csv)
        mon = {"Nov": 11, "Dec": 12, "Jan": 13, "Feb": 14}
        key = lambda s: (mon[s[2:]], int(s[:2]))  # noqa: E731
        g = df["date"].to_numpy()
        yy = df["y"].to_numpy()
        is_tr = np.array([key(d) <= key(a.cut) for d in g])
        tr, te = np.where(is_tr)[0], np.where(~is_tr)[0]
        _, sp, bp, _ = B.run_fold(X, yy, g, tr, te, "SNV", "PLS-DA", quick=False)
        P = pd.Series(np.clip(sp, 0, 1), index=df["Sample ID"].iloc[te].str.upper())
        tid = sorted(set(P.index) & set.intersection(*[set(s.index.str.upper()) for s in ty.values()]))
        up = lambda S: {k: pd.Series(v.to_numpy(), index=v.index.str.upper()) for k, v in S.items()}  # noqa: E731
        ty, ts, ta = up(ty), up(ts), up(ta)
        yt = pd.Series(yy[te], index=P.index).loc[tid].to_numpy()
        say(f"latih {len(tr)} telur / {len(set(g[tr]))} tanggal; uji {len(tid)} telur / {len(set(g[te]))} tanggal; PLS {bp}")
        pt = P.loc[tid].to_numpy()
        T = {"PLS-DA": pt, f"YOLO11s ({len(ty)} seed)": ens(ty, tid)}
        if ts:
            T[f"Kontrol ({len(ts)} seed)"] = ens(ts, tid)
        T["Fusi PLS+YOLO11s"] = (pt + T[f"YOLO11s ({len(ty)} seed)"]) / 2
        if ta:
            T[f"Anchor-YOLO11s ({len(ta)} seed)"] = ens(ta, tid)
        rows = [line(nm, yt, s, pt if nm != "PLS-DA" else None, "PLS-DA") for nm, s in T.items()]
        pd.DataFrame(rows).to_csv(out / "F_temporal.csv", index=False)
        say("  Fusi vs PLS: " + fmt_paired(paired(yt, T["Fusi PLS+YOLO11s"], pt)))
        if ta:
            say("  Anchor vs PLS: " + fmt_paired(paired(yt, T[f"Anchor-YOLO11s ({len(ta)} seed)"], pt)))
        if ts:
            say("  YOLO vs kontrol: " + fmt_paired(paired(yt, T[f"YOLO11s ({len(ty)} seed)"], T[f"Kontrol ({len(ts)} seed)"])))

    pd.DataFrame({"y": y, "pls": pls, **Y}, index=ids).to_csv(out / "oof_nca.csv")
    (out / "summary.md").write_text("\n".join(L))
    say(f"\nTersimpan di {out}/")


if __name__ == "__main__":
    main()
