#!/usr/bin/env python3
"""
Tahap 2b — Analisis confound flok (label = kandang ber-pejantan vs tanpa pejantan).

Pertanyaan: apakah sinyal spektral benar-benar 'fertilitas', atau hanya perbedaan flok
yang juga terlihat di morfometri (massa, diameter, tebal & kekuatan cangkang)?

Semua dengan LODO CV (tanggal = grup), preprocessing SNV, model PLS-DA dan LogReg (tuning inner GroupKFold):
  1. Spektrum saja
  2. Morfometri saja
  3. Spektrum + morfometri  -> nilai tambah spektrum di atas morfometri (bootstrap berpasangan selisih AUC)
  4. Spektrum yang sudah 'dibersihkan' dari morfometri (residualisasi linier, di-fit di fold latih saja)
  5. Subset matched: tiap tanggal, pasangan F-IF 1:1 nearest-neighbour pada morfometri terstandar
     (caliper). Kalau matching berhasil, morfometri-saja ~0.5; spektrum tetap tinggi = sinyal bukan dari morfometri.

Pemakaian:
  python confound.py --csv spectra_reference.csv --out results_confound --n_jobs 25
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import baseline as B  # noqa: E402

MORPH = ["Mass (g)", "Major dia (mm)", "Minor dia (mm)", "Thickness (mm)", "Shell strength (N)"]


def fold(Xs, M, y, g, tr, te, mode, model, quick):
    """mode: spec | morph | both | resid"""
    snv = B.SNV()
    S_tr, S_te = snv.transform(Xs[tr]), snv.transform(Xs[te])
    ms = StandardScaler().fit(M[tr])
    M_tr, M_te = ms.transform(M[tr]), ms.transform(M[te])
    if mode == "spec":
        A_tr, A_te = S_tr, S_te
    elif mode == "morph":
        A_tr, A_te = M_tr, M_te
    elif mode == "both":
        ss = StandardScaler().fit(S_tr)
        w = np.sqrt(S_tr.shape[1] / M_tr.shape[1])  # beri bobot setara blok spektrum vs morfometri
        A_tr = np.c_[ss.transform(S_tr), w * M_tr]
        A_te = np.c_[ss.transform(S_te), w * M_te]
    elif mode == "resid":
        lr = LinearRegression().fit(M_tr, S_tr)
        A_tr, A_te = S_tr - lr.predict(M_tr), S_te - lr.predict(M_te)
    else:
        raise ValueError(mode)
    est, grid = B.model_grid(model, quick, A_tr.shape[1])
    n_inner = min(5, len(np.unique(g[tr])))
    gs = GridSearchCV(est, grid, cv=GroupKFold(n_inner), scoring="balanced_accuracy", n_jobs=1)
    gs.fit(A_tr, y[tr], groups=g[tr])
    return te, B.scores_of(gs.best_estimator_, A_te)


def lodo(Xs, M, y, g, mode, model, quick, n_jobs):
    folds = [(np.where(g != d)[0], np.where(g == d)[0]) for d in np.unique(g)]
    res = Parallel(n_jobs=n_jobs)(delayed(fold)(Xs, M, y, g, tr, te, mode, model, quick) for tr, te in folds)
    s = np.full(len(y), np.nan)
    for te, sc in res:
        s[te] = sc
    return s


def paired_auc_diff(y, s1, s2, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    d = np.array(d)
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5]), float((d <= 0).mean())


def smd(M, y):
    a, b = M[y == 1], M[y == 0]
    return (a.mean(0) - b.mean(0)) / np.sqrt((a.var(0) + b.var(0)) / 2 + 1e-12)


def match_within_date(Mz, y, g, caliper):
    """Greedy 1:1 nearest-neighbour F<->IF per tanggal, jarak Euclid pada morfometri terstandar."""
    keep = []
    for d in np.unique(g):
        f = np.where((g == d) & (y == 1))[0]
        i = np.where((g == d) & (y == 0))[0]
        if len(f) == 0 or len(i) == 0:
            continue
        D = np.linalg.norm(Mz[f][:, None, :] - Mz[i][None, :, :], axis=2)
        used_f, used_i = set(), set()
        for flat in np.argsort(D, axis=None):
            a, b = np.unravel_index(flat, D.shape)
            if D[a, b] > caliper:
                break
            if a in used_f or b in used_i:
                continue
            used_f.add(a)
            used_i.add(b)
            keep += [f[a], i[b]]
    return np.array(sorted(keep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results_confound")
    ap.add_argument("--n_jobs", type=int, default=25)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--calipers", type=float, nargs="+", default=[1.0, 1.5],
                    help="jarak Euclid maks pada 5 morfometri terstandar (SD)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def say(m):
        print(m, flush=True)
        log.append(m)

    T0 = time.time()
    df, X, wl, _ = B.load(a.csv)
    mc = [c for c in MORPH if c in df.columns]
    ok = ~df[mc].isna().any(axis=1).to_numpy()
    df, X = df[ok].reset_index(drop=True), X[ok]
    M = df[mc].to_numpy(float)
    y, g = df["y"].to_numpy(), df["date"].to_numpy()
    say(f"# Analisis confound flok\n\nn={len(y)} (baris tanpa morfometri dibuang: {int((~ok).sum())}), "
        f"morfometri: {mc}")

    rows, S = [], {}
    models = ["PLS-DA", "LogReg"]
    say("\n## 1-4. LODO: spektrum vs morfometri vs gabungan vs spektrum-residual")
    for model in models:
        for mode in ["spec", "morph", "both", "resid"]:
            s = lodo(X, M, y, g, mode, model, a.quick, a.n_jobs)
            S[(model, mode)] = s
            mt = B.metrics(y, s)
            lo, hi = B.boot_ci(y, s)
            rows.append({"set": "full", "model": model, "mode": mode, **mt, "bal_acc_lo": lo, "bal_acc_hi": hi})
            say(f"{model:7s} {mode:6s} bal_acc={mt['bal_acc']:.3f} [{lo:.3f},{hi:.3f}] auc={mt['auc']:.3f}")
        d, ci, p = paired_auc_diff(y, S[(model, "both")], S[(model, "morph")])
        say(f"  -> nilai tambah spektrum di atas morfometri ({model}): dAUC={d:+.3f} "
            f"[{ci[0]:+.3f},{ci[1]:+.3f}], P(dAUC<=0)={p:.4f}")
        d2, ci2, p2 = paired_auc_diff(y, S[(model, "resid")], S[(model, "spec")])
        say(f"  -> efek residualisasi ({model}): dAUC(resid-spec)={d2:+.3f} [{ci2[0]:+.3f},{ci2[1]:+.3f}]")

    say("\n## 5. Subset matched per tanggal (morfometri seimbang)")
    Mz = (M - M.mean(0)) / M.std(0)
    for cal in sorted(a.calipers):
        idx = match_within_date(Mz, y, g, cal)
        ym, gm = y[idx], g[idx]
        say(f"\nCaliper {cal} SD: {len(idx)} telur ({int(ym.sum())} F / {int((1-ym).sum())} IF), "
            f"{len(np.unique(gm))} tanggal")
        before, after = smd(M, y), smd(M[idx], ym)
        for c, b0, b1 in zip(mc, before, after):
            say(f"  SMD {c:20s} sebelum={b0:+.2f}  sesudah={b1:+.2f}")
        for model in models:
            for mode in ["spec", "morph"]:
                s = lodo(X[idx], M[idx], ym, gm, mode, model, a.quick, a.n_jobs)
                mt = B.metrics(ym, s)
                lo, hi = B.boot_ci(ym, s)
                rows.append({"set": f"matched_cal{cal}", "model": model, "mode": mode, **mt,
                             "bal_acc_lo": lo, "bal_acc_hi": hi, "n": len(idx)})
                say(f"  {model:7s} {mode:6s} bal_acc={mt['bal_acc']:.3f} [{lo:.3f},{hi:.3f}] auc={mt['auc']:.3f}")

    pd.DataFrame(rows).to_csv(out / "confound_metrics.csv", index=False)
    say(f"\nSelesai dalam {(time.time()-T0)/60:.1f} menit.")
    say("\nCara membaca: kalau (a) 'both' > 'morph' secara signifikan, (b) 'resid' tidak jauh di bawah 'spec', "
        "dan (c) di subset matched 'morph' ~0.5 sementara 'spec' tetap tinggi -> sinyal spektral tidak "
        "dijelaskan oleh morfometri flok.")
    (out / "summary.md").write_text("\n".join(log))


if __name__ == "__main__":
    main()
