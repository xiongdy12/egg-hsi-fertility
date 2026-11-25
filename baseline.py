#!/usr/bin/env python3
"""
Tahap 2a — Baseline spektral dari CSV (spektrum rata-rata per telur), dataset UIUC IDB-8141497.

Protokol utama: Leave-One-Date-Out (LODO) CV. Telur dikelompokkan per tanggal koleksi
(ID 'Egg-F-03Dec-0096' -> grup '03Dec'), sehingga model diuji pada hari yang tidak pernah dilihat.
Semua preprocessing yang 'fit' (MSC, scaler) dan SELEKSI BAND dilakukan di dalam fold latih saja.

Isi:
  A. Reproduksi protokol paper asli (split acak 70/15/15, PLS-DA turunan-2) -> pembanding 90,3%
  B. LODO CV: PLS-DA, LogReg, SVM-RBF, HistGradientBoosting, RF, (opsional) 1D-CNN
     x preprocessing {raw, SNV, MSC, SG1, SG2}
  C. Seleksi band di dalam fold: VIP-PLS, SPA, Fisher; k in {3,5,8,10} + RGB-simulasi (CIE 1931 -> sRGB)
  D. Uji confound: apakah morfometri (massa, diameter, tebal & kekuatan cangkang) saja bisa menebak kelas?
  E. Statistik: bootstrap CI 95%, McNemar exact vs baseline PLS-DA, Wilcoxon per-tanggal

Pemakaian:
  python baseline.py --csv spectra_reference.csv --out results_baseline --n_jobs 64
  python baseline.py --csv ... --quick          # uji cepat (grid kecil)
  python baseline.py --csv ... --cnn            # tambahkan 1D-CNN (butuh torch, pakai GPU jika ada)
"""
import argparse
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from scipy.stats import binomtest, wilcoxon
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ============================================================================ data

MORPH_COLS = ["Mass (g)", "Major dia (mm)", "Minor dia (mm)", "Thickness (mm)", "Shell strength (N)"]


def load(csv):
    df = pd.read_csv(csv)
    df["Sample ID"] = df["Sample ID"].astype(str).str.strip().str.strip("'\"")
    m = df["Sample ID"].str.extract(r"Egg-([A-Za-z]+)-(\d{2}[A-Za-z]{3})-(\d+)")
    df["cls_code"], df["date"], df["num"] = m[0], m[1], m[2].astype(int)
    df["y"] = (df["cls_code"].str.upper() == "F").astype(int)  # 1 = fertil
    spec_cols = [c for c in df.columns if re.fullmatch(r"\s*\d+(\.\d+)?\s*", str(c))]
    wl = np.array([float(c) for c in spec_cols])
    X = df[spec_cols].to_numpy(float)
    # cek konsistensi label ID vs kolom 'Fertility status'
    chk = {}
    if "Fertility status" in df.columns:
        chk = pd.crosstab(df["cls_code"], df["Fertility status"].astype(str)).to_dict()
    return df, X, wl, chk

# ============================================================================ preprocessing


class SNV(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)


class MSC(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.ref_ = np.asarray(X, float).mean(0)
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        out = np.empty_like(X)
        for i, x in enumerate(X):
            b, a = np.polyfit(self.ref_, x, 1)
            out[i] = (x - a) / (b + 1e-12)
        return out


class SG(BaseEstimator, TransformerMixin):
    def __init__(self, deriv=1, window=15, poly=2):
        self.deriv, self.window, self.poly = deriv, window, poly

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return savgol_filter(np.asarray(X, float), self.window, self.poly, deriv=self.deriv, axis=1)


class Identity(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, float)


PREPROC = {
    "raw": lambda: Identity(),
    "SNV": lambda: SNV(),
    "MSC": lambda: MSC(),
    "SG1": lambda: SG(1),
    "SG2": lambda: SG(2),
}

# ============================================================================ model


class PLSDA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=10):
        self.n_components = n_components

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        nc = int(min(self.n_components, X.shape[1], X.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=nc, scale=True).fit(X, y.astype(float))
        return self

    def decision_function(self, X):
        return self.pls_.predict(X).ravel()

    def predict_proba(self, X):
        s = np.clip(self.decision_function(X), 0, 1)
        return np.c_[1 - s, s]

    def predict(self, X):
        return (self.decision_function(X) > 0.5).astype(int)


def model_grid(name, quick, n_feat):
    if name == "PLS-DA":
        comps = [2, 5, 10, 15, 20, 27, 30] if not quick else [5, 15]
        comps = sorted({c for c in comps if c < n_feat} | {min(n_feat, 2)})
        return Pipeline([("m", PLSDA())]), {"m__n_components": comps}
    if name == "LogReg":
        return (Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=5000))]),
                {"m__C": [1e-3, 1e-2, 1e-1, 1, 10] if not quick else [1e-2, 1]})
    if name == "SVM-RBF":
        return (Pipeline([("s", StandardScaler()), ("m", SVC(kernel="rbf", probability=True))]),
                {"m__C": [0.1, 1, 10, 100] if not quick else [1, 10],
                 "m__gamma": ["scale", 1e-3, 1e-2] if not quick else ["scale"]})
    if name == "HGB":
        return (Pipeline([("m", HistGradientBoostingClassifier(random_state=0))]),
                {"m__learning_rate": [0.05, 0.1] if not quick else [0.1],
                 "m__max_depth": [3, None] if not quick else [3]})
    if name == "RF":
        return (Pipeline([("m", RandomForestClassifier(n_estimators=500 if not quick else 100,
                                                        random_state=0, n_jobs=1))]),
                {"m__max_features": ["sqrt", 0.1] if not quick else ["sqrt"]})
    raise ValueError(name)


def scores_of(est, X):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)

# ============================================================================ seleksi band


def vip_scores(X, y, n_comp=10):
    pls = PLSRegression(n_components=min(n_comp, X.shape[1]), scale=True).fit(X, y.astype(float))
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p = w.shape[0]
    ss = np.diag(t.T @ t * (q.T @ q)) if q.ndim == 2 else np.diag(t.T @ t) * q.ravel() ** 2
    wn = w / np.linalg.norm(w, axis=0, keepdims=True)
    return np.sqrt(p * (wn ** 2 @ ss) / ss.sum())


def fisher_scores(X, y):
    a, b = X[y == 1], X[y == 0]
    return (a.mean(0) - b.mean(0)) ** 2 / (a.var(0) + b.var(0) + 1e-12)


def spa(X, k, start):
    """Successive Projections Algorithm: rantai k band dengan kolinearitas minimum."""
    Xc = X - X.mean(0)
    sel = [start]
    P = Xc.copy()
    for _ in range(k - 1):
        xj = P[:, sel[-1]][:, None]
        P = P - xj @ (xj.T @ P) / (xj.T @ xj + 1e-12)
        norms = np.linalg.norm(P, axis=0)
        norms[sel] = -1
        sel.append(int(np.argmax(norms)))
    return sel


def min_sep(idx_sorted, scores, k, gap):
    """Ambil k band teratas dengan jarak minimal 'gap' indeks (hindari band bertetangga yang redundan)."""
    chosen = []
    for j in idx_sorted:
        if all(abs(j - c) >= gap for c in chosen):
            chosen.append(int(j))
        if len(chosen) == k:
            break
    return chosen


def select_bands(method, Xtr, ytr, k):
    if method == "VIP":
        s = vip_scores(Xtr, ytr)
        return min_sep(np.argsort(-s), s, k, gap=5)
    if method == "Fisher":
        s = fisher_scores(Xtr, ytr)
        return min_sep(np.argsort(-s), s, k, gap=5)
    if method == "SPA":
        return spa(Xtr, k, start=int(np.argmax(fisher_scores(Xtr, ytr))))
    raise ValueError(method)


def cie_rgb_weights(wl):
    """Simulasi kamera RGB: CIE 1931 (aproksimasi analitik Wyman dkk. 2013) -> XYZ -> sRGB linear."""
    def g(x, mu, s1, s2):
        s = np.where(x < mu, s1, s2)
        return np.exp(-0.5 * ((x - mu) / s) ** 2)
    xb = 1.056 * g(wl, 599.8, 37.9, 31.0) + 0.362 * g(wl, 442.0, 16.0, 26.7) - 0.065 * g(wl, 501.1, 20.4, 26.2)
    yb = 0.821 * g(wl, 568.8, 46.9, 40.5) + 0.286 * g(wl, 530.9, 16.3, 31.1)
    zb = 1.217 * g(wl, 437.0, 11.8, 36.0) + 0.681 * g(wl, 459.0, 26.0, 13.8)
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    W = M @ np.vstack([xb, yb, zb])  # 3 x B
    return W.T  # B x 3

# ============================================================================ evaluasi


def metrics(y, s, thr=0.5):
    p = (s > thr).astype(int)
    out = {"acc": accuracy_score(y, p), "bal_acc": balanced_accuracy_score(y, p),
           "f1": f1_score(y, p), "mcc": matthews_corrcoef(y, p),
           "sens": ((p == 1) & (y == 1)).sum() / max((y == 1).sum(), 1),
           "spec": ((p == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)}
    try:
        out["auc"] = roc_auc_score(y, s)
    except ValueError:
        out["auc"] = np.nan
    return out


def boot_ci(y, s, key="bal_acc", n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(metrics(y[i], s[i])[key])
    return np.percentile(vals, [2.5, 97.5])


def mcnemar(y, pa, pb):
    a_ok, b_ok = pa == y, pb == y
    n01, n10 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    if n01 + n10 == 0:
        return 1.0, n01, n10
    return binomtest(n01, n01 + n10, 0.5).pvalue, n01, n10


def run_fold(Xall, y, groups, tr, te, prep_name, model_name, quick, band_fn=None, inner_splits=5):
    """Satu fold luar: (opsional) pilih band dari data latih -> preprocessing -> tuning inner GroupKFold -> prediksi."""
    Xtr, Xte = Xall[tr], Xall[te]
    prep = PREPROC[prep_name]().fit(Xtr, y[tr])
    Ptr, Pte = prep.transform(Xtr), prep.transform(Xte)
    bands = None
    if band_fn is not None:
        bands = band_fn(Ptr, y[tr])
        if isinstance(bands, np.ndarray) and bands.ndim == 2:  # matriks bobot (RGB-sim)
            Ptr, Pte = Ptr @ bands, Pte @ bands
        else:
            Ptr, Pte = Ptr[:, bands], Pte[:, bands]
    est, grid = model_grid(model_name, quick, Ptr.shape[1])
    n_inner = min(inner_splits, len(np.unique(groups[tr])))
    gs = GridSearchCV(est, grid, cv=GroupKFold(n_inner), scoring="balanced_accuracy", n_jobs=1, refit=True)
    gs.fit(Ptr, y[tr], groups=groups[tr])
    return te, scores_of(gs.best_estimator_, Pte), gs.best_params_, bands


def lodo(Xall, y, groups, prep, model, quick, n_jobs, band_fn=None):
    dates = np.unique(groups)
    folds = [(np.where(groups != d)[0], np.where(groups == d)[0]) for d in dates]
    res = Parallel(n_jobs=n_jobs)(
        delayed(run_fold)(Xall, y, groups, tr, te, prep, model, quick, band_fn) for tr, te in folds)
    s = np.full(len(y), np.nan)
    params, bands = [], []
    for te, sc, bp, b in res:
        s[te] = sc
        params.append(bp)
        bands.append(b)
    per_date = {d: accuracy_score(y[groups == d], (s[groups == d] > 0.5).astype(int)) for d in dates}
    return s, params, bands, per_date

# ============================================================================ 1D-CNN (opsional)


def cnn_lodo(Xall, y, groups, prep_name, epochs, seed=0):
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    def make(nb):
        return nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))

    s = np.full(len(y), np.nan)
    for d in np.unique(groups):
        tr, te = groups != d, groups == d
        prep = PREPROC[prep_name]().fit(Xall[tr])
        sc = StandardScaler().fit(prep.transform(Xall[tr]))
        f = lambda X: torch.tensor(sc.transform(prep.transform(X)), dtype=torch.float32, device=dev)[:, None]
        Xtr, ytr = f(Xall[tr]), torch.tensor(y[tr], dtype=torch.float32, device=dev)
        net = make(Xall.shape[1]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
        lossf = nn.BCEWithLogitsLoss()
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(Xtr), device=dev)
            for i in range(0, len(perm), 64):
                b = perm[i:i + 64]
                opt.zero_grad()
                lossf(net(Xtr[b]).ravel(), ytr[b]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            s[te] = torch.sigmoid(net(f(Xall[te])).ravel()).cpu().numpy()
    return s

# ============================================================================ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results_baseline")
    ap.add_argument("--n_jobs", type=int, default=32)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cnn", action="store_true")
    ap.add_argument("--cnn_epochs", type=int, default=150)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T0 = time.time()
    log = []

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    df, X, wl, chk = load(a.csv)
    y, groups = df["y"].to_numpy(), df["date"].to_numpy()
    say(f"# Baseline spektral (CSV)\n\nData: {len(y)} telur, {X.shape[1]} band ({wl.min():.1f}-{wl.max():.1f} nm), "
        f"fertil={int(y.sum())}, infertil={int((1 - y).sum())}, tanggal={len(np.unique(groups))}")
    say(f"Cek label ID vs kolom 'Fertility status': {chk}")
    tab = pd.crosstab(df["date"], df["y"]).rename(columns={0: "IF", 1: "F"})
    tab.to_csv(out / "date_class_table.csv")
    one_cls = tab[(tab == 0).any(axis=1)].index.tolist()
    say(f"Tanggal dengan satu kelas saja: {one_cls}")

    rows, oof = [], {}

    # ------------------------------------------------------------- A. reproduksi protokol paper
    say("\n## A. Reproduksi protokol paper (split acak stratified 70/15/15, PLS-DA SG2), 10 ulangan")
    rep = []
    for r in range(10):
        idx = np.arange(len(y))
        tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=r)
        va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=r)
        prep = SG(2).fit(X[tr])
        Ptr, Pva, Pte = (prep.transform(X[i]) for i in (tr, va, te))
        best, bestc = -1, None
        for c in ([5, 10, 15, 20, 27, 30] if not a.quick else [10, 27]):
            m = PLSDA(c).fit(Ptr, y[tr])
            acc = accuracy_score(y[va], m.predict(Pva))
            if acc > best:
                best, bestc = acc, c
        m = PLSDA(bestc).fit(np.r_[Ptr, Pva], np.r_[y[tr], y[va]])
        rep.append(accuracy_score(y[te], m.predict(Pte)))
    say(f"Akurasi test acak: {np.mean(rep):.3f} ± {np.std(rep):.3f} (paper: 0.903)")
    rows.append({"block": "A_random_split", "prep": "SG2", "model": "PLS-DA", "acc": np.mean(rep),
                 "acc_std": np.std(rep)})

    # ------------------------------------------------------------- B. LODO, spektrum penuh
    say("\n## B. LODO CV, spektrum penuh")
    models = ["PLS-DA", "LogReg", "SVM-RBF", "HGB", "RF"]
    preps = ["raw", "SNV", "MSC", "SG1", "SG2"] if not a.quick else ["SNV", "SG2"]
    if a.quick:
        models = ["PLS-DA", "LogReg", "SVM-RBF"]
    for pr in preps:
        for mdl in models:
            t0 = time.time()
            s, params, _, per_date = lodo(X, y, groups, pr, mdl, a.quick, a.n_jobs)
            mt = metrics(y, s)
            lo, hi = boot_ci(y, s)
            key = f"B|{pr}|{mdl}"
            oof[key] = (s, per_date)
            rows.append({"block": "B_full", "prep": pr, "model": mdl, **mt, "bal_acc_lo": lo, "bal_acc_hi": hi,
                         "per_date_acc_mean": np.mean(list(per_date.values())),
                         "per_date_acc_std": np.std(list(per_date.values())),
                         "params": json.dumps(params[:3], default=str), "sec": time.time() - t0})
            say(f"{pr:4s} {mdl:8s} bal_acc={mt['bal_acc']:.3f} [{lo:.3f},{hi:.3f}] auc={mt['auc']:.3f} "
                f"({time.time()-t0:.0f}s)")

    if a.cnn:
        for pr in (["SNV", "SG1"] if not a.quick else ["SNV"]):
            t0 = time.time()
            s = cnn_lodo(X, y, groups, pr, a.cnn_epochs if not a.quick else 10)
            per_date = {d: accuracy_score(y[groups == d], (s[groups == d] > 0.5).astype(int)) for d in np.unique(groups)}
            mt = metrics(y, s)
            lo, hi = boot_ci(y, s)
            oof[f"B|{pr}|1D-CNN"] = (s, per_date)
            rows.append({"block": "B_full", "prep": pr, "model": "1D-CNN", **mt, "bal_acc_lo": lo,
                         "bal_acc_hi": hi, "per_date_acc_mean": np.mean(list(per_date.values())),
                         "per_date_acc_std": np.std(list(per_date.values())), "sec": time.time() - t0})
            say(f"{pr:4s} 1D-CNN   bal_acc={mt['bal_acc']:.3f} [{lo:.3f},{hi:.3f}] auc={mt['auc']:.3f}")

    # ------------------------------------------------------------- C. seleksi band (k kecil = sensor murah)
    say("\n## C. Seleksi band di dalam fold (k band) dan RGB-simulasi")
    full_B = pd.DataFrame([r for r in rows if r["block"] == "B_full"])
    best_prep = full_B.sort_values("bal_acc", ascending=False).iloc[0]["prep"]
    say(f"Preprocessing terbaik dari blok B: {best_prep} (dipakai untuk seleksi band)")
    sel_models = ["LogReg", "SVM-RBF"] if not a.quick else ["LogReg"]
    ks = [3, 5, 8, 10] if not a.quick else [3, 5]
    band_log = []
    for meth in ["VIP", "SPA", "Fisher"]:
        for k in ks:
            fn = (lambda Xt, yt, meth=meth, k=k: select_bands(meth, Xt, yt, k))
            for mdl in sel_models:
                s, _, bands, per_date = lodo(X, y, groups, best_prep, mdl, a.quick, a.n_jobs, band_fn=fn)
                mt = metrics(y, s)
                lo, hi = boot_ci(y, s)
                oof[f"C|{meth}{k}|{mdl}"] = (s, per_date)
                flat = [wl[j] for b in bands for j in b]
                freq = pd.Series(np.round(flat, 0)).value_counts().head(k)
                band_log.append({"method": meth, "k": k, "model": mdl,
                                 "most_frequent_nm": freq.index.tolist(),
                                 "stability_top_k_share": float(freq.sum() / len(flat))})
                rows.append({"block": "C_bands", "prep": best_prep, "model": mdl, "bands": f"{meth}-{k}",
                             **mt, "bal_acc_lo": lo, "bal_acc_hi": hi,
                             "per_date_acc_mean": np.mean(list(per_date.values()))})
                say(f"{meth:6s} k={k:2d} {mdl:8s} bal_acc={mt['bal_acc']:.3f} [{lo:.3f},{hi:.3f}] "
                    f"band~{sorted(freq.index.tolist())}")
    # RGB-simulasi: pakai spektrum MENTAH (kamera melihat intensitas, bukan turunan)
    W = cie_rgb_weights(wl)
    for mdl in sel_models:
        s, _, _, per_date = lodo(X, y, groups, "raw", mdl, a.quick, a.n_jobs, band_fn=lambda Xt, yt: W)
        mt = metrics(y, s)
        lo, hi = boot_ci(y, s)
        oof[f"C|RGBsim|{mdl}"] = (s, per_date)
        rows.append({"block": "C_bands", "prep": "raw", "model": mdl, "bands": "RGB-sim(CIE)", **mt,
                     "bal_acc_lo": lo, "bal_acc_hi": hi, "per_date_acc_mean": np.mean(list(per_date.values()))})
        say(f"RGB-sim     {mdl:8s} bal_acc={mt['bal_acc']:.3f} [{lo:.3f},{hi:.3f}]")
    pd.DataFrame(band_log).to_csv(out / "band_selection.csv", index=False)

    # ------------------------------------------------------------- D. uji confound morfometri
    say("\n## D. Uji confound: morfometri saja (tanpa spektrum)")
    mcols = [c for c in MORPH_COLS if c in df.columns]
    M = df[mcols].to_numpy(float)
    ok = ~np.isnan(M).any(1)
    from scipy.stats import mannwhitneyu
    for c in mcols:
        v = df[c].to_numpy(float)
        g1, g0 = v[(y == 1) & ~np.isnan(v)], v[(y == 0) & ~np.isnan(v)]
        p = mannwhitneyu(g1, g0).pvalue
        say(f"{c:20s} F={g1.mean():.2f}±{g1.std():.2f}  IF={g0.mean():.2f}±{g0.std():.2f}  Mann-Whitney p={p:.2e}")
    for mdl in ["LogReg", "HGB"]:
        s, _, _, _ = lodo(M[ok], y[ok], groups[ok], "raw", mdl, a.quick, a.n_jobs)
        mt = metrics(y[ok], s)
        rows.append({"block": "D_confound", "prep": "raw", "model": mdl, "bands": "morfometri", **mt})
        say(f"Morfometri {mdl:8s} bal_acc={mt['bal_acc']:.3f} auc={mt['auc']:.3f} "
            f"(~0.5 = tidak ada confound; tinggi = perbedaan flok terdeteksi tanpa spektrum)")

    # ------------------------------------------------------------- E. uji statistik vs baseline
    say("\n## E. Uji statistik vs baseline PLS-DA SG2 (protokol paper, di bawah LODO)")
    base_key = "B|SG2|PLS-DA"
    if base_key in oof:
        sb, pdb = oof[base_key]
        pb = (sb > 0.5).astype(int)
        ranked = sorted(oof.items(), key=lambda kv: -balanced_accuracy_score(y, (kv[1][0] > 0.5).astype(int)))
        for key, (s, pdd) in ranked[:8]:
            if key == base_key:
                continue
            p_mc, n01, n10 = mcnemar(y, (s > 0.5).astype(int), pb)
            common = sorted(set(pdd) & set(pdb))
            diff = np.array([pdd[d] - pdb[d] for d in common])
            p_w = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            say(f"{key:28s} McNemar p={p_mc:.3g} (benar-salah {n01}/{n10}); Wilcoxon per-tanggal p={p_w:.3g}")

    # ------------------------------------------------------------- simpan
    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics_all.csv", index=False)
    oof_df = pd.DataFrame({"Sample ID": df["Sample ID"], "date": groups, "y": y})
    for k, (s, _) in oof.items():
        oof_df[k] = s
    oof_df.to_csv(out / "oof_scores.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        Z = SNV().transform(X)
        for lab, nm in [(1, "fertil"), (0, "infertil")]:
            mu, sd = Z[y == lab].mean(0), Z[y == lab].std(0)
            ax[0].plot(wl, mu, label=nm)
            ax[0].fill_between(wl, mu - sd, mu + sd, alpha=.2)
        ax[0].set_title("Spektrum rata-rata (SNV)")
        ax[0].set_xlabel("nm")
        ax[0].legend()
        ax[1].plot(wl, fisher_scores(PREPROC[best_prep]().fit(X).transform(X), y))
        ax[1].set_title(f"Fisher score per band ({best_prep}, seluruh data — hanya ilustrasi)")
        ax[1].set_xlabel("nm")
        fig.tight_layout()
        fig.savefig(out / "spectra_fisher.png", dpi=130)
    except Exception as e:
        say(f"(plot gagal: {e})")
    say(f"\nSelesai dalam {(time.time()-T0)/60:.1f} menit. Hasil: {out}/")
    top = res[res.block.isin(["B_full", "C_bands"])].sort_values("bal_acc", ascending=False).head(10)
    say("\n## 10 teratas (LODO)\n" + top[["block", "prep", "model", "bands", "bal_acc", "bal_acc_lo",
                                           "bal_acc_hi", "auc", "mcc"]].to_string(index=False))
    (out / "summary.md").write_text("\n".join(log))


if __name__ == "__main__":
    main()
