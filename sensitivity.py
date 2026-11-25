#!/usr/bin/env python3
"""
Analisis sensitivitas: apakah hasil utama bertahan tanpa telur bertanda (tulisan/coretan tinta, retak, citra rusak)?

  A. Evaluasi ulang prediksi yang SUDAH ADA (dilatih dengan semua telur) hanya pada telur bersih.
  B. Latih ulang PLS-DA SNV (CSV) dengan LODO tanpa telur bertanda.
  C. Jika runs_clean/ ada (YOLOv11s & kontrol spektral dilatih ulang tanpa telur bertanda, beberapa seed):
     rata-rata seed, spasial vs non-spasial, dan fusi dengan PLS-DA bersih (uji berpasangan).

Pemakaian:
  python sensitivity.py --csv .../spectra_reference.csv --exclude results_ink/audit_visual.csv \
      --runs_clean runs_clean --out results_sensitivity
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import baseline as B  # noqa: E402
from final_analysis import fmt_paired, load_oof, paired  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--exclude", required=True)
    ap.add_argument("--baseline_oof", default="results_baseline/oof_scores.csv")
    ap.add_argument("--seedavg", default="results_final/seedavg_oof_scores.csv")
    ap.add_argument("--runs_clean", default="runs_clean")
    ap.add_argument("--models", nargs="+", default=["yolo11s_all16", "specpool_all16"])
    ap.add_argument("--n_jobs", type=int, default=25)
    ap.add_argument("--out", default="results_sensitivity")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    L = []

    def say(m=""):
        print(m, flush=True)
        L.append(m)

    ex = set(pd.read_csv(a.exclude)["Sample ID"].astype(str))
    say(f"# Analisis sensitivitas tanpa telur bertanda\n\nTelur dikeluarkan: {len(ex)}")

    # ---------------------------------------------------------------- A
    say("\n## A. Prediksi lama (dilatih dengan semua telur), dievaluasi pada semua vs bersih")
    sa = pd.read_csv(a.seedavg, index_col=0)
    sa["fusion"] = (sa["pls"] + sa["yolo11s_all16"]) / 2
    clean = ~sa.index.isin(ex)
    for col, nm in [("pls", "PLS-DA SNV"), ("specpool_all16", "Kontrol spektral (5 seed)"),
                    ("yolo11s_all16", "YOLOv11s (5 seed)"), ("fusion", "Fusi PLS + YOLO")]:
        r = []
        for sel, lab in [(np.ones(len(sa), bool), "semua"), (clean, "bersih"), (~clean, "bertanda")]:
            y, s = sa.loc[sel, "y"].to_numpy(), sa.loc[sel, col].to_numpy()
            r.append(f"{lab} n={sel.sum()}: bal_acc={bacc(y, s > .5):.3f}" +
                     (f" AUC={roc_auc_score(y, s):.3f}" if len(np.unique(y)) > 1 else ""))
        say(f"{nm:26s} | " + " | ".join(r))
    yc = sa.loc[clean, "y"].to_numpy()
    say("Pada telur bersih: fusi vs PLS -> " + fmt_paired(paired(yc, sa.loc[clean, "fusion"].to_numpy(),
                                                                  sa.loc[clean, "pls"].to_numpy())))
    say("Pada telur bersih: YOLO vs kontrol spektral -> " + fmt_paired(paired(
        yc, sa.loc[clean, "yolo11s_all16"].to_numpy(), sa.loc[clean, "specpool_all16"].to_numpy())))

    # ---------------------------------------------------------------- B
    say("\n## B. PLS-DA SNV dilatih ulang tanpa telur bertanda (LODO)")
    df, X, wl, _ = B.load(a.csv)
    keep = ~df["Sample ID"].isin(ex).to_numpy()
    dfk, Xk = df[keep].reset_index(drop=True), X[keep]
    y, g = dfk["y"].to_numpy(), dfk["date"].to_numpy()
    s, _, _, _ = B.lodo(Xk, y, g, "SNV", "PLS-DA", False, a.n_jobs)
    mt = B.metrics(y, s)
    lo, hi = B.boot_ci(y, s)
    say(f"PLS-DA SNV bersih: n={len(y)} bal_acc={mt['bal_acc']:.3f} [{lo:.3f},{hi:.3f}] AUC={mt['auc']:.3f}")
    pls_clean = pd.Series(np.clip(s, 0, 1), index=dfk["Sample ID"])
    pd.DataFrame({"y": y, "pls_clean": pls_clean.to_numpy()}, index=dfk["Sample ID"]).to_csv(out / "pls_clean_oof.csv")

    # ---------------------------------------------------------------- C
    rc = Path(a.runs_clean)
    if rc.exists():
        say("\n## C. Model kubus dilatih ulang tanpa telur bertanda")
        avg = {}
        for mname in a.models:
            tags = sorted(t.name for t in rc.iterdir() if t.name.startswith(mname))
            S = {t: load_oof(rc / t) for t in tags}
            S = {t: v for t, v in S.items() if v is not None and len(v) > 0}
            if not S:
                continue
            ids = sorted(set.intersection(*[set(v.index) for v in S.values()]) & set(pls_clean.index))
            yy = dfk.set_index("Sample ID").loc[ids, "y"].to_numpy()
            per = [(bacc(yy, v.loc[ids] > .5), roc_auc_score(yy, v.loc[ids])) for v in S.values()]
            ba = np.array([p[0] for p in per])
            say(f"{mname}: {len(S)} seed, n={len(ids)}, bal_acc per seed {ba.mean():.3f} ± "
                f"{ba.std(ddof=1) if len(ba) > 1 else 0:.3f}")
            avg[mname] = pd.concat([v.loc[ids] for v in S.values()], axis=1).mean(1)
            m = B.metrics(yy, avg[mname].to_numpy())
            lo, hi = B.boot_ci(yy, avg[mname].to_numpy())
            say(f"  rata-rata seed: bal_acc={m['bal_acc']:.3f} [{lo:.3f},{hi:.3f}] AUC={m['auc']:.3f}")
        if len(avg) == 2:
            ids = sorted(set(avg[a.models[0]].index) & set(avg[a.models[1]].index))
            yy = dfk.set_index("Sample ID").loc[ids, "y"].to_numpy()
            p0 = pls_clean.loc[ids].to_numpy()
            m1, m2 = (avg[m].loc[ids].to_numpy() for m in a.models)
            say("Spasial vs non-spasial: " + fmt_paired(paired(yy, m1, m2)))
            for nm, mm in [(a.models[0], m1), (a.models[1], m2)]:
                f = (p0 + mm) / 2
                m = B.metrics(yy, f)
                lo, hi = B.boot_ci(yy, f)
                say(f"PLS bersih + {nm}: bal_acc={m['bal_acc']:.3f} [{lo:.3f},{hi:.3f}] AUC={m['auc']:.3f}")
                say("  vs PLS bersih: " + fmt_paired(paired(yy, f, p0)))
    else:
        say(f"\n(C dilewati: {rc} belum ada)")
    (out / "summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
