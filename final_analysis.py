#!/usr/bin/env python3
"""
Analisis akhir: variasi antar-seed, rata-rata seed, fusi dengan PLS-DA (CSV), dan uji berpasangan.

Pertanyaan yang dijawab:
  Q1 Stabilitas: bal_acc/AUC tiap seed (mean ± SD) untuk YOLOv11s-all16 dan kontrol specpool-all16.
  Q2 Spasial vs non-spasial: rata-rata seed YOLO vs rata-rata seed specpool (selisih AUC berpasangan).
  Q3 Komplementer: PLS-DA + YOLO(rata-rata seed) vs PLS-DA saja; dan kontrol PLS-DA + specpool.
Semua bobot fusi = rata-rata sederhana, ditetapkan a priori (tidak dituning).

Pemakaian:
  python final_analysis.py --runs runs --baseline_oof results_baseline/oof_scores.csv --out results_final
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


def load_oof(run_dir):
    recs = [json.load(open(f)) for f in sorted(Path(run_dir).glob("fold_*.json"))]
    if not recs:
        return None
    return pd.Series(sum((r["score"] for r in recs), []), index=sum((r["ids"] for r in recs), []))


def paired(y, s_new, s_ref, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    dA, dB = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        dA.append(roc_auc_score(y[i], s_new[i]) - roc_auc_score(y[i], s_ref[i]))
        dB.append(bacc(y[i], s_new[i] > .5) - bacc(y[i], s_ref[i] > .5))
    out = {}
    for k, d in (("dAUC", np.array(dA)), ("dBalAcc", np.array(dB))):
        out[k] = (d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), float((d <= 0).mean()))
    p, n01, n10 = B.mcnemar(y, (s_new > .5).astype(int), (s_ref > .5).astype(int))
    out["mcnemar"] = (p, n01, n10)
    return out


def fmt_paired(o):
    a, b = o["dAUC"], o["dBalAcc"]
    return (f"dAUC={a[0]:+.4f} [{a[1]:+.4f},{a[2]:+.4f}] P(<=0)={a[3]:.4f} | "
            f"dBalAcc={b[0]:+.4f} [{b[1]:+.4f},{b[2]:+.4f}] P(<=0)={b[3]:.4f} | "
            f"McNemar p={o['mcnemar'][0]:.3g} ({o['mcnemar'][1]}/{o['mcnemar'][2]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--baseline_oof", default="results_baseline/oof_scores.csv")
    ap.add_argument("--baseline_key", default="B|SNV|PLS-DA")
    ap.add_argument("--models", nargs="+", default=["yolo11s_all16", "specpool_all16"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--out", default="results_final")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    L = []

    def say(m=""):
        print(m, flush=True)
        L.append(m)

    b = pd.read_csv(a.baseline_oof).set_index("Sample ID")
    avg = {}
    say("# Analisis akhir\n\n## Q1. Stabilitas antar-seed")
    for mname in a.models:
        tags = [mname] + [f"{mname}_s{s}" for s in a.seeds]
        S = {t: load_oof(Path(a.runs) / t) for t in tags}
        S = {t: s for t, s in S.items() if s is not None}
        if not S:
            say(f"{mname}: tidak ada hasil")
            continue
        ids = sorted(set.intersection(*[set(s.index) for s in S.values()]) & set(b.index))
        y = b.loc[ids, "y"].to_numpy()
        per = []
        for t, s in S.items():
            v = s.loc[ids].to_numpy()
            per.append((t, bacc(y, v > .5), roc_auc_score(y, v)))
        ba, au = np.array([p[1] for p in per]), np.array([p[2] for p in per])
        say(f"{mname}: {len(S)} seed, n={len(ids)} | bal_acc {ba.mean():.3f} ± {ba.std(ddof=1) if len(ba)>1 else 0:.3f} "
            f"(min {ba.min():.3f}, maks {ba.max():.3f}) | AUC {au.mean():.3f} ± {au.std(ddof=1) if len(au)>1 else 0:.3f}")
        sm = pd.concat([s.loc[ids] for s in S.values()], axis=1).mean(1)
        avg[mname] = sm
        m = B.metrics(y, sm.to_numpy())
        lo, hi = B.boot_ci(y, sm.to_numpy())
        say(f"  rata-rata {len(S)} seed: bal_acc={m['bal_acc']:.3f} [{lo:.3f},{hi:.3f}] AUC={m['auc']:.3f} MCC={m['mcc']:.3f}")

    ids = sorted(set(b.index).intersection(*[set(s.index) for s in avg.values()]))
    y = b.loc[ids, "y"].to_numpy()
    pls = np.clip(b.loc[ids, a.baseline_key].to_numpy(), 0, 1)
    mp = B.metrics(y, pls)
    say(f"\nPembanding PLS-DA SNV (CSV): bal_acc={mp['bal_acc']:.3f} AUC={mp['auc']:.3f} (n={len(ids)})")

    if len(a.models) >= 2 and all(m in avg for m in a.models[:2]):
        m1, m2 = a.models[:2]
        say(f"\n## Q2. Spasial vs non-spasial: {m1} vs {m2} (rata-rata seed)")
        say(fmt_paired(paired(y, avg[m1].loc[ids].to_numpy(), avg[m2].loc[ids].to_numpy())))

    say("\n## Q3. Fusi dengan PLS-DA (rata-rata sederhana, a priori)")
    rows = []
    for mname, sm in avg.items():
        f = (pls + sm.loc[ids].to_numpy()) / 2
        m = B.metrics(y, f)
        lo, hi = B.boot_ci(y, f)
        o = paired(y, f, pls)
        say(f"PLS + {mname}: bal_acc={m['bal_acc']:.3f} [{lo:.3f},{hi:.3f}] AUC={m['auc']:.3f} MCC={m['mcc']:.3f}")
        say("  vs PLS saja: " + fmt_paired(o))
        alone = B.metrics(y, sm.loc[ids].to_numpy())
        rows.append({"model": mname, "alone_bal_acc": alone["bal_acc"], "alone_auc": alone["auc"],
                     "fusion_bal_acc": m["bal_acc"], "fusion_lo": lo, "fusion_hi": hi, "fusion_auc": m["auc"],
                     "dAUC": o["dAUC"][0], "dAUC_lo": o["dAUC"][1], "dAUC_hi": o["dAUC"][2],
                     "dBalAcc": o["dBalAcc"][0], "dBalAcc_lo": o["dBalAcc"][1], "dBalAcc_hi": o["dBalAcc"][2],
                     "mcnemar_p": o["mcnemar"][0]})
    pd.DataFrame(rows).to_csv(out / "fusion_table.csv", index=False)
    fus = pd.DataFrame({"y": y, "pls": pls, **{k: v.loc[ids].to_numpy() for k, v in avg.items()}}, index=ids)
    fus.to_csv(out / "seedavg_oof_scores.csv")
    (out / "summary.md").write_text("\n".join(L))
    say(f"\nTersimpan di {out}/")


if __name__ == "__main__":
    main()
