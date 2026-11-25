#!/usr/bin/env python3
"""
Tahap 3c — Rekap hasil train_cube.py: metrik OOF (LODO), CI bootstrap, uji vs baseline CSV, band terpilih.

Pemakaian:
  python aggregate_cube.py --runs runs --baseline_oof results_baseline/oof_scores.csv --out results_cube
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
import baseline as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--baseline_oof", default="results_baseline/oof_scores.csv")
    ap.add_argument("--baseline_key", default="B|SNV|PLS-DA")
    ap.add_argument("--out", default="results_cube")
    ap.add_argument("--n_folds", type=int, default=25)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = None
    if Path(a.baseline_oof).exists():
        bo = pd.read_csv(a.baseline_oof)
        if a.baseline_key in bo.columns:
            base = bo.set_index("Sample ID")[[a.baseline_key, "y", "date"]]

    rows, lines, oof_all = [], ["# Hasil model kubus (LODO)\n"], {}
    for d in sorted(Path(a.runs).iterdir()):
        files = sorted(d.glob("fold_*.json"))
        if not files:
            continue
        cfg = json.load(open(d / "config.json"))
        recs = [json.load(open(f)) for f in files]
        ids = sum((r["ids"] for r in recs), [])
        y = np.array(sum((r["y"] for r in recs), []))
        s = np.array(sum((r["score"] for r in recs), []))
        oof_all[d.name] = pd.Series(s, index=ids)
        mt = B.metrics(y, s)
        lo, hi = B.boot_ci(y, s)
        per_fold = {r["fold"]: accuracy_score(r["y"], (np.array(r["score"]) > 0.5).astype(int)) for r in recs}
        row = {"tag": d.name, "backbone": cfg["backbone"], "selector": cfg["selector"], "k": cfg.get("k"),
               "folds": f"{len(files)}/{a.n_folds}", "n": len(y), **mt, "bal_acc_lo": lo, "bal_acc_hi": hi,
               "fold_acc_mean": np.mean(list(per_fold.values())), "fold_acc_std": np.std(list(per_fold.values())),
               "sec_per_fold": np.mean([r["sec"] for r in recs])}
        nms = [r["selected_nm"] for r in recs if r.get("selected_nm")]
        if nms:
            flat = [round(v / 5) * 5 for nm in nms for v in nm]  # bin 5 nm
            row["bands_top"] = [b for b, _ in Counter(flat).most_common(cfg.get("k", 3) + 2)]
            row["bands_stability"] = round(max(Counter(flat).values()) / len(nms), 2)
        if base is not None:
            common = [i for i in ids if i in base.index]
            if common:
                yb = base.loc[common, "y"].to_numpy()
                pb = (base.loc[common, a.baseline_key].to_numpy() > 0.5).astype(int)
                pm = (oof_all[d.name].loc[common].to_numpy() > 0.5).astype(int)
                p_mc, n01, n10 = B.mcnemar(yb, pm, pb)
                row.update({"mcnemar_p_vs_csv": p_mc, "win_loss_vs_csv": f"{n01}/{n10}"})
                bdates = base.loc[common, "date"].to_numpy()
                diffs = []
                for fd, acc in per_fold.items():
                    sel = bdates == fd
                    if sel.any():
                        diffs.append(acc - accuracy_score(yb[sel], pb[sel]))
                diffs = np.array(diffs)
                row["wilcoxon_p_vs_csv"] = wilcoxon(diffs).pvalue if np.any(diffs != 0) else 1.0
        rows.append(row)

    if not rows:
        sys.exit("Belum ada hasil fold.")
    res = pd.DataFrame(rows).sort_values("bal_acc", ascending=False)
    res.to_csv(out / "cube_metrics.csv", index=False)
    pd.DataFrame(oof_all).to_csv(out / "cube_oof_scores.csv")
    cols = [c for c in ["tag", "folds", "bal_acc", "bal_acc_lo", "bal_acc_hi", "auc", "mcc", "fold_acc_std",
                        "bands_top", "mcnemar_p_vs_csv", "win_loss_vs_csv", "wilcoxon_p_vs_csv", "sec_per_fold"]
            if c in res.columns]
    if base is not None:
        yb = base["y"].to_numpy()
        sb = base[a.baseline_key].to_numpy()
        lines.append(f"Pembanding CSV ({a.baseline_key}): bal_acc={B.metrics(yb, sb)['bal_acc']:.3f}\n")
    lines.append(res[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    lines.append("\nCatatan: win_loss = (benar di model kubus & salah di CSV) / (sebaliknya).")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
