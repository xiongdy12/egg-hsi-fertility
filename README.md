# Spatial–spectral deep learning for pre-incubation egg fertility screening

Code accompanying the manuscript

> T. A. Prasetyo, *Does spatial context help hyperspectral egg fertility screening? A leakage-controlled evaluation of YOLO11 with matched non-spatial controls*, submitted to *Signal, Image and Video Processing*.

All experiments use the public transmission hyperspectral egg dataset of Ahmed et al.
(Illinois Data Bank, <https://doi.org/10.13012/B2IDB-8141497_V1>, CC0; data descriptor:
*Scientific Data* 13:237, 2026, <https://doi.org/10.1038/s41597-026-06556-1>).
The data are **not** redistributed here; download them from the Illinois Data Bank.

## What is in this repository

| Step | Script(s) | Output | Paper section |
|---|---|---|---|
| 1. Download (parallel, resumable) | `download_data.sh`, `pdownload.py`, `inspect_data.py` | `raw/` | 2.1 |
| 2. Mean-spectrum baselines (LODO, nested tuning) | `baseline.py` | `results_baseline/` | 2.3, 3.1 |
| 3. Confound analysis (morphometrics, residuals, matching) | `confound.py` | `results_confound/` | 3.1 |
| 4. File-linkage audit by spectral fingerprint | `audit_ids.py` | `audit_ids.csv` | 2.1 |
| 5. Cube preprocessing (destriping, mask, crop, pack) | `prepare_cubes.py` | `cubes/` | 2.4 |
| 6. Cube models, 20 screening configurations | `train_cube.py`, `run_cube.sh`, `aggregate_cube.py` | `runs/`, `results_cube/` | 2.4, 3.2–3.4 |
| 7. Seeds, spatial vs. non-spatial, fusion | `final_analysis.py` | `results_final/` | 3.3, 3.5 |
| 8. Latency/FLOPs, figures, Grad-CAM and spatial perturbations | `benchmark_latency.py`, `make_figures.py`, `gradcam_spatial.py`, `run_extras.sh` | `results_latency/`, `figures/`, `results_spatial/` | 2.6, 3.3 |
| 9. Shell-mark audit and sensitivity analysis | `ink_check.py`, `contact_sheets.py`, `sensitivity.py`, `run_sensitivity.sh` | `results_ink/`, `results_sensitivity/` | 2.1, 3.7 |
| 10. Extended experiments (backbones, HybridSN, width ablation, residual anchoring, temporal split, calibration) | `run_nca.sh`, `nca_analysis.py` | `results_nca/`, `results_latency_nca/` | 3.2–3.6 |

Usage examples are given in the header of each script. Every data-dependent step
(preprocessing, band selection, tuning, BatchNorm statistics, fusion parameters) is fitted on the
training dates of each leave-one-date-out fold only.

`supplementary/` contains the 45 re-linked cube files (`audit_ids.csv`) and the 57 eggs flagged
in the shell-mark audit (`audit_visual.csv`). `results/` contains the summary files reported in the paper.

## Environment

Experiments were run on Ubuntu (NVIDIA DGX, B200 GPUs) with Python 3.12; exact package versions are
listed in `requirements.txt`.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Processed cubes need about 48 GB of disk; training at 128×128 px needs about 20 GB of GPU memory per process.

## Citation

If you use this code, please cite the paper and this software archive (see `CITATION.cff`), and the
dataset of Ahmed et al.

## Licence

Code: MIT licence (see `LICENSE`). Dataset: CC0, © the original authors.
