#!/usr/bin/env bash
# Grad-CAM + uji spasial (4 GPU) -> benchmark latency (CPU, setelah GPU selesai) -> rekap -> semua gambar.
# Pemakaian (dari ~/egg_hsi, venv aktif):
#   nohup bash run_extras.sh /raid/home/tegar/egg_hsi/cubes /raid/home/tegar/egg_hsi/raw/spectra_reference.csv "4 5 6 7" > extras.log 2>&1 &
set -euo pipefail
DATA="${1:?folder cubes}"
CSV="${2:?spectra_reference.csv}"
GPUS=(${3:-4 5 6 7})
NG=${#GPUS[@]}
cd "$(dirname "$0")"
mkdir -p logs

echo "[$(date +%H:%M)] 1/4 Grad-CAM + uji spasial di GPU ${GPUS[*]}"
for r in $(seq 0 $((NG-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$r]} python -u gradcam_spatial.py --data "$DATA" --rank "$r" --world "$NG" \
    --res 128 --out results_spatial > "logs/spatial_rank$r.log" 2>&1 &
done
wait
python gradcam_spatial.py --data "$DATA" --report --out results_spatial

echo "[$(date +%H:%M)] 2/4 Benchmark latency (CPU 1 & 4 thread, inti CPU 0-3; GPU ${GPUS[0]})"
CUDA_VISIBLE_DEVICES=${GPUS[0]} taskset -c 0-3 python -u benchmark_latency.py --configs configs.jsonl \
  --res 128 --out results_latency --gpu

echo "[$(date +%H:%M)] 3/4 Gambar"
python make_figures.py --csv "$CSV" --out figures

echo "[$(date +%H:%M)] 4/4 Paket hasil"
cp results_spatial/fig_*.pdf results_spatial/fig_*.png figures/ 2>/dev/null || true
zip -qr hasil_gambar_latency.zip figures results_latency results_spatial/summary.md results_final \
  results_cube/summary.md results_baseline/summary.md results_confound/summary.md
echo "[$(date +%H:%M)] SELESAI -> ~/egg_hsi/hasil_gambar_latency.zip"
