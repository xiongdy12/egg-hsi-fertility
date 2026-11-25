#!/usr/bin/env bash
# Eksperimen tambahan untuk Neural Computing and Applications (LODO + uji temporal + latensi + analisis).
# Pemakaian (dari ~/egg_hsi, venv aktif):
#   nohup bash run_nca.sh /raid/home/tegar/egg_hsi/cubes /raid/home/tegar/egg_hsi/raw/spectra_reference.csv "6 7" > nca.log 2>&1 &
#   argumen 3 = daftar GPU yang kosong (cek dulu dengan nvidia-smi)
set -euo pipefail
DATA="${1:?folder cubes}"
CSV="${2:?spectra_reference.csv}"
GPUS=(${3:-7})
NG=${#GPUS[@]}
cd "$(dirname "$0")"
mkdir -p logs
python train_cube.py --make_configs_nca configs_nca.jsonl configs_temporal.jsonl

echo "[$(date +%H:%M)] 1/4 LODO: 22 konfigurasi di GPU ${GPUS[*]}"
for r in $(seq 0 $((NG-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$r]} python -u train_cube.py --data "$DATA" --configs configs_nca.jsonl \
    --out runs --rank "$r" --world "$NG" --res 128 --anchor_csv "$CSV" > "logs/nca_rank$r.log" 2>&1 &
done
wait

echo "[$(date +%H:%M)] 2/4 Uji temporal: 13 konfigurasi"
for r in $(seq 0 $((NG-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$r]} python -u train_cube.py --data "$DATA" --configs configs_temporal.jsonl \
    --out runs_temporal --split temporal --cut 18Dec --rank "$r" --world "$NG" --res 128 \
    --anchor_csv "$CSV" > "logs/temporal_rank$r.log" 2>&1 &
done
wait

echo "[$(date +%H:%M)] 3/4 Latensi model baru"
CUDA_VISIBLE_DEVICES=${GPUS[0]} taskset -c 0-3 python benchmark_latency.py --configs configs_nca.jsonl \
  --res 128 --out results_latency_nca --gpu > logs/latency_nca.log 2>&1

echo "[$(date +%H:%M)] 4/4 Analisis"
python nca_analysis.py --runs runs --runs_temporal runs_temporal --baseline_oof results_baseline/oof_scores.csv \
  --csv "$CSV" --out results_nca
echo "[$(date +%H:%M)] SELESAI -> results_nca/summary.md dan results_latency_nca/summary.md"
