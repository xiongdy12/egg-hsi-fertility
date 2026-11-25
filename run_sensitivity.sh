#!/usr/bin/env bash
# Latih ulang YOLOv11s (300 band) & kontrol spektral TANPA telur bertanda, lalu analisis sensitivitas.
# Pemakaian (dari ~/egg_hsi, venv aktif):
#   nohup bash run_sensitivity.sh /raid/home/tegar/egg_hsi/cubes /raid/home/tegar/egg_hsi/raw/spectra_reference.csv \
#         "7" 3 > sensitivity.log 2>&1 &
#   argumen 3 = daftar GPU (mis. "6 7"), argumen 4 = jumlah seed (default 3)
set -euo pipefail
DATA="${1:?folder cubes}"
CSV="${2:?spectra_reference.csv}"
GPUS=(${3:-7})
NSEED="${4:-3}"
EXC=results_ink/audit_visual.csv
cd "$(dirname "$0")"
mkdir -p logs
python - "$NSEED" <<'EOF'
import json, sys
n = int(sys.argv[1])
B = {json.loads(l)["tag"]: json.loads(l) for l in open("configs.jsonl")}
with open("configs_clean.jsonl", "w") as f:
    for t in ["yolo11s_all16", "specpool_all16"]:
        for s in range(n):
            c = dict(B[t]); c["tag"] = f"{t}_s{s}"; c["seed"] = s; f.write(json.dumps(c) + "\n")
print(f"configs_clean.jsonl: {2*n} konfigurasi")
EOF
NG=${#GPUS[@]}
echo "[$(date +%H:%M)] Latih ulang tanpa $(($(wc -l < $EXC)-1)) telur bertanda di GPU ${GPUS[*]}"
for r in $(seq 0 $((NG-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$r]} python -u train_cube.py --data "$DATA" --configs configs_clean.jsonl \
    --out runs_clean --rank "$r" --world "$NG" --res 128 --exclude "$EXC" > "logs/clean_rank$r.log" 2>&1 &
done
wait
echo "[$(date +%H:%M)] Analisis sensitivitas"
python sensitivity.py --csv "$CSV" --exclude "$EXC" --runs_clean runs_clean --out results_sensitivity
echo "[$(date +%H:%M)] SELESAI -> results_sensitivity/summary.md"
