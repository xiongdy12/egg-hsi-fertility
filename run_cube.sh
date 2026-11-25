#!/usr/bin/env bash
# Jalankan semua konfigurasi train_cube.py di semua GPU (satu proses per GPU).
# Pemakaian:  bash run_cube.sh /raid/home/tegar/egg_hsi/cubes [jumlah_gpu] [argumen tambahan train_cube.py, mis. --res 128]
set -euo pipefail
DATA="${1:?path folder cubes}"
NG="${2:-$(nvidia-smi -L | wc -l)}"
EXTRA=("${@:3}")
cd "$(dirname "$0")"
[ -f configs.jsonl ] || python train_cube.py --make_configs configs.jsonl

# unduh bobot pretrained SEKALI (hindari 8 proses mengunduh bersamaan)
python - <<'EOF'
from ultralytics import YOLO
for n in ("yolo11n-cls", "yolo11s-cls", "yolo11m-cls"):
    YOLO(f"{n}.pt")
import torchvision.models as m
for f, w in [(m.mobilenet_v3_small, "IMAGENET1K_V1"), (m.efficientnet_b0, "IMAGENET1K_V1"), (m.resnet18, "IMAGENET1K_V1")]:
    try: f(weights=w)
    except Exception as e: print("gagal unduh", f.__name__, e)
print("bobot siap")
EOF

mkdir -p logs
for r in $(seq 0 $((NG-1))); do
  nohup python -u train_cube.py --data "$DATA" --configs configs.jsonl --out runs \
       --rank "$r" --world "$NG" "${EXTRA[@]}" > "logs/rank$r.log" 2>&1 &
done
echo "Berjalan di $NG GPU. Pantau: tail -f logs/rank0.log   |  rekap: python aggregate_cube.py"
wait
echo "SEMUA SELESAI"
