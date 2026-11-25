#!/usr/bin/env bash
# Unduh dataset HSI telur UIUC (Illinois Data Bank IDB-8141497, CC0) ke server B200.
# Pemakaian:  bash download_data.sh /path/ke/data_dir
# Ukuran: ~219 GB arsip .rar + ruang ekstraksi ~219 GB -> siapkan >= 500 GB kosong.
set -euo pipefail

DATA_DIR="${1:-$HOME/egg_hsi_data}"
PAGE="https://databank.illinois.edu/datasets/IDB-8141497"
mkdir -p "$DATA_DIR/raw"
cd "$DATA_DIR/raw"

echo "== Cek ruang disk di $DATA_DIR =="
df -h "$DATA_DIR"
AVAIL_GB=$(df -BG --output=avail "$DATA_DIR" | tail -1 | tr -dc '0-9')
if [ "$AVAIL_GB" -lt 500 ]; then
  echo "PERINGATAN: ruang kosong ${AVAIL_GB} GB < 500 GB. Ekstraksi penuh bisa gagal."
fi

echo "== Ambil daftar file dari halaman dataset =="
curl -sSL "$PAGE" -o dataset_page.html
# Link unduhan Illinois Data Bank berpola /datafiles/<id>/download
grep -oE '/datafiles/[A-Za-z0-9_-]+/download' dataset_page.html | sort -u > datafile_links.txt || true

if [ ! -s datafile_links.txt ]; then
  cat <<EOF
Link file tidak terbaca otomatis (halaman mungkin dirender JavaScript).
Lakukan manual:
  1. Buka $PAGE di browser.
  2. Klik kanan tombol Download tiap file -> Copy link.
  3. Jalankan:  wget -c --content-disposition "<link>"   (di folder $DATA_DIR/raw)
  Alternatif untuk file 219 GB: tombol Globus di halaman yang sama (lebih tahan putus).
EOF
  exit 1
fi

echo "Ditemukan $(wc -l < datafile_links.txt) file:"
cat datafile_links.txt

# Unduh file kecil dulu (readme, csv), lalu arsip besar. -c = bisa dilanjutkan kalau putus.
while read -r rel; do
  url="https://databank.illinois.edu${rel}"
  echo "-> $url"
  wget -c --content-disposition --tries=20 --timeout=60 "$url"
done < datafile_links.txt

echo "== Selesai unduh. Isi folder: =="
ls -lh

cat <<'EOF'

Langkah berikut (JANGAN ekstrak semua dulu):
  # lihat isi arsip tanpa ekstrak
  7z l all_data.rar | head -50            # atau: unrar l all_data.rar | head -50
  # ekstrak ~40 file contoh untuk inspeksi
  7z l -slt all_data.rar | grep '^Path = .*\.mat$' | head -40 | sed 's/Path = //' > sample_list.txt
  7z x all_data.rar -o../extracted @sample_list.txt
Kalau 7z belum ada:  sudo apt install p7zip-full p7zip-rar   (atau: conda install -c conda-forge p7zip)
EOF
