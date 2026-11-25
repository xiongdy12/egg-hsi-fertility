#!/usr/bin/env python3
"""
Pengunduh paralel (banyak koneksi) dengan HTTP Range, bisa dilanjutkan. Hanya pustaka standar.

Versi 3:
- Potongan yang terputus DILANJUTKAN dari byte terakhir (tidak diulang dari nol).
- URL S3 presigned dipakai ulang sampai mendekati kedaluwarsa (dibaca dari parameter X-Amz-*).
- Sumber URL baru: (1) file --url_file (bisa diisi dari laptop lain), (2) databank.
  Selama URL lama masih berlaku, kegagalan databank tidak menghentikan unduhan.

Pemakaian (state lama dengan --chunk_mb yang sama tetap dipakai):
  python3 pdownload.py --url https://databank.illinois.edu/datafiles/9teio/download \
      --out all_data.rar --workers 256 --chunk_mb 64 --url_file ~/egg_hsi/s3url.txt
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")


def ts():
    return time.strftime("%H:%M:%S")


def s3_expiry(url):
    """Waktu kedaluwarsa (epoch) dari X-Amz-Date + X-Amz-Expires; fallback 1 jam."""
    try:
        q = parse_qs(urlparse(url).query)
        t0 = datetime.strptime(q["X-Amz-Date"][0], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
        return t0 + int(q["X-Amz-Expires"][0])
    except Exception:
        return time.time() + 3600


class UrlCache:
    def __init__(self, src, timeout, url_file, retry_every, margin=600, min_valid=60):
        self.src, self.timeout, self.url_file = src, timeout, url_file
        self.retry_every, self.margin, self.min_valid = retry_every, margin, min_valid
        self.url, self.exp, self.gen, self.next_try = None, 0.0, 0, 0.0
        self.lock = threading.Lock()
        self.hinted = False

    def _from_file(self):
        if not self.url_file or not os.path.exists(self.url_file):
            return None
        u = open(self.url_file).read().strip()
        if u.startswith("http") and u != self.url and s3_expiry(u) > time.time() + self.min_valid:
            return u
        return None

    def _from_databank(self):
        req = urllib.request.Request(self.src, headers={"User-Agent": UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.geturl()

    def _try_refresh(self, block):
        while True:
            if time.time() >= self.next_try:
                new, how = self._from_file(), "file"
                if new is None:
                    try:
                        new, how = self._from_databank(), "databank"
                    except Exception as e:
                        print(f"[{ts()}] databank menolak ({e})", flush=True)
                        if not self.hinted and self.url_file:
                            print(f"[{ts()}] -> perbarui URL lewat laptop: tulis URL S3 baru ke {self.url_file}",
                                  flush=True)
                            self.hinted = True
                if new:
                    self.url, self.exp, self.gen = new, s3_expiry(new), self.gen + 1
                    left = (self.exp - time.time()) / 60
                    print(f"[{ts()}] URL S3 baru dari {how} (gen {self.gen}), berlaku ~{left:.0f} menit", flush=True)
                    return True
                self.next_try = time.time() + self.retry_every
            if not block:
                return False
            time.sleep(5)

    def get(self):
        with self.lock:
            now = time.time()
            if self.url is None or now > self.exp - self.min_valid:
                self._try_refresh(block=True)          # tidak ada URL yang berlaku: tunggu
            elif now > self.exp - self.margin:
                self._try_refresh(block=False)         # masih berlaku: coba perbarui tanpa menghentikan
            return self.url, self.gen

    def invalidate(self, gen):
        with self.lock:
            if gen == self.gen:
                self.exp = 0  # anggap kedaluwarsa; get() berikutnya akan menunggu URL baru


def get_size(url, timeout, tries=10):
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                m = re.search(r"/(\d+)", r.headers.get("Content-Range", ""))
                if not m:
                    sys.exit(f"Server tidak mendukung Range (status={r.status})")
                return int(m.group(1))
        except (OSError, IOError) as e:
            print(f"cek ukuran gagal ({e}), coba lagi...", flush=True)
            time.sleep(5)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--chunk_mb", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--url_file", default=os.path.expanduser("~/egg_hsi/s3url.txt"))
    ap.add_argument("--retry_every", type=float, default=120, help="detik antar percobaan minta URL baru")
    ap.add_argument("--margin", type=float, default=600, help=argparse.SUPPRESS)
    ap.add_argument("--min_valid", type=float, default=60, help=argparse.SUPPRESS)
    ap.add_argument("--size", type=int, default=None, help="ukuran file (byte) jika databank tak bisa dihubungi")
    a = ap.parse_args()

    state_path = a.out + ".parts.json"
    st = json.load(open(state_path)) if os.path.exists(state_path) else None
    cache = UrlCache(a.url, a.timeout, a.url_file, a.retry_every, a.margin, a.min_valid)
    size = a.size or (st or {}).get("size")
    if size is None:
        u, _ = cache.get()
        size = get_size(u, a.timeout)
    chunk = a.chunk_mb * 2**20
    n = (size + chunk - 1) // chunk
    done = set()
    if st:
        if st.get("size") == size and st.get("chunk") == chunk:
            done = set(st["done"])
        else:
            sys.exit("File state tidak cocok (ukuran/chunk berbeda). Pakai --chunk_mb yang sama.")
    f = open(a.out, "r+b" if os.path.exists(a.out) else "w+b")
    if os.path.getsize(a.out) != size:
        f.truncate(size)
    fd = f.fileno()

    lock = threading.Lock()
    stop = threading.Event()
    stats = {"bytes": 0, "errors": 0}
    err_kinds = {}

    def save_state():
        tmp = state_path + ".tmp"
        with open(tmp, "w") as g:
            json.dump({"size": size, "chunk": chunk, "done": sorted(done)}, g)
        os.replace(tmp, state_path)

    def fetch(i):
        start, end = i * chunk, min(size, (i + 1) * chunk) - 1
        pos, fails, quick = start, 0, 0
        while pos <= end and not stop.is_set():
            u, gen = cache.get()
            try:
                req = urllib.request.Request(u, headers={"User-Agent": UA, "Range": f"bytes={pos}-{end}"})
                with urllib.request.urlopen(req, timeout=a.timeout) as r:
                    if r.status != 206:
                        raise IOError(f"status {r.status}")
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        os.pwrite(fd, buf, pos)
                        pos += len(buf)
                        fails = 0  # ada kemajuan -> reset jeda
                        with lock:
                            stats["bytes"] += len(buf)
                if pos <= end:
                    raise IOError("terpotong")
            except Exception as e:
                code = getattr(e, "code", None)
                kind = f"HTTP {code}" if code else type(e).__name__
                with lock:
                    stats["errors"] += 1
                    err_kinds[kind] = err_kinds.get(kind, 0) + 1
                if code in (400, 403):
                    cache.invalidate(gen)
                    if quick < 3:
                        quick += 1
                        time.sleep(0.5)
                        continue
                fails += 1
                time.sleep(min(60, 2 ** min(fails, 6)) * (0.5 + random.random()))
        if pos > end:
            with lock:
                done.add(i)
                save_state()

    todo = [i for i in range(n) if i not in done]
    print(f"Ukuran {size/2**30:.1f} GiB, {n} potongan x {a.chunk_mb} MB, selesai {len(done)}, "
          f"sisa {len(todo)}, workers {a.workers}", flush=True)

    def reporter():
        last_b, last_t = 0, time.time()
        while not stop.is_set():
            time.sleep(30)
            with lock:
                b, nd, err, kinds = stats["bytes"], len(done), stats["errors"], dict(err_kinds)
                prefix = 0
                while prefix in done:
                    prefix += 1
            now = time.time()
            rate = (b - last_b) / max(now - last_t, 1e-9)
            last_b, last_t = b, now
            remain = size - nd * chunk - 0
            eta = remain / rate / 3600 if rate > 0 else float("inf")
            url_left = (cache.exp - now) / 60
            print(f"[{ts()}] {nd}/{n} ({100*nd/n:.1f}%), {rate/2**20:.2f} MB/s, ETA {eta:.1f} jam, "
                  f"awal-file utuh {min(prefix*chunk, size)/2**30:.1f} GiB, URL sisa {url_left:.0f} mnt, "
                  f"error {err} {kinds}", flush=True)

    threading.Thread(target=reporter, daemon=True).start()
    try:
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(fetch, todo))
    except KeyboardInterrupt:
        stop.set()
    finally:
        with lock:
            save_state()
        f.flush()
        os.fsync(fd)
        f.close()
    print("SELESAI: semua potongan lengkap." if len(done) == n
          else f"Belum lengkap: {len(done)}/{n}. Jalankan ulang untuk melanjutkan.", flush=True)


if __name__ == "__main__":
    main()
