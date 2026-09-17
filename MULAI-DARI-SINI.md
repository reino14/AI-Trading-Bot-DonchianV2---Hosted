# MULAI DARI SINI

Bingung dokumennya kebanyakan? Wajar. Ini penjelasan singkatnya.

**Semuanya cuma SATU folder: `trading-bot/`.** Folder yang sedang Anda buka ini.
Tidak ada proyek kedua. Dokumen HTML, file Excel, dan kode Python semuanya ada
di dalam folder ini.

---

## Isi folder ini, dan mana yang dijalankan

| Isi | Apa ini | Dijalankan? |
|---|---|---|
| `docs/panduan-proyek.html` | Penjelasan tujuan proyek dan kamus istilah | Tidak, dibaca di browser |
| `docs/framework-kode.html` | Rencana 12 minggu dan penjelasan tiap file kode | Tidak, dibaca di browser |
| `docs/monitoring-proyek-bot-trading.xlsx` | Checklist progres, gate, log harian | Tidak, dibuka di Excel |
| `scaffold.py` | Pembuat folder dan file kosong | Ya, sekali di awal |
| `src/`, `scripts/` | Kode program yang sesungguhnya | Ya |
| `config/config.yaml` | Semua pengaturan di satu tempat | Tidak, diedit saja |

Tiga yang pertama adalah **kertas**. Sisanya adalah **program**.

---

## Apa itu scaffold.py, dan kenapa ada

`scaffold.py` bukan proyek terpisah. Dia cuma membuat **file-file kosong** di
dalam folder yang sama ini.

Analoginya: sebelum mulai menulis skripsi, Anda membuat dulu file
`bab1.docx`, `bab2.docx`, `bab3.docx` yang masih kosong. Isinya belum ada,
tapi kerangkanya sudah terlihat. Itu yang dilakukan `scaffold.py` — dia membuat
42 file Python kosong, masing-masing sudah berisi keterangan "file ini nanti
untuk apa" dan "dikerjakan di sprint berapa".

Kenapa perlu? Supaya saat membuka folder, langsung terlihat pekerjaan yang
tersisa. Tanpa itu, folder cuma berisi 4 file dan tidak jelas apa lagi yang
harus dibuat.

**Aman dijalankan berkali-kali.** File yang sudah berisi kode tidak akan
ditimpa. Kalau dijalankan ulang, dia cuma bilang "sudah ada, dilewati".

---

## Status sekarang: 4 dari 42 file sudah berisi kode

Yang sudah jadi dan bisa langsung dijalankan:

```
src/core/cost_model.py       menghitung ongkos transaksi
src/data/fetch_history.py    mengunduh data harga dari bursa
src/data/store.py            membaca data yang sudah diunduh
scripts/analyze_costs.py     membandingkan pergerakan harga vs ongkos
```

Empat file ini adalah **Hari 1**. Sisanya (38 file) masih kosong dan akan
diisi di Hari 2 sampai Hari 7.

---

## Yang dilakukan hari ini, langkah demi langkah

Buka Terminal (Mac/Linux) atau PowerShell (Windows), masuk ke folder ini.

### Langkah 1 — siapkan Python

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Kalau berhasil, di awal baris terminal muncul tulisan `(.venv)`.

### Langkah 2 — pasang library

```bash
pip install ccxt pandas numpy pyarrow pyyaml openpyxl
```

Tunggu sampai selesai. Butuh koneksi internet.

### Langkah 3 — buat kerangka file

```bash
python scaffold.py
```

Akan muncul daftar file yang dibuat. Kalau sebelumnya sudah pernah dijalankan,
akan muncul "Dibuat: 0 file, Dilewati: 42 file" — itu normal dan benar.

### Langkah 4 — lihat ongkos transaksi

```bash
python -m src.core.cost_model
```

Ini perintah pertama yang menghasilkan sesuatu. Akan muncul tiga skenario
ongkos. Perhatikan selisih antara baris pertama dan ketiga.

### Langkah 5 — unduh data harga

```bash
python -m src.data.fetch_history "BTC/USDT:USDT" --days 540
```

Butuh 10–20 menit. Akan terlihat progres berjalan. Hasilnya masuk ke
`data/raw/`. Ulangi untuk instrumen kedua:

```bash
python -m src.data.fetch_history "ETH/USDT:USDT" --days 540
```

### Langkah 6 — jalankan analisisnya

```bash
python -m scripts.analyze_costs "BTC/USDT:USDT" "ETH/USDT:USDT"
```

**Ini hasil yang menentukan.** Cari baris `RASIO gerak/ongkos`:

- Di bawah 2× → berhenti, ganti instrumen, jangan lanjut ke Hari 2
- 2× sampai 4× → lanjut ke Hari 2
- Di atas 4× → lanjut, tapi periksa dulu likuiditas instrumennya

### Langkah 7 — catat hasilnya

Buka `docs/monitoring-proyek-bot-trading.xlsx`:

1. Sheet **Sprint 1 Harian** — ubah kolom Status jadi "Selesai" untuk tugas
   Hari 1 yang sudah dikerjakan
2. Sheet **Gate Outcome** — isi kolom Aktual di baris **G1** dengan angka rasio
   yang barusan didapat. Kolom Lulus akan menghitung sendiri
3. Sheet **Keputusan** — isi baris terakhir dengan instrumen yang dipilih dan
   alasannya

Sel berwarna kuning adalah sel yang diisi manual. Sisanya terhitung otomatis.

Terakhir, buka `lessons.md` dan tulis satu paragraf: instrumen apa yang dipilih
dan kenapa.

---

## Kalau muncul error

**`No module named ccxt`**
Library belum terpasang, atau virtual environment belum aktif. Pastikan ada
tulisan `(.venv)` di terminal, lalu ulangi Langkah 2.

**`No module named src`**
Terminal sedang tidak berada di folder `trading-bot/`. Jalankan `pwd`
(Windows: `cd`) untuk mengecek, lalu pindah ke folder yang benar.

**`FileNotFoundError` saat menjalankan analyze_costs**
Data belum diunduh. Jalankan Langkah 5 dulu.

**Unduhan data terhenti di tengah**
Jalankan ulang perintah yang sama. Data yang sudah masuk tidak hilang.

---

## Setelah Hari 1 selesai

Kabari hasil rasionya. Hari 2 adalah menulis strategi dan mesin simulasi —
5 file, dan itu bagian yang paling banyak kodenya.

Kalau rasionya di bawah 2×, jangan lanjut. Tiga hal yang bisa dicoba dulu:
coba instrumen lain, ubah asumsi eksekusi dari taker ke maker di
`scripts/analyze_costs.py`, atau naikkan `hold_minutes` di `cost_model.py`
dari 5 ke 30 supaya pergerakan yang ditangkap lebih besar dibanding ongkosnya.
