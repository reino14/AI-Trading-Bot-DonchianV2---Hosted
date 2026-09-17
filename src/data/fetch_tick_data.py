"""
data/fetch_tick_data.py

LANGKAH 1 dari rencana replikasi Chris: tarik data trade-level (bukan
bar OHLCV) untuk BTCUSDT & ETHUSDT perpetual -- fondasi untuk footprint
candle, delta, dan volume profile di langkah-langkah berikutnya.

SUMBER: data.binance.vision (arsip publik gratis, tanpa API key), lewat
paket `binance-vision` (BUKAN saya tulis sendiri parsernya -- sudah
diverifikasi menangani kuirk nyata: skema kolom futures 7-kolom vs
spot 8-kolom, header yang tidak konsisten antar file, satuan epoch yang
berubah ms->mikrodetik pertengahan 2025, dan verifikasi checksum
SHA-256 per file).

KENAPA PER BULAN, BUKAN SEKALIGUS
-----------------------------------
Data tick JAUH lebih besar dari bar harian yang sudah kita tarik untuk
proyek cross-sectional. BTCUSDT aggTrades sehari saja bisa ratusan ribu
baris; setahun bisa puluhan juta baris, beberapa GB. Menarik 2+ tahun
sekaligus ke satu DataFrame akan membebani memori laptop mana pun.
Skrip ini menarik SATU BULAN, simpan ke parquet, BUANG dari memori,
lanjut ke bulan berikutnya -- supaya penggunaan memori puncak tetap
terkendali berapa pun panjang rentang yang diminta.

SARAN: mulai dari rentang PENDEK (1-3 bulan terakhir) untuk membangun
dan menguji logika footprint/POC di langkah berikutnya. Baru perbesar
rentang setelah logikanya benar -- menarik 2 tahun tick data untuk
kode yang ternyata masih ada bug adalah waktu terbuang paling mahal
di seluruh proyek ini.

KETERBATASAN YANG PERLU DIINGAT
---------------------------------
Order book DEPTH (bukan cuma top bid/ask) historis TIDAK tersedia
gratis di mana pun untuk masa lalu -- itu bukan kuirk sumber data ini
saja, itu batasan struktural (Binance tidak menyimpan L2 historis
untuk didistribusikan gratis). Yang tersedia gratis: bookTicker
(top-of-book bid/ask saja, sejak Mei 2023) dan bookDepth (depth
teragregasi per persentase dari mid-price, harian). Keduanya lebih
tipis dari DOM ladder yang Chris lihat langsung, tapi cukup untuk
konteks order-flow dasar -- lihat --include-book-ticker.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from binance_vision import fetch_data


def _month_range(start: date, end: date) -> list[tuple[date, date]]:
    """Pecah rentang jadi daftar (awal_bulan, akhir_bulan_atau_end)."""
    periods = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        if cursor.month == 12:
            next_month = date(cursor.year + 1, 1, 1)
        else:
            next_month = date(cursor.year, cursor.month + 1, 1)
        period_start = max(cursor, start)
        period_end = min(date.fromordinal(next_month.toordinal() - 1), end)
        periods.append((period_start, period_end))
        cursor = next_month
    return periods


def fetch_symbol_tick_data(
    symbol: str,
    start: date,
    end: date,
    out_dir: Path,
    data_type: str = "aggTrades",
    market: str = "um",
) -> None:
    out_subdir = out_dir / data_type / symbol
    out_subdir.mkdir(parents=True, exist_ok=True)

    for period_start, period_end in _month_range(start, end):
        label = period_start.strftime("%Y-%m")
        out_path = out_subdir / f"{label}.parquet"
        if out_path.exists():
            print(f"  [{symbol}/{data_type}] {label}: sudah ada, dilewati "
                  f"(hapus filenya kalau mau tarik ulang)")
            continue

        result = fetch_data(
            ticker=symbol,
            start_date=period_start.isoformat(),
            end_date=period_end.isoformat(),
            market=market,
            data_type=data_type,
        )

        if result.data.empty:
            print(f"  [{symbol}/{data_type}] {label}: KOSONG "
                  f"(missing={result.missing}, failed={result.failed})")
            continue

        result.data.to_parquet(out_path)
        size_mb = out_path.stat().st_size / 1_000_000
        print(f"  [{symbol}/{data_type}] {label}: {len(result.data):,} baris, "
              f"{size_mb:.1f} MB -> {out_path}")
        if result.missing:
            print(f"    CATATAN: periode hilang di sumber: {result.missing}")
        if result.failed:
            print(f"    PERINGATAN: gagal diunduh (bukan 'kosong', tapi ERROR): "
                  f"{result.failed}")

        # Buang dari memori sebelum lanjut bulan berikutnya -- ini intinya.
        del result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--out-dir", default="data/raw/tick")
    p.add_argument("--include-book-ticker", action="store_true",
                    help="juga tarik top-of-book bid/ask (tersedia sejak Mei 2023)")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_dir = Path(args.out_dir)

    data_types = ["aggTrades"] + (["bookTicker"] if args.include_book_ticker else [])

    for symbol in args.symbols:
        for dt in data_types:
            print(f"\n=== {symbol} / {dt} ({start} s.d. {end}) ===")
            fetch_symbol_tick_data(symbol, start, end, out_dir, data_type=dt)

    print("\nSelesai. Ingat: ini baru trade prints (+ top-of-book kalau diminta).")
    print("Volume profile (POC/value area) dan footprint/delta dihitung dari ini")
    print("di langkah berikutnya -- belum ada di file yang baru ditarik ini.")


if __name__ == "__main__":
    main()