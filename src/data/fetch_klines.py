"""
data/fetch_klines.py

LANGKAH 3 (bagian data): tarik kline 1 jam & 4 jam untuk BTCUSDT &
ETHUSDT -- dipakai buat higher-timeframe bias (pengganti "market
structure" yang Chris tentukan dari chart 1H/4H).

SENGAJA TIDAK diturunkan dari data tick yang sudah ditarik di langkah
1. Itu akan berarti memproses ulang jutaan baris trade cuma untuk
dapat bar 1 jam -- boros besar-besaran. Kline sudah diagregasi Binance
sendiri dan ukurannya kecil (ribuan baris untuk beberapa tahun, bukan
puluhan juta) -- ditarik langsung lewat sumber yang sama
(data.binance.vision, paket binance-vision yang sudah diverifikasi).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from binance_vision import fetch_data


def fetch_klines(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    out_dir: Path,
    market: str = "um",
) -> Path:
    result = fetch_data(
        ticker=symbol, start_date=start, end_date=end,
        market=market, data_type="klines", interval=interval,
    )
    out_subdir = out_dir / "klines" / symbol
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / f"{interval}.parquet"
    result.data.to_parquet(out_path)
    print(f"  [{symbol}/{interval}] {len(result.data):,} bar -> {out_path}")
    if result.missing:
        print(f"    CATATAN: periode hilang: {result.missing}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--intervals", nargs="+", default=["1h", "4h"])
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out-dir", default="data/raw")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    for symbol in args.symbols:
        for interval in args.intervals:
            print(f"=== {symbol} / {interval} ===")
            fetch_klines(symbol, interval, args.start, args.end, out_dir)


if __name__ == "__main__":
    main()