"""
Unduh bar 1 menit saham AS lewat yfinance.

Keluarannya sengaja dibuat SAMA PERSIS dengan fetch_history.py (kripto),
sehingga store.py dan analyze_holding.py bisa dipakai tanpa diubah.

BATASAN PENTING:
    Yahoo Finance hanya menyediakan bar 1 menit untuk sekitar 30 hari terakhir,
    dan hanya bisa diambil per potongan 7 hari.

    Ini CUKUP untuk menjawab pertanyaan Hari 1 (seberapa besar harga bergerak
    per menit dibanding ongkos), tapi TIDAK CUKUP untuk backtest Hari 2 yang
    butuh 18 bulan data.

    Untuk Hari 2, data historis panjang harus dibeli. Databento atau Polygon,
    sekali beli, kisaran Rp 200.000 - 500.000 untuk satu ticker.

Cara pakai:
    pip install yfinance
    python -m src.data.fetch_stock_history NVDA
    python -m src.data.fetch_stock_history NVDA TSLA AMD --days 30
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")

# Yahoo membatasi permintaan bar 1 menit ke 7 hari per panggilan.
CHUNK_DAYS = 7
MAX_DAYS = 30


def fetch_stock_ohlcv(ticker: str, days: int = MAX_DAYS) -> pd.DataFrame:
    """
    Tarik bar 1 menit, potongan 7 hari sekali, lalu digabung.

    Mengembalikan DataFrame dengan index waktu UTC dan kolom:
        open, high, low, close, volume
    """
    import yfinance as yf

    days = min(days, MAX_DAYS)
    end = datetime.now(timezone.utc)
    frames: list[pd.DataFrame] = []

    remaining = days
    while remaining > 0:
        span = min(CHUNK_DAYS, remaining)
        start = end - timedelta(days=span)

        chunk = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1m",
            progress=False,
            auto_adjust=False,
        )

        if not chunk.empty:
            frames.append(chunk)
            print(f"  ...{start:%Y-%m-%d} sampai {end:%Y-%m-%d}: {len(chunk):,} bar")

        end = start
        remaining -= span

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames)

    # yfinance kadang mengembalikan kolom bertingkat kalau ticker berupa list.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]]

    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="first")].sort_index()

    return df.dropna()


def save(df: pd.DataFrame, ticker: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{ticker}.parquet"
    df.to_parquet(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Unduh bar 1 menit saham AS")
    parser.add_argument("tickers", nargs="+", help="contoh: NVDA TSLA")
    parser.add_argument("--days", type=int, default=MAX_DAYS)
    args = parser.parse_args()

    for ticker in args.tickers:
        print(f"\nMengunduh {ticker}, {min(args.days, MAX_DAYS)} hari terakhir...")
        df = fetch_stock_ohlcv(ticker, days=args.days)

        if df.empty:
            print(f"  Tidak ada data untuk {ticker}. Cek nama tickernya.")
            continue

        path = save(df, ticker)
        hari_bursa = df.index.normalize().nunique()

        print(f"\n  Tersimpan: {path}")
        print(f"    {len(df):,} bar dari {hari_bursa} hari bursa")
        print(f"    {df.index.min():%Y-%m-%d %H:%M} sampai {df.index.max():%Y-%m-%d %H:%M} UTC")
        print(
            "    Catatan: saham hanya diperdagangkan sekitar 390 menit per hari,\n"
            "    jauh lebih sedikit dari kripto yang 1.440 menit."
        )


if __name__ == "__main__":
    main()