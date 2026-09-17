"""
Unduh bar dari bursa dan simpan ke file Parquet.

Dijalankan manual, sekali di awal proyek lalu sesekali untuk menambah data baru.

Cara pakai:
    python -m src.data.fetch_history "BTC/USDT:USDT" --days 540
    python -m src.data.fetch_history "BTC/USDT" --days 3000 --timeframe 1d --exchange binance

Catatan: bursa membatasi jumlah bar per permintaan (biasanya 500-1500),
jadi data ditarik sepotong-sepotong dalam loop, bukan sekali ambil.

PENTING soal --timeframe: kalau butuh window PANJANG (harian, mingguan),
UNDUH LANGSUNG di resolusi itu (--timeframe 1d), JANGAN unduh bar 1-menit
bertahun-tahun lalu diresample manual -- itu jutaan baris data yang gak
perlu, boros waktu unduh dan storage, padahal ujungnya cuma dipakai jadi
ribuan baris harian.
"""

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")

# Berapa bar yang diminta per satu panggilan API.
# Batas tiap bursa beda; 1000 aman untuk sebagian besar.
LIMIT_PER_CALL = 1000

# Peta timeframe ccxt -> alias frekuensi pandas, dipakai report_gaps()
# supaya bisa cek bar yang hilang di resolusi APAPUN, bukan cuma menit.
_TIMEFRAME_TO_PANDAS_FREQ = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h",
    "1d": "1D", "1w": "1W",
}


def _symbol_to_filename(symbol: str, timeframe: str) -> str:
    """
    timeframe == "1m" (default lama) -> pola LAMA, TIDAK BERUBAH:
        "BTC/USDT:USDT" -> "BTC-USDT-USDT.parquet"
        Ini WAJIB sama persis dengan src/data/store.py:_symbol_to_filename()
        default (timeframe=None) -- supaya load_bars(symbol) tanpa
        timeframe tetap menemukan file yang sudah ada.

    timeframe lain (mis. "1d") -> pola BARU:
        "BTC/USDT" -> "BTC-USDT_1d.parquet"
        Cocok dengan load_bars(symbol, timeframe="1d") di store.py.
    """
    base = symbol.replace("/", "-").replace(":", "-")
    if timeframe == "1m":
        return base + ".parquet"
    return f"{base}_{timeframe}.parquet"


def fetch_ohlcv(
    symbol: str,
    days: int = 540,
    timeframe: str = "1m",
    exchange_id: str = "binanceusdm",
    testnet: bool = False,
) -> pd.DataFrame:
    """
    Tarik bar dari bursa, dari `days` hari lalu sampai sekarang.

    Mengembalikan DataFrame dengan kolom:
        timestamp (index, UTC), open, high, low, close, volume
    """
    import ccxt  # diimpor di dalam fungsi supaya file ini bisa dites tanpa ccxt

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    if testnet:
        exchange.set_sandbox_mode(True)

    since = int(
        (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    rows: list[list] = []
    cursor = since

    while cursor < now_ms:
        batch = exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=cursor, limit=LIMIT_PER_CALL
        )
        if not batch:
            break

        rows.extend(batch)

        # Maju ke bar berikutnya setelah bar terakhir yang didapat.
        last_ts = batch[-1][0]
        if last_ts <= cursor:
            break  # bursa tidak maju — hentikan supaya tidak loop selamanya
        cursor = last_ts + 1

        done = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        print(f"  ...sampai {done:%Y-%m-%d %H:%M} ({len(rows):,} bar)", end="\r")

        # enableRateLimit sudah menahan laju, ini jeda tambahan untuk aman.
        time.sleep(exchange.rateLimit / 1000)

    print()

    df = pd.DataFrame(
        rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    return df


def report_gaps(df: pd.DataFrame, timeframe: str = "1m") -> pd.DataFrame:
    """
    Cari bar yang hilang dari data, di RESOLUSI timeframe yang diminta
    (bukan cuma menit lagi -- digeneralisasi supaya bekerja untuk bar
    harian/mingguan juga).

    Penting: celah data yang tidak diketahui akan dibaca backtest sebagai
    harga yang melompat, dan bisa memunculkan "keuntungan" palsu.
    """
    freq = _TIMEFRAME_TO_PANDAS_FREQ.get(timeframe)
    if freq is None:
        print(f"  (Peringatan: timeframe '{timeframe}' tidak dikenal untuk cek data hilang -- dilewati.)")
        return pd.DataFrame({"missing_timestamp": []})

    expected = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    missing = expected.difference(df.index)
    return pd.DataFrame({"missing_timestamp": missing})


def save(df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / _symbol_to_filename(symbol, timeframe)
    df.to_parquet(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Unduh bar dari bursa")
    parser.add_argument("symbol", help="contoh: BTC/USDT:USDT")
    parser.add_argument("--days", type=int, default=540, help="default 540 (~18 bulan)")
    parser.add_argument(
        "--timeframe",
        default="1m",
        help="resolusi bar (1m, 5m, 1h, 4h, 1d, 1w, dst) -- default '1m' (perilaku lama). "
        "Untuk horizon panjang (harian/mingguan), UNDUH LANGSUNG di resolusi itu, "
        "jangan unduh 1m bertahun-tahun lalu resample manual.",
    )
    parser.add_argument("--exchange", default="binanceusdm")
    parser.add_argument("--testnet", action="store_true")
    args = parser.parse_args()

    print(f"Mengunduh {args.symbol} dari {args.exchange}, timeframe {args.timeframe}, {args.days} hari terakhir...")
    df = fetch_ohlcv(
        args.symbol, days=args.days, timeframe=args.timeframe,
        exchange_id=args.exchange, testnet=args.testnet,
    )

    if df.empty:
        print("Tidak ada data yang didapat. Cek nama simbol, timeframe, dan nama bursa.")
        return

    gaps = report_gaps(df, timeframe=args.timeframe)
    path = save(df, args.symbol, args.timeframe)

    print(f"\nTersimpan: {path}")
    print(f"  {len(df):,} bar")
    print(f"  {df.index.min():%Y-%m-%d %H:%M} sampai {df.index.max():%Y-%m-%d %H:%M} UTC")
    print(f"  Bar yang hilang: {len(gaps):,} ({len(gaps) / max(len(df), 1) * 100:.3f}%)")

    if len(gaps) / max(len(df), 1) > 0.01:
        print(
            "\n  PERHATIAN: lebih dari 1% data hilang. Periksa dulu sebelum dipakai\n"
            "  backtest — celah data bisa memunculkan hasil yang menyesatkan."
        )


if __name__ == "__main__":
    main()