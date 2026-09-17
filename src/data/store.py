"""
Satu pintu untuk membaca data historis.

Semua lapisan lain membaca bar lewat file ini, bukan langsung dari Parquet.
Alasannya: kalau nanti format penyimpanan diganti (misalnya ke DuckDB),
yang berubah cukup file ini — kode strategi dan backtest tidak tersentuh.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def _symbol_to_filename(symbol: str, timeframe: str | None = None) -> str:
    """
    timeframe=None (default) -> pola LAMA, TIDAK BERUBAH:
        "BTC/USDT:USDT" -> "BTC-USDT-USDT.parquet"
        Ini menjaga kompatibilitas penuh dengan semua file yang sudah
        ada dan semua pemanggilan load_bars(symbol) tanpa timeframe.

    timeframe diisi eksplisit (mis. "1d") -> pola BARU:
        "BTC/USDT" -> "BTC-USDT_1d.parquet"
        Dipakai untuk data yang diunduh di resolusi selain 1 menit
        (lihat src/data/fetch_history.py) -- supaya tidak menimpa file
        1 menit yang sudah ada di simbol yang sama.
    """
    base = symbol.replace("/", "-").replace(":", "-")
    if timeframe is None:
        return base + ".parquet"
    return f"{base}_{timeframe}.parquet"


def load_bars(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    timeframe: str | None = None,
) -> pd.DataFrame:
    """
    Baca bar untuk satu instrumen.

    timeframe=None (default, PERILAKU LAMA TIDAK BERUBAH): baca file
    1-menit seperti sebelumnya -- semua pemanggilan load_bars(symbol)
    yang sudah ada di proyek ini TIDAK PERLU diubah, tetap bekerja
    persis sama.

    timeframe diisi eksplisit (mis. "1d", "1h"): baca file yang
    diunduh di resolusi itu lewat
    `python -m src.data.fetch_history SYMBOL --timeframe 1d` --
    dipakai untuk backtest horizon panjang (harian/mingguan) tanpa
    perlu resample dari data 1-menit yang jutaan baris.

    start / end opsional, format bebas yang dikenali pandas ("2025-01-01").
    Dipakai untuk memisahkan data penyetelan parameter dari data uji.

    Mengembalikan DataFrame ber-index waktu UTC, terurut naik.
    """
    path = RAW_DIR / _symbol_to_filename(symbol, timeframe)
    if not path.exists():
        tf_hint = f" --timeframe {timeframe}" if timeframe else ""
        raise FileNotFoundError(
            f"Data untuk {symbol} (timeframe={timeframe or '1m default'}) belum ada di {path}.\n"
            f"Jalankan dulu: python -m src.data.fetch_history '{symbol}'{tf_hint}"
        )

    df = pd.read_parquet(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom hilang di {path}: {missing}")

    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]

    return df.sort_index()


def available_symbols() -> list[str]:
    """Daftar instrumen yang datanya sudah diunduh."""
    if not RAW_DIR.exists():
        return []
    return sorted(p.stem for p in RAW_DIR.glob("*.parquet"))


def split_train_test(
    df: pd.DataFrame, test_fraction: float = 0.33
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bagi data menjadi bagian penyetelan (lama) dan bagian uji (baru).

    Pembagiannya BERURUTAN WAKTU, tidak diacak. Mengacak data harga adalah
    kesalahan klasik: potongan masa depan akan bocor ke bagian penyetelan,
    dan hasilnya terlihat jauh lebih bagus dari kenyataan.

    Bagian uji tidak boleh disentuh sampai parameter dikunci.
    """
    split_at = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_at], df.iloc[split_at:]