"""
scripts/run_donchian_capital_backtest.py

Backtest modal awal -> modal akhir untuk strategi Donchian close-only.

Konfigurasi default:
- Symbol          : BTCUSDT
- Timeframe       : 5 menit
- Periode uji     : 7 hari terakhir yang tersedia di parquet
- Lookback        : 150 candle
- Modal awal      : Rp10.000.000
- Position sizing : 100%, dibandingkan dengan fraksi konservatif 20%

Pada timeframe 5 menit:
- 1 hari idealnya memiliki 288 candle
- 7 hari idealnya memiliki 2.016 candle
- Lookback 150 mewakili sekitar 12,5 jam

PERINGATAN:
Periode 7 hari terlalu pendek untuk menyimpulkan bahwa strategi profitable
atau siap digunakan secara live.

Pengujian ini lebih tepat digunakan untuk:
1. Memastikan strategi berjalan pada timeframe kecil.
2. Melihat frekuensi sinyal dan trade.
3. Memeriksa biaya transaksi.
4. Memeriksa apakah konfigurasi menghasilkan cukup trade.

Setelah konfigurasi awal ditemukan, strategi sebaiknya divalidasi kembali
menggunakan periode yang lebih panjang dan beberapa kondisi pasar.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from src.microstructure.performance_metrics import summarize_performance
from src.strategy.donchian_close import compute_donchian_signal
from src.strategy.trade_dependence import extract_trades


INTERVAL_TO_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1_440,
}


def parse_arguments() -> argparse.Namespace:
    """Membaca argument command line."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Symbol yang diuji. Default: BTCUSDT",
    )

    parser.add_argument(
        "--interval",
        default="5m",
        help="Timeframe parquet. Default: 5m",
    )

    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Direktori utama data. Default: data/raw",
    )

    parser.add_argument(
        "--days",
        type=float,
        default=7.0,
        help=(
            "Jumlah hari terakhir dari parquet yang digunakan. "
            "Gunakan 0 untuk memakai seluruh data. Default: 7"
        ),
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=150,
        help="Jumlah candle untuk Donchian lookback. Default: 150",
    )

    parser.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help=(
            "Estimasi biaya transaksi dalam basis point per unit "
            "sesuai kontrak extract_trades(). Default: 10 bps"
        ),
    )

    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10_000_000,
        help="Modal awal simulasi. Default: 10000000",
    )

    parser.add_argument(
        "--position-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraksi modal yang digunakan per trade. "
            "1.0 berarti 100 persen. Default: 1.0"
        ),
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Memastikan nilai argument masuk akal."""

    if args.days < 0:
        raise ValueError(
            "--days tidak boleh negatif. "
            "Gunakan 0 untuk memakai seluruh data."
        )

    if args.lookback < 2:
        raise ValueError("--lookback minimal 2 candle.")

    if args.cost_bps < 0:
        raise ValueError("--cost-bps tidak boleh negatif.")

    if args.initial_capital <= 0:
        raise ValueError("--initial-capital harus lebih besar dari 0.")

    if not 0 < args.position_fraction <= 1:
        raise ValueError(
            "--position-fraction harus lebih besar dari 0 "
            "dan maksimal 1.0."
        )


def normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Memastikan data memiliki DatetimeIndex bernama open_time.

    Mendukung:
    1. open_time sebagai kolom.
    2. open_time sebagai index.
    3. DatetimeIndex tanpa nama.
    """

    result = df.copy()

    result.columns = [
        str(column).lower()
        for column in result.columns
    ]

    if "open_time" in result.columns:
        result["open_time"] = pd.to_datetime(
            result["open_time"],
            errors="coerce",
            utc=True,
        )
        result = result.set_index("open_time")

    elif result.index.name == "open_time":
        result.index = pd.to_datetime(
            result.index,
            errors="coerce",
            utc=True,
        )

    elif isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(
            result.index,
            errors="coerce",
            utc=True,
        )
        result.index.name = "open_time"

    else:
        raise ValueError(
            "Parquet tidak memiliki kolom atau DatetimeIndex "
            "'open_time'."
        )

    # Hapus timestamp rusak dan urutkan secara kronologis.
    result = result.loc[~result.index.isna()]
    result = result.sort_index()

    # Jika timestamp identik, gunakan bar terakhir.
    result = result.loc[
        ~result.index.duplicated(keep="last")
    ]

    return result


def load_market_data(
    data_dir: str,
    symbol: str,
    interval: str,
) -> tuple[pd.DataFrame, Path]:
    """Memuat dan membersihkan parquet."""

    parquet_path = (
        Path(data_dir)
        / "klines"
        / symbol
        / f"{interval}.parquet"
    )

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"File parquet tidak ditemukan:\n{parquet_path}\n\n"
            f"Pastikan data timeframe {interval} sudah tersedia."
        )

    df = pd.read_parquet(parquet_path)

    if df.empty:
        raise ValueError(
            f"Parquet kosong: {parquet_path}"
        )

    df = normalize_datetime_index(df)

    if "close" not in df.columns:
        raise ValueError(
            f"Kolom 'close' tidak ditemukan dalam {parquet_path}.\n"
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    # Pastikan close berupa angka.
    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    df = df.dropna(subset=["close"])

    if df.empty:
        raise ValueError(
            "Tidak ada data close valid setelah "
            "pembersihan parquet."
        )

    return df, parquet_path


def filter_recent_period(
    df: pd.DataFrame,
    days: float,
) -> pd.DataFrame:
    """
    Membatasi data ke sejumlah hari terakhir.

    days = 0 berarti seluruh data digunakan.
    """

    if days == 0:
        return df.copy()

    end_time = df.index.max()
    start_time = end_time - pd.Timedelta(days=days)

    filtered = df.loc[
        (df.index > start_time)
        & (df.index <= end_time)
    ].copy()

    return filtered


def interval_minutes(interval: str) -> Optional[int]:
    """Mengubah string interval menjadi menit jika dikenali."""

    return INTERVAL_TO_MINUTES.get(interval.lower())


def format_duration(duration: pd.Timedelta) -> str:
    """Mengubah Timedelta menjadi teks hari, jam, dan menit."""

    total_minutes = int(
        duration.total_seconds() // 60
    )

    days, remaining_minutes = divmod(
        total_minutes,
        1_440,
    )

    hours, minutes = divmod(
        remaining_minutes,
        60,
    )

    parts: list[str] = []

    if days:
        parts.append(f"{days} hari")

    if hours:
        parts.append(f"{hours} jam")

    if minutes or not parts:
        parts.append(f"{minutes} menit")

    return " ".join(parts)


def print_dataset_information(
    df_full: pd.DataFrame,
    df_test: pd.DataFrame,
    parquet_path: Path,
    interval: str,
    days: float,
    lookback: int,
) -> None:
    """Mencetak informasi data agar periode pengujian transparan."""

    minutes = interval_minutes(interval)

    full_duration = (
        df_full.index.max()
        - df_full.index.min()
    )

    test_duration = (
        df_test.index.max()
        - df_test.index.min()
    )

    print("\n" + "=" * 70)
    print("INFORMASI DATA BACKTEST")
    print("=" * 70)

    print(f"  File parquet        = {parquet_path}")
    print(f"  Timeframe           = {interval}")
    print(f"  Total data parquet  = {len(df_full):,} bar")
    print(f"  Awal seluruh data   = {df_full.index.min()}")
    print(f"  Akhir seluruh data  = {df_full.index.max()}")
    print(
        f"  Durasi seluruh data = "
        f"{format_duration(full_duration)}"
    )

    print()

    filter_text = (
        "Semua data"
        if days == 0
        else f"{days:g} hari terakhir"
    )

    print(f"  Filter hari         = {filter_text}")
    print(f"  Awal periode uji    = {df_test.index.min()}")
    print(f"  Akhir periode uji   = {df_test.index.max()}")
    print(
        f"  Durasi aktual uji   = "
        f"{format_duration(test_duration)}"
    )
    print(f"  Bar periode uji     = {len(df_test):,}")

    if minutes is not None:
        lookback_minutes = lookback * minutes
        lookback_duration = pd.Timedelta(
            minutes=lookback_minutes
        )

        print(f"  Lookback            = {lookback} candle")
        print(
            f"  Durasi lookback     = "
            f"{format_duration(lookback_duration)}"
        )

        if days > 0:
            expected_bars = round(
                days * 24 * 60 / minutes
            )

            data_coverage = (
                len(df_test) / expected_bars
                if expected_bars > 0
                else float("nan")
            )

            print(
                f"  Bar ideal periode   = "
                f"{expected_bars:,}"
            )

            print(
                f"  Cakupan data        = "
                f"{data_coverage:.1%}"
            )

            if len(df_test) < expected_bars * 0.95:
                print(
                    "  PERINGATAN           = Jumlah candle jauh "
                    "di bawah jumlah ideal. Periksa kemungkinan "
                    "gap data."
                )

    else:
        print(f"  Lookback            = {lookback} candle")
        print(
            "  Durasi lookback     = Tidak dihitung karena "
            "interval belum dikenali."
        )

    print()


def get_datetime_series(
    trades: pd.DataFrame,
    candidates: list[str],
) -> Optional[pd.Series]:
    """
    Mencari kolom waktu berdasarkan beberapa nama kandidat.
    """

    for column in candidates:
        if column not in trades.columns:
            continue

        converted = pd.to_datetime(
            trades[column],
            errors="coerce",
            utc=True,
        )

        if converted.notna().any():
            return converted

    return None


def print_trade_timing_statistics(
    trades: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> None:
    """Mencetak frekuensi entry dan holding period."""

    duration_days = (
        test_end - test_start
    ).total_seconds() / 86_400

    n_trades = len(trades)

    print("=" * 70)
    print("FREKUENSI DAN DURASI TRADE")
    print("=" * 70)

    print(
        f"  Jumlah trade selesai       = {n_trades}"
    )

    if duration_days > 0:
        trades_per_day = n_trades / duration_days
        trades_per_week = trades_per_day * 7

        print(
            f"  Trade selesai per hari     = "
            f"{trades_per_day:.2f}"
        )

        print(
            f"  Estimasi trade per minggu  = "
            f"{trades_per_week:.2f}"
        )

        if n_trades > 0:
            calendar_interval = (
                duration_days / n_trades
            )

            print(
                f"  Rasio kalender per trade   = "
                f"{calendar_interval:.2f} hari"
            )

    entry_times = get_datetime_series(
        trades,
        [
            "entry_time",
            "entry_timestamp",
            "entry_date",
            "open_time",
            "start_time",
        ],
    )

    exit_times = get_datetime_series(
        trades,
        [
            "exit_time",
            "exit_timestamp",
            "exit_date",
            "close_time",
            "end_time",
        ],
    )

    if entry_times is not None:
        valid_entries = (
            entry_times
            .dropna()
            .sort_values()
        )

        if len(valid_entries) >= 2:
            entry_gaps = (
                valid_entries.diff()
                .dropna()
            )

            print(
                f"  Rata-rata jarak entry      = "
                f"{format_duration(entry_gaps.mean())}"
            )

            print(
                f"  Median jarak entry         = "
                f"{format_duration(entry_gaps.median())}"
            )

        else:
            print(
                "  Jarak antar-entry          = "
                "Tidak cukup entry untuk dihitung"
            )

    else:
        print(
            "  Jarak antar-entry          = "
            "Kolom entry time tidak ditemukan"
        )

    if entry_times is not None and exit_times is not None:
        holding_periods = (
            exit_times - entry_times
        )

        holding_periods = holding_periods.loc[
            holding_periods.notna()
            & (
                holding_periods
                >= pd.Timedelta(0)
            )
        ]

        if not holding_periods.empty:
            print(
                f"  Rata-rata holding period   = "
                f"{format_duration(holding_periods.mean())}"
            )

            print(
                f"  Median holding period      = "
                f"{format_duration(holding_periods.median())}"
            )

            print(
                f"  Holding period terpendek   = "
                f"{format_duration(holding_periods.min())}"
            )

            print(
                f"  Holding period terlama     = "
                f"{format_duration(holding_periods.max())}"
            )

        else:
            print(
                "  Holding period             = "
                "Timestamp entry/exit tidak valid"
            )

    else:
        print(
            "  Holding period             = "
            "Kolom entry/exit time tidak ditemukan"
        )

    print()


def print_performance_summary(
    trades: pd.DataFrame,
    initial_capital: float,
    position_fraction: float,
) -> None:
    """Menghitung dan mencetak metrik performa."""

    trades_for_summary = trades.copy()

    # performance_metrics.py menggunakan net_pnl_bps.
    trades_for_summary["net_pnl_bps"] = (
        trades_for_summary["return_pct"] * 10_000.0
    )

    configurations = [
        (
            f"MODAL (position_fraction={position_fraction:.0%})",
            position_fraction,
        ),
        (
            "PEMBANDING: FRAKSI KONSERVATIF 20%",
            0.2,
        ),
    ]

    for label, fraction in configurations:
        summary = summarize_performance(
            trades_for_summary,
            initial_capital,
            fraction,
        )

        print("=" * 70)
        print(label)
        print("=" * 70)

        print(
            f"  Jumlah trade       = "
            f"{summary['n_trades']}"
        )

        print(
            f"  Win rate           = "
            f"{summary['win_rate']:.1%}"
        )

        print(
            f"  Win/loss ratio     = "
            f"{summary['win_loss_ratio']:.2f}"
        )

        print(
            f"  Sharpe ratio       = "
            f"{summary['sharpe_ratio']:.2f}"
        )

        print(
            f"  Max drawdown       = "
            f"{summary['max_drawdown_pct']:.1%}"
        )

        print(
            f"  Jumlah episode DD  = "
            f"{summary['n_drawdown_episodes']}"
        )

        print(
            f"  Rata recovery      = "
            f"{summary['avg_recovery_days']:.1f} hari"
        )

        print(
            f"  Recovery terlama   = "
            f"{summary['max_recovery_days']:.1f} hari"
        )

        print(
            f"  Masih drawdown?    = "
            f"{summary['still_in_drawdown_at_end']}"
        )

        print(
            f"  Modal awal         = "
            f"Rp{initial_capital:,.0f}"
        )

        print(
            f"  Modal akhir        = "
            f"Rp{summary['final_capital']:,.0f}"
        )

        print(
            f"  Return             = "
            f"{summary['total_return_pct']:.1%}"
        )

        print()


def print_trade_concentration_warning(
    trades: pd.DataFrame,
) -> None:
    """Mencetak kontribusi trade terbaik."""

    returns = pd.to_numeric(
        trades["return_pct"],
        errors="coerce",
    ).dropna()

    if returns.empty:
        print(
            "Tidak dapat menghitung konsentrasi trade."
        )
        return

    best_trade = returns.max()
    arithmetic_total_return = returns.sum()

    print("=" * 70)
    print("KONSENTRASI HASIL TRADE")
    print("=" * 70)

    print(
        f"  Return trade terbesar      = "
        f"{best_trade:.2%}"
    )

    print(
        f"  Total return aritmetis     = "
        f"{arithmetic_total_return:.2%}"
    )

    if arithmetic_total_return > 0:
        best_contribution = (
            best_trade / arithmetic_total_return
        )

        print(
            f"  Kontribusi trade terbesar  = "
            f"{best_contribution:.1%}"
        )

        if best_contribution > 0.40:
            print()
            print(
                "PERINGATAN: Lebih dari 40% total return "
                "aritmetis berasal dari satu trade."
            )

            print(
                "Hasil strategi sangat bergantung pada "
                "kemunculan trade besar tersebut."
            )

    else:
        print(
            "  Kontribusi trade terbesar  = Tidak relevan "
            "karena total return aritmetis tidak positif"
        )

    print()

    print(
        "Catatan: strategi trend-following dapat menang "
        "jarang tetapi menghasilkan beberapa kemenangan besar."
    )

    print(
        "Max drawdown tetap sangat dipengaruhi urutan trade "
        "dan position sizing."
    )

    print(
        "Total return aritmetis berbeda dari return modal "
        "compounded pada performance summary."
    )


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)

    print(
        f"Memuat {args.symbol} {args.interval} "
        f"dari direktori {args.data_dir}..."
    )

    df_full, parquet_path = load_market_data(
        data_dir=args.data_dir,
        symbol=args.symbol,
        interval=args.interval,
    )

    df_test = filter_recent_period(
        df=df_full,
        days=args.days,
    )

    if df_test.empty:
        raise ValueError(
            "Tidak ada candle dalam periode pengujian "
            "yang dipilih."
        )

    if len(df_test) <= args.lookback:
        raise ValueError(
            f"Data periode uji hanya memiliki "
            f"{len(df_test):,} candle, sedangkan lookback "
            f"adalah {args.lookback:,}.\n"
            "Perbesar --days atau perkecil --lookback."
        )

    print_dataset_information(
        df_full=df_full,
        df_test=df_test,
        parquet_path=parquet_path,
        interval=args.interval,
        days=args.days,
        lookback=args.lookback,
    )

    close = df_test["close"].astype(float)

    print("=" * 70)
    print("MENJALANKAN STRATEGI")
    print("=" * 70)

    print(f"  Symbol              = {args.symbol}")
    print(f"  Timeframe           = {args.interval}")
    print(f"  Lookback            = {args.lookback}")
    print(f"  Biaya               = {args.cost_bps:.2f} bps")
    print(
        f"  Modal awal          = "
        f"Rp{args.initial_capital:,.0f}"
    )
    print(
        f"  Position fraction   = "
        f"{args.position_fraction:.0%}"
    )
    print()

    signal = compute_donchian_signal(
        close,
        lookback=args.lookback,
    )

    trades = extract_trades(
        signal,
        close,
        cost_bps_per_unit=args.cost_bps,
    )

    print(f"  Trade selesai       = {len(trades)}")
    print()

    if trades.empty:
        print("=" * 70)
        print("TIDAK ADA TRADE SELESAI")
        print("=" * 70)

        print(
            "Strategi tidak menghasilkan trade selesai "
            "dalam periode data yang dipilih."
        )

        print()
        print("Pilihan eksperimen berikutnya:")
        print("  1. Perbesar periode, misalnya --days 30.")
        print("  2. Perkecil lookback, misalnya --lookback 72.")
        print("  3. Periksa apakah signal pernah berubah posisi.")
        print()

        print(
            "Jangan menyimpulkan strategi rusak hanya "
            "berdasarkan satu minggu tanpa trade."
        )

        return

    if "return_pct" not in trades.columns:
        raise ValueError(
            "Output extract_trades() tidak memiliki "
            "kolom 'return_pct'.\n"
            f"Kolom yang tersedia: {list(trades.columns)}"
        )

    print_trade_timing_statistics(
        trades=trades,
        test_start=df_test.index.min(),
        test_end=df_test.index.max(),
    )

    print_performance_summary(
        trades=trades,
        initial_capital=args.initial_capital,
        position_fraction=args.position_fraction,
    )

    print_trade_concentration_warning(trades)


if __name__ == "__main__":
    main()