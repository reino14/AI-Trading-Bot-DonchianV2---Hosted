"""
scripts/market_regime.py

Alat diagnostik terpisah -- BUKAN bagian dari 7 file resmi Hari 2.
Menjawab pertanyaan yang berbeda dari backtest: bukan "apakah strategi
X profit", tapi "pasar ini sebenarnya sedang tren atau sideways?" --
dihitung LANGSUNG dari data harga, bukan ditebak tidak langsung dari
hasil P&L strategi (itu cara yang lemah dan berisiko muter-muter --
P&L strategi dipengaruhi banyak hal, bukan cuma regime pasar semata).

Metrik: Kaufman's Efficiency Ratio (ER), indikator klasik untuk
mengukur "seberapa efisien" harga bergerak ke satu arah.

    ER = |perubahan bersih N bar| / (jumlah |perubahan tiap bar| N bar)

    ER mendekati 1 -> harga bergerak "lurus" ke satu arah (tren kuat,
    sedikit bolak-balik) -- kondisi yang secara teori menguntungkan
    strategi momentum/breakout seperti Donchian.

    ER mendekati 0 -> harga banyak bolak-balik tanpa progres bersih
    (choppy/ranging) -- kondisi yang secara teori menguntungkan
    mean-reversion seperti VWAP, BUKAN breakout.

Cara pakai:
    python -m scripts.market_regime "BTC/USDT:USDT" "ETH/USDT:USDT" --window 120
"""

import argparse

import pandas as pd

from src.backtest.validate import resample_bars, split_out_of_sample
from src.data.store import load_bars


def efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    net_change = (close - close.shift(window)).abs()
    path_length = close.diff().abs().rolling(window).sum()
    return net_change / path_length


def summarize_regime(df: pd.DataFrame, window: int, label: str) -> None:
    if len(df) <= window:
        print(f"  {label}: data tidak cukup untuk hitung ER (window={window})")
        return
    er = efficiency_ratio(df["close"], window).dropna()
    print(f"  {label}  ({len(df):,} bar)")
    print(f"    ER rata-rata                          : {er.mean():.3f}")
    print(f"    ER median                             : {er.median():.3f}")
    print(f"    % waktu ER > 0.5 (cenderung TREN)      : {(er > 0.5).mean() * 100:.1f}%")
    print(f"    % waktu ER < 0.3 (cenderung RANGING)   : {(er < 0.3).mean() * 100:.1f}%")


def run_one(symbol: str, window: int, er_window: int, test_months: int) -> None:
    print(f"\n{'=' * 78}")
    print(f"  {symbol}  (bar {window} menit, Efficiency Ratio window {er_window} bar)")
    print("=" * 78)

    raw = load_bars(symbol)
    bars = resample_bars(raw, window)
    train_df, test_df = split_out_of_sample(bars, test_months=test_months)

    summarize_regime(bars, er_window, "Seluruh periode data")
    print()
    summarize_regime(train_df, er_window, "In-sample (train)")
    print()
    summarize_regime(test_df, er_window, "Out-of-sample (test)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baca kondisi pasar (tren vs ranging) langsung dari data harga, bukan dari hasil backtest"
    )
    parser.add_argument("symbols", nargs="+")
    parser.add_argument("--window", type=int, default=120, help="ukuran bar dalam menit")
    parser.add_argument("--er-window", type=int, default=20, help="jumlah bar untuk hitung Efficiency Ratio")
    parser.add_argument("--test-months", type=int, default=12)
    args = parser.parse_args()

    for symbol in args.symbols:
        try:
            run_one(symbol, args.window, args.er_window, args.test_months)
        except FileNotFoundError as e:
            print(f"\n{e}\n")


if __name__ == "__main__":
    main()