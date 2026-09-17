"""
scripts/diagnose_direction.py

Diagnostik cepat, BUKAN kandidat strategi final: apakah VWAP reversion
gagal karena ARAHNYA salah (seharusnya momentum, ikut penyimpangan --
bukan reversion, lawan penyimpangan), atau karena masalah lain (mis.
exit asimetris, ongkos)?

Cara kerja: VwapReversionStrategy dan cerminnya (VwapMomentumDiagnostic
-- entry dibalik, exit tetap sama) dijalankan berdampingan, di data
yang sama.

SENGAJA dijalankan di data IN-SAMPLE (train) SAJA, bukan out-of-sample.
Ini pertanyaan diagnostik ("ke arah mana bias sebenarnya?"), bukan uji
kelayakan strategi final -- data out-of-sample harus dijaga supaya
tidak "diintip" berulang kali untuk pertanyaan semacam ini, atau kita
diam-diam overfit ke periode 6 bulan itu lewat banyak percobaan. Kandidat
final (apa pun hasil diagnostik ini nanti) baru diuji SEKALI di
out-of-sample lewat scripts/run_backtest.py.

Cara pakai:
    python -m scripts.diagnose_direction "BTC/USDT:USDT" "ETH/USDT:USDT" --window 30 --test-months 6
"""

import argparse

import pandas as pd

from src.backtest.engine import run_backtest, trades_to_dataframe
from src.backtest.metrics import compute_metrics
from src.backtest.validate import resample_bars, split_out_of_sample
from src.core.cost_model import CostConfig
from src.data.store import load_bars
from src.strategy.base import Position, Strategy
from src.strategy.vwap_reversion import VwapReversionParams, VwapReversionStrategy


class VwapMomentumDiagnostic(Strategy):
    """
    Cermin dari VwapReversionStrategy: entry mengikuti arah penyimpangan
    (bertaruh tren berlanjut), bukan melawannya. Exit TETAP SAMA (kembali
    ke VWAP atau timeout) -- supaya perbandingan ini murni soal ARAH
    entry, bukan tercampur dengan perubahan logika exit.

    HANYA untuk diagnostik. Kalau ini yang menang, itu petunjuk kuat
    untuk pindah ke pola momentum/breakout -- BUKAN berarti kelas ini
    sendiri siap dipakai jadi strategi final (exit-nya belum didesain
    ulang untuk logika momentum, cuma dipinjam dari reversion).
    """

    def __init__(self, params: VwapReversionParams | None = None):
        super().__init__(params or VwapReversionParams())

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p: VwapReversionParams = self.params

        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        pv = typical_price * df["volume"]
        rolling_pv = pv.rolling(p.vwap_window_bars, min_periods=p.vwap_window_bars).sum()
        rolling_vol = df["volume"].rolling(p.vwap_window_bars, min_periods=p.vwap_window_bars).sum()
        vwap = rolling_pv / rolling_vol
        deviation_bps = (df["close"] - vwap) / vwap * 10_000

        signals = pd.Series(Position.FLAT, index=df.index, dtype=int)
        position = Position.FLAT
        bars_held = 0

        for i in range(len(df)):
            dev = deviation_bps.iloc[i]
            if pd.isna(dev):
                signals.iloc[i] = Position.FLAT
                continue

            if position == Position.FLAT:
                # DIBALIK dari VwapReversionStrategy: ikut arah, bukan lawan.
                if dev >= p.entry_threshold_bps:
                    position = Position.LONG
                    bars_held = 0
                elif dev <= -p.entry_threshold_bps:
                    position = Position.SHORT
                    bars_held = 0

            elif position == Position.LONG:
                bars_held += 1
                reverted = dev <= 0
                timed_out = bars_held >= p.max_hold_bars
                if reverted or timed_out:
                    position = Position.FLAT
                    bars_held = 0

            elif position == Position.SHORT:
                bars_held += 1
                reverted = dev >= 0
                timed_out = bars_held >= p.max_hold_bars
                if reverted or timed_out:
                    position = Position.FLAT
                    bars_held = 0

            signals.iloc[i] = position

        return signals


def run_one(
    symbol: str, window: int, test_months: int, entry_is_maker: bool, exit_is_maker: bool
) -> None:
    print(f"\n{'=' * 78}")
    print(f"  {symbol}  -- diagnostik arah (data IN-SAMPLE/train SAJA)")
    print("=" * 78)

    raw = load_bars(symbol)
    bars = resample_bars(raw, window)
    train_df, _ = split_out_of_sample(bars, test_months=test_months)

    if len(train_df) == 0:
        print("\n  Data in-sample kosong. Cek --test-months.")
        return

    print(f"  In-sample (train): {len(train_df):,} bar, {train_df.index.min():%Y-%m-%d} s/d {train_df.index.max():%Y-%m-%d}")

    cfg = CostConfig()
    params = VwapReversionParams()

    for label, strat in [
        ("REVERSION (strategi asli, lawan arah penyimpangan)", VwapReversionStrategy(params)),
        ("MOMENTUM  (cermin diagnostik, ikut arah penyimpangan)", VwapMomentumDiagnostic(params)),
    ]:
        trades = run_backtest(
            train_df,
            strat,
            cfg,
            bar_minutes=window,
            entry_is_maker=entry_is_maker,
            exit_is_maker=exit_is_maker,
        )
        m = compute_metrics(trades_to_dataframe(trades))
        print(f"\n  {label}")
        print(f"    Transaksi    : {m.n_trades}")
        print(f"    Win rate     : {m.win_rate:.1%}")
        print(f"    Net P&L rata : {m.mean_net_pnl_bps:.2f} bps")
        print(f"    t-stat       : {m.t_stat:.2f}")
        print(f"    Drawdown     : {m.max_drawdown_pct:.2f}%")

    print("\n  CARA BACA:")
    print("  - Kalau MOMENTUM positif & REVERSION negatif (atau sebaliknya):")
    print("    petunjuk kuat arah mana yang cocok untuk instrumen/window ini")
    print("    di periode in-sample -- pindah desain ke pola yang menang.")
    print("  - Kalau DUA-DUANYA negatif: masalahnya kemungkinan bukan cuma")
    print("    arah -- exit asimetris atau ongkos yang perlu dibenahi dulu,")
    print("    sebelum mencoba pola sama sekali baru.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostik arah: reversion vs momentum, di data in-sample saja"
    )
    parser.add_argument("symbols", nargs="+")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument(
        "--test-months",
        type=int,
        default=6,
        help="HARUS sama dengan run_backtest.py, supaya batas train/test konsisten",
    )
    parser.add_argument("--maker-entry", action="store_true")
    parser.add_argument("--maker-exit", action="store_true")
    args = parser.parse_args()

    for symbol in args.symbols:
        try:
            run_one(symbol, args.window, args.test_months, args.maker_entry, args.maker_exit)
        except FileNotFoundError as e:
            print(f"\n{e}\n")


if __name__ == "__main__":
    main()