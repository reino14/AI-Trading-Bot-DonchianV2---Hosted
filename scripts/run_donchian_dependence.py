"""
scripts/run_donchian_dependence.py

Jalankan satu lookback: sinyal -> return -> trade -> runs test ->
bandingkan strategi asli vs versi "cuma masuk setelah kalah".

CATATAN DATA: video aslinya backtest 2018-2022 (~4,5 tahun) di 1H.
Kita cuma punya ~900 hari (2,5 tahun) 1H BTC -- hasil di sini TIDAK
akan sekuat demonstrasi videonya, terutama untuk lookback besar
(sedikit trade). Ini bukan salah kode, itu keterbatasan data yang ada.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.strategy.donchian_close import compute_donchian_signal
from src.strategy.trade_dependence import (
    extract_trades,
    filter_by_previous_trade,
    runs_test,
    summarize_trades,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--lookback", type=int, default=72, help="72 = 3 hari di 1H, sesuai pilihan video")
    p.add_argument("--cost-bps", type=float, default=10.0, help="ongkos per sisi -- default cocok CRYPTO_SPOT taker")
    args = p.parse_args()

    print(f"Memuat {args.symbol} {args.interval}...")
    df = pd.read_parquet(Path(args.data_dir) / "klines" / args.symbol / f"{args.interval}.parquet")
    df = df.set_index("open_time")
    df.columns = [c.lower() for c in df.columns]
    close = df["close"]
    print(f"  {len(close):,} bar ({close.index[0].date()} s.d. {close.index[-1].date()})\n")

    print(f"Lookback = {args.lookback}")
    signal = compute_donchian_signal(close, lookback=args.lookback)
    trades = extract_trades(signal, close, cost_bps_per_unit=args.cost_bps)
    print(f"  Ongkos per unit perubahan sinyal: {args.cost_bps} bps "
          f"(reversal = 2 unit, entry pertama dari flat = 1 unit)")

    print(f"\n{'='*70}\nSTRATEGI ASLI (semua trade)\n{'='*70}")
    summary = summarize_trades(trades)
    for k, v in summary.items():
        print(f"  {k:20s} = {v:.4f}" if isinstance(v, float) else f"  {k:20s} = {v}")

    if not trades.empty:
        total_gross = trades["gross_return_pct"].sum()
        total_cost = trades["cost_pct"].sum()
        print(f"\n  Total return KOTOR (sebelum ongkos) = {total_gross:.4f}")
        print(f"  Total ongkos                        = {total_cost:.4f}")
        print(f"  Total return BERSIH (sudah dikurangi)= {total_gross - total_cost:.4f}")

    signs = trades["return_pct"].apply(lambda r: 1 if r > 0 else (-1 if r < 0 else 0)).to_numpy()
    nonzero = signs[signs != 0]
    z = runs_test(nonzero)
    print(f"\n  Runs test z-score = {z:.4f}")
    print("  (positif = pemenang/pecundang cenderung berselang-seling lebih dari acak;")
    print("   ini dasar statistik untuk aturan 'cuma masuk setelah kalah')")

    print(f"\n{'='*70}\nFILTER: cuma masuk kalau trade SEBELUMNYA KALAH\n{'='*70}")
    mask_after_loser = filter_by_previous_trade(trades, only_after="loser")
    trades_after_loser = trades[mask_after_loser]
    summary_loser = summarize_trades(trades_after_loser)
    for k, v in summary_loser.items():
        print(f"  {k:20s} = {v:.4f}" if isinstance(v, float) else f"  {k:20s} = {v}")

    print(f"\n{'='*70}\nPEMBANDING: cuma masuk kalau trade SEBELUMNYA MENANG\n{'='*70}")
    mask_after_winner = filter_by_previous_trade(trades, only_after="winner")
    trades_after_winner = trades[mask_after_winner]
    summary_winner = summarize_trades(trades_after_winner)
    for k, v in summary_winner.items():
        print(f"  {k:20s} = {v:.4f}" if isinstance(v, float) else f"  {k:20s} = {v}")

    if summary.get("n_trades", 0) < 30:
        print("\nPERHATIAN: n_trades < 30 -- terlalu sedikit untuk kesimpulan apa pun, "
              "apalagi setelah difilter jadi lebih sedikit lagi. Ini konsekuensi "
              "langsung dari rentang data yang lebih pendek dari video aslinya.")


if __name__ == "__main__":
    main()