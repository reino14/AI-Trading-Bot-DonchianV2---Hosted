"""
scripts/run_donchian_scan.py

Pemindaian penuh lookback 12-672 -- persis rentang video pertama.
Untuk TIAP lookback: profit factor, Sharpe, Sortino (dari deret return
per-bar yang SUDAH dipotong ongkos -- lihat catatan di bawah kenapa
ini beda dari profit factor yang berbasis per-trade), z-score runs
test, DAN perbandingan filter "masuk setelah kalah" -- LENGKAP DENGAN
pemeriksaan kerapuhan (t-stat + dominasi trade tunggal) di SETIAP
lookback, bukan cuma satu titik seperti sebelumnya.

KENAPA SHARPE/SORTINO DARI RETURN PER-BAR, BUKAN PER-TRADE
------------------------------------------------------------------------
Video bilang sharpe/sortino "annualized dengan mengalikan akar jumlah
bar per tahun" -- itu cuma masuk akal kalau basisnya deret return
PER-BAR (setiap bar text 1h berkontribusi satu observasi), bukan per-
trade (jumlah trade tidak proporsional dengan waktu kalender). Profit
factor TETAP per-trade (definisi videonya eksplisit "sum winning
returns / sum losing returns" dari trade, bukan bar).

OUTPUT: satu file CSV lengkap (semua lookback, semua metrik) untuk
Anda plot sendiri (Excel/Python), PLUS ringkasan teks di terminal
dengan deteksi plateau paling dasar (rentang lookback berturut-turut
dengan profit_factor > 1).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.strategy.donchian_close import compute_donchian_signal, compute_strategy_returns
from src.strategy.trade_dependence import extract_trades, filter_by_previous_trade, runs_test


def _profit_factor(returns: pd.Series) -> float:
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if losses.empty or losses.sum() == 0:
        return float("inf") if not wins.empty else float("nan")
    return float(wins.sum() / abs(losses.sum()))


def scan_one_lookback(close: pd.Series, lookback: int, cost_bps: float, bars_per_year: float) -> dict | None:
    signal = compute_donchian_signal(close, lookback=lookback)
    if signal.dropna().empty:
        return None

    bar_returns = compute_strategy_returns(close, signal, cost_bps_per_unit=cost_bps)
    net = bar_returns["strategy_return_net"].dropna()

    trades = extract_trades(signal, close, cost_bps_per_unit=cost_bps)
    if len(trades) < 2:
        return None

    sharpe = (
        float(net.mean() / net.std() * math.sqrt(bars_per_year))
        if net.std() > 0 else float("nan")
    )
    downside = net[net < 0]
    sortino = (
        float(net.mean() / downside.std() * math.sqrt(bars_per_year))
        if len(downside) > 1 and downside.std() > 0 else float("nan")
    )

    profit_factor = _profit_factor(trades["return_pct"])
    avg_trade_pct = float(trades["return_pct"].mean())
    n_trades = len(trades)

    signs = np.sign(trades["return_pct"].to_numpy())
    signs_nonzero = signs[signs != 0]
    z = runs_test(signs_nonzero) if len(signs_nonzero) >= 2 else float("nan")

    mask_loser = filter_by_previous_trade(trades, only_after="loser")
    filtered = trades[mask_loser]
    if len(filtered) >= 2:
        pf_filtered = _profit_factor(filtered["return_pct"])
        mean_f = filtered["return_pct"].mean()
        std_f = filtered["return_pct"].std()
        t_filtered = float(mean_f / (std_f / math.sqrt(len(filtered)))) if std_f > 0 else float("nan")
        total_f = filtered["return_pct"].sum()
        best_f = filtered["return_pct"].max()
        best_pct_f = float(best_f / total_f) if total_f != 0 else float("nan")
    else:
        pf_filtered = t_filtered = best_pct_f = float("nan")

    return {
        "lookback": lookback, "n_trades": n_trades, "profit_factor": profit_factor,
        "sharpe": sharpe, "sortino": sortino, "avg_trade_pct": avg_trade_pct,
        "runs_z": z, "n_filtered": len(filtered), "pf_filtered": pf_filtered,
        "t_filtered": t_filtered, "filtered_best_trade_pct": best_pct_f,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--lookback-min", type=int, default=12)
    p.add_argument("--lookback-max", type=int, default=672)
    p.add_argument("--step", type=int, default=4, help="lompat tiap N lookback -- 660 titik satu-satu bisa lambat")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--out-csv", default="donchian_scan_results.csv")
    args = p.parse_args()

    print(f"Memuat {args.symbol} {args.interval}...")
    df = pd.read_parquet(Path(args.data_dir) / "klines" / args.symbol / f"{args.interval}.parquet")
    df = df.set_index("open_time")
    df.columns = [c.lower() for c in df.columns]
    close = df["close"]
    bars_per_year = {"1h": 24 * 365, "4h": 6 * 365, "15m": 96 * 365, "5m": 288 * 365}.get(args.interval, 24 * 365)
    print(f"  {len(close):,} bar ({close.index[0].date()} s.d. {close.index[-1].date()})\n")

    lookbacks = list(range(args.lookback_min, args.lookback_max + 1, args.step))
    print(f"Memindai {len(lookbacks)} nilai lookback ({args.lookback_min}-{args.lookback_max}, "
          f"step {args.step})...\n")

    results = []
    for i, lb in enumerate(lookbacks):
        r = scan_one_lookback(close, lb, args.cost_bps, bars_per_year)
        if r is not None:
            results.append(r)
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(lookbacks)} selesai")

    if not results:
        print("Tidak ada lookback yang menghasilkan cukup trade untuk dianalisis.")
        return

    scan_df = pd.DataFrame(results).set_index("lookback")
    scan_df.to_csv(args.out_csv)
    print(f"\nHasil lengkap disimpan ke {args.out_csv} ({len(scan_df)} baris)\n")

    print("=" * 90)
    print("RINGKASAN STRATEGI ASLI (semua lookback)")
    print("=" * 90)
    n_profitable = (scan_df["profit_factor"] > 1.0).sum()
    print(f"  Lookback dengan profit_factor > 1.0 (bersih ongkos): {n_profitable}/{len(scan_df)}")
    print(f"  Sharpe median: {scan_df['sharpe'].median():.3f}")
    print(f"  Sharpe range : {scan_df['sharpe'].min():.3f} s.d. {scan_df['sharpe'].max():.3f}")

    # Deteksi plateau paling dasar: rentang berturut-turut dengan profit_factor>1.
    is_profitable = scan_df["profit_factor"] > 1.0
    plateaus = []
    start = None
    for lb, ok in is_profitable.items():
        if ok and start is None:
            start = lb
        elif not ok and start is not None:
            plateaus.append((start, prev_lb))
            start = None
        prev_lb = lb
    if start is not None:
        plateaus.append((start, prev_lb))
    plateaus.sort(key=lambda pr: pr[1] - pr[0], reverse=True)

    print(f"\n  Plateau terpanjang (profit_factor>1 berturut-turut): ", end="")
    if plateaus:
        best_start, best_end = plateaus[0]
        print(f"lookback {best_start}-{best_end} ({best_end-best_start} langkah)")
    else:
        print("TIDAK ADA -- tidak satu pun rentang lookback konsisten profit_factor>1")

    print("\n" + "=" * 90)
    print("RINGKASAN FILTER 'MASUK SETELAH KALAH' -- DENGAN PEMERIKSAAN KERAPUHAN")
    print("=" * 90)
    # PENTING: t_filtered POSITIF signifikan = aturan Turtle sungguhan
    # membantu. t_filtered NEGATIF signifikan = aturan itu sungguhan
    # MERUGIKAN (bukan "lolos" juga) -- dua hal ini WAJIB dipisah,
    # bukan digabung lewat abs() begitu saja.
    reliably_positive = scan_df[
        (scan_df["t_filtered"] >= 2.0) & (scan_df["filtered_best_trade_pct"].abs() < 0.5)
    ]
    reliably_negative = scan_df[
        (scan_df["t_filtered"] <= -2.0) & (scan_df["filtered_best_trade_pct"].abs() < 0.5)
    ]
    print(f"  Lookback dengan t_filtered POSITIF signifikan (aturan sungguhan membantu, "
          f"bukan kebetulan trade tunggal): {len(reliably_positive)}/{len(scan_df)}")
    print(f"  Lookback dengan t_filtered NEGATIF signifikan (aturan sungguhan MERUGIKAN): "
          f"{len(reliably_negative)}/{len(scan_df)}")

    if not reliably_positive.empty:
        print("\n  Lookback yang MENDUKUNG aturan Turtle (t positif signifikan):")
        print(reliably_positive[["n_filtered", "pf_filtered", "t_filtered", "filtered_best_trade_pct"]].to_string())
    else:
        print("\n  TIDAK ADA lookback dengan bukti POSITIF signifikan untuk aturan Turtle --")
        print("  bukan cuma di lookback=72 yang sudah dicoba, tapi di seluruh rentang yang dipindai.")

    print(f"\n  Runs z-score median (semua lookback): {scan_df['runs_z'].median():.3f}")
    n_significant_z = (scan_df["runs_z"] >= 1.96).sum()
    print(f"  Lookback dengan runs_z >= 1.96 (signifikan sendiri): {n_significant_z}/{len(scan_df)}")


if __name__ == "__main__":
    main()