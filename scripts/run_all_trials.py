"""
scripts/run_all_trials.py

Satu file untuk jalankan SEMUA kombinasi trial yang sudah kita sepakati
sekaligus -- menggantikan tempel-tempel perintah satu baris tiap kali
menambah kombinasi baru. Load data SEKALI, pakai ulang untuk seluruh
grid trial, cetak tabel perbandingan di akhir.

n_trials dihitung OTOMATIS dari jumlah kombinasi yang benar-benar
dijalankan di grid ini -- kalau Anda tambah kombinasi baru ke TRIALS
di bawah, ambang terkoreksi ikut naik otomatis, tidak perlu dihitung
manual lagi.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.core.cost_model import CRYPTO_PERP
from src.microstructure.absorption import calibrate_absorption_thresholds, detect_absorption
from src.microstructure.market_structure import build_structure_series
from src.microstructure.signal_generator import (
    compute_participation_threshold,
    generate_setups,
)
from src.microstructure.trade_simulator import simulate_trades

_EULER_MASCHERONI = 0.5772156649015329


def deflated_t_threshold(n_trials: int, base_t: float = 2.0) -> float:
    """
    Disalin dari src/backtest/portfolio_metrics.py proyek "Traits 2" --
    lihat penjelasan lengkap di sana soal kenapa ambang t naik seiring
    jumlah percobaan (Bailey & Lopez de Prado, deflated Sharpe ratio).
    """
    if n_trials <= 1:
        return base_t
    nd = NormalDist()
    alpha_single = 2.0 * (1.0 - nd.cdf(base_t))
    alpha_adj = 1.0 - (1.0 - alpha_single) ** (1.0 / n_trials)
    return nd.inv_cdf(1.0 - alpha_adj / 2.0)


def load_context(data_dir: Path, symbol: str):
    print(f"Memuat data untuk {symbol}...")
    footprint = pd.read_parquet(data_dir / "tick" / f"{symbol.lower().replace('usdt','')}_footprint_5min.parquet")

    abs_th = calibrate_absorption_thresholds(footprint)
    footprint["is_absorption"] = detect_absorption(footprint, abs_th)
    print(f"  Absorption: volume>={abs_th.volume_threshold:.1f}, "
          f"|delta_pct|<={abs_th.delta_pct_threshold:.3f}")

    part_th = compute_participation_threshold(footprint, window_bars=12, percentile=0.25)
    print(f"  Ambang partisipasi (kuartil bawah rolling 1 jam): {part_th:.1f}")

    df_1h = pd.read_parquet(data_dir / "klines" / symbol / "1h.parquet").set_index("open_time")
    df_1h.columns = [c.lower() for c in df_1h.columns]
    structure_1h = build_structure_series(df_1h, lookback=3)

    df_4h = pd.read_parquet(data_dir / "klines" / symbol / "4h.parquet").set_index("open_time")
    df_4h.columns = [c.lower() for c in df_4h.columns]
    structure_4h = build_structure_series(df_4h, lookback=3)

    daily_profile = pd.read_parquet(data_dir / "tick" / f"{symbol.lower().replace('usdt','')}_daily_profile.parquet")

    print(f"  Footprint: {len(footprint):,} bar\n")
    return footprint, structure_1h, structure_4h, daily_profile, part_th


def run_trial(label, footprint, structure_1h, structure_4h, daily_profile, **kwargs):
    setups = generate_setups(footprint, structure_1h, structure_4h, daily_profile, **kwargs)
    trades = simulate_trades(setups, footprint, CRYPTO_PERP, bar_minutes=5, max_holding_bars=288)

    row = {"trial": label, "n_setups": len(setups), "n": len(trades)}
    if len(trades) < 2:
        row.update(win_rate=float("nan"), mean_bps=float("nan"),
                    total_bps=float("nan"), t_stat=float("nan"), best_pct=float("nan"))
        return row, trades

    mean = trades["net_pnl_bps"].mean()
    std = trades["net_pnl_bps"].std(ddof=1)
    total = trades["net_pnl_bps"].sum()
    t_stat = mean / (std / math.sqrt(len(trades))) if std > 0 else float("nan")
    best = trades["net_pnl_bps"].max()
    best_pct = best / total if total != 0 else float("nan")

    row.update(
        win_rate=(trades["net_pnl_bps"] > 0).mean(),
        mean_bps=mean, total_bps=total, t_stat=t_stat, best_pct=best_pct,
    )
    return row, trades


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--verbose", action="store_true", help="cetak daftar trade lengkap per trial")
    args = p.parse_args()

    footprint, structure_1h, structure_4h, daily_profile, part_th = load_context(
        Path(args.data_dir), args.symbol
    )

    # GRID TRIAL -- tambah/kurangi kombinasi di sini, n_trials otomatis
    # menyesuaikan. Tiap baris SATU HIPOTESIS TERPISAH -- lihat catatan
    # n_trials di signal_generator.py.
    TRIALS = []
    for target_mode in ("prev_poc", "swing"):
        for two_attempts in (False, True):
            for use_participation in (False, True):
                label = f"target={target_mode}"
                kwargs = {"target_mode": target_mode, "require_two_attempts": two_attempts}
                if two_attempts:
                    label += " +2attempt"
                if use_participation:
                    label += " +partisipasi"
                    kwargs["min_avg_volume"] = part_th
                TRIALS.append((label, kwargs))

    n_trials = len(TRIALS)
    threshold = deflated_t_threshold(n_trials)
    print(f"Menjalankan {n_trials} trial. Ambang t terkoreksi: ±{threshold:.2f}\n")

    results = []
    all_trades = {}
    for label, kwargs in TRIALS:
        row, trades = run_trial(label, footprint, structure_1h, structure_4h, daily_profile, **kwargs)
        results.append(row)
        all_trades[label] = trades
        if args.verbose and not trades.empty:
            print(f"\n--- {label} ---")
            cols = ["direction", "exit_reason", "hold_bars", "gross_pnl_bps", "net_pnl_bps"]
            print(trades[cols].sort_values("net_pnl_bps", ascending=False).to_string())

    summary = pd.DataFrame(results)
    summary["signifikan?"] = summary["t_stat"].abs() >= threshold

    print("\n" + "=" * 100)
    print(f"RINGKASAN {n_trials} TRIAL -- ambang t terkoreksi: ±{threshold:.2f}")
    print("=" * 100)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(summary.to_string(index=False))

    n_small_sample = (summary["n"] < 30).sum()
    if n_small_sample:
        print(f"\nPERHATIAN: {n_small_sample} trial punya n<30 -- t-stat-nya TIDAK BISA "
              f"dipercaya berapa pun nilainya, terlepas dari tanda plus/minus. Lihat "
              f"kolom best_pct: kalau angkanya jauh di atas 100% atau negatif ekstrem, "
              f"itu tanda satu trade mendominasi seluruh hasil.")


if __name__ == "__main__":
    main()