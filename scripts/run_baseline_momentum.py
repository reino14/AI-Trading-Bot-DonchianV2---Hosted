"""
scripts/run_baseline_momentum.py

LANGKAH 4 dari rencana: jalankan xs_momentum.py APA ADANYA (parameter
default, tidak disetel) di panel sungguhan dengan ongkos sungguhan.
Tujuannya BUKAN mencari edge -- tujuannya memastikan pipeline data
sampai gate tersambung benar ujung ke ujung, dan mengkalibrasi
ekspektasi Anda terhadap angka yang keluar dari data asli (bukan data
acak seperti di smoke test).

Kalau angkanya kelihatan terlalu bagus (Sharpe > 3), curigai data,
bukan rayakan strateginya -- hampir selalu itu bug universe atau NaN.
Kalau gagal gate, itu hasil sah, BUKAN alasan mengubah parameter di
sini sampai lolos (itu persis overfitting yang proyek ini hindari
sejak awal).

Default market='futures' -- XsMomentumStrategy market-neutral (long
DAN short berimbang) butuh kemampuan short, yang cuma tersedia di
panel futures. Versi spot-only (long-only, tanpa short) adalah
strategi konstruksi berbeda yang belum dibangun -- lihat catatan di
xs_momentum.py soal ALLOWS_SHORT.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.cost_adapter import build_cost_spec
from src.backtest.portfolio_engine import run_portfolio_backtest
from src.backtest.portfolio_metrics import format_summary, gate, summarize
from src.core.cost_model import CRYPTO_PERP, CRYPTO_SPOT
from src.strategy.xs_momentum import XsMomentumParams, XsMomentumStrategy


def _diagnose_panel(close: pd.DataFrame) -> None:
    """
    Pemeriksaan kewarasan SEBELUM backtest jalan -- supaya angka aneh
    di hasil akhir punya penjelasan, bukan misteri.
    """
    n_bars, n_assets = close.shape
    print(f"Panel: {n_bars} bar x {n_assets} aset "
          f"({close.index[0].date()} s.d. {close.index[-1].date()})")

    fully_nan = close.columns[close.isna().all()].tolist()
    if fully_nan:
        print(f"  PERINGATAN: {len(fully_nan)} kolom seluruhnya NaN "
              f"(kemungkinan gagal tarik data): {fully_nan}")

    daily_ret = close.pct_change()
    vol = daily_ret.std()
    near_zero_vol = vol[vol < 0.001].index.tolist()  # <0.1% std harian
    if near_zero_vol:
        print(f"  CATATAN: {len(near_zero_vol)} aset volatilitas nyaris nol "
              f"(kemungkinan stablecoin, ikut disertakan karena Anda pilih "
              f"'pakai apa adanya'): {near_zero_vol}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--market", choices=["spot", "futures"], default="futures")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--lookback", type=int, default=30)
    p.add_argument("--rebalance", type=int, default=1)
    p.add_argument("--top-fraction", type=float, default=0.2)
    p.add_argument("--n-trials", type=int, default=1,
                    help="untuk gate -- 1 karena ini baseline pertama, "
                         "bukan bagian dari sweep")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    close_path = data_dir / f"{args.market}_close.parquet"
    close = pd.read_parquet(close_path)
    print(f"Memuat {close_path}")
    _diagnose_panel(close)

    cfg = CRYPTO_SPOT if args.market == "spot" else CRYPTO_PERP
    cost_spec = build_cost_spec(cfg, entry_is_maker=False, exit_is_maker=False)
    print(f"Ongkos ({args.market}, taker/taker): "
          f"per_side={cost_spec.per_side_bps:.2f}bps, "
          f"financing={cost_spec.financing_bps_per_day:.3f}bps/hari\n")

    strategy = XsMomentumStrategy(
        XsMomentumParams(
            lookback_bars=args.lookback,
            rebalance_bars=args.rebalance,
            top_fraction=args.top_fraction,
        )
    )
    print(f"Strategi: {strategy.describe()} -- PARAMETER DEFAULT, TIDAK DISETEL\n")

    result = run_portfolio_backtest(close, strategy, cost_spec, bar_minutes=1440)
    summary = summarize(result.net_returns, 1440, result.gross_returns, result.turnover)

    print("=== HASIL (baseline, tanpa disetel) ===")
    print(format_summary(summary))

    passed, reasons = gate(summary, n_trials=args.n_trials)
    print(f"\nGATE (n_trials={args.n_trials}): {'LOLOS' if passed else 'GAGAL'}")
    for r in reasons:
        print(f"  - {r}")

    if summary["sharpe_annual"] > 3.0:
        print("\nPERHATIAN: Sharpe > 3 di baseline TANPA disetel sangat tidak "
              "biasa. Curigai data (survivorship, look-ahead, atau bug di "
              "panel) sebelum menganggap ini strategi bagus -- lihat catatan "
              "di kepala file ini.")


if __name__ == "__main__":
    main()