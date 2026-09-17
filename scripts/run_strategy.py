"""
scripts/run_strategy.py

LANGKAH 4-5 dari rencana: jalankan satu strategi dari alpha bank APA
ADANYA (parameter default, tidak disetel) di panel sungguhan dengan
ongkos sungguhan, lalu cetak verdict gate. Menggantikan
run_baseline_momentum.py -- sekarang generik untuk ketiga strategi
(momentum, reversion, funding_carry), bukan strategi tunggal, supaya
loading panel tidak diduplikasi tiga kali.

PENTING -- n_trials HARUS DIHITUNG MANUAL OLEH ANDA:
Skrip ini TIDAK melacak berapa kali Anda sudah menjalankannya dengan
parameter berbeda. Tiap kali Anda mengubah lookback/rebalance/
top_fraction dan menjalankan lagi, itu SATU PERCOBAAN LAGI --
tambahkan ke --n-trials di panggilan berikutnya. Lihat penjelasan
deflated_t_threshold() di portfolio_metrics.py kalau lupa kenapa ini
penting: t-stat yang sama bisa "lolos" di n_trials=1 tapi "gagal" di
n_trials=25, dan keduanya benar tergantung berapa kali Anda sebenarnya
mencoba.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.backtest.cost_adapter import build_cost_spec
from src.backtest.portfolio_engine import run_portfolio_backtest
from src.backtest.portfolio_metrics import format_summary, gate, summarize
from src.core.cost_model import CRYPTO_PERP, CRYPTO_SPOT
from src.strategy.funding_carry import FundingCarryParams, FundingCarryStrategy
from src.strategy.xs_momentum import XsMomentumParams, XsMomentumStrategy
from src.strategy.xs_reversion import XsReversionParams, XsReversionStrategy

STRATEGIES = {
    "momentum": (XsMomentumStrategy, XsMomentumParams),
    "reversion": (XsReversionStrategy, XsReversionParams),
    "funding_carry": (FundingCarryStrategy, FundingCarryParams),
}


def _diagnose_panel(close: pd.DataFrame) -> None:
    n_bars, n_assets = close.shape
    print(f"Panel: {n_bars} bar x {n_assets} aset "
          f"({close.index[0].date()} s.d. {close.index[-1].date()})")

    fully_nan = close.columns[close.isna().all()].tolist()
    if fully_nan:
        print(f"  PERINGATAN: {len(fully_nan)} kolom seluruhnya NaN "
              f"(kemungkinan gagal tarik data): {fully_nan}")

    daily_ret = close.pct_change()
    vol = daily_ret.std()
    near_zero_vol = vol[vol < 0.001].index.tolist()
    if near_zero_vol:
        print(f"  CATATAN: {len(near_zero_vol)} aset volatilitas nyaris nol "
              f"(kemungkinan stablecoin): {near_zero_vol}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", choices=list(STRATEGIES), required=True)
    p.add_argument("--market", choices=["spot", "futures"], default="futures")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--lookback", type=int, default=None,
                    help="override lookback_bars (momentum/reversion) atau "
                         "smooth_bars (funding_carry); default pakai bawaan strategi")
    p.add_argument("--rebalance", type=int, default=None)
    p.add_argument("--top-fraction", type=float, default=None)
    p.add_argument("--n-trials", type=int, default=1,
                    help="WAJIB Anda hitung manual -- lihat catatan di kepala file")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    close = pd.read_parquet(data_dir / f"{args.market}_close.parquet")
    print(f"Memuat {data_dir / f'{args.market}_close.parquet'}")
    _diagnose_panel(close)

    extra = None
    if args.strategy == "funding_carry":
        if args.market != "futures":
            raise SystemExit(
                "funding_carry butuh data futures (funding tidak ada di spot) "
                "-- pakai --market futures."
            )
        funding_path = data_dir / "futures_funding_bps.parquet"
        funding = pd.read_parquet(funding_path)
        print(f"Memuat {funding_path}: {funding.shape}\n")
        extra = {"funding": funding}

    cfg = CRYPTO_SPOT if args.market == "spot" else CRYPTO_PERP
    cost_spec = build_cost_spec(cfg, entry_is_maker=False, exit_is_maker=False)
    print(f"Ongkos ({args.market}, taker/taker): "
          f"per_side={cost_spec.per_side_bps:.2f}bps, "
          f"financing={cost_spec.financing_bps_per_day:.3f}bps/hari\n")

    strategy_cls, params_cls = STRATEGIES[args.strategy]
    field_name = "smooth_bars" if args.strategy == "funding_carry" else "lookback_bars"
    kwargs = {}
    if args.lookback is not None:
        kwargs[field_name] = args.lookback
    if args.rebalance is not None:
        kwargs["rebalance_bars"] = args.rebalance
    if args.top_fraction is not None:
        kwargs["top_fraction"] = args.top_fraction
    strategy = strategy_cls(params_cls(**kwargs))
    tuned_note = "PARAMETER DEFAULT, TIDAK DISETEL" if not kwargs else f"DISETEL: {kwargs}"
    print(f"Strategi: {strategy.describe()} -- {tuned_note}\n")

    result = run_portfolio_backtest(close, strategy, cost_spec, bar_minutes=1440, extra=extra)
    summary = summarize(result.net_returns, 1440, result.gross_returns, result.turnover)

    print("=== HASIL ===")
    print(format_summary(summary))

    passed, reasons = gate(summary, n_trials=args.n_trials)
    print(f"\nGATE (n_trials={args.n_trials}): {'LOLOS' if passed else 'GAGAL'}")
    for r in reasons:
        print(f"  - {r}")

    if summary["sharpe_annual"] > 3.0:
        print("\nPERHATIAN: Sharpe > 3 sangat tidak biasa untuk baseline tanpa "
              "disetel. Curigai data sebelum menganggap ini strategi bagus.")


if __name__ == "__main__":
    main()