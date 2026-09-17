"""
scripts/run_sweep.py

HARI 2 -- uji sensitivitas parameter: kriteria lolos keempat. Menggeser
tiap parameter strategi +-20% satu per satu (field lain tetap di nilai
default/terkunci yang sama dengan run_backtest.py), lalu cek apakah
expectancy net masih positif di SEMUA varian.

Mendukung pola strategi apa pun yang terdaftar di scripts/run_backtest.py
(STRATEGIES) -- pilih lewat --strategy, sama seperti run_backtest.py.

Dijalankan di data OUT-OF-SAMPLE yang sama seperti run_backtest.py
(--strategy, --window, --test-months, --param HARUS sama persis),
supaya hasilnya konsisten dan bisa dibandingkan langsung.

Cara pakai:
    python -m scripts.run_sweep "BTC/USDT:USDT" --strategy donchian_breakout --window 240
"""

import argparse

from src.backtest.validate import check_robustness, resample_bars, split_out_of_sample
from scripts.run_backtest import STRATEGIES, build_params
from src.core.cost_model import CostConfig
from src.data.store import load_bars


def run_one(
    symbol: str,
    window: int,
    test_months: int,
    pct: float,
    entry_is_maker: bool,
    exit_is_maker: bool,
    strategy_name: str,
    param_overrides: list[str],
) -> None:
    print(f"\n{'=' * 78}")
    print(f"  {symbol}  -- uji sensitivitas +-{pct:.0%}  (pola: {strategy_name})")
    print("=" * 78)

    raw = load_bars(symbol)
    bars = resample_bars(raw, window)
    _, test_df = split_out_of_sample(bars, test_months=test_months)

    if len(test_df) == 0:
        print("\n  Data out-of-sample kosong. Cek --test-months atau jumlah data yang diunduh.")
        return

    strategy_cls, params_cls = STRATEGIES[strategy_name]
    base_params = build_params(params_cls, param_overrides)  # HARUS sama dengan run_backtest.py
    cfg = CostConfig()

    result = check_robustness(
        strategy_cls,
        base_params,
        test_df,
        cfg,
        bar_minutes=window,
        entry_is_maker=entry_is_maker,
        exit_is_maker=exit_is_maker,
        pct=pct,
    )

    print(f"\n  Parameter dasar (terkunci): {base_params}")
    print(f"\n  {'Varian':<48}{'Transaksi':>10}{'Net P&L':>10}{'t-stat':>9}  Status")
    print("  " + "-" * 84)

    base_status = "OK" if result.base_metrics.mean_net_pnl_bps > 0 else "RUNTUH"
    print(
        f"  {'(dasar, tanpa geseran)':<48}{result.base_metrics.n_trades:>10}"
        f"{result.base_metrics.mean_net_pnl_bps:>10.2f}{result.base_metrics.t_stat:>9.2f}  {base_status}"
    )

    for label, m in result.variant_results:
        status = "OK" if m.mean_net_pnl_bps > 0 else "RUNTUH"
        print(f"  {label:<48}{m.n_trades:>10}{m.mean_net_pnl_bps:>10.2f}{m.t_stat:>9.2f}  {status}")

    print()
    if result.overall_robust:
        print(f"  >>> ROBUST -- expectancy net tetap positif di semua varian +-{pct:.0%}.")
        print("      Kriteria ke-4 LULUS.")
    else:
        print(f"  >>> TIDAK ROBUST -- setidaknya satu varian +-{pct:.0%} jadi rugi.")
        print("      Kriteria ke-4 GAGAL, apa pun hasil run_backtest.py.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Uji sensitivitas parameter +-20% (Hari 2, kriteria ke-4)")
    parser.add_argument("symbols", nargs="+", help="contoh: BTC/USDT:USDT ETH/USDT:USDT")
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="donchian_breakout")
    parser.add_argument("--window", type=int, default=240, help="HARUS sama dengan run_backtest.py")
    parser.add_argument("--test-months", type=int, default=6, help="HARUS sama dengan run_backtest.py")
    parser.add_argument("--pct", type=float, default=0.2, help="besar geseran parameter, default 0.2 (=20%%)")
    parser.add_argument("--maker-entry", action="store_true")
    parser.add_argument("--maker-exit", action="store_true")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="HARUS sama dengan run_backtest.py, format key=value, bisa diulang",
    )
    args = parser.parse_args()

    for symbol in args.symbols:
        try:
            run_one(
                symbol,
                args.window,
                args.test_months,
                args.pct,
                args.maker_entry,
                args.maker_exit,
                args.strategy,
                args.param,
            )
        except FileNotFoundError as e:
            print(f"\n{e}\n")


if __name__ == "__main__":
    main()