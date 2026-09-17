"""
scripts/smoke_alpha_bank.py

Uji asap untuk xs_reversion.py dan funding_carry.py -- pola yang sama
seperti tes 12-13 di smoke_portfolio_engine.py: kontrol NOL (random
walk, strategi TIDAK BOLEH menemukan edge) dan kontrol POSITIF (edge
ditanam sesuai jenis sinyalnya, strategi WAJIB menemukannya).

Kenapa kontrol positifnya beda per strategi: momentum butuh persistensi
return (ditanam lewat AR(1) drift, lihat smoke_portfolio_engine.py),
reversion butuh OVERREAKSI (harga yang habis melonjak tajam lalu
terkoreksi -- pola yang secara struktural BERLAWANAN dengan momentum),
funding_carry butuh PEMBEDAAN FUNDING yang konsisten (bukan pola
harga sama sekali -- exposure yang funding-nya tinggi harus punya
return rata-rata sedikit lebih rendah, meniru pembayaran funding itu
sendiri).
"""

import sys

import numpy as np
import pandas as pd

from src.backtest.portfolio_engine import run_portfolio_backtest
from src.backtest.portfolio_metrics import summarize
from src.strategy.funding_carry import FundingCarryParams, FundingCarryStrategy
from src.strategy.xs_reversion import XsReversionParams, XsReversionStrategy

FAILURES: list[str] = []
FREE_COST = None  # diisi setelah import CostSpec di bawah


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def make_random_walk(n_bars=1200, n_assets=40, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_bars, freq="1D", tz="UTC")
    cols = [f"SYM{i:02d}" for i in range(n_assets)]
    rets = rng.normal(0, 0.03, (n_bars, n_assets))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=cols)


def main() -> int:
    from src.backtest.portfolio_engine import CostSpec
    free = CostSpec(per_side_bps=0.0)

    # ============================================================
    print("== REVERSION: kontrol nol (random walk) ==")
    close_rw = make_random_walk(seed=1)
    strat = XsReversionStrategy(XsReversionParams(lookback_bars=2, rebalance_bars=1))
    r = run_portfolio_backtest(close_rw, strat, free, bar_minutes=1440)
    s = summarize(r.net_returns, 1440)
    check("tidak ada edge di random walk", abs(s["t_stat"]) < 2.5, f"t={s['t_stat']:.2f}")

    print("\n== REVERSION: kontrol positif (overreaksi ditanam) ==")
    # Tanam pola: return besar di satu bar HAMPIR SELALU diikuti
    # koreksi sebagian di bar berikutnya -- overreaksi klasik.
    rng = np.random.default_rng(21)
    n_bars, n_assets = 1200, 40
    shock = rng.normal(0, 0.04, (n_bars, n_assets))
    correction = -0.4 * np.roll(shock, 1, axis=0)  # koreksi 40% dari shock kemarin
    correction[0] = 0
    noise = rng.normal(0, 0.015, (n_bars, n_assets))
    rets = shock + correction + noise
    close_edge = pd.DataFrame(
        100 * np.exp(np.cumsum(rets, axis=0)),
        index=pd.date_range("2022-01-01", periods=n_bars, freq="1D", tz="UTC"),
        columns=[f"SYM{i:02d}" for i in range(n_assets)],
    )
    r = run_portfolio_backtest(close_edge, strat, CostSpec(per_side_bps=2.0), bar_minutes=1440)
    s = summarize(r.net_returns, 1440, r.gross_returns, r.turnover)
    check("overreaksi yang ditanam terdeteksi", s["t_stat"] > 3.0,
          f"t={s['t_stat']:.2f}, Sharpe={s['sharpe_annual']:.2f}")
    print(f"    Sharpe={s['sharpe_annual']:.2f}, drawdown={s['max_drawdown']:.1%}")

    # ============================================================
    print("\n== FUNDING_CARRY: menolak tanpa extra['funding'] ==")
    strat_fc = FundingCarryStrategy(FundingCarryParams(smooth_bars=7, rebalance_bars=1))
    try:
        run_portfolio_backtest(close_rw, strat_fc, free, bar_minutes=1440, extra=None)
        check("menolak saat funding tidak disediakan", False, "tidak melempar error")
    except ValueError:
        check("menolak saat funding tidak disediakan", True)

    print("\n== FUNDING_CARRY: kontrol nol (random walk harga, funding acak) ==")
    funding_rw = pd.DataFrame(
        np.random.default_rng(2).normal(0, 1.0, close_rw.shape),
        index=close_rw.index, columns=close_rw.columns,
    )
    r = run_portfolio_backtest(close_rw, strat_fc, free, bar_minutes=1440,
                                extra={"funding": funding_rw})
    s = summarize(r.net_returns, 1440)
    check("tidak ada edge saat funding tidak berhubungan dengan return",
          abs(s["t_stat"]) < 2.5, f"t={s['t_stat']:.2f}")

    print("\n== FUNDING_CARRY: kontrol positif (funding tinggi -> return rendah, ditanam) ==")
    rng = np.random.default_rng(31)
    n_bars, n_assets = 1200, 40
    # Tiap aset punya "level funding" tetap sepanjang waktu (meniru rezim
    # crowding yang persisten), dan return rata-ratanya SEDIKIT lebih
    # rendah kalau funding-nya tinggi -- persis mekanisme carry.
    funding_level = rng.normal(0, 3.0, n_assets)  # bps per bar, level per aset
    daily_funding = np.tile(funding_level, (n_bars, 1)) + rng.normal(0, 0.5, (n_bars, n_assets))
    drag = -0.0002 * funding_level  # funding tinggi -> drift return sedikit negatif
    rets = np.tile(drag, (n_bars, 1)) + rng.normal(0, 0.02, (n_bars, n_assets))
    idx = pd.date_range("2022-01-01", periods=n_bars, freq="1D", tz="UTC")
    cols = [f"SYM{i:02d}" for i in range(n_assets)]
    close_fc = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)
    funding_fc = pd.DataFrame(daily_funding, index=idx, columns=cols)

    r = run_portfolio_backtest(close_fc, strat_fc, CostSpec(per_side_bps=1.0),
                                bar_minutes=1440, extra={"funding": funding_fc})
    s = summarize(r.net_returns, 1440, r.gross_returns, r.turnover)
    check("carry yang ditanam terdeteksi", s["t_stat"] > 3.0,
          f"t={s['t_stat']:.2f}, Sharpe={s['sharpe_annual']:.2f}")
    print(f"    Sharpe={s['sharpe_annual']:.2f}, drawdown={s['max_drawdown']:.1%}")

    print("\n== FUNDING_CARRY: menolak kalau extra['funding'] tidak sejajar ==")
    misaligned = funding_fc.iloc[:-5]  # index lebih pendek dari close
    try:
        run_portfolio_backtest(close_fc, strat_fc, free, bar_minutes=1440,
                                extra={"funding": misaligned})
        check("menolak funding tidak sejajar", False, "tidak melempar error")
    except ValueError:
        check("menolak funding tidak sejajar", True)

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())