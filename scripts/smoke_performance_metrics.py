"""
scripts/smoke_performance_metrics.py

Uji asap microstructure/performance_metrics.py -- kurva ekuitas
sintetis yang jawabannya saya hitung tangan: Sharpe, drawdown, dan
episode waktu pulih (termasuk kasus BELUM pulih sampai data habis).
"""

import sys

import numpy as np
import pandas as pd

from src.microstructure.performance_metrics import (
    build_daily_equity_curve,
    find_drawdown_episodes,
    max_drawdown_pct,
    sharpe_ratio,
    summarize_performance,
    win_loss_ratio,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    print("== 1. build_daily_equity_curve: P&L diterapkan di exit_time, bukan entry ==")
    trades = pd.DataFrame({
        "exit_time": pd.to_datetime(["2026-01-02", "2026-01-04"], utc=True),
        "net_pnl_bps": [1000.0, -500.0],  # +10% lalu -5%
    })
    curve = build_daily_equity_curve(
        trades, initial_capital=100.0, position_fraction=1.0,
        start=pd.Timestamp("2026-01-01", tz="UTC"), end=pd.Timestamp("2026-01-05", tz="UTC"),
    )
    check("hari 1 (sebelum trade 1) = modal awal (100)",
          abs(curve.loc["2026-01-01"] - 100.0) < 1e-6, f"dapat {curve.loc['2026-01-01']}")
    check("hari 2 (setelah trade 1, +10%) = 110",
          abs(curve.loc["2026-01-02"] - 110.0) < 1e-6, f"dapat {curve.loc['2026-01-02']}")
    check("hari 3 (di antara trade, belum ada trade baru) TETAP 110 (ffill)",
          abs(curve.loc["2026-01-03"] - 110.0) < 1e-6, f"dapat {curve.loc['2026-01-03']}")
    check("hari 4 (setelah trade 2, -5% dari 110) = 104.5",
          abs(curve.loc["2026-01-04"] - 104.5) < 1e-6, f"dapat {curve.loc['2026-01-04']}")

    print("\n== 2. max_drawdown_pct dihitung tangan ==")
    equity = pd.Series([100.0, 120.0, 90.0, 95.0, 130.0])
    # Puncak 120 -> lembah 90 -> drawdown = (120-90)/120 = 0.25
    dd = max_drawdown_pct(equity)
    check("drawdown = 25.0%", abs(dd - 0.25) < 1e-9, f"dapat {dd:.4f}")

    print("\n== 3. find_drawdown_episodes: satu episode, PULIH ==")
    idx3 = pd.date_range("2026-01-01", periods=5, freq="1D", tz="UTC")
    equity3 = pd.Series([100.0, 120.0, 90.0, 95.0, 130.0], index=idx3)
    episodes3 = find_drawdown_episodes(equity3)
    check("1 episode terdeteksi", len(episodes3) == 1, f"dapat {len(episodes3)}")
    if episodes3:
        e = episodes3[0]
        check("puncak = hari ke-2 (120)", e.peak_date == idx3[1])
        check("lembah = hari ke-3 (90)", e.trough_date == idx3[2])
        check("pulih di hari ke-5 (130 > 120)", e.recovery_date == idx3[4])
        check("waktu pulih = 2 hari (dari lembah ke pemulihan)", e.recovery_days == 2,
              f"dapat {e.recovery_days}")

    print("\n== 4. find_drawdown_episodes: BELUM PULIH sampai data habis ==")
    idx4 = pd.date_range("2026-01-01", periods=4, freq="1D", tz="UTC")
    equity4 = pd.Series([100.0, 120.0, 90.0, 95.0], index=idx4)  # tidak pernah balik ke 120
    episodes4 = find_drawdown_episodes(equity4)
    check("1 episode terdeteksi, recovery_date=None", len(episodes4) == 1 and episodes4[0].recovery_date is None)
    check("recovery_days=None (bukan angka palsu)", episodes4[0].recovery_days is None)

    print("\n== 5. win_loss_ratio dihitung tangan ==")
    trades5 = pd.DataFrame({"net_pnl_bps": [100.0, 200.0, -50.0, -50.0]})
    # rata untung=150, rata rugi=-50 -> rasio=150/50=3.0
    wl = win_loss_ratio(trades5)
    check("win/loss ratio = 3.0", abs(wl - 3.0) < 1e-9, f"dapat {wl}")

    print("\n== 6. sharpe_ratio: return konstan positif -> Sharpe TINGGI ==")
    idx6 = pd.date_range("2026-01-01", periods=100, freq="1D", tz="UTC")
    # Return harian konstan 0.1% -- variasinya nyaris nol, Sharpe harus sangat tinggi.
    equity6 = pd.Series(100 * (1.001 ** np.arange(100)), index=idx6)
    sr = sharpe_ratio(equity6)
    check("Sharpe sangat tinggi untuk return konstan (variasi ~nol)", sr > 20, f"dapat {sr:.2f}")

    print("\n== 7. sharpe_ratio: return acak rata-rata nol -> Sharpe dekat nol ==")
    rng = np.random.default_rng(5)
    idx7 = pd.date_range("2026-01-01", periods=300, freq="1D", tz="UTC")
    rets7 = rng.normal(0, 0.02, 300)
    equity7 = pd.Series(100 * np.cumprod(1 + rets7), index=idx7)
    sr7 = sharpe_ratio(equity7)
    check("Sharpe di rentang wajar untuk random walk (bukan ekstrem)",
          -3 < sr7 < 3, f"dapat {sr7:.2f}")

    print("\n== 8. summarize_performance: end-to-end tidak crash, angka konsisten ==")
    trades8 = pd.DataFrame({
        "exit_time": pd.to_datetime(
            ["2026-01-02", "2026-01-05", "2026-01-10", "2026-01-15"], utc=True
        ),
        "net_pnl_bps": [500.0, -300.0, 800.0, -200.0],
    })
    summary = summarize_performance(trades8, initial_capital=1000.0, position_fraction=1.0)
    check("n_trades = 4", summary["n_trades"] == 4)
    check("win_rate = 50%", abs(summary["win_rate"] - 0.5) < 1e-9)
    check("final_capital dan total_return_pct konsisten satu sama lain",
          abs(summary["total_return_pct"] - (summary["final_capital"] / 1000.0 - 1)) < 1e-9)

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())