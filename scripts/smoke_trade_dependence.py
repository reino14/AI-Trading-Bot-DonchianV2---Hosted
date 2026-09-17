"""
scripts/smoke_trade_dependence.py

Uji asap strategy/trade_dependence.py -- ekstraksi trade dengan return
yang bisa dihitung tangan, runs test dengan contoh yang saya hitung
manual dulu di luar kode, filter dependence, dan ringkasan.
"""

import sys

import numpy as np
import pandas as pd

from src.strategy.trade_dependence import (
    extract_trades,
    filter_by_previous_trade,
    runs_test,
    summarize_trades,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    print("== 1. extract_trades: long lalu short, return dihitung tangan ==")
    idx = pd.date_range("2026-01-01", periods=5, freq="1D", tz="UTC")
    close = pd.Series([100.0, 110.0, 121.0, 100.0, 90.0], index=idx)
    signal = pd.Series([1.0, 1.0, -1.0, -1.0, 1.0], index=idx)
    # Trade1: long, entry bar0(100) -> exit bar2(121) [signal berubah ke -1 di bar2]
    #   return = (121/100 - 1)*1 = 0.21
    # Trade2: short, entry bar2(121) -> exit bar4(90) [signal berubah ke 1 di bar4]
    #   return = (90/121 - 1)*(-1) = 1 - 90/121 = 0.256198...
    # Posisi terbuka setelah bar4 (long lagi) TIDAK dihitung -- belum ada exit.
    trades = extract_trades(signal, close)
    check("2 trade selesai terekstrak (posisi terbuka terakhir TIDAK dihitung)",
          len(trades) == 2, f"dapat {len(trades)}")
    if len(trades) == 2:
        t1, t2 = trades.iloc[0], trades.iloc[1]
        check("trade1: long, entry=100, exit=121, return=0.21",
              t1["direction"] == 1 and abs(t1["return_pct"] - 0.21) < 1e-9,
              f"dapat dir={t1['direction']}, return={t1['return_pct']:.4f}")
        check("trade2: short, entry=121, exit=90, return=(1-90/121)",
              t2["direction"] == -1 and abs(t2["return_pct"] - (1 - 90/121)) < 1e-9,
              f"dapat dir={t2['direction']}, return={t2['return_pct']:.4f}")
        check("hold_bars trade1 = 2 (bar0 ke bar2)", t1["hold_bars"] == 2)

    print("\n== 2. runs_test: contoh dihitung tangan (n1=6, n2=6, 4 runs) ==")
    # [+,+,+,-,-,-,+,+,+,-,-,-] -- 4 runs (+++, ---, +++, ---).
    # expected_runs = 2*6*6/12+1 = 7. var = 2*6*6*(72-12)/(144*11) = 4320/1584 = 2.72727...
    # std = 1.65145... z = (4-7)/1.65145 = -1.8166...
    signs = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1])
    z = runs_test(signs)
    check("z-score = -1.8166 (dihitung tangan)", abs(z - (-1.8166)) < 0.001, f"dapat {z:.4f}")

    print("\n== 3. runs_test: alternating sempurna (+,-,+,-,...) -> z SANGAT POSITIF ==")
    # Alternating sempurna = jumlah runs MAKSIMUM mungkin -- z harus besar positif.
    signs_alt = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
    z_alt = runs_test(signs_alt)
    check("z-score besar POSITIF untuk alternating sempurna", z_alt > 2.5, f"dapat {z_alt:.4f}")

    print("\n== 4. runs_test: streak panjang (semua + dulu, baru semua -) -> z SANGAT NEGATIF ==")
    signs_streak = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    z_streak = runs_test(signs_streak)
    check("z-score besar NEGATIF untuk 2 streak panjang (cuma 2 runs)",
          z_streak < -2.5, f"dapat {z_streak:.4f}")

    print("\n== 5. filter_by_previous_trade: trade PERTAMA selalu dibuang (tidak ada pembanding) ==")
    trades5 = pd.DataFrame({"return_pct": [0.1, -0.05, 0.2, -0.1, 0.3]})
    mask_loser = filter_by_previous_trade(trades5, only_after="loser")
    check("trade0 (pertama) SELALU False", mask_loser.iloc[0] == False)  # noqa: E712
    check("trade1 (setelah trade0 UNTUNG) -> False (only_after=loser)",
          mask_loser.iloc[1] == False)  # noqa: E712
    check("trade2 (setelah trade1 RUGI) -> True", mask_loser.iloc[2] == True)  # noqa: E712
    check("trade3 (setelah trade2 UNTUNG) -> False", mask_loser.iloc[3] == False)  # noqa: E712
    check("trade4 (setelah trade3 RUGI) -> True", mask_loser.iloc[4] == True)  # noqa: E712

    mask_winner = filter_by_previous_trade(trades5, only_after="winner")
    check("mode 'winner' adalah KEBALIKAN dari mode 'loser' (untuk trade 1-4)",
          (mask_winner.iloc[1:] == ~mask_loser.iloc[1:]).all())

    print("\n== 6. summarize_trades: profit factor dihitung tangan ==")
    trades6 = pd.DataFrame({"return_pct": [0.10, 0.20, -0.05, -0.05]})
    # profit_factor = (0.10+0.20) / abs(-0.05-0.05) = 0.30/0.10 = 3.0
    summary = summarize_trades(trades6)
    check("profit_factor = 3.0", abs(summary["profit_factor"] - 3.0) < 1e-9,
          f"dapat {summary['profit_factor']}")
    check("win_rate = 50%", abs(summary["win_rate"] - 0.5) < 1e-9)

    print("\n== 7. Trade kosong -> tidak crash ==")
    empty_signal = pd.Series([np.nan, np.nan], index=pd.date_range("2026-01-01", periods=2, freq="1D", tz="UTC"))
    empty_close = pd.Series([100.0, 101.0], index=empty_signal.index)
    empty_trades = extract_trades(empty_signal, empty_close)
    check("DataFrame kosong untuk sinyal yang semuanya NaN", empty_trades.empty)
    check("summarize_trades tidak crash untuk trades kosong",
          summarize_trades(empty_trades) == {"n_trades": 0})

    print("\n== 8. Ongkos: trade PERTAMA (dari flat) kena 1 unit, trade SETELAHNYA (reversal) kena 2 unit ==")
    idx8 = pd.date_range("2026-01-01", periods=4, freq="1D", tz="UTC")
    close8 = pd.Series([100.0, 110.0, 100.0, 90.0], index=idx8)
    signal8 = pd.Series([1.0, 1.0, -1.0, 1.0], index=idx8)
    trades8 = extract_trades(signal8, close8, cost_bps_per_unit=10.0)
    check("2 trade selesai terekstrak", len(trades8) == 2, f"dapat {len(trades8)}")
    if len(trades8) == 2:
        t1_8, t2_8 = trades8.iloc[0], trades8.iloc[1]
        check("trade1 (dari flat): units_changed_at_entry = 1.0",
              abs(t1_8["units_changed_at_entry"] - 1.0) < 1e-9, f"dapat {t1_8['units_changed_at_entry']}")
        check("trade1: cost_pct = 1 * 10bps = 0.001",
              abs(t1_8["cost_pct"] - 0.001) < 1e-9, f"dapat {t1_8['cost_pct']}")
        check("trade2 (reversal long->short): units_changed_at_entry = 2.0",
              abs(t2_8["units_changed_at_entry"] - 2.0) < 1e-9, f"dapat {t2_8['units_changed_at_entry']}")
        check("trade2: cost_pct = 2 * 10bps = 0.002",
              abs(t2_8["cost_pct"] - 0.002) < 1e-9, f"dapat {t2_8['cost_pct']}")
        check("return_pct (net) = gross_return_pct - cost_pct, untuk KEDUA trade",
              abs(t1_8["return_pct"] - (t1_8["gross_return_pct"] - t1_8["cost_pct"])) < 1e-9
              and abs(t2_8["return_pct"] - (t2_8["gross_return_pct"] - t2_8["cost_pct"])) < 1e-9)
        check("net SELALU <= gross (ongkos tidak pernah negatif/menambah untung)",
              (trades8["return_pct"] <= trades8["gross_return_pct"] + 1e-12).all())

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())