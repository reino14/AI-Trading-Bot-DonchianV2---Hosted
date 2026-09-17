"""
scripts/smoke_trade_simulator.py

Uji asap microstructure/trade_simulator.py -- lima skenario dengan
jawaban yang saya tahu pasti, termasuk tie-break konservatif dan
penanganan setup yang tumpang tindih dengan posisi masih terbuka.
"""

import sys

import pandas as pd

from src.core.cost_model import CostConfig
from src.microstructure.trade_simulator import simulate_trades

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


FREE = CostConfig(maker_bps=0, taker_bps=0, spread_bps=0, slippage_bps=0,
                   funding_bps_per_8h=0)


def make_bars(rows: list[tuple], start="2026-01-01 00:00:00", freq="5min") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(rows, columns=["high", "low", "close"], index=idx)


def main() -> int:
    print("== 1. LONG: target kena bersih di bar ke-2 (tanpa nyentuh stop) ==")
    setups = pd.DataFrame({
        "direction": ["long"], "entry_price": [100.0],
        "stop_price": [95.0], "target_price": [110.0],
        "val_price": [0.0], "vah_price": [0.0],
    }, index=pd.DatetimeIndex([pd.Timestamp("2026-01-01 00:00:00", tz="UTC")]))
    bars = make_bars([
        (102, 98, 101),    # bar konfirmasi itu sendiri -- TIDAK dicek (mulai dari bar berikutnya)
        (105, 99, 104),    # bar 1 setelah entry -- belum kena apa pun
        (112, 108, 111),   # bar 2 -- target (110) kena, stop (95) tidak
    ])
    trades = simulate_trades(setups, bars, FREE, max_holding_bars=10)
    check("1 trade dihasilkan", len(trades) == 1)
    if len(trades) == 1:
        t = trades.iloc[0]
        check("exit_reason = target", t["exit_reason"] == "target")
        check("exit_price = 110.0 (harga target, bukan high bar)", t["exit_price"] == 110.0)
        check("hold_bars = 2", t["hold_bars"] == 2, f"dapat {t['hold_bars']}")

    print("\n== 2. SHORT: stop kena bersih ==")
    setups2 = pd.DataFrame({
        "direction": ["short"], "entry_price": [100.0],
        "stop_price": [105.0], "target_price": [90.0],
        "val_price": [0.0], "vah_price": [0.0],
    }, index=pd.DatetimeIndex([pd.Timestamp("2026-01-01 00:00:00", tz="UTC")]))
    bars2 = make_bars([
        (101, 98, 100),
        (106, 99, 103),  # high 106 >= stop 105 -> stop kena
    ])
    trades2 = simulate_trades(setups2, bars2, FREE, max_holding_bars=10)
    check("exit_reason = stop", trades2.iloc[0]["exit_reason"] == "stop")
    check("exit_price = 105.0", trades2.iloc[0]["exit_price"] == 105.0)

    print("\n== 3. TIE-BREAK: satu bar kena stop DAN target -> HARUS menang stop ==")
    setups3 = pd.DataFrame({
        "direction": ["long"], "entry_price": [100.0],
        "stop_price": [95.0], "target_price": [110.0],
        "val_price": [0.0], "vah_price": [0.0],
    }, index=pd.DatetimeIndex([pd.Timestamp("2026-01-01 00:00:00", tz="UTC")]))
    bars3 = make_bars([
        (101, 99, 100),
        (115, 90, 105),  # range bar ini mencakup stop(95) DAN target(110) sekaligus
    ])
    trades3 = simulate_trades(setups3, bars3, FREE, max_holding_bars=10)
    check("tie-break memilih stop (konservatif), BUKAN target",
          trades3.iloc[0]["exit_reason"] == "stop",
          f"dapat '{trades3.iloc[0]['exit_reason']}'")

    print("\n== 4. TIMEOUT: stop maupun target tidak pernah kena ==")
    setups4 = pd.DataFrame({
        "direction": ["long"], "entry_price": [100.0],
        "stop_price": [50.0], "target_price": [150.0],
        "val_price": [0.0], "vah_price": [0.0],
    }, index=pd.DatetimeIndex([pd.Timestamp("2026-01-01 00:00:00", tz="UTC")]))
    bars4 = make_bars([(101, 99, 100)] * 5)
    trades4 = simulate_trades(setups4, bars4, FREE, max_holding_bars=3)
    check("exit_reason = timeout", trades4.iloc[0]["exit_reason"] == "timeout")
    check("hold_bars = max_holding_bars (3)", trades4.iloc[0]["hold_bars"] == 3)

    print("\n== 5. Setup kedua tumpang tindih posisi terbuka -> DIABAIKAN ==")
    setups5 = pd.DataFrame({
        "direction": ["long", "long"], "entry_price": [100.0, 100.0],
        "stop_price": [50.0, 50.0], "target_price": [150.0, 150.0],
        "val_price": [0.0, 0.0], "vah_price": [0.0, 0.0],
    }, index=pd.DatetimeIndex([
        pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("2026-01-01 00:05:00", tz="UTC"),  # muncul SEBELUM posisi 1 selesai
    ]))
    bars5 = make_bars([(101, 99, 100)] * 10)
    trades5 = simulate_trades(setups5, bars5, FREE, max_holding_bars=8)
    check("cuma 1 trade dieksekusi (setup ke-2 diabaikan)", len(trades5) == 1,
          f"dapat {len(trades5)} trade")

    print("\n== 6. Ongkos benar-benar mengurangi net_pnl_bps dibanding gross ==")
    perp_cost = CostConfig(maker_bps=2, taker_bps=5, spread_bps=1, slippage_bps=2,
                            funding_bps_per_8h=1)
    trades_cost = simulate_trades(setups, bars, perp_cost, max_holding_bars=10)
    t = trades_cost.iloc[0]
    check("net_pnl_bps < gross_pnl_bps (ongkos beneran dipotong)",
          t["net_pnl_bps"] < t["gross_pnl_bps"],
          f"gross={t['gross_pnl_bps']:.2f}, net={t['net_pnl_bps']:.2f}")

    print("\n" + "=" * 62)
    print("move_to_breakeven / trail_after_breakeven -- fitur baru")
    print("=" * 62)

    print("\n== 7. Breakeven ter-trigger begitu harga reclaim value area, lalu STOP di entry ==")
    # LONG entry=100, VAL=95 (asal, entry di bawah VAL -- discount).
    # Bar1: close=96 (>= VAL 95) -> reclaim, stop pindah ke entry(100).
    # Bar2: low turun ke 99 -- BUKAN kena stop LAMA (95), tapi KENA stop BARU (100).
    setups_be = pd.DataFrame({
        "direction": ["long"], "entry_price": [100.0],
        "stop_price": [90.0], "target_price": [130.0],
        "val_price": [95.0], "vah_price": [999.0],
    }, index=pd.DatetimeIndex([pd.Timestamp("2026-01-01 00:00:00", tz="UTC")]))
    bars_be = make_bars([
        (101, 99, 100),   # bar konfirmasi -- tidak dicek
        (97, 95, 96),     # bar1: close=96 >= VAL(95) -> reclaim, stop->100
        (100, 99, 99.5),  # bar2: low=99 < stop_lama(90) TIDAK relevan, tapi < stop_baru(100) -> KENA
    ])
    trades_be = simulate_trades(setups_be, bars_be, FREE, max_holding_bars=10, move_to_breakeven=True)
    check("exit_reason = breakeven (bukan 'stop' biasa)",
          trades_be.iloc[0]["exit_reason"] == "breakeven",
          f"dapat '{trades_be.iloc[0]['exit_reason']}'")
    check("exit_price = entry_price (100.0) -- PnL kotor nyaris nol",
          trades_be.iloc[0]["exit_price"] == 100.0)
    check("keluar di bar ke-2 (stop lama 90 tidak akan pernah kena di data ini)",
          trades_be.iloc[0]["hold_bars"] == 2, f"dapat {trades_be.iloc[0]['hold_bars']}")

    print("\n== 8. Tanpa move_to_breakeven -- stop LAMA tetap dipakai (perilaku default) ==")
    trades_no_be = simulate_trades(setups_be, bars_be, FREE, max_holding_bars=10)
    check("exit_reason = timeout (stop lama 90 tidak pernah kena, breakeven tidak aktif)",
          trades_no_be.iloc[0]["exit_reason"] == "timeout",
          f"dapat '{trades_no_be.iloc[0]['exit_reason']}'")

    print("\n== 9. Trailing: stop mengetat ke low N-bar terakhir, HANYA ke arah untung ==")
    # LONG entry=100, VAL=95. Bar1 reclaim -> breakeven(100). Bar2-4 naik
    # terus, trailing (lookback=2) harus mengikuti low 2 bar terakhir,
    # TIDAK PERNAH turun kembali walau ada bar yang low-nya lebih rendah belakangan.
    setups_trail = pd.DataFrame({
        "direction": ["long"], "entry_price": [100.0],
        "stop_price": [90.0], "target_price": [200.0],
        "val_price": [95.0], "vah_price": [999.0],
    }, index=pd.DatetimeIndex([pd.Timestamp("2026-01-01 00:00:00", tz="UTC")]))
    bars_trail = make_bars([
        (101, 99, 100),    # bar konfirmasi
        (97, 96, 96.5),    # bar1: close=96.5>=95 -> reclaim, stop=100
        (115, 110, 112),   # bar2: trail window(lookback=2)=[bar1,bar2] low=min(96,110)=96 -- TIDAK > stop(100), stop TETAP 100
        (130, 125, 128),   # bar3: window=[bar2,bar3] low=min(110,125)=110 -- 110>100 -> stop naik ke 110
        (108, 105, 106),   # bar4: low=105 < stop_baru(110) -> KENA stop trailing di 110 (BUKAN 90 atau 100)
    ])
    trades_trail = simulate_trades(
        setups_trail, bars_trail, FREE, max_holding_bars=10,
        move_to_breakeven=True, trail_after_breakeven=True, trail_lookback_bars=2,
    )
    check("exit_price = 110.0 (level trailing, bukan breakeven 100 atau stop asal 90)",
          abs(trades_trail.iloc[0]["exit_price"] - 110.0) < 1e-9,
          f"dapat {trades_trail.iloc[0]['exit_price']}")
    check("exit_reason = 'stop' (bukan 'breakeven' -- exit_price sudah beda dari entry)",
          trades_trail.iloc[0]["exit_reason"] == "stop")

    print("\n== 10. Trailing TIDAK PERNAH melonggar mundur walau harga sempat turun dulu ==")
    # Setelah stop naik ke level tertentu, bar berikutnya dengan low
    # lebih RENDAH dari stop saat ini tidak boleh menurunkan stop lagi.
    setups_never_loosen = pd.DataFrame({
        "direction": ["long"], "entry_price": [100.0],
        "stop_price": [90.0], "target_price": [200.0],
        "val_price": [95.0], "vah_price": [999.0],
    }, index=pd.DatetimeIndex([pd.Timestamp("2026-01-01 00:00:00", tz="UTC")]))
    bars_nl = make_bars([
        (101, 99, 100),
        (97, 96, 96.5),     # reclaim -> stop=100
        (140, 135, 138),    # window(lookback=2)=[96,135] -> low=96, stop TETAP 100 (96<100)
        (145, 140, 142),    # window=[135,140] -> low=135 -> stop naik ke 135
        (137, 120, 130),    # low=120 < stop(135) -> harus KENA di 135, BUKAN turun mengikuti 120
    ])
    trades_nl = simulate_trades(
        setups_never_loosen, bars_nl, FREE, max_holding_bars=10,
        move_to_breakeven=True, trail_after_breakeven=True, trail_lookback_bars=2,
    )
    check("exit_price = 135.0 (stop trailing tidak pernah turun ke 120)",
          abs(trades_nl.iloc[0]["exit_price"] - 135.0) < 1e-9,
          f"dapat {trades_nl.iloc[0]['exit_price']}")

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())