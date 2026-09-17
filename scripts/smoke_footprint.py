"""
scripts/smoke_footprint.py

Uji asap microstructure/footprint.py -- delta yang bisa dihitung
tangan, batas bar yang benar (tidak bocor ke bar tetangga), dan bar
kosong TIDAK muncul di hasil (bukan diisi nol).
"""

import sys

import pandas as pd

from src.microstructure.footprint import compute_footprint_bars

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    print("== 1. Delta yang bisa dihitung tangan ==")
    # Bar 1 (menit 0-4): 3 beli (qty 2+3+1=6), 2 jual (qty 1+1=2) -> delta=4
    # Bar 2 (menit 5-9): 1 beli (qty 1), 4 jual (qty 2+2+2+2=8) -> delta=-7
    trades = pd.DataFrame({
        "transact_time": pd.to_datetime([
            "2026-01-01 00:00:00", "2026-01-01 00:01:00", "2026-01-01 00:02:00",
            "2026-01-01 00:03:00", "2026-01-01 00:04:00",
            "2026-01-01 00:05:00", "2026-01-01 00:06:00", "2026-01-01 00:07:00",
            "2026-01-01 00:08:00", "2026-01-01 00:09:00",
        ], utc=True),
        "price": [100, 101, 100.5, 100.2, 100.8, 105, 104, 103, 102, 101],
        "quantity": [2, 3, 1, 1, 1, 1, 2, 2, 2, 2],
        "is_buyer_maker": [False, False, False, True, True, False, True, True, True, True],
    })
    bars = compute_footprint_bars(trades, bar_freq="5min", n_bins=4)
    check("2 bar terbentuk", len(bars) == 2, f"dapat {len(bars)}")

    bar1 = bars.iloc[0]
    bar2 = bars.iloc[1]
    check("delta bar 1 = 6-2 = 4", bar1["delta"] == 4, f"dapat {bar1['delta']}")
    check("delta bar 2 = 1-8 = -7", bar2["delta"] == -7, f"dapat {bar2['delta']}")
    check("delta_pct bar 1 = 4/8 = 0.5", abs(bar1["delta_pct"] - 0.5) < 1e-9,
          f"dapat {bar1['delta_pct']}")
    check("volume bar 1 = 8 (6+2)", bar1["volume"] == 8, f"dapat {bar1['volume']}")
    check("volume bar 2 = 9 (1+8)", bar2["volume"] == 9, f"dapat {bar2['volume']}")

    print("\n== 2. OHLC bar diambil dari urutan waktu, bukan urutan baris ==")
    check("open bar 1 = harga trade PERTAMA (100)", bar1["open"] == 100,
          f"dapat {bar1['open']}")
    check("close bar 1 = harga trade TERAKHIR (100.8)", bar1["close"] == 100.8,
          f"dapat {bar1['close']}")
    check("high bar 1 = 101, low bar 1 = 100",
          bar1["high"] == 101 and bar1["low"] == 100,
          f"high={bar1['high']}, low={bar1['low']}")

    print("\n== 3. Batas bar: trade PERSIS di menit ke-5 masuk bar KEDUA ==")
    check("bar 1 timestamp = 00:00, bar 2 = 00:05",
          bars.index[0] == pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
          and bars.index[1] == pd.Timestamp("2026-01-01 00:05:00", tz="UTC"))

    print("\n== 4. Ada celah waktu (tidak ada transaksi) -> bar TIDAK muncul, bukan nol ==")
    trades_gap = pd.DataFrame({
        "transact_time": pd.to_datetime([
            "2026-01-01 00:00:00", "2026-01-01 00:20:00",  # celah 20 menit
        ], utc=True),
        "price": [100.0, 105.0],
        "quantity": [1.0, 1.0],
        "is_buyer_maker": [False, False],
    })
    bars_gap = compute_footprint_bars(trades_gap, bar_freq="5min", n_bins=4)
    check("cuma 2 bar (bukan 5 bar grid seragam)", len(bars_gap) == 2,
          f"dapat {len(bars_gap)} bar: {list(bars_gap.index)}")

    print("\n== 5. Tidak ada transaksi sama sekali -> DataFrame kosong, bukan error ==")
    empty = pd.DataFrame({"transact_time": pd.Series([], dtype="datetime64[ns, UTC]"),
                           "price": [], "quantity": [], "is_buyer_maker": []})
    bars_empty = compute_footprint_bars(empty, bar_freq="5min")
    check("DataFrame kosong, bukan exception", bars_empty.empty)

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())