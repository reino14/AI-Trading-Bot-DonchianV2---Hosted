"""
scripts/smoke_smc_zones.py

Uji asap microstructure/smc_zones.py -- equilibrium yang bisa dihitung
tangan, label premium/discount benar, "unknown" sebelum swing ada,
plus kenari look-ahead.
"""

import sys

import pandas as pd

from src.microstructure.smc_zones import compute_premium_discount

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def make_df(rows: list[tuple], start="2026-01-01", freq="1h") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def main() -> int:
    print("== 1. Equilibrium dihitung tangan: swing high=100, swing low=50 -> titik tengah=75 ==")
    # Swing low BERSIH di index3 (low=50) -- lookback=3 butuh index>=3
    # supaya ada cukup bar di kedua sisi untuk dicek sama sekali.
    # Swing high BERSIH di index9 (high=100).
    rows = [
        (72, 73, 71, 72),
        (71, 72, 70, 71),
        (70, 71, 69, 70),
        (69, 70, 50, 65),   # swing low: low=50 (index3)
        (65, 66, 60, 64),
        (64, 66, 60, 65),
        (65, 80, 60, 78),
        (78, 90, 75, 88),
        (88, 95, 85, 92),
        (92, 100, 90, 95),  # swing high: high=100 (index9)
        (95, 96, 92, 93),
        (93, 94, 91, 92),
        (92, 93, 90, 91),
        # bar UJI: close di ATAS equilibrium(75) -> "premium"
        (91, 92, 88, 85),
    ]
    df = make_df(rows)
    zone = compute_premium_discount(df, lookback=3)
    last_zone = zone.iloc[-1]
    check("bar terakhir (close=85 > equilibrium 75) berlabel 'premium'",
          last_zone == "premium", f"dapat '{last_zone}'")

    print("\n== 2. Bar dengan close DI BAWAH equilibrium -> 'discount' ==")
    rows2 = rows[:-1] + [(91, 92, 88, 70)]  # close=70 < equilibrium(75)
    df2 = make_df(rows2)
    zone2 = compute_premium_discount(df2, lookback=3)
    check("close=70 < equilibrium 75 -> 'discount'", zone2.iloc[-1] == "discount",
          f"dapat '{zone2.iloc[-1]}'")

    print("\n== 3. Sebelum swing high ATAU swing low terkonfirmasi -> 'unknown', bukan tebakan ==")
    early_zone = zone.iloc[0]
    check("bar sangat awal (belum ada swing terkonfirmasi) -> 'unknown'",
          early_zone == "unknown", f"dapat '{early_zone}'")

    print("\n== 4. KENARI LOOK-AHEAD: ubah bar SETELAH suatu titik -> label lama tidak berubah ==")
    cutoff = 10  # titik setelah swing high (index9) terkonfirmasi (confirm_pos=9+3=12)
    label_before = zone.iloc[cutoff]

    df_future = df.copy()
    df_future.iloc[cutoff + 1:, df_future.columns.get_loc("high")] = 0.002
    df_future.iloc[cutoff + 1:, df_future.columns.get_loc("low")] = 0.0005
    df_future.iloc[cutoff + 1:, df_future.columns.get_loc("close")] = 0.001
    zone_after = compute_premium_discount(df_future, lookback=3)
    label_after = zone_after.iloc[cutoff]

    check("label di titik cutoff SAMA walau semua data SETELAHNYA diubah total",
          label_before == label_after, f"sebelum='{label_before}', sesudah='{label_after}'")

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())