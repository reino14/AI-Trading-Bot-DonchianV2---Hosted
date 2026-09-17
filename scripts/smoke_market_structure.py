"""
scripts/smoke_market_structure.py

Uji asap microstructure/market_structure.py -- deret harga sintetis
dengan jawaban yang saya tahu pasti (zigzag naik murni, turun murni,
mendatar), plus kenari look-ahead (struktur di bar t tidak boleh
berubah kalau data SETELAH bar t diubah).
"""

import sys

import numpy as np
import pandas as pd

from src.microstructure.market_structure import (
    build_structure_series,
    classify_structure,
    find_swing_points,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "LULUS" if condition else "GAGAL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def make_zigzag(highs: list[float], lows: list[float], bars_per_leg: int = 8) -> pd.DataFrame:
    """
    Bangun deret OHLC zigzag EKSPLISIT: naik ke highs[0], turun ke
    lows[0], naik ke highs[1], dst -- bergantian, dengan swing point
    yang JELAS dan JAUH dari noise supaya terdeteksi tanpa ambigu.
    """
    rows = []
    level = 0.0
    points = []
    for h, l in zip(highs, lows):
        points.append(h)
        points.append(l)

    prev = 0.0
    for target in points:
        for i in range(bars_per_leg):
            frac = (i + 1) / bars_per_leg
            price = prev + (target - prev) * frac
            rows.append({"high": price + 0.01, "low": price - 0.01, "close": price})
        prev = target

    idx = pd.date_range("2026-01-01", periods=len(rows), freq="1h", tz="UTC")
    return pd.DataFrame(rows, index=idx)


def main() -> int:
    print("== 1. Uptrend murni: higher high + higher low berturut-turut ==")
    df_up = make_zigzag(highs=[10, 20, 30, 40], lows=[2, 8, 18, 28])
    swings = find_swing_points(df_up, lookback=3)
    n_highs = sum(1 for s in swings if s.kind == "high")
    n_lows = sum(1 for s in swings if s.kind == "low")
    check("swing high terdeteksi (harus 4)", n_highs == 4, f"dapat {n_highs}")
    check("swing low terdeteksi (harus 3, 1 di awal & akhir tidak lengkap)",
          n_lows >= 3, f"dapat {n_lows}")

    structure = build_structure_series(df_up, lookback=3)
    label_near_end = structure.iloc[-5]
    check("struktur mendekati akhir uptrend terklasifikasi 'up'",
          label_near_end == "up", f"dapat '{label_near_end}'")

    print("\n== 2. Downtrend murni: lower high + lower low berturut-turut ==")
    df_down = make_zigzag(highs=[40, 30, 20, 10], lows=[28, 18, 8, 2])
    structure_down = build_structure_series(df_down, lookback=3)
    label_down = structure_down.iloc[-5]
    check("struktur mendekati akhir downtrend terklasifikasi 'down'",
          label_down == "down", f"dapat '{label_down}'")

    print("\n== 3. Sideways: high stabil, low stabil (rentang tetap) ==")
    df_side = make_zigzag(highs=[20, 20, 20, 20], lows=[10, 10, 10, 10])
    structure_side = build_structure_series(df_side, lookback=3)
    label_side = structure_side.iloc[-5]
    check("struktur rentang tetap terklasifikasi 'sideways' (bukan HH/HL murni)",
          label_side == "sideways", f"dapat '{label_side}'")

    print("\n== 4. Awal deret (belum cukup swing) -> 'sideways', bukan menebak ==")
    early_label = structure = build_structure_series(df_up, lookback=3).iloc[2]
    check("bar sangat awal -> 'sideways' (netral, bukan salah tebak arah)",
          early_label == "sideways", f"dapat '{early_label}'")

    print("\n== 5. KENARI LOOK-AHEAD: struktur di bar t tidak berubah kalau ==")
    print("      data SETELAH bar t diubah total")
    df_a = make_zigzag(highs=[10, 20, 30], lows=[2, 8, 18])
    cutoff = len(df_a) - 10  # titik jauh sebelum akhir data
    label_before = build_structure_series(df_a, lookback=3).iloc[cutoff]

    df_b = df_a.copy()
    # Ubah drastis SEMUA bar SETELAH cutoff -- jatuhkan ke nol.
    df_b.iloc[cutoff + 1:, df_b.columns.get_loc("high")] = 0.001
    df_b.iloc[cutoff + 1:, df_b.columns.get_loc("low")] = 0.0001
    df_b.iloc[cutoff + 1:, df_b.columns.get_loc("close")] = 0.0005
    label_after = build_structure_series(df_b, lookback=3).iloc[cutoff]

    check("label di titik cutoff SAMA walau masa depan diubah total",
          label_before == label_after,
          f"sebelum='{label_before}', sesudah='{label_after}'")
    print("    (kalau GAGAL, classify_structure() mengintip swing masa depan --")
    print("     bug serius untuk dipakai di backtest)")

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"GAGAL: {len(FAILURES)} tes -> {FAILURES}")
        return 1
    print("SEMUA TES LULUS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())