"""
microstructure/smc_zones.py

SMC LANGKAH 4 (terakhir dari toolkit): Premium/Discount zone.
Reuse find_swing_points() lagi -- sama fondasi seperti BOS/CHoCH,
Order Block, dan Liquidity Sweep.

DEFINISI (interpretasi saya)
------------------------------------------------------------------------
Equilibrium (titik tengah) = (swing_high + swing_low) / 2, dari swing
high DAN swing low TERKONFIRMASI TERBARU -- masing-masing dicari
independen, TIDAK perlu berurutan tertentu (beda dari zona Fibonacci
di proyek Chris yang butuh pasangan low->high searah tren). Ini
mencerminkan cara SMC memandang "dealing range" saat ini: rentang
antara ekstrem tertinggi dan terendah yang baru saja terbentuk,
apa pun urutan kemunculannya.

close > equilibrium -> "premium" (mahal, kandidat cari SHORT).
close < equilibrium -> "discount" (murah, kandidat cari LONG).
Belum ada swing high ATAU swing low yang terkonfirmasi -> "unknown"
(BUKAN ditebak jadi premium atau discount).
"""

from __future__ import annotations

import pandas as pd

from src.microstructure.market_structure import find_swing_points


def compute_premium_discount(df: pd.DataFrame, lookback: int = 3) -> pd.Series:
    """
    df: WAJIB punya kolom high, low, close, index datetime UTC urut naik.

    Return: Series label "premium"/"discount"/"unknown" per bar, index
        sama dengan df. TIDAK ADA LOOK-AHEAD: swing yang dipakai di bar
        t cuma yang SUDAH terkonfirmasi pada bar_pos <= t (index+lookback
        <= bar_pos), persis disiplin yang sama seperti smc_structure.py
        dan smc_liquidity.py.
    """
    swings = find_swing_points(df, lookback=lookback)
    highs_by_confirm: dict[int, float] = {}
    lows_by_confirm: dict[int, float] = {}
    for s in swings:
        confirm_pos = s.index + lookback
        if s.kind == "high":
            highs_by_confirm[confirm_pos] = s.price
        else:
            lows_by_confirm[confirm_pos] = s.price

    labels = []
    last_high: float | None = None
    last_low: float | None = None

    for bar_pos, (t, bar) in enumerate(df.iterrows()):
        if bar_pos in highs_by_confirm:
            last_high = highs_by_confirm[bar_pos]
        if bar_pos in lows_by_confirm:
            last_low = lows_by_confirm[bar_pos]

        if last_high is None or last_low is None:
            labels.append("unknown")
            continue

        equilibrium = (last_high + last_low) / 2.0
        close = float(bar["close"])
        labels.append("premium" if close > equilibrium else "discount")

    return pd.Series(labels, index=df.index, name="zone")