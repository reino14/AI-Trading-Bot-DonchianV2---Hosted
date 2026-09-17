"""
microstructure/smc_structure.py

SMC LANGKAH 1: Break of Structure (BOS) dan Change of Character (CHoCH)
-- KONSEP INI SUDAH PUNYA FONDASI DI KODE KITA. BOS/CHoCH secara
mekanis SAMA PERSIS dengan swing HH/HL yang sudah dibangun di
market_structure.py -- yang baru di sini cuma LABEL PERISTIWA
("tembus swing tertentu") di atas fondasi yang sudah teruji, bukan
konsep baru dari nol.

DEFINISI (interpretasi saya -- SMC punya beberapa varian definisi
antar edukator, ini bukan satu-satunya yang "benar")
------------------------------------------------------------------------
BOS (Break of Structure): harga close TEMBUS swing high/low SEARAH
    tren yang sedang berjalan -- KONFIRMASI kelanjutan tren.
    Tren naik + tembus swing high terbaru = BOS bullish.
    Tren turun + tembus swing low terbaru = BOS bearish.

CHoCH (Change of Character): harga close TEMBUS swing high/low
    BERLAWANAN dengan tren yang sedang berjalan -- SINYAL PERTAMA
    kemungkinan tren berbalik.
    Tren naik + tembus swing LOW terbaru (bukan high) = CHoCH bearish.
    Tren turun + tembus swing HIGH terbaru = CHoCH bullish.
    Tren "sideways" (belum ada arah) + tembus high ATAU low = CHoCH
    ke arah itu (sinyal awal tren baru mulai terbentuk).

MEKANISME "PATAH SEKALI, BUKAN TIAP BAR" -- state machine
------------------------------------------------------------------------
Swing high/low TERBARU yang TERKONFIRMASI jadi "level yang diawasi".
Begitu level itu tertembus, peristiwa dicatat SEKALI, level itu
ditandai "sudah tertembus" -- TIDAK terus-menerus mencatat peristiwa
baru tiap bar selama harga tetap di atas/bawah level itu. Level yang
diawasi baru berganti begitu swing BARU terkonfirmasi.

TIDAK ADA LOOK-AHEAD: level yang diawasi cuma diperbarui begitu swing
BENAR-BENAR terkonfirmasi (index+lookback <= bar_pos, persis definisi
find_swing_points()) -- peristiwa di bar t tidak pernah bergantung
pada swing yang baru pasti setelah bar t.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.microstructure.market_structure import build_structure_series, find_swing_points


@dataclass
class StructureEvent:
    timestamp: pd.Timestamp
    bar_pos: int
    event_type: str  # "BOS_bullish", "BOS_bearish", "CHoCH_bullish", "CHoCH_bearish"
    broken_price: float  # level swing yang ditembus
    close_price: float  # harga close bar yang menembus
    trend_before: str  # trend SEBELUM peristiwa ("up"/"down"/"sideways")


def detect_bos_choch(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    df: WAJIB punya kolom high, low, close, index datetime UTC urut naik
        (persis skema yang dipakai market_structure.py).

    Return: DataFrame peristiwa, urut waktu, kolom sesuai StructureEvent.
        KOSONG kalau tidak ada peristiwa terdeteksi -- bukan error.
    """
    swings = find_swing_points(df, lookback=lookback)
    trend_series = build_structure_series(df, lookback=lookback)

    # Kelompokkan swing berdasarkan bar_pos di mana ia TERKONFIRMASI
    # (bukan bar_pos kemunculannya sendiri) -- supaya bisa diperbarui
    # tepat pada saat konfirmasi, bukan lebih awal.
    confirm_at: dict[int, list] = {}
    for s in swings:
        confirm_pos = s.index + lookback
        confirm_at.setdefault(confirm_pos, []).append(s)

    pending_high_price: float | None = None
    pending_high_broken = True  # True = belum ada level valid untuk diawasi
    pending_low_price: float | None = None
    pending_low_broken = True

    events: list[StructureEvent] = []

    for bar_pos, (t, bar) in enumerate(df.iterrows()):
        for s in confirm_at.get(bar_pos, []):
            if s.kind == "high":
                pending_high_price = s.price
                pending_high_broken = False
            else:
                pending_low_price = s.price
                pending_low_broken = False

        trend = trend_series.iloc[bar_pos]
        close = float(bar["close"])

        if not pending_high_broken and close > pending_high_price:
            event_type = "BOS_bullish" if trend == "up" else "CHoCH_bullish"
            events.append(StructureEvent(
                timestamp=t, bar_pos=bar_pos, event_type=event_type,
                broken_price=pending_high_price, close_price=close, trend_before=trend,
            ))
            pending_high_broken = True

        if not pending_low_broken and close < pending_low_price:
            event_type = "BOS_bearish" if trend == "down" else "CHoCH_bearish"
            events.append(StructureEvent(
                timestamp=t, bar_pos=bar_pos, event_type=event_type,
                broken_price=pending_low_price, close_price=close, trend_before=trend,
            ))
            pending_low_broken = True

    if not events:
        return pd.DataFrame(columns=[
            "timestamp", "bar_pos", "event_type", "broken_price", "close_price", "trend_before",
        ])
    return pd.DataFrame([vars(e) for e in events]).set_index("timestamp").sort_index()