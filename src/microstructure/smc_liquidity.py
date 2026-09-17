"""
microstructure/smc_liquidity.py

SMC LANGKAH 3: Liquidity Sweep -- komponen terakhir dari daftar awal.
Reuse find_swing_points() (sama seperti smc_structure.py), TAPI
logikanya BEDA dari BOS/CHoCH: bukan "tembus dan bertahan", melainkan
"tembus lewat sumbu (wick) lalu DITOLAK kembali sebelum bar tutup" --
sinyal stop-hunt/likuidasi, bukan breakout sungguhan.

DEFINISI (interpretasi saya)
------------------------------------------------------------------------
sweep_high (sinyal BEARISH -- rejection dari atas): HIGH bar menembus
    swing high yang diawasi, TAPI CLOSE bar itu kembali di BAWAH level
    itu. Diartikan: harga "mengambil" stop-loss/liquidity di atas
    swing high, lalu ditolak -- calon pembalikan turun.
sweep_low (sinyal BULLISH): cermin -- LOW menembus swing low yang
    diawasi, CLOSE kembali di ATAS level itu.

KENAPA STATE-NYA TERPISAH DARI detect_bos_choch() (bukan digabung)
------------------------------------------------------------------------
Sweep TIDAK "mengonsumsi" level yang diawasi -- beda dari BOS yang
begitu tertembus (close-through) langsung ditandai selesai. Level yang
di-sweep TETAP diawasi setelahnya, karena secara struktur belum benar-
benar ditembus (cuma sumbu-nya, bukan closing-nya). Ini SENGAJA
dibiarkan bisa memicu BERKALI-KALI pada level yang sama kalau harga
berulang kali menguji dan ditolak di level itu -- setiap wick-reject
adalah peristiwa tersendiri, beda dari BOS yang cuma sekali per level.
Menggabungkan state dengan detect_bos_choch() berisiko mengganggu
logika BOS yang sudah teruji, jadi dibuat modul independen.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.microstructure.market_structure import find_swing_points


@dataclass
class LiquiditySweepEvent:
    timestamp: pd.Timestamp
    bar_pos: int
    event_type: str  # "sweep_high" (bearish) atau "sweep_low" (bullish)
    swept_level: float
    wick_extreme: float  # high (sweep_high) atau low (sweep_low) bar itu
    close_price: float


def detect_liquidity_sweeps(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    df: WAJIB punya kolom high, low, close, index datetime UTC urut naik.

    Return: DataFrame urut waktu peristiwa. Kosong kalau tidak ada
        sweep terdeteksi -- bukan error. Level yang di-sweep TETAP
        "berlaku" untuk sweep berikutnya (lihat catatan modul) --
        cuma diperbarui begitu swing BARU terkonfirmasi, sama seperti
        detect_bos_choch().
    """
    swings = find_swing_points(df, lookback=lookback)

    confirm_at: dict[int, list] = {}
    for s in swings:
        confirm_at.setdefault(s.index + lookback, []).append(s)

    pending_high_price: float | None = None
    pending_low_price: float | None = None

    events: list[LiquiditySweepEvent] = []

    for bar_pos, (t, bar) in enumerate(df.iterrows()):
        for s in confirm_at.get(bar_pos, []):
            if s.kind == "high":
                pending_high_price = s.price
            else:
                pending_low_price = s.price

        high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])

        if pending_high_price is not None and high > pending_high_price and close < pending_high_price:
            events.append(LiquiditySweepEvent(
                timestamp=t, bar_pos=bar_pos, event_type="sweep_high",
                swept_level=pending_high_price, wick_extreme=high, close_price=close,
            ))

        if pending_low_price is not None and low < pending_low_price and close > pending_low_price:
            events.append(LiquiditySweepEvent(
                timestamp=t, bar_pos=bar_pos, event_type="sweep_low",
                swept_level=pending_low_price, wick_extreme=low, close_price=close,
            ))

    if not events:
        return pd.DataFrame(columns=[
            "timestamp", "bar_pos", "event_type", "swept_level", "wick_extreme", "close_price",
        ])
    return pd.DataFrame([vars(e) for e in events]).set_index("timestamp").sort_index()