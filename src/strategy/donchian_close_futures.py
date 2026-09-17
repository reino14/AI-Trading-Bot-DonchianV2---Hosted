"""
strategy/donchian_close_futures.py

Strategi Donchian close-only, dibungkus supaya cocok dengan kontrak
strategy/base.py milik Nero -- LOGIKA SINYALNYA REUSE LANGSUNG dari
src/strategy/donchian_close.py yang sudah diuji (scripts/smoke_donchian_close.py),
TIDAK ditulis ulang di sini. Satu-satunya hal baru di file ini adalah
PEMBUNGKUS: ubah +1.0/-1.0/NaN jadi Position.LONG/SHORT/FLAT sesuai
kontrak Strategy, dan deklarasi ALLOWS_SHORT.

ALLOWS_SHORT = True -- EKSPLISIT, bukan cuma mengandalkan default kelas
Strategy. Ini strategi FUTURES (selalu di pasar, long/short bergantian),
BUKAN strategi spot -- deklarasi eksplisit ini yang membuat jaring
pengaman di backtest/engine.py TIDAK menahan sinyal SHORT-nya (beda
dari strategi spot lain di proyek ini yang WAJIB ALLOWS_SHORT=False).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategy.donchian_close import compute_donchian_signal
from src.strategy.base import Position, Strategy, StrategyParams


@dataclass(frozen=True)
class DonchianCloseFuturesParams(StrategyParams):
    lookback: int = 238  # default = tengah plateau tervalidasi dari scan 12-672


class DonchianCloseFuturesStrategy(Strategy):
    """
    Selalu di pasar begitu breakout pertama terjadi -- PERSIS logika
    yang sudah divalidasi lewat run_donchian_capital_backtest.py.
    """

    ALLOWS_SHORT: bool = True

    def __init__(self, params: DonchianCloseFuturesParams | None = None):
        super().__init__(params or DonchianCloseFuturesParams())

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        raw = compute_donchian_signal(df["close"], lookback=self.params.lookback)
        # NaN (belum ada breakout pertama, masa pemanasan) -> Position.FLAT.
        # Ini pemetaan yang BENAR, bukan tebakan: "belum tahu arah" secara
        # semantik SAMA dengan "tidak ada posisi" di kontrak Strategy.
        mapped = raw.map({1.0: Position.LONG, -1.0: Position.SHORT})
        return mapped.fillna(Position.FLAT).astype(int)