"""
strategy/donchian_close_live.py

Versi LIVE (per-bar) dari logika di donchian_close.py yang sudah
diuji lewat scripts/smoke_donchian_close.py. TIDAK menimpa file batch
yang sudah ada -- ini pembungkus tambahan untuk dipanggil tiap ada
candle baru yang closed, bukan sekali untuk seluruh histori.

CATATAN PENTING: fungsi ini FUNGSI MURNI (tidak simpan state internal)
supaya gampang diverifikasi -- caller (runner/broker Anda) yang
bertanggung jawab menyimpan `last_signal` antar pemanggilan dan
menyediakan histori close yang cukup (minimal `lookback+1` bar
TERAKHIR yang sudah closed, TIDAK termasuk candle yang sedang
berjalan/belum closed).
"""

from __future__ import annotations

import numpy as np


def next_signal(recent_closes: list[float] | np.ndarray, lookback: int, last_signal: float | None) -> float | None:
    """
    recent_closes: WAJIB minimal `lookback + 1` harga close TERBARU
        yang SUDAH CLOSED, urut waktu (elemen terakhir = candle yang
        baru saja closed). JANGAN masukkan candle yang belum closed.
    lookback: SAMA PERSIS dengan yang dipakai backtest (jangan diam-
        diam beda dari nilai yang sudah divalidasi).
    last_signal: posisi saat ini (+1.0 / -1.0 / None kalau belum ada
        posisi sama sekali).

    Return: sinyal BARU (+1.0/-1.0), atau `last_signal` kalau tidak
        ada breakout di candle yang baru closed (posisi dipertahankan).
        None kalau data belum cukup (< lookback+1 close).

    SETARA MATEMATIS dengan compute_donchian_signal() versi batch --
    channel dihitung dari `lookback` close SEBELUM candle yang baru
    closed (TIDAK termasuk candle itu sendiri), dibandingkan dengan
    close candle itu.
    """
    closes = np.asarray(recent_closes, dtype=float)
    if len(closes) < lookback + 1:
        return None  # data belum cukup -- JANGAN menebak, jangan kasih sinyal palsu

    window = closes[-(lookback + 1):-1]  # `lookback` bar SEBELUM candle terakhir
    latest_close = closes[-1]

    upper = window.max()
    lower = window.min()

    if latest_close > upper:
        return 1.0
    if latest_close < lower:
        return -1.0
    return last_signal  # tidak ada breakout -- pertahankan posisi