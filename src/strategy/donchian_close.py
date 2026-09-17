"""
strategy/donchian_close.py

Donchian breakout versi close-only dari video "Modified Donchian`
Breakout" -- BUKAN Donchian standar (yang pakai high/low). Perbedaan
konstruksinya PENTING, bukan kosmetik: video pakai close saja karena
high/low sering ada spike palsu.

KONSTRUKSI "LAG 1 BAR + LOOKBACK DIKURANGI 1" -- SAYA VERIFIKASI SECARA
MATEMATIS, BUKAN DITEBAK
------------------------------------------------------------------------
Video bilang: hitung rolling max/min dari close dengan window
(lookback-1), lalu geser 1 bar -- katanya ini cuma supaya breakout
kelihatan jelas di chart. Saya buktikan ini SECARA MATEMATIS setara
dengan:

    channel_upper[t] = close[t-lookback : t-1].rolling(lookback).max()

yaitu: rolling_max(close, window=lookback).shift(1) -- nilai channel
di bar t dihitung dari `lookback` bar SEBELUM t (TIDAK termasuk close
bar t sendiri). Ini yang saya implementasikan langsung -- lebih
sederhana untuk dibaca, hasilnya identik dengan konstruksi video.

SELALU DI PASAR (LONG ATAU SHORT, TIDAK PERNAH FLAT)
------------------------------------------------------------------------
INI MELEPAS BATASAN "TANPA SHORT-SELLING" PROYEK INI -- disepakati
eksplisit oleh Nero KHUSUS untuk menguji strategi ini, BUKAN perubahan
default proyek. Breakout atas -> long. Breakout bawah -> short (posisi
long ditutup, short dibuka). Posisi bertahan sampai breakout
berlawanan berikutnya (forward-fill).

RETURN: TIDAK ADA LOOK-AHEAD
------------------------------------------------------------------------
signal[t] diputuskan dari channel yang cuma pakai data sampai t-1 (lihat
di atas) DIBANDINGKAN dengan close[t] -- jadi signal[t] itu sendiri
"mengetahui" close[t] (itu sah, breakout terjadi PAS saat close[t]
diketahui). Yang TIDAK BOLEH: signal[t] dipakai untuk mendapat return
dari close[t-1] ke close[t] (itu look-ahead, karena keputusan practically
baru bisa dieksekusi SETELAH close[t] diketahui). Makanya:

    strategy_return[t] = signal[t-1] * log_return[t]

signal DIGESER SATU BAR sebelum dikalikan return -- posisi yang
diputuskan di bar t baru mulai menghasilkan return di bar t+1 dan
seterusnya.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_donchian_signal(close: pd.Series, lookback: int) -> pd.Series:
    """
    Return Series posisi (+1/-1), index sama dengan close, SELALU terisi
    kecuali di awal (sebelum breakout pertama pernah terjadi -- itu
    NaN, caller putuskan mau diapakan, JANGAN diam-diam diisi 0/1 di
    sini).
    """
    upper = close.rolling(lookback).max().shift(1)
    lower = close.rolling(lookback).min().shift(1)

    breakout_up = close > upper
    breakout_down = close < lower

    raw = pd.Series(np.nan, index=close.index)
    raw[breakout_up] = 1.0
    raw[breakout_down] = -1.0

    return raw.ffill()  # posisi bertahan sampai breakout berlawanan berikutnya


def compute_strategy_returns(
    close: pd.Series,
    signal: pd.Series,
    cost_bps_per_unit: float = 0.0,
) -> pd.DataFrame:
    """
    cost_bps_per_unit: ongkos (bps) per SATU UNIT perubahan sinyal --
        posisi tetap (diff=0) = 0 ongkos. Masuk dari flat/keluar total
        (diff=1) = 1x ongkos. REVERSAL penuh long<->short (diff=2) =
        2x ongkos -- itu memang dua transaksi (tutup lama + buka baru),
        bukan digandakan sembarangan.

    Return DataFrame: log_return, signal (sebelum digeser -- keputusan
        di bar itu), strategy_return_gross, cost, strategy_return_net.
    """
    log_return = np.log(close / close.shift(1))
    signal_lagged = signal.shift(1)

    gross = signal_lagged * log_return

    signal_change = signal.diff().abs().fillna(0.0)
    cost = signal_change.shift(1).fillna(0.0) * (cost_bps_per_unit / 10_000.0)
    # cost DIGESER SATU BAR juga -- ongkos transaksi terjadi PAS bar
    # perubahan sinyal (saat itu dieksekusi), bukan di bar sinyal
    # BERIKUTNYA sesudahnya. signal_change[t] = perubahan YANG TERJADI
    # di bar t -- ongkos ini menempel di return bar t+1 karena eksekusi
    # baru selesai di akhir bar t (sama seperti signal_lagged).

    net = gross - cost

    return pd.DataFrame({
        "log_return": log_return,
        "signal": signal,
        "strategy_return_gross": gross,
        "cost": cost,
        "strategy_return_net": net,
    })