"""
microstructure/market_structure.py

LANGKAH 3: klasifikasi struktur value up/down/sideways dari swing
high/low -- padanan objektif untuk "environment" yang Chris tentukan
dengan mata dari chart 1H/4H.

DEFINISI SWING POINT (fractal sederhana, pilihan saya -- ada varian
lain di literatur teknikal, ini bukan satu-satunya definisi "benar")
--------------------------------------------------------------------
Bar ke-i adalah SWING HIGH kalau high-nya lebih tinggi dari `lookback`
bar sebelum DAN sesudahnya. Swing LOW simetris (lebih rendah dari
sekitarnya). `lookback` default 3 -- semakin besar, semakin sedikit
swing point yang terdeteksi (lebih menyaring noise), semakin kecil,
semakin sensitif.

KLASIFIKASI STRUKTUR
---------------------
Higher High + Higher Low berturut-turut (dibanding swing SEBELUMNYA
dengan jenis sama) -> "up". Lower High + Lower Low -> "down". Kalau
tidak keduanya konsisten -> "sideways". Ini penyederhanaan -- Chris di
transkrip cuma bilang "higher highs, higher lows" tanpa mendefinisikan
berapa banyak swing berturut-turut yang dibutuhkan sebelum menyebut
struktur "berubah". Saya pakai 2 swing high + 2 swing low terakhir
sebagai jendela minimal -- INI PARAMETER, bukan aturan pasti Chris.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SwingPoint:
    index: int
    timestamp: pd.Timestamp
    price: float
    kind: str  # "high" atau "low"


def find_swing_points(df: pd.DataFrame, lookback: int = 3) -> list[SwingPoint]:
    """
    df: WAJIB punya kolom 'high', 'low', index datetime UTC terurut naik.

    Bar di ujung (kurang dari `lookback` bar di salah satu sisi) tidak
    bisa dinilai -- dilewati, bukan dipaksa true/false.
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    points: list[SwingPoint] = []

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback: i + lookback + 1]
        if highs[i] == window_h.max() and np.sum(window_h == highs[i]) == 1:
            points.append(SwingPoint(i, df.index[i], float(highs[i]), "high"))
        window_l = lows[i - lookback: i + lookback + 1]
        if lows[i] == window_l.min() and np.sum(window_l == lows[i]) == 1:
            points.append(SwingPoint(i, df.index[i], float(lows[i]), "low"))

    return sorted(points, key=lambda p: p.index)


def classify_structure(
    swings: list[SwingPoint],
    as_of_index: int,
) -> str:
    """
    Klasifikasi struktur pada titik waktu `as_of_index`, memakai HANYA
    swing point yang index-nya < as_of_index (tidak boleh mengintip
    swing yang baru terkonfirmasi setelah titik ini -- swing high/low
    baru bisa dikonfirmasi `lookback` bar SETELAH titik itu sendiri,
    jadi ini juga menghindari look-ahead kalau dipakai di backtest).

    Return: "up", "down", atau "sideways".
    """
    visible = [s for s in swings if s.index < as_of_index]
    recent_highs = [s for s in visible if s.kind == "high"][-2:]
    recent_lows = [s for s in visible if s.kind == "low"][-2:]

    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return "sideways"  # belum cukup data -- default netral, bukan menebak

    higher_high = recent_highs[-1].price > recent_highs[-2].price
    higher_low = recent_lows[-1].price > recent_lows[-2].price
    lower_high = recent_highs[-1].price < recent_highs[-2].price
    lower_low = recent_lows[-1].price < recent_lows[-2].price

    if higher_high and higher_low:
        return "up"
    if lower_high and lower_low:
        return "down"
    return "sideways"


def build_structure_series(df: pd.DataFrame, lookback: int = 3) -> pd.Series:
    """
    Hitung label struktur ("up"/"down"/"sideways") untuk SETIAP bar di
    df, dari data yang sudah bisa dilihat sampai bar itu -- aman dipakai
    sebagai fitur backtest (tidak mengintip masa depan).
    """
    swings = find_swing_points(df, lookback=lookback)
    labels = [classify_structure(swings, i) for i in range(len(df))]
    return pd.Series(labels, index=df.index, name="structure")