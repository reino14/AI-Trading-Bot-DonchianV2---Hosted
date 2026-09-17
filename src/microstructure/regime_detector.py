"""
microstructure/regime_detector.py

Detektor regime pasar: trending / sideways / volatile per bar.
Dipakai regime_router.py untuk mengarahkan bar ke konfigurasi
generate_setups() yang berbeda -- "otak pemilih" sesuai istilah Nero,
bukan AI yang bebas memilih sendiri.

DEFINISI (interpretasi saya, bukan standar tunggal)
------------------------------------------------------
1. VOLATILE menang prioritas: kalau volatilitas realized (std return
   5 menit, rolling `vol_window` bar SEBELUM bar ini -- shift(1) dulu,
   pola sama seperti perbaikan bug filter partisipasi) berada di atas
   ambang kalibrasi -- regime = "volatile", TERLEPAS dari arah struktur.
2. Kalau tidak volatile: TRENDING kalau struktur 1H "up" atau "down".
3. Selain itu (tidak volatile, struktur "sideways"): SIDEWAYS.

Ambang volatilitas WAJIB dikalibrasi dari data sendiri -- disiplin
sama seperti calibrate_absorption_thresholds() dan
compute_participation_threshold(): kalibrasi terpisah dari deteksi,
supaya train/test split nanti tidak bocor.
"""

from __future__ import annotations

import pandas as pd


def _rolling_realized_vol(footprint_df: pd.DataFrame, vol_window: int) -> pd.Series:
    """
    Std return 5-menit dari `vol_window` bar SEBELUM bar ini -- TIDAK
    TERMASUK bar itu sendiri (shift(1) dulu baru rolling), sama seperti
    _rolling_avg_volume() di signal_generator.py. Ini penting supaya
    volatilitas bar itu sendiri (kalau kebetulan bar besar) tidak ikut
    menaikkan pengukuran "volatilitas sekitarnya".
    """
    returns = footprint_df["close"].pct_change()
    return returns.shift(1).rolling(vol_window, min_periods=vol_window).std()


def compute_volatility_threshold(
    footprint_df: pd.DataFrame,
    vol_window: int = 24,
    percentile: float = 0.80,
) -> float:
    """
    Kalibrasi ambang volatilitas dari DataFrame yang diberikan.
    PANGGIL HANYA DENGAN DATA TRAIN kalau dipakai bertahap.

    vol_window=24 default: 24 bar x 5 menit = 2 jam.
    percentile=0.80: ambang = kuartil atas (top 20% momen paling
    volatil dari seluruh riwayat) -- default konservatif, cuma
    menandai ekstrem sungguhan sebagai "volatile", bukan variasi biasa.
    """
    vol = _rolling_realized_vol(footprint_df, vol_window)
    return float(vol.dropna().quantile(percentile))


def classify_regime(
    footprint_df: pd.DataFrame,
    structure_1h: pd.Series,
    vol_threshold: float,
    vol_window: int = 24,
) -> pd.Series:
    """
    Return Series label "trending"/"sideways"/"volatile" per bar di
    footprint_df, index sama. Bar yang belum punya cukup riwayat untuk
    volatilitas (vol_window pertama) ATAU belum punya struktur 1H
    diberi label "sideways" -- default netral, BUKAN menebak arah.

    Memakai pd.merge_asof (vektor, bukan loop per-bar) supaya tetap
    cepat untuk 100rb+ bar -- secara semantik SAMA PERSIS dengan
    "ambil nilai structure_1h TERAKHIR yang timestampnya <= bar ini"
    (persis _structure_as_of() di signal_generator.py), cuma jalannya
    jauh lebih cepat.
    """
    vol = _rolling_realized_vol(footprint_df, vol_window)

    struct_df = structure_1h.rename("bias").reset_index()
    struct_df.columns = ["timestamp", "bias"]
    struct_df = struct_df.sort_values("timestamp")

    fp_times = pd.DataFrame({"timestamp": footprint_df.index}).sort_values("timestamp")
    merged = pd.merge_asof(fp_times, struct_df, on="timestamp", direction="backward")
    bias_at_bar = merged.set_index("timestamp")["bias"].reindex(footprint_df.index)

    is_volatile = vol.notna() & (vol >= vol_threshold)
    is_trending = bias_at_bar.isin(["up", "down"])

    labels = pd.Series("sideways", index=footprint_df.index, name="regime")
    labels[is_trending & ~is_volatile] = "trending"
    labels[is_volatile] = "volatile"
    return labels