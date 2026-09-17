"""
microstructure/absorption.py

LANGKAH 6: definisi "absorption" -- INI INTERPRETASI SAYA, BUKAN ATURAN
PASTI CHRIS. Dia tidak pernah memberi angka di transkrip. Definisi di
sini adalah hipotesis yang bisa diuji, dikalibrasi dari sebaran data
sungguhan Anda sendiri (bukan angka bulat yang saya karang) -- tapi
kalau nanti gagal gate, itu bisa berarti definisi saya yang salah,
BUKAN berarti konsep Chris tidak berlaku.

DEFINISI: volume di kuartil ATAS (tinggi) DAN |delta_pct| di kuartil
BAWAH (nyaris seimbang beli-jual) = absorption. Intuisinya: banyak
transaksi terjadi, tapi pembeli dan penjual agresif kira-kira
seimbang -- salah satu penjelasan paling umum untuk itu adalah satu
sisi (mis. penjual) sedang "diserap" oleh sisi lain tanpa berhasil
mendorong harga bergerak searah dorongannya.

KENAPA KALIBRASI AMBANG DIPISAH DARI DETEKSI (calibrate vs detect)
--------------------------------------------------------------------
calibrate_absorption_thresholds() menghitung angka ambang dari
DataFrame yang Anda kasih. detect_absorption() cuma menerima angka
ambang siap pakai, TIDAK menghitung sendiri dari data yang sedang
dites. Ini SENGAJA dipisah: kalau Anda nanti pakai train/test split
(langkah 7 rencana besar), ambang HARUS dikalibrasi dari periode TRAIN
saja, lalu diterapkan ke TEST tanpa dihitung ulang -- memisahkan dua
fungsi ini membuat kebocoran informasi (menghitung ambang dari data
yang sama yang sedang diuji) jadi kesalahan yang harus dilakukan
SENGAJA, bukan default yang diam-diam terjadi.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AbsorptionThresholds:
    volume_threshold: float
    delta_pct_threshold: float  # ambang untuk |delta_pct|, selalu positif


def calibrate_absorption_thresholds(
    footprint_df: pd.DataFrame,
    volume_percentile: float = 0.75,
    delta_pct_percentile: float = 0.25,
) -> AbsorptionThresholds:
    """
    Hitung ambang dari DataFrame yang diberikan. PANGGIL INI HANYA
    DENGAN DATA TRAIN kalau dipakai untuk backtest bertahap -- lihat
    catatan di kepala file.

    volume_percentile=0.75: ambang volume = persentil 75 (kuartil atas).
    delta_pct_percentile=0.25: ambang |delta_pct| = persentil 25 (kuartil
        bawah) DARI |delta_pct| itu sendiri -- bukan dari delta_pct
        mentah (yang bisa negatif).
    """
    volume_threshold = float(footprint_df["volume"].quantile(volume_percentile))
    delta_pct_threshold = float(footprint_df["delta_pct"].abs().quantile(delta_pct_percentile))
    return AbsorptionThresholds(volume_threshold, delta_pct_threshold)


def detect_absorption(
    footprint_df: pd.DataFrame,
    thresholds: AbsorptionThresholds,
) -> pd.Series:
    """
    Return Series boolean, index sama dengan footprint_df: True kalau
    bar itu memenuhi definisi absorption (volume tinggi + delta nyaris
    seimbang) MENURUT ambang yang diberikan -- tidak menghitung ambang
    apa pun dari footprint_df ini sendiri.
    """
    high_volume = footprint_df["volume"] >= thresholds.volume_threshold
    low_delta = footprint_df["delta_pct"].abs() <= thresholds.delta_pct_threshold
    return (high_volume & low_delta).rename("is_absorption")