"""
microstructure/regime_router.py

"Otak pemilih strategi" sesuai istilah Nero -- BUKAN AI yang bebas
memilih. Router ini cuma menjalankan generate_setups() dengan
konfigurasi BERBEDA per regime, lalu MENYARING hasilnya supaya tiap
setup cuma dipakai kalau bar tempat ia muncul memang diklasifikasi
regime yang sesuai. Semua keputusan "kapan pakai strategi apa" sudah
Anda tentukan di depan lewat regime_configs -- router tidak menebak
apa pun sendiri.

KENAPA CARANYA "JALANKAN PENUH, LALU SARING" -- BUKAN "POTONG DATA
DULU PER REGIME"
-------------------------------------------------------------------
generate_setups() butuh KONTEKS MASA LALU PENUH untuk fungsi
look-ahead-safe-nya (rolling volume, swing point, referensi hari
sebelumnya, state machine dua-percobaan). Kalau data dipotong dulu per
regime SEBELUM dipanggil, semua jendela rolling dan referensi masa
lalu itu rusak (mis. "bar sebelumnya" jadi bukan bar sebelumnya yang
sungguhan). Jadi router menjalankan generate_setups() di SELURUH data
historis untuk tiap konfigurasi regime, dan BARU SETELAH ITU membuang
setup yang bar konfirmasinya ternyata bukan regime yang dimaksud.

Ini AMAN dari look-ahead: regime_series sendiri sudah causal (lihat
regime_detector.py, diverifikasi kenari look-ahead sendiri), jadi
menyaring "apakah bar t regime X" tidak pernah melihat masa depan.
"""

from __future__ import annotations

import pandas as pd

from src.microstructure.signal_generator import generate_setups


def route_by_regime(
    footprint_df: pd.DataFrame,
    structure_1h: pd.Series,
    structure_4h: pd.Series,
    daily_profile: pd.DataFrame,
    regime_series: pd.Series,
    regime_configs: dict[str, dict],
) -> pd.DataFrame:
    """
    regime_series: hasil classify_regime() -- label per bar, index
        SAMA PERSIS dengan footprint_df.
    regime_configs: {label_regime: kwargs_untuk_generate_setups}.
        Regime yang TIDAK ADA di dict ini otomatis TIDAK PERNAH trading
        (mis. kalau Anda cuma isi "trending" dan "volatile", bar
        "sideways" tidak akan pernah menghasilkan setup lewat jalur
        ini -- kecuali Anda sengaja isi kwargs untuk "sideways" juga,
        mis. bias_mode="none").

    Return: gabungan semua setup dari semua regime, urut waktu.
        Kalau dua regime kebetulan menghasilkan setup PERSIS di bar
        yang sama (jarang, tapi mungkin kalau kwargs tumpang tindih),
        KEDUANYA tetap muncul -- trade_simulator.py yang nanti
        memutuskan mana yang dieksekusi lebih dulu lewat aturan
        "satu posisi pada satu waktu" yang sudah ada.
    """
    all_setups = []

    for regime_label, gen_kwargs in regime_configs.items():
        setups = generate_setups(
            footprint_df, structure_1h, structure_4h, daily_profile, **gen_kwargs
        )
        if setups.empty:
            continue

        bar_regime = regime_series.reindex(setups.index)
        matches = setups[bar_regime == regime_label].copy()
        matches["regime"] = regime_label
        if not matches.empty:
            all_setups.append(matches)

    if not all_setups:
        return pd.DataFrame()

    combined = pd.concat(all_setups).sort_index()
    return combined