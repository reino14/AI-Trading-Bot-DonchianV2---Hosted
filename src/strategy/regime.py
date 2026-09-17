"""
strategy/regime.py

Utilitas regime pasar yang dipakai BERSAMA oleh lebih dari satu strategi
(donchian_breakout.py, time_series_momentum.py, dst) -- dipisah ke sini
supaya tidak duplikat logika yang sama di tiap file strategi baru.

Lihat scripts/market_regime.py untuk penjelasan lengkap Efficiency Ratio
dan temuan kunci proyek ini: BTC/ETH ranging ~67% waktu, trending kuat
cuma ~8-9% waktu, bahkan di window 2 jam.
"""

import pandas as pd

#: Ambang Efficiency Ratio minimal supaya entry MOMENTUM/BREAKOUT
#: diambil -- TETAP, bukan parameter yang disetel per strategi. Sama
#: persis dengan ambang "ranging" di scripts/market_regime.py, supaya
#: konsisten lintas strategi dan bukan angka yang dicari-cari.
MIN_ENTRY_EFFICIENCY_RATIO = 0.3

#: Ambang Efficiency Ratio MAKSIMAL supaya entry MEAN-REVERSION diambil
#: -- KEBALIKAN dari ambang di atas. Reversion secara teori butuh
#: kondisi RANGING (ER rendah), bukan trending -- kalau market sedang
#: trending kuat, harga yang "menyimpang jauh" justru mungkin memulai
#: tren baru, bukan bersiap kembali ke rata-rata. Nilai SAMA (0.3),
#: cuma arah pembandingnya dibalik -- bukan angka baru yang dicari-cari.
MAX_ENTRY_EFFICIENCY_RATIO_REVERSION = 0.3


def efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    """Kaufman's Efficiency Ratio -- lihat scripts/market_regime.py untuk penjelasan lengkap."""
    net_change = (close - close.shift(window)).abs()
    path_length = close.diff().abs().rolling(window).sum()
    return net_change / path_length