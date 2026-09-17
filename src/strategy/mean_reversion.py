"""
strategy/mean_reversion.py

Strategi: mean-reversion LONG-ONLY untuk spot -- taruhannya harga yang
menyimpang terlalu jauh DI BAWAH rata-rata bergerak akan kembali naik
mendekat. Kebalikan filosofi dari donchian_breakout.py dan
time_series_momentum.py (dua-duanya sudah gagal menunjukkan edge yang
meyakinkan di BTC/ETH 2 jam, sampel cukup besar).

KENAPA DICOBA SEKARANG: scripts/market_regime.py menemukan BTC/ETH
ranging (Efficiency Ratio < 0.3) sekitar 67% waktu, trending kuat cuma
~8-9% waktu -- bahkan di window 2 jam. Dua pola trend-following yang
sudah dicoba sama-sama gagal. Mean-reversion secara teori lebih cocok
dengan kondisi ranging yang dominan ini.

BEDA DARI strategy/vwap_reversion.py (kandidat #1, gagal telak di
window 30 MENIT):
  1. Window jauh lebih panjang (2 jam+, bukan 30 menit) -- pelajaran
     dari Donchian/momentum bahwa window sangat menentukan hasil.
  2. LONG-ONLY untuk spot (v1 dulu dua arah, dirancang untuk futures).
  3. Filter regime EFISIENSI RENDAH (KEBALIKAN dari filter
     Donchian/momentum yang minta efisiensi TINGGI) -- reversion
     butuh ranging, bukan trending.
  4. Stop-loss berbasis HARGA (pelajaran dari kegagalan VWAP v1 yang
     exit murni berbasis sinyal/waktu, membiarkan kerugian membesar
     tak terkendali).

VERSI LONG-ONLY -- untuk spot trading. ALLOWS_SHORT=False. Ini cuma
menangkap SEPARUH mean-reversion klasik (beli saat harga terlalu
RENDAH, bukan short saat harga terlalu TINGGI) -- konsekuensi wajar
dari larangan short-selling di spot.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategy.base import Position, Strategy, StrategyParams
from src.strategy.regime import MAX_ENTRY_EFFICIENCY_RATIO_REVERSION, efficiency_ratio

#: Jaring pengaman terakhir, bukan parameter yang disetel.
SAFETY_MAX_HOLD_BARS = 200


@dataclass(frozen=True)
class MeanReversionParams(StrategyParams):
    """
    Tiga parameter, sesuai batas Hari 2.

    lookback_bars:
        Window (bar) untuk hitung rata-rata bergerak (SMA) sebagai
        acuan harga "wajar". Default 20 -- disamakan skalanya dengan
        entry_lookback_bars/entry_window_bars strategi sebelumnya
        supaya perbandingan antar pola tetap adil.

    entry_threshold_bps:
        Seberapa jauh harga harus DI BAWAH rata-rata (dalam bps)
        sebelum dianggap "terlalu murah" dan memicu entry. Default
        30.0 -- skala sama dengan threshold strategi sebelumnya,
        BUKAN hasil fitting ke data ini.

    stop_loss_bps:
        Batas kerugian dari harga entry, dalam bps. Pelajaran dari
        strategy/vwap_reversion.py v1: exit murni berbasis sinyal/waktu
        membiarkan kerugian membesar tak terkendali kalau harga terus
        turun alih-alih reversion. Default 100.0 -- tebakan awal
        (bukan fitting), lebih longgar dari VWAP dulu (20 bps) karena
        window di sini jauh lebih panjang (2 jam vs 30 menit) --
        pergerakan wajar per window jauh lebih besar.
    """

    lookback_bars: int = 20
    entry_threshold_bps: float = 30.0
    stop_loss_bps: float = 100.0


class MeanReversionStrategy(Strategy):
    """
    Entry:
        - Close di bawah SMA(lookback_bars) sejauh >= entry_threshold_bps
          DAN Efficiency Ratio menunjukkan market RANGING (filter
          regime, TETAP -- KEBALIKAN dari filter Donchian/momentum:
          reversion butuh ranging, bukan trending) -> LONG.

    Exit (salah satu terjadi lebih dulu):
        - Harga kembali ke/atas SMA (deviasi >= 0) -> reversion
          tercapai, keluar.
        - Kerugian mencapai stop_loss_bps dari harga entry -> stop-loss.
        - SAFETY_MAX_HOLD_BARS tercapai -> jaring pengaman terakhir.
    """

    ALLOWS_SHORT = False  # spot -- cuma menangkap separuh reversion (beli saat murah)

    def __init__(self, params: MeanReversionParams | None = None):
        super().__init__(params or MeanReversionParams())

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p: MeanReversionParams = self.params

        sma = df["close"].rolling(p.lookback_bars).mean()
        deviation_bps = (df["close"] - sma) / sma * 10_000

        er = efficiency_ratio(df["close"], p.lookback_bars).shift(1)

        signals = pd.Series(Position.FLAT, index=df.index, dtype=int)

        position = Position.FLAT
        entry_price = None
        bars_held = 0

        for i in range(len(df)):
            close_price = df["close"].iloc[i]
            dev = deviation_bps.iloc[i]
            er_i = er.iloc[i]

            if position == Position.FLAT:
                if pd.isna(dev) or pd.isna(er_i):
                    signals.iloc[i] = Position.FLAT
                    continue

                ranging_enough = er_i <= MAX_ENTRY_EFFICIENCY_RATIO_REVERSION
                if ranging_enough and dev <= -p.entry_threshold_bps:
                    position = Position.LONG
                    entry_price = close_price
                    bars_held = 0
                # Deviasi positif kuat (harga terlalu TINGGI) TIDAK
                # diproses -- tidak ada cabang SHORT (spot, long-only).

            else:  # position == Position.LONG
                bars_held += 1

                move_bps = (close_price - entry_price) / entry_price * 10_000
                reverted = dev >= 0  # harga sudah balik ke/atas rata-rata
                stopped_out = move_bps <= -p.stop_loss_bps
                timed_out = bars_held >= SAFETY_MAX_HOLD_BARS

                if reverted or stopped_out or timed_out:
                    position = Position.FLAT
                    entry_price = None
                    bars_held = 0

            signals.iloc[i] = position

        return signals


if __name__ == "__main__":
    # Uji asap pakai RANDOM WALK MURNI (bukan dibuat mean-reverting
    # buatan) -- kalau strategi ini "menang" di data acak murni, itu
    # tanda ada bug, bukan tanda strategi bagus.
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2026-01-01", periods=n, freq="2h", tz="UTC")
    price = 100_000 + np.cumsum(rng.normal(0, 50, n))
    df = pd.DataFrame(
        {
            "open": price,
            "high": price + rng.uniform(0, 40, n),
            "low": price - rng.uniform(0, 40, n),
            "close": price + rng.normal(0, 15, n),
            "volume": rng.uniform(100, 1000, n),
        },
        index=idx,
    )

    strat = MeanReversionStrategy()
    sig = strat.generate_signals(df)

    print(f"Strategi: {strat.describe()}")
    print(f"ALLOWS_SHORT: {strat.ALLOWS_SHORT}")
    print(f"Bar dengan sinyal LONG : {(sig == Position.LONG).sum()}")
    print(f"Bar dengan sinyal SHORT: {(sig == Position.SHORT).sum()}")
    print(f"Bar dengan sinyal FLAT : {(sig == Position.FLAT).sum()}")