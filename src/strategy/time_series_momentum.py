"""
strategy/time_series_momentum.py

Strategi: time-series momentum murni (return-based), BUKAN channel
breakout. Pola ini yang paling didukung riset akademik lintas aset
selama puluhan tahun (Moskowitz, Ooi & Pedersen 2012; Hurst, Ooi &
Pedersen 2017 -- data 1880-2016, rasio Sharpe ~0.4 per dekade).

BEDA DARI src/strategy/donchian_breakout.py (kandidat #2, gagal --
lihat catatan proyek): Donchian mengecek apakah harga MENEMBUS REKOR
channel N-bar -- sensitif ke SATU bar ekstrem (satu wick bisa memicu
atau menggagalkan sinyal). Time-series momentum di sini mengecek RETURN
rata-rata N-bar -- lebih smooth, tidak segampang "ketipu" satu candle
nyasar. Ini menguji apakah kegagalan Donchian karena momentum memang
tidak ada di BTC/ETH window ini, atau karena CARA DETEKSI-nya (channel
breakout) yang terlalu kasar.

VERSI LONG-ONLY -- untuk spot trading (tidak ada short-selling, tidak
ada leverage), sama seperti Donchian versi terbaru. ALLOWS_SHORT=False.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategy.base import Position, Strategy, StrategyParams
from src.strategy.regime import MIN_ENTRY_EFFICIENCY_RATIO, efficiency_ratio

#: Jaring pengaman terakhir, bukan parameter yang disetel -- sama pola
#: seperti di donchian_breakout.py.
SAFETY_MAX_HOLD_BARS = 200


@dataclass(frozen=True)
class TimeSeriesMomentumParams(StrategyParams):
    """
    Tiga parameter, sesuai batas Hari 2.

    entry_lookback_bars:
        Window (bar) untuk hitung return MASUK: (close sekarang - close
        entry_lookback_bars yang lalu) / close saat itu. Default 20,
        SENGAJA disamakan skalanya dengan entry_window_bars Donchian
        yang sudah diuji -- supaya perbandingan hasil dua pola ini adil
        (beda karena CARA deteksi momentum, bukan beda karena window).

    exit_lookback_bars:
        Window LEBIH PENDEK untuk hitung return KELUAR. Default 10,
        pola asimetris yang sama seperti Donchian (keluar lebih cepat
        bereaksi daripada masuk, supaya tidak menunggu reversal penuh).

    entry_threshold_bps:
        Ambang minimal besar return MASUK, dalam bps, supaya momentum
        yang terlalu lemah/noise tidak memicu entry. Default 30.0 --
        tebakan awal masuk akal (skala sama dengan entry_threshold_bps
        VWAP dulu), BUKAN hasil fitting ke data ini.
    """

    entry_lookback_bars: int = 20
    exit_lookback_bars: int = 10
    entry_threshold_bps: float = 30.0


class TimeSeriesMomentumStrategy(Strategy):
    """
    Entry:
        - Return entry_lookback_bars bar terakhir >= entry_threshold_bps
          (momentum naik cukup kuat) DAN Efficiency Ratio menunjukkan
          market sedang trending (filter regime, TETAP, sama seperti
          Donchian) -> LONG.

    Exit (salah satu terjadi lebih dulu):
        - Return exit_lookback_bars bar terakhir berbalik NEGATIF
          (momentum jangka pendek sudah membalik) -> keluar.
        - SAFETY_MAX_HOLD_BARS tercapai -> jaring pengaman terakhir.
    """

    ALLOWS_SHORT = False  # spot -- tidak ada short-selling

    def __init__(self, params: TimeSeriesMomentumParams | None = None):
        super().__init__(params or TimeSeriesMomentumParams())

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p: TimeSeriesMomentumParams = self.params

        # Return N-bar: pakai close SEKARANG langsung (bukan di-shift) --
        # ini memang niat sinyal itu sendiri: "seberapa jauh harga sudah
        # bergerak sampai close bar ini", sama seperti Donchian membanding-
        # kan close_price sekarang ke channel historis -- bukan look-ahead
        # bias (tidak ada info MASA DEPAN yang dipakai).
        entry_return_bps = (
            (df["close"] - df["close"].shift(p.entry_lookback_bars))
            / df["close"].shift(p.entry_lookback_bars)
            * 10_000
        )
        exit_return_bps = (
            (df["close"] - df["close"].shift(p.exit_lookback_bars))
            / df["close"].shift(p.exit_lookback_bars)
            * 10_000
        )
        # Filter regime -- di-shift(1) supaya ER yang dilihat di bar i
        # dihitung dari bar-bar SEBELUM bar i, konsisten dengan
        # donchian_breakout.py.
        er = efficiency_ratio(df["close"], p.entry_lookback_bars).shift(1)

        signals = pd.Series(Position.FLAT, index=df.index, dtype=int)

        position = Position.FLAT
        bars_held = 0

        for i in range(len(df)):
            entry_ret = entry_return_bps.iloc[i]
            exit_ret = exit_return_bps.iloc[i]
            er_i = er.iloc[i]

            if position == Position.FLAT:
                if pd.isna(entry_ret) or pd.isna(er_i):
                    signals.iloc[i] = Position.FLAT
                    continue

                trending_enough = er_i >= MIN_ENTRY_EFFICIENCY_RATIO
                if trending_enough and entry_ret >= p.entry_threshold_bps:
                    position = Position.LONG
                    bars_held = 0
                # Momentum negatif kuat TIDAK diproses sama sekali --
                # tidak ada cabang SHORT (spot, long-only).

            else:  # position == Position.LONG
                bars_held += 1

                momentum_faded = (not pd.isna(exit_ret)) and exit_ret < 0
                timed_out = bars_held >= SAFETY_MAX_HOLD_BARS

                if momentum_faded or timed_out:
                    position = Position.FLAT
                    bars_held = 0

            signals.iloc[i] = position

        return signals


if __name__ == "__main__":
    # Uji asap pakai data acak (window 4 jam) -- cuma memastikan
    # strategi tidak error dan bentuk sinyalnya masuk akal.
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    price = 100_000 + np.cumsum(rng.normal(0, 100, n))
    df = pd.DataFrame(
        {
            "open": price,
            "high": price + rng.uniform(0, 80, n),
            "low": price - rng.uniform(0, 80, n),
            "close": price + rng.normal(0, 30, n),
            "volume": rng.uniform(100, 1000, n),
        },
        index=idx,
    )

    strat = TimeSeriesMomentumStrategy()
    sig = strat.generate_signals(df)

    print(f"Strategi: {strat.describe()}")
    print(f"ALLOWS_SHORT: {strat.ALLOWS_SHORT}")
    print(f"Bar dengan sinyal LONG : {(sig == Position.LONG).sum()}")
    print(f"Bar dengan sinyal SHORT: {(sig == Position.SHORT).sum()}")
    print(f"Bar dengan sinyal FLAT : {(sig == Position.FLAT).sum()}")