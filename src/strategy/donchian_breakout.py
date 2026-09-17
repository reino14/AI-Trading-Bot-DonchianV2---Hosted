"""
strategy/donchian_breakout.py

Strategi: Donchian channel breakout (ala Turtle Trading System).

Gagasan: kalau harga menembus level TERTINGGI dalam N bar terakhir,
itu tanda tren naik baru dimulai -> ikut arah (LONG). Kalau menembus
level TERENDAH dalam N bar terakhir -> ikut arah turun (SHORT).

Ini KEBALIKAN filosofi dari strategy/vwap_reversion.py: bukan bertaruh
harga "kembali" ke rata-rata, tapi bertaruh harga yang sudah mencetak
rekor baru akan TERUS ke arah itu (momentum/tren).

=== CATATAN: EXIT BERBASIS WAKTU (max_hold_bars) ===

Exit pakai max_hold_bars (bukan stop-loss harga) atas permintaan
eksplisit -- PENTING: ini mengembalikan risiko yang sama seperti
strategy/vwap_reversion.py versi awal: kerugian per transaksi TIDAK
PUNYA BATAS ATAS eksplisit, cuma dibatasi channel keluar atau waktu,
dua-duanya tidak menjamin kerugian kecil kalau harga bergerak jauh
sebelum salah satu syarat terpenuhi.

=== TAMBAHAN: FILTER REGIME (Efficiency Ratio) ===

Temuan dari scripts/market_regime.py: BTC dan ETH ternyata punya
karakter regime yang HAMPIR SAMA (bukan penyebab beda hasil BTC vs
ETH seperti dugaan awal) -- dan market cuma benar-benar trending
(ER > 0.5) sekitar 8-9% waktu, RANGING (ER < 0.3) sekitar 67% waktu.
Breakout yang terjadi di kondisi ranging kemungkinan besar breakout
palsu (whipsaw) -- konsisten dengan win rate rendah (20-27%) yang
kita lihat di semua percobaan Donchian sejauh ini.

Filter ditambahkan: entry HANYA diambil kalau Efficiency Ratio (bar-bar
yang sama dipakai buat cek breakout) sudah di atas ambang MINIMAL --
menyaring breakout yang terjadi saat market jelas-jelas sedang
ranging. Ambang (MIN_ENTRY_EFFICIENCY_RATIO) SENGAJA dibuat TETAP,
bukan parameter ke-4 yang bisa disetel -- nilainya PERSIS 0.3, sama
dengan ambang "ranging" yang sudah dipakai di market_regime.py,
supaya bukan angka yang dicari-cari sampai hasil bagus.

Pendekatan ini sejalan dengan riset yang ditemukan sebelumnya: sistem
Donchian breakout di kripto BTC/ETH yang menambahkan filter tren
(garis tengah channel periode lebih panjang sebagai penyaring arah)
terbukti memperbaiki hasil signifikan dibanding tanpa filter.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategy.base import Position, Strategy, StrategyParams
from src.strategy.regime import MIN_ENTRY_EFFICIENCY_RATIO, efficiency_ratio as _efficiency_ratio


@dataclass(frozen=True)
class DonchianBreakoutParams(StrategyParams):
    """
    Tiga parameter, sesuai batas Hari 2.

    entry_window_bars:
        Lookback (jumlah bar) untuk channel breakout MASUK. Harga
        menembus tertinggi/terendah N bar SEBELUM bar ini -> entry ke
        arah tembusan. Default 20, nilai klasik Turtle System 1.

    exit_window_bars:
        Lookback (jumlah bar) untuk channel KELUAR, sengaja lebih
        pendek dari entry_window_bars. Harga balik menembus channel
        pendek ini ke arah berlawanan -> tren kehabisan tenaga, keluar.
        Default 10, juga nilai klasik Turtle System 1.

    max_hold_bars:
        Batas berapa lama posisi dipegang kalau channel keluar belum
        juga tersentuh. Keluar paksa di titik ini, TIDAK PEDULI seberapa
        besar untung/rugi saat itu -- lihat peringatan di docstring
        modul ini soal risiko yang ditimbulkan. Default 20 -- tebakan
        kasar (skala mirip entry_window_bars), bukan hasil fitting.
    """

    entry_window_bars: int = 20
    exit_window_bars: int = 10
    max_hold_bars: int = 20


class DonchianBreakoutStrategy(Strategy):
    """
    VERSI LONG-ONLY -- untuk spot trading (tidak ada short-selling,
    tidak ada leverage). Ini perubahan struktural, bukan cuma "kebetulan
    gak pernah short" -- kemampuan short DIHAPUS TOTAL dari kode ini,
    ditandai eksplisit lewat ALLOWS_SHORT = False di bawah.

    Entry:
        - Close menembus (>) rolling max(high, entry_window_bars bar
          SEBELUM bar ini) -> LONG.
        - SYARAT TAMBAHAN (filter regime, TETAP): Efficiency Ratio bar
          entry_window_bars sebelumnya harus >= MIN_ENTRY_EFFICIENCY_RATIO
          (0.3). Kalau market sedang jelas ranging, sinyal tembus
          channel diabaikan -- kemungkinan besar breakout palsu.

        Tembus channel BAWAH (yang di versi futures dulu memicu SHORT)
        SEKARANG DIABAIKAN SEPENUHNYA -- tidak ada tindakan apa pun,
        karena spot tidak bisa untung dari harga turun tanpa short.

    Exit (salah satu terjadi lebih dulu):
        - Close jatuh di bawah rolling min(low, exit_window_bars) ->
          tren kehabisan tenaga, keluar (jual aset yang dimiliki).
        - Sudah dipegang max_hold_bars bar -> keluar paksa, TANPA
          melihat besar untung/rugi saat itu.
    """

    #: Deklarasi eksplisit -- lihat strategy/base.py. Ini strategi TIDAK
    #: PERNAH menghasilkan Position.SHORT, dan backtest/engine.py akan
    #: menahan (clip ke FLAT) kalau ternyata ada yang lolos -- jaring
    #: pengaman kedua, bukan cuma andalan kode di bawah ini benar.
    ALLOWS_SHORT = False

    def __init__(self, params: DonchianBreakoutParams | None = None):
        super().__init__(params or DonchianBreakoutParams())

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p: DonchianBreakoutParams = self.params

        # .shift(1): channel dihitung dari bar-bar SEBELUM bar saat ini
        # -- supaya tidak "mengintip" high/low bar ini sendiri (no
        # look-ahead bias).
        entry_high = df["high"].rolling(p.entry_window_bars).max().shift(1)
        exit_low = df["low"].rolling(p.exit_window_bars).min().shift(1)
        # Filter regime -- lihat MIN_ENTRY_EFFICIENCY_RATIO di atas.
        er = _efficiency_ratio(df["close"], p.entry_window_bars).shift(1)

        signals = pd.Series(Position.FLAT, index=df.index, dtype=int)

        position = Position.FLAT
        bars_held = 0

        for i in range(len(df)):
            close_price = df["close"].iloc[i]
            e_high = entry_high.iloc[i]
            x_low = exit_low.iloc[i]
            er_i = er.iloc[i]

            if position == Position.FLAT:
                # Entry CUMA butuh data entry_window_bars + ER siap --
                # TIDAK boleh ikut ketahan cuma karena exit_window_bars
                # (biasanya lebih besar) belum siap.
                if pd.isna(e_high) or pd.isna(er_i):
                    signals.iloc[i] = Position.FLAT
                    continue

                trending_enough = er_i >= MIN_ENTRY_EFFICIENCY_RATIO
                if trending_enough and close_price > e_high:
                    position = Position.LONG
                    bars_held = 0
                # Tembus channel bawah TIDAK diproses sama sekali --
                # tidak ada cabang SHORT lagi.

            else:  # position == Position.LONG (satu-satunya kemungkinan selain FLAT)
                bars_held += 1

                trend_exhausted = (not pd.isna(x_low)) and close_price < x_low
                timed_out = bars_held >= p.max_hold_bars

                if trend_exhausted or timed_out:
                    position = Position.FLAT
                    bars_held = 0

            signals.iloc[i] = position

        return signals


if __name__ == "__main__":
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

    strat = DonchianBreakoutStrategy()
    sig = strat.generate_signals(df)

    print(f"Strategi: {strat.describe()}")
    print(f"Bar dengan sinyal LONG : {(sig == Position.LONG).sum()}")
    print(f"Bar dengan sinyal SHORT: {(sig == Position.SHORT).sum()}")
    print(f"Bar dengan sinyal FLAT : {(sig == Position.FLAT).sum()}")