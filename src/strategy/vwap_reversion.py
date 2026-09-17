"""
strategy/vwap_reversion.py

Strategi: VWAP reversion.

Gagasan: VWAP (volume-weighted average price) adalah harga rata-rata di
suatu window waktu, ditimbang oleh volume transaksi tiap bar -- bukan
rata-rata biasa, tapi rata-rata yang lebih "percaya" pada bar dengan
volume besar. Taruhannya: kalau harga menyimpang cukup jauh dari
VWAP-nya, ada kecenderungan harga tertarik balik mendekat (reversion).

Ini BUKAN hukum fisika -- cuma pola empiris yang kadang muncul di pasar
yang sedang tidak tren kuat.

VERSI INI (v2) mengganti exit berbasis WAKTU (max_hold_bars) dengan
exit berbasis HARGA (stop_loss_bps). Alasannya bukan selera -- versi
pertama (v1) didiagnosis lewat scripts/diagnose_direction.py dan
terbukti rugi di DUA arah (reversion maupun momentum) dengan pola yang
sama: untung dibatasi kecil (keluar begitu harga sentuh VWAP), tapi
rugi dibiarkan membesar sampai batas WAKTU tercapai -- bukan batas
SEBERAPA JAUH harga sudah bergerak melawan posisi. stop_loss_bps
langsung menyasar akar masalah itu, bukan cuma menyamarkannya.

INI MASIH BISA GAGAL LOLOS KRITERIA HARI 2 -- itu tetap jawaban yang
sah. Kalau gagal lagi, itu tanda pola VWAP reversion (apa pun exit-nya)
memang tidak cocok untuk instrumen/window ini, bukan alasan untuk terus
menambah-nambah aturan sampai angkanya terlihat bagus.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategy.base import Position, Strategy, StrategyParams

#: Batas waktu pegang posisi, dalam BAR (bukan menit). SENGAJA bukan
#: parameter yang disetel (bukan field dataclass) -- cuma jaring
#: pengaman terakhir untuk kondisi aneh (mis. data nyaris flat lama).
#: Kalau stop_loss_bps bekerja seperti seharusnya, batas ini hampir
#: tidak pernah tersentuh -- exit sudah terjadi lebih dulu lewat
#: reversion atau stop-loss.
SAFETY_MAX_HOLD_BARS = 96  # 96 bar x 30 menit = 48 jam


@dataclass(frozen=True)
class VwapReversionParams(StrategyParams):
    """
    Tiga parameter, sesuai batas Hari 2 -- tidak lebih.

    vwap_window_bars:
        Berapa bar dipakai untuk menghitung VWAP bergerak (rolling).
        Default 48 bar x 30 menit = 24 jam.

    entry_threshold_bps:
        Seberapa jauh harga (close) harus menyimpang dari VWAP, dalam
        bps, sebelum strategi membuka posisi.

    stop_loss_bps:
        Batas kerugian, dalam bps dari harga entry. Begitu harga
        bergerak MELAWAN posisi sejauh ini, keluar paksa -- tidak
        peduli sudah berapa lama dipegang. Ini yang membatasi kerugian
        terburuk per transaksi, melengkapi keuntungan yang memang sudah
        dibatasi oleh target reversion ke VWAP.
    """

    vwap_window_bars: int = 48
    entry_threshold_bps: float = 30.0
    stop_loss_bps: float = 20.0


class VwapReversionStrategy(Strategy):
    """
    Entry:
        - Hitung VWAP bergerak sepanjang vwap_window_bars bar terakhir.
        - Close menyimpang DI ATAS VWAP >= entry_threshold_bps
          -> buka SHORT (bertaruh harga turun kembali ke VWAP).
        - Close menyimpang DI BAWAH VWAP >= entry_threshold_bps
          -> buka LONG (bertaruh harga naik kembali ke VWAP).

    Exit (salah satu terjadi lebih dulu):
        - Harga kembali menyentuh/melewati VWAP -> reversion tercapai
          (target/take-profit).
        - Harga bergerak MELAWAN posisi >= stop_loss_bps dari harga
          entry -> stop-loss, batasi kerugian.
        - Sudah dipegang SAFETY_MAX_HOLD_BARS bar (jaring pengaman,
          bukan parameter yang disetel) -> keluar paksa.
    """

    def __init__(self, params: VwapReversionParams | None = None):
        super().__init__(params or VwapReversionParams())

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p: VwapReversionParams = self.params

        # Typical price = pendekatan umum harga "representatif" satu bar,
        # dipakai luas untuk hitung VWAP (bukan cuma pakai close saja).
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        pv = typical_price * df["volume"]

        rolling_pv = pv.rolling(p.vwap_window_bars, min_periods=p.vwap_window_bars).sum()
        rolling_vol = df["volume"].rolling(p.vwap_window_bars, min_periods=p.vwap_window_bars).sum()
        vwap = rolling_pv / rolling_vol

        deviation_bps = (df["close"] - vwap) / vwap * 10_000

        signals = pd.Series(Position.FLAT, index=df.index, dtype=int)

        # Loop eksplisit (bukan vectorized) karena exit bergantung pada
        # STATE: posisi terbuka atau tidak, harga entry, dan sudah
        # berapa lama dipegang.
        position = Position.FLAT
        entry_price = None
        bars_held = 0

        for i in range(len(df)):
            dev = deviation_bps.iloc[i]
            close_price = df["close"].iloc[i]

            if pd.isna(dev):
                signals.iloc[i] = Position.FLAT
                continue

            if position == Position.FLAT:
                if dev >= p.entry_threshold_bps:
                    position = Position.SHORT
                    entry_price = close_price
                    bars_held = 0
                elif dev <= -p.entry_threshold_bps:
                    position = Position.LONG
                    entry_price = close_price
                    bars_held = 0

            else:
                bars_held += 1

                if position == Position.LONG:
                    # move_bps positif = harga bergerak MENGUNTUNGKAN posisi LONG.
                    move_bps = (close_price - entry_price) / entry_price * 10_000
                    reverted = dev >= 0  # harga sudah balik ke/atas VWAP
                else:  # SHORT
                    move_bps = (entry_price - close_price) / entry_price * 10_000
                    reverted = dev <= 0  # harga sudah balik ke/bawah VWAP

                stopped_out = move_bps <= -p.stop_loss_bps
                timed_out = bars_held >= SAFETY_MAX_HOLD_BARS

                if reverted or stopped_out or timed_out:
                    position = Position.FLAT
                    entry_price = None
                    bars_held = 0

            signals.iloc[i] = position

        return signals


if __name__ == "__main__":
    # Uji asap (smoke test) pakai data acak -- cuma memastikan strategi
    # tidak error dan bentuk sinyalnya masuk akal. INI BUKAN BACKTEST
    # SUNGGUHAN.
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2026-01-01", periods=n, freq="30min", tz="UTC")
    price = 100_000 + np.cumsum(rng.normal(0, 50, n))
    df = pd.DataFrame(
        {
            "open": price,
            "high": price + rng.uniform(0, 30, n),
            "low": price - rng.uniform(0, 30, n),
            "close": price + rng.normal(0, 10, n),
            "volume": rng.uniform(100, 1000, n),
        },
        index=idx,
    )

    strat = VwapReversionStrategy()
    sig = strat.generate_signals(df)

    print(f"Strategi: {strat.describe()}")
    print(f"Bar dengan sinyal LONG : {(sig == Position.LONG).sum()}")
    print(f"Bar dengan sinyal SHORT: {(sig == Position.SHORT).sum()}")
    print(f"Bar dengan sinyal FLAT : {(sig == Position.FLAT).sum()}")