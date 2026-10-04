"""
src/strategy/regime_filtered.py

Pembungkus (wrapper) strategi: strategi apa pun (mis. DonchianCloseFuturesStrategy)
tetap menghasilkan sinyal seperti biasa, lalu ENTRY-nya disaring pakai
market regime dari src/strategy/regime_detector.py.

ATURAN FILTER
-------------
Satu "run" = rentetan bar dengan sinyal mentah yang sama (mis. LONG terus).
  - Posisi dalam run itu BARU diizinkan mulai bar pertama di mana regime
    TREND searah sinyal (LONG butuh TREND_UP, SHORT butuh TREND_DOWN).
    Sebelum itu sinyal ditahan jadi FLAT.
  - Sekali diizinkan, posisi DIPEGANG sampai sinyal mentah berubah --
    regime yang berubah jadi sideways TIDAK menutup posisi paksa.
    (Kalau sinyal cuma di-set 0 tiap kali sideways, PaperRunner akan
    langsung menutup posisi terbuka -- itu yang dihindari di sini.)
  - Exit dan pembalikan arah tetap mengikuti strategi asli. Untuk strategi
    selalu-di-pasar (long <-> short), pembalikan di kondisi tidak trending
    berarti: posisi lama ditutup, lalu FLAT sampai regime mengonfirmasi.

Tetap FUNGSI MURNI dan tanpa look-ahead (kontrak strategy/base.py):
sinyal mentah dan regime di bar t sama-sama hanya memakai data <= t.

BATASAN (disampaikan terbuka)
-----------------------------
Strategi dievaluasi ulang dari buffer PaperRunner tiap bar. Kalau satu run
lebih panjang dari buffer dan SEMUA bar "trend searah" di run itu sudah
keluar dari buffer, izin hilang dan posisi ditutup. Karena itu launcher
memperbesar buffer_size saat filter aktif.
"""

from dataclasses import asdict

import pandas as pd

from src.strategy.base import Position, Strategy
from src.strategy.regime_detector import RegimeConfig, detect_regime


def apply_regime_filter(raw: pd.Series, regime: pd.Series) -> pd.Series:
    """Fungsi murni -- bisa diuji tanpa strategi sungguhan."""
    run_id = (raw != raw.shift()).cumsum()
    match = ((raw == Position.LONG) & (regime == "TREND_UP")) | (
        (raw == Position.SHORT) & (regime == "TREND_DOWN")
    )
    allowed = match.astype(int).groupby(run_id).cummax().astype(bool)
    return raw.where(allowed, Position.FLAT).astype(int)


class RegimeFilteredStrategy(Strategy):
    def __init__(self, inner: Strategy, cfg: RegimeConfig | None = None):
        super().__init__(inner.params)
        self.inner = inner
        self.cfg = cfg or RegimeConfig()
        self.ALLOWS_SHORT = inner.ALLOWS_SHORT

    @property
    def name(self) -> str:
        return f"RegimeFiltered[{self.inner.name}]"

    def describe(self) -> str:
        return f"{self.inner.describe()} + filter regime({asdict(self.cfg)})"

    def regime_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return detect_regime(df, self.cfg)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        raw = self.inner.generate_signals(df).astype(int)
        regime = self.regime_frame(df)["regime"]
        return apply_regime_filter(raw, regime)