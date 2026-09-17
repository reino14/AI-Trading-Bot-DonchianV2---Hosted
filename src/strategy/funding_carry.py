"""
strategy/funding_carry.py

Kandidat ketiga, dan satu-satunya di alpha bank yang punya DASAR
EKONOMIS, bukan pola statistik yang diharap bertahan. Momentum dan
reversion bertaruh pada perilaku harga masa depan yang MIRIP masa
lalu -- itu asumsi statistik, bisa berhenti berlaku kapan saja.
Funding carry bertaruh pada sesuatu yang didefinisikan secara
mekanis oleh desain kontrak perpetual itu sendiri: funding rate
POSITIF berarti trader long MEMBAYAR trader short setiap 8 jam --
harga yang dibayar pasar yang crowded-bullish untuk memegang
eksposur leverage. Funding NEGATIF, sebaliknya, short yang membayar
long.

Strategi: SHORT aset dengan funding rata-rata paling POSITIF (mahal
untuk long, jadi kita ambil sisi yang dibayar), LONG aset dengan
funding paling NEGATIF (murah/disubsidi untuk long). Ini BUKAN
taruhan arah harga -- di pasar datar sekalipun, carry ini tetap
menghasilkan selama pembedaan funding-nya konsisten.

KENAPA BUTUH `extra["funding"]`
--------------------------------
generate_weights() di sini WAJIB menerima panel funding lewat
parameter `extra` (lihat kontrak di cross_sectional_base.py) --
bukan dihitung dari `close` sama sekali. Panel ini harus punya
index & kolom SAMA PERSIS dengan `close`, berisi funding dalam bps
PER BAR (bukan per 8 jam) -- itu persis yang dihasilkan
align_funding_to_bars() di fetch_universe.py, disimpan ke
futures_funding_bps.parquet.

Kalau `extra` tidak diisi atau tidak punya key 'funding',
generate_weights() melempar ValueError yang jelas -- DIAM-DIAM
mengembalikan bobot nol akan terlihat seperti "strategi ini
tidak menghasilkan sinyal", padahal sebenarnya cuma lupa
menyambungkan data funding-nya.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategy.cross_sectional_base import (
    CrossSectionalParams,
    CrossSectionalStrategy,
)


@dataclass(frozen=True)
class FundingCarryParams(CrossSectionalParams):
    smooth_bars: int = 7  # rata-rata bergulir -- funding harian cukup berisik
    rebalance_bars: int = 1
    top_fraction: float = 0.2


class FundingCarryStrategy(CrossSectionalStrategy):
    ALLOWS_SHORT = True  # WAJIB True -- carry ini secara desain butuh short

    def __init__(self, params: FundingCarryParams | None = None):
        super().__init__(params or FundingCarryParams())

    def generate_weights(
        self,
        close: pd.DataFrame,
        extra: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        p: FundingCarryParams = self.params

        if extra is None or "funding" not in extra:
            raise ValueError(
                f"{self.name}: butuh extra['funding'] (panel funding bps per "
                f"bar, index & kolom sama persis dengan close) -- ini strategi "
                f"carry, sinyalnya BUKAN dari harga. Lihat "
                f"futures_funding_bps.parquet dari fetch_universe.py."
            )
        funding = extra["funding"]
        if list(funding.columns) != list(close.columns) or list(funding.index) != list(close.index):
            raise ValueError(
                f"{self.name}: extra['funding'] tidak sejajar dengan close "
                f"(index/kolom harus identik). Reindex dulu sebelum dipanggil."
            )

        # Rata-rata bergulir HANYA memakai data masa lalu -- funding.rolling()
        # dengan window berakhir di baris t memakai baris [t-smooth_bars+1, t],
        # tidak menyentuh masa depan. Tidak perlu shift tambahan di sini
        # karena portfolio_engine.py sendiri yang menggeser SELURUH bobot
        # (termasuk yang dari fungsi ini) lewat execution_lag_bars.
        smoothed = funding.rolling(p.smooth_bars, min_periods=max(1, p.smooth_bars // 2)).mean()

        weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for i in range(p.smooth_bars, len(close)):
            if (i - p.smooth_bars) % p.rebalance_bars != 0:
                continue

            row = smoothed.iloc[i].dropna()
            row = row[np.isfinite(row)]
            n = len(row)
            if n < 10:
                continue

            k = max(1, int(round(n * p.top_fraction)))
            ranked = row.sort_values()
            # Funding paling NEGATIF (ranked awal) -> LONG (disubsidi untuk long).
            # Funding paling POSITIF (ranked akhir) -> SHORT (mahal untuk long,
            # kita ambil sisi yang menerima pembayaran).
            cheap_to_long = ranked.index[:k]
            expensive_to_long = ranked.index[-k:]

            weights.loc[weights.index[i], cheap_to_long] = 0.5 / k
            weights.loc[weights.index[i], expensive_to_long] = -0.5 / k

        rebalance_rows = weights.abs().sum(axis=1) > 0
        weights = weights.where(rebalance_rows, np.nan).ffill().fillna(0.0)

        return weights