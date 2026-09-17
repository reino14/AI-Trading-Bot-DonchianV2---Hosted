"""
strategy/xs_reversion.py

Kandidat kedua di alpha bank: reversion jangka pendek.

Dasarnya BEDA dari xs_momentum.py, bukan sekadar dibalik tanda begitu
saja secara kebetulan: momentum bertaruh pada PERSISTENSI (pemenang
30 hari cenderung tetap unggul), reversion bertaruh pada OVERREAKSI
jangka SANGAT pendek (pemenang 1-3 hari terakhir sering terkoreksi,
karena lonjakan tajam dalam 1-3 hari kerap didorong likuidasi paksa,
berita sesaat, atau tekanan likuiditas sesaat -- bukan informasi baru
yang permanen). Dua horizon waktu yang berbeda ini yang membuat
keduanya berpotensi tidak saling tumpang tindih (cek korelasi return
kedua strategi sebelum digabung -- lihat langkah 6 di rencana).

Konstruksi mekanis: SAMA seperti xs_momentum (ranking, top/bottom
fraction, bobot rata), tanda arahnya saja yang dibalik -- LONG yang
performanya paling BURUK belakangan, SHORT yang paling BAIK.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategy.cross_sectional_base import (
    CrossSectionalParams,
    CrossSectionalStrategy,
)


@dataclass(frozen=True)
class XsReversionParams(CrossSectionalParams):
    lookback_bars: int = 2  # SENGAJA pendek -- 1-3 hari, beda rezim dari momentum
    rebalance_bars: int = 1
    top_fraction: float = 0.2


class XsReversionStrategy(CrossSectionalStrategy):
    ALLOWS_SHORT = True

    def __init__(self, params: XsReversionParams | None = None):
        super().__init__(params or XsReversionParams())

    def generate_weights(
        self,
        close: pd.DataFrame,
        extra: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        p: XsReversionParams = self.params

        recent_return = close.pct_change(p.lookback_bars)

        weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for i in range(p.lookback_bars, len(close)):
            if (i - p.lookback_bars) % p.rebalance_bars != 0:
                continue

            row = recent_return.iloc[i].dropna()
            row = row[np.isfinite(row)]
            n = len(row)
            if n < 10:
                continue

            k = max(1, int(round(n * p.top_fraction)))
            ranked = row.sort_values()
            # DIBALIK dari momentum: yang TURUN paling tajam (ranked
            # awal) di-LONG (bertaruh koreksi naik), yang NAIK paling
            # tajam (ranked akhir) di-SHORT (bertaruh koreksi turun).
            losers = ranked.index[:k]
            winners = ranked.index[-k:]

            weights.loc[weights.index[i], losers] = 0.5 / k
            weights.loc[weights.index[i], winners] = -0.5 / k

        rebalance_rows = weights.abs().sum(axis=1) > 0
        weights = weights.where(rebalance_rows, np.nan).ffill().fillna(0.0)

        return weights