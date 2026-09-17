"""
strategy/xs_momentum.py

Strategi cross-sectional pertama: momentum relatif.

Idenya satu kalimat: tiap rebalance, urutkan seluruh alam semesta aset
berdasarkan return `lookback_bars` terakhir, long kelompok teratas,
short kelompok terbawah, bobot rata. Tidak ada prediksi arah pasar sama
sekali -- yang dipertaruhkan hanya bahwa peringkat relatif punya sedikit
daya prediksi terhadap peringkat relatif berikutnya.

Ini yang membedakannya dari empat strategi kemarin: Donchian, VWAP
reversion, dan kawan-kawan semuanya bertaruh pada ARAH satu aset.
Yang ini netral terhadap arah pasar (long dan short seimbang), jadi
naik-turunnya BTC sebagian besar hilang dari P&L. Yang tersisa cuma
kualitas rankingnya -- sinyal yang jauh lebih lemah per taruhan, tapi
dengan taruhan jauh lebih banyak.

TIGA PARAMETER, sesuai batas Hari 2.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategy.cross_sectional_base import (
    CrossSectionalParams,
    CrossSectionalStrategy,
)


@dataclass(frozen=True)
class XsMomentumParams(CrossSectionalParams):
    lookback_bars: int = 30
    rebalance_bars: int = 1
    top_fraction: float = 0.2  # 0.2 = long 20% teratas, short 20% terbawah


class XsMomentumStrategy(CrossSectionalStrategy):
    ALLOWS_SHORT = True

    def __init__(self, params: XsMomentumParams | None = None):
        super().__init__(params or XsMomentumParams())

    def generate_weights(
        self,
        close: pd.DataFrame,
        extra: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        p: XsMomentumParams = self.params

        # Return lookback. pct_change memakai HANYA data masa lalu sampai
        # baris t -- tidak ada shift negatif di mana pun di fungsi ini.
        momentum = close.pct_change(p.lookback_bars)

        weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for i in range(p.lookback_bars, len(close)):
            if (i - p.lookback_bars) % p.rebalance_bars != 0:
                continue  # bukan hari rebalance -- diisi lewat ffill di bawah

            row = momentum.iloc[i].dropna()
            row = row[np.isfinite(row)]
            n = len(row)
            if n < 10:
                continue  # alam semesta terlalu kecil untuk ranking bermakna

            k = max(1, int(round(n * p.top_fraction)))
            ranked = row.sort_values()
            shorts = ranked.index[:k]
            longs = ranked.index[-k:]

            weights.loc[weights.index[i], longs] = 0.5 / k
            weights.loc[weights.index[i], shorts] = -0.5 / k

        # Di antara tanggal rebalance, bobot dipertahankan (bukan direset
        # ke nol). Ini penting untuk turnover: strategi yang bobotnya
        # jatuh ke nol tiap bar akan membayar ongkos dua kali lipat
        # secara palsu.
        rebalance_rows = weights.abs().sum(axis=1) > 0
        weights = weights.where(rebalance_rows, np.nan).ffill().fillna(0.0)

        return weights