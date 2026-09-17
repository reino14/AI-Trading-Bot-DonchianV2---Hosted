"""
strategy/cross_sectional_base.py

Kontrak dasar untuk strategi CROSS-SECTIONAL (banyak aset sekaligus).

BEDANYA DENGAN strategy/base.py YANG LAMA
----------------------------------------
base.py lama: strategi mengembalikan NIAT ARAH di satu aset
    (Position.LONG / FLAT / SHORT) -> engine menerjemahkannya jadi
    transaksi buka/tutup.

File ini: strategi mengembalikan BOBOT TARGET di banyak aset sekaligus
    (mis. +0.04 di SOL, -0.04 di ADA, 0 di sisanya) -> engine
    menerjemahkan PERUBAHAN bobot jadi turnover, dan turnover jadi
    ongkos.

Kenapa harus beda, bukan sekadar dipanggil berulang per aset?
Karena keputusan cross-sectional bersifat RELATIF: "SOL bagus" tidak
punya arti sendiri, yang punya arti adalah "SOL bagus RELATIF terhadap
99 aset lain hari ini". Ranking itu cuma bisa dihitung kalau strategi
melihat seluruh panel sekaligus. Memanggil strategi per aset satu-satu
secara struktural tidak bisa menghasilkan sinyal jenis ini.

base.py lama TIDAK diubah dan TIDAK dihapus -- src/runner/paper.py yang
sedang live di VPS tetap memakainya. Ini jalur paralel, bukan pengganti.

ATURAN YANG TETAP SAMA
----------------------
Strategi tetap FUNGSI MURNI: terima panel data, kembalikan bobot.
Tidak tahu fee, tidak tahu slippage, tidak tahu order. Semua itu
tanggung jawab backtest/portfolio_engine.py.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class CrossSectionalParams:
    """
    Kelas dasar parameter. Tiap strategi konkret bikin subclass sendiri.

    Batas MAKSIMAL 3 PARAMETER dari Hari 2 tetap berlaku, dan justru
    makin penting di sini: strategi cross-sectional punya lebih banyak
    tempat untuk menyelipkan parameter tersembunyi (jumlah decile,
    panjang lookback, ambang likuiditas, frekuensi rebalance). Tiap
    tambahan satu, ambang t-stat yang harus dilewati ikut naik -- lihat
    deflated_t_threshold() di backtest/portfolio_metrics.py.
    """

    pass


class CrossSectionalStrategy(ABC):
    """Kontrak yang wajib dipenuhi semua strategi cross-sectional."""

    #: Sama artinya seperti di base.py lama. False = spot / tanpa short.
    #: portfolio_engine.py akan MEMOTONG (clip ke 0) semua bobot negatif
    #: dari strategi yang ALLOWS_SHORT=False -- jaring pengaman kedua,
    #: bukan pengganti kehati-hatian di generate_weights().
    ALLOWS_SHORT: bool = True

    def __init__(self, params: CrossSectionalParams):
        self.params = params

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def generate_weights(
        self,
        close: pd.DataFrame,
        extra: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """
        close: DataFrame harga penutupan. index = datetime UTC (satu baris
            = satu bar), kolom = simbol. NaN artinya aset belum listing
            atau datanya bolong -- engine akan memaksa bobotnya jadi 0 di
            baris itu, tapi strategi sebaiknya tetap menanganinya sendiri.

        extra: panel tambahan sejajar (index & kolom sama persis dengan
            close), mis. {"volume": ..., "funding": ...}. Ini pintu masuk
            untuk sinyal funding carry nanti, tanpa mengubah kontrak.

        Return: DataFrame bobot target. index & kolom SAMA PERSIS dengan
            close. Nilai = fraksi ekuitas. +0.05 artinya long 5% ekuitas,
            -0.05 short 5%.

        KONVENSI WAKTU (ini bagian paling gampang bikin salah):
            Baris t berisi bobot yang diputuskan memakai data SAMPAI DAN
            TERMASUK penutupan bar t. Engine akan mengeksekusinya untuk
            dipegang selama bar t+1. Jadi bobot baris t hanya boleh
            menyentuh close.iloc[:t+1] -- tidak boleh satu baris pun ke
            depan.

            Engine menerapkan penggeseran ini SECARA STRUKTURAL (lihat
            execution_lag_bars). Tapi penggeseran itu TIDAK bisa
            menyelamatkan strategi yang mengintip ke depan di dalam
            generate_weights() itu sendiri -- mis. memakai
            close.pct_change().shift(-1). Itu tetap tanggung jawab Anda,
            dan scripts/smoke_portfolio_engine.py punya tes kenari untuk
            memastikan engine-nya sendiri tidak bocor.
        """
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name}({asdict(self.params)})"