"""
strategy/base.py

Kerangka dasar untuk semua strategi.

ATURAN PENTING (Hari 2):
Strategi adalah FUNGSI MURNI. Ia menerima data harga dan mengembalikan
sinyal posisi (long/flat/short) di tiap bar. Ia TIDAK menyentuh order,
tidak tahu soal fee, tidak tahu soal slippage, tidak tahu apakah order
benar-benar terisi. Semua urusan eksekusi itu tanggung jawab
backtest/engine.py (dan nanti kode live).

Kenapa dipisah begini? Dua alasan:
  1. Strategi bisa diuji dan diganti-ganti tanpa menyentuh logika
     eksekusi sama sekali.
  2. Logika eksekusi (yang menentukan untung/rugi secara nyata, lewat
     cost_model.py) dipakai SAMA PERSIS oleh backtest maupun live nanti
     — supaya angka backtest tidak "berbohong" karena pakai asumsi
     eksekusi yang berbeda dari yang sungguhan dipakai saat live.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import pandas as pd


class Position:
    """
    Nilai sinyal posisi yang DIINGINKAN di suatu bar.

    Ini BUKAN order. Ini niat arah — backtest/engine.py yang nanti
    menerjemahkan perubahan niat ini jadi order beli/jual dan menghitung
    ongkosnya lewat cost_model.py.
    """

    LONG = 1
    FLAT = 0
    SHORT = -1


@dataclass(frozen=True)
class StrategyParams:
    """
    Kelas dasar kosong untuk parameter strategi.

    Tiap strategi konkret membuat subclass dataclass sendiri, MAKSIMAL
    3 field angka (lihat catatan Hari 2: "Satu pola, maksimal tiga
    parameter"). Batasan ini disengaja — makin banyak parameter yang
    bisa disetel, makin gampang strategi "cocok" ke data historis
    secara kebetulan (overfitting), padahal tidak ada keunggulan nyata.
    """

    pass


class Strategy(ABC):
    """
    Kontrak yang wajib dipenuhi semua strategi konkret.
    """

    #: Deklarasi eksplisit: apakah strategi ini BOLEH menghasilkan sinyal
    #: Position.SHORT. Default True (perilaku lama, cocok untuk futures
    #: yang memang bisa short). Strategi untuk SPOT trading WAJIB set ini
    #: False -- lihat backtest/engine.py, yang akan menahan (clip ke FLAT)
    #: sinyal SHORT dari strategi manapun yang ALLOWS_SHORT-nya False.
    #: Ini jaring pengaman STRUKTURAL: walau suatu saat ada strategi baru
    #: yang lupa/salah menghapus cabang SHORT dari generate_signals(),
    #: sistem tetap tidak akan pernah mengeksekusi short di mode spot.
    ALLOWS_SHORT: bool = True

    def __init__(self, params: StrategyParams):
        self.params = params

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        df: OHLCV dengan kolom open, high, low, close, volume,
            index datetime UTC (satu baris = satu bar, mis. 30 menit).

        Return: pd.Series bernilai Position.LONG / FLAT / SHORT,
            index SAMA PERSIS dengan df.

        WAJIB tidak melihat ke depan (no look-ahead bias): nilai sinyal
        di baris t hanya boleh dihitung dari data sampai dan termasuk
        baris t. Kalau ini dilanggar, backtest akan menunjukkan hasil
        bagus yang mustahil dicapai secara live — jenis kesalahan yang
        paling sering membuat orang percaya diri palsu pada strategi
        yang sebetulnya tidak berfungsi.
        """
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name}({asdict(self.params)})"