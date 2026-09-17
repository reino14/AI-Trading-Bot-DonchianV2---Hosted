"""
backtest/engine.py

Mesin simulasi. Memproses data bar demi bar, memanggil strategi untuk
dapat sinyal (strategy/base.py), dan mencatat tiap transaksi berikut
ongkosnya lewat cost_model.py -- FUNGSI YANG SAMA PERSIS dipakai Hari 1
untuk cek kelayakan instrumen.

Prinsip penting: engine ini TIDAK tahu apa-apa soal strategi tertentu.
Ia cuma menerjemahkan PERUBAHAN sinyal (Position.LONG/FLAT/SHORT) jadi
transaksi buka/tutup posisi, lalu menghitung untung/rugi bersih tiap
transaksi setelah ongkos. Strategi dan mesin sengaja dipisah (lihat
strategy/base.py) supaya bagian ini tidak perlu diubah sama sekali
walau strategi diganti-ganti.
"""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.core.cost_model import CostConfig, round_trip_cost
from src.strategy.base import Position, Strategy


@dataclass
class Trade:
    """Satu transaksi bolak-balik: buka posisi, lalu tutup lagi."""

    entry_time: datetime
    exit_time: datetime
    direction: int  # Position.LONG (1) atau Position.SHORT (-1)
    entry_price: float
    exit_price: float
    hold_bars: int
    gross_pnl_bps: float  # untung/rugi KOTOR, sebelum ongkos
    cost_bps: float  # ongkos bolak-balik dari cost_model
    net_pnl_bps: float  # gross_pnl_bps - cost_bps -- ANGKA YANG SEBENARNYA PENTING


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    cost_cfg: CostConfig,
    bar_minutes: int = 30,
    entry_is_maker: bool = False,
    exit_is_maker: bool = False,
) -> list[Trade]:
    """
    Jalankan satu strategi di satu deret data, hasilkan daftar transaksi.

    bar_minutes:
        Lama satu bar dalam menit. WAJIB cocok dengan resolusi data yang
        dipakai (mis. 30, sesuai keputusan window Hari 1) -- ini dipakai
        untuk menghitung hold_minutes yang sebenarnya, bukan cuma jumlah
        bar, supaya komponen funding di cost_model dihitung benar.

    entry_is_maker / exit_is_maker:
        Skenario eksekusi untuk menghitung ongkos tiap transaksi.
        Pilih yang realistis untuk cara strategi ini akan dijalankan
        live nanti -- taker kalau butuh eksekusi segera, maker kalau
        strategi bisa sabar menunggu limit order terisi. Skenario ini
        MENENTUKAN, bukan detail kecil: lihat selisih rasio taker vs
        limit di hasil Hari 1 Anda.
    """
    signals = strategy.generate_signals(df)

    # JARING PENGAMAN STRUKTURAL: kalau strategi mendeklarasikan
    # ALLOWS_SHORT=False (lihat strategy/base.py -- wajib untuk strategi
    # spot, karena spot tidak bisa short-selling), tapi ENTAH BAGAIMANA
    # tetap menghasilkan sinyal SHORT (mis. bug di generate_signals()),
    # sinyal itu DITAHAN (clip ke FLAT) DI SINI -- bukan cuma dipercaya
    # begitu saja dari strategi. Ini level pertahanan KEDUA, bukan
    # pengganti kehati-hatian di generate_signals() itu sendiri.
    if not strategy.ALLOWS_SHORT and (signals == Position.SHORT).any():
        n_clipped = int((signals == Position.SHORT).sum())
        print(
            f"  PERINGATAN: strategi {strategy.name} mendeklarasikan ALLOWS_SHORT=False "
            f"tapi menghasilkan {n_clipped} sinyal SHORT -- ditahan jadi FLAT. "
            f"Ini kemungkinan BUG di generate_signals(), periksa strategi tersebut."
        )
        signals = signals.replace(Position.SHORT, Position.FLAT)

    trades: list[Trade] = []

    open_direction = Position.FLAT
    open_entry_time = None
    open_entry_price = None
    open_bar_index = None

    def _close_trade(exit_index: int, exit_price: float, exit_time) -> None:
        nonlocal open_direction, open_entry_time, open_entry_price, open_bar_index

        hold_bars = exit_index - open_bar_index
        hold_minutes = hold_bars * bar_minutes

        if open_direction == Position.LONG:
            gross_pnl_bps = (exit_price - open_entry_price) / open_entry_price * 10_000
        else:  # SHORT
            gross_pnl_bps = (open_entry_price - exit_price) / open_entry_price * 10_000

        cost = round_trip_cost(
            cost_cfg,
            entry_is_maker=entry_is_maker,
            exit_is_maker=exit_is_maker,
            hold_minutes=hold_minutes,
        )

        trades.append(
            Trade(
                entry_time=open_entry_time,
                exit_time=exit_time,
                direction=open_direction,
                entry_price=open_entry_price,
                exit_price=exit_price,
                hold_bars=hold_bars,
                gross_pnl_bps=gross_pnl_bps,
                cost_bps=cost.total_bps,
                net_pnl_bps=gross_pnl_bps - cost.total_bps,
            )
        )

        open_direction = Position.FLAT
        open_entry_time = None
        open_entry_price = None
        open_bar_index = None

    for i in range(len(df)):
        current_signal = signals.iloc[i]
        timestamp = df.index[i]
        price = df["close"].iloc[i]

        if open_direction == Position.FLAT and current_signal != Position.FLAT:
            # Tidak ada posisi terbuka, sinyal minta buka -> buka posisi.
            open_direction = current_signal
            open_entry_time = timestamp
            open_entry_price = price
            open_bar_index = i

        elif open_direction != Position.FLAT and current_signal != open_direction:
            # Posisi sedang terbuka, sinyal berubah (ke FLAT atau ke arah
            # berlawanan) -> tutup posisi yang berjalan.
            _close_trade(i, price, timestamp)

            # Kalau sinyal baru LANGSUNG ke arah berlawanan (bukan FLAT
            # dulu), buka posisi baru di bar yang sama juga.
            if current_signal != Position.FLAT:
                open_direction = current_signal
                open_entry_time = timestamp
                open_entry_price = price
                open_bar_index = i

    # Posisi yang masih terbuka di akhir data SENGAJA tidak dihitung
    # sebagai transaksi selesai -- kita tidak tahu harga exit-nya, dan
    # lebih jujur dibuang daripada ditebak jadi seolah-olah closed di
    # harga terakhir (itu akan mendistorsi metrik, terutama kalau
    # kebetulan closed di harga yang menguntungkan).

    return trades


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    """Ubah daftar Trade jadi DataFrame, untuk dipakai backtest/metrics.py."""
    if not trades:
        return pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "direction",
                "entry_price",
                "exit_price",
                "hold_bars",
                "gross_pnl_bps",
                "cost_bps",
                "net_pnl_bps",
            ]
        )
    return pd.DataFrame([vars(t) for t in trades])