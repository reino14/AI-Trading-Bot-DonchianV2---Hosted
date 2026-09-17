"""
microstructure/trade_simulator.py

LANGKAH 8: dari kandidat setup (signal_generator.py) ke hasil trade
SUNGGUHAN -- jalan bar demi bar mulai bar SETELAH konfirmasi, cek mana
yang kena duluan: stop atau target.

DEPENDENSI: butuh src/core/cost_model.py -- SALIN dari proyek "Traits 2"
ke sini (src/core/cost_model.py di folder "Chris Strategy" ini). Bukan
ditulis ulang di sini SENGAJA -- file itu sendiri menyatakan dirinya
"satu-satunya sumber kebenaran soal ongkos di seluruh proyek", jadi
proyek baru ini pun wajib memakainya, bukan membuat sumber ongkos kedua
yang bisa diam-diam berbeda angkanya.

ATURAN PENTING YANG PERLU ANDA SADARI
----------------------------------------
1. SATU POSISI PADA SATU WAKTU. Selama posisi masih terbuka, setup
   baru yang muncul DIABAIKAN -- persis seperti Chris bilang dia tidak
   duduk scalping bolak-balik, cuma pegang beberapa trade sehari.
2. TIE-BREAK KONSERVATIF: kalau satu bar range-nya mencakup stop DAN
   target sekaligus (bar itu volatil, menyentuh dua-duanya), asumsi
   default SELALU stop duluan -- ini asumsi PESIMISTIS standar untuk
   backtest yang cuma punya OHLC bar, bukan urutan tick sungguhan di
   dalam bar itu. Membuat hasil terlihat lebih jelek dari kenyataan,
   bukan lebih bagus -- arah bias yang aman untuk keputusan go/no-go.
3. Masuk baru bisa dicek MULAI BAR SETELAH bar konfirmasi (entry_price
   dari setup dianggap terisi di CLOSE bar konfirmasi -- realistis
   untuk order yang dieksekusi begitu bar itu selesai).

MANAJEMEN DINAMIS (move_to_breakeven, trail_after_breakeven) -- LANGKAH
BARU, DIAMBIL LANGSUNG DARI KATA-KATA CHRIS, BUKAN DIKARANG BEBAS
------------------------------------------------------------------------
Dia bilang eksplisit: begitu harga berhasil balik ke DALAM value area
(dari discount/premium), dia pindah stop ke breakeven -- kalau GAGAL
balik, dia potong rugi atau tetap di stop awal. Ini terjemahan
mekanisnya: reclaim = harga close balik ke sisi value area (>=VAL
untuk long yang entry di bawah VAL, <=VAH untuk short yang entry di
atas VAH). Begitu reclaim terjadi, stop dipindah PERSIS ke entry_price
-- tidak pernah dilonggarkan lagi sesudahnya.

Trailing sesudah breakeven: dia bilang "trail stop di jalan naik"
begitu mendekati target. Saya kodekan sebagai trailing ke ekstrem
`trail_lookback_bars` bar terakhir -- HANYA mengetat ke arah yang
menguntungkan (never loosen), tidak pernah melonggar mundur.

Default KEDUANYA False -- perilaku lama (stop/target statis) tidak
berubah sama sekali kalau tidak diaktifkan secara eksplisit.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.core.cost_model import CostConfig, round_trip_cost


@dataclass
class RealizedTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    exit_reason: str  # "stop", "breakeven", "target", atau "timeout"
    hold_bars: int
    gross_pnl_bps: float
    cost_bps: float
    net_pnl_bps: float


def simulate_trades(
    setups: pd.DataFrame,
    price_bars: pd.DataFrame,
    cost_cfg: CostConfig,
    bar_minutes: int = 5,
    max_holding_bars: int = 288,  # default 288 bar x 5menit = 1 hari penuh
    entry_is_maker: bool = False,
    exit_is_maker: bool = False,
    move_to_breakeven: bool = False,
    trail_after_breakeven: bool = False,
    trail_lookback_bars: int = 12,
) -> pd.DataFrame:
    """
    setups: hasil generate_setups() -- index timestamp, kolom direction,
        entry_price, stop_price, target_price, val_price, vah_price.
    price_bars: DataFrame bar (SAMA frekuensi dengan yang dipakai
        generate_setups(), mis. footprint 5-menit), WAJIB punya kolom
        high, low, close, index datetime urut naik -- dipakai untuk
        menelusuri apakah stop/target kena.
    max_holding_bars: kalau stop maupun target belum kena setelah
        sekian bar, posisi ditutup paksa di harga close bar terakhir
        ("timeout") -- MENCEGAH satu posisi "menggantung" tak
        terbatas kalau harga diam di tengah selamanya.
    move_to_breakeven: False (default, perilaku lama) atau True --
        lihat penjelasan panjang di kepala file.
    trail_after_breakeven: False (default) atau True -- trailing HANYA
        aktif setelah breakeven ter-trigger (butuh move_to_breakeven=True
        untuk pernah benar-benar jalan).
    trail_lookback_bars: jendela bar untuk trailing.
    """
    if setups.empty:
        return pd.DataFrame()

    bar_index = price_bars.index
    trades: list[RealizedTrade] = []
    blocked_until: pd.Timestamp | None = None

    for entry_time, setup in setups.iterrows():
        if blocked_until is not None and entry_time <= blocked_until:
            continue  # masih ada posisi terbuka dari setup sebelumnya

        # Cari posisi bar konfirmasi di price_bars, mulai telusuri dari
        # bar SETELAHNYA.
        pos = bar_index.searchsorted(entry_time, side="right")
        if pos >= len(bar_index):
            continue  # tidak ada bar setelah entry di data ini

        direction = setup["direction"]
        entry_price = float(setup["entry_price"])
        current_stop = float(setup["stop_price"])
        target_price = float(setup["target_price"])
        val_price = float(setup["val_price"]) if "val_price" in setup.index and pd.notna(setup["val_price"]) else None
        vah_price = float(setup["vah_price"]) if "vah_price" in setup.index and pd.notna(setup["vah_price"]) else None

        breakeven_triggered = False
        exit_price = None
        exit_time = None
        exit_reason = None
        hold_bars = 0

        for i in range(pos, min(pos + max_holding_bars, len(bar_index))):
            bar = price_bars.iloc[i]
            hold_bars = i - pos + 1

            hit_stop = bar["low"] <= current_stop if direction == "long" else bar["high"] >= current_stop
            hit_target = bar["high"] >= target_price if direction == "long" else bar["low"] <= target_price

            if hit_stop or hit_target:
                # Tie-break konservatif: stop menang kalau dua-duanya kena bar yang sama.
                if hit_stop:
                    exit_price = current_stop
                    is_be = breakeven_triggered and abs(current_stop - entry_price) < 1e-9
                    exit_reason = "breakeven" if is_be else "stop"
                else:
                    exit_price, exit_reason = target_price, "target"
                exit_time = bar_index[i]
                break

            # Belum keluar -- evaluasi manajemen dinamis pakai CLOSE bar
            # ini, berlaku untuk bar BERIKUTNYA (bukan retroaktif ke bar
            # yang baru saja dicek).
            if move_to_breakeven and not breakeven_triggered and val_price is not None and vah_price is not None:
                reclaimed = (
                    bar["close"] >= val_price
                    if direction == "long"
                    else bar["close"] <= vah_price
                )
                if reclaimed:
                    breakeven_triggered = True
                    current_stop = entry_price

            if trail_after_breakeven and breakeven_triggered:
                window = price_bars.iloc[max(pos, i - trail_lookback_bars + 1): i + 1]
                if direction == "long":
                    candidate = float(window["low"].min())
                    if candidate > current_stop:
                        current_stop = candidate
                else:
                    candidate = float(window["high"].max())
                    if candidate < current_stop:
                        current_stop = candidate

        if exit_price is None:
            # Timeout -- tutup di close bar terakhir yang ditelusuri.
            last_i = min(pos + max_holding_bars, len(bar_index)) - 1
            exit_price = float(price_bars.iloc[last_i]["close"])
            exit_time = bar_index[last_i]
            exit_reason = "timeout"
            hold_bars = last_i - pos + 1

        if direction == "long":
            gross_pnl_bps = (exit_price - entry_price) / entry_price * 10_000
        else:
            gross_pnl_bps = (entry_price - exit_price) / entry_price * 10_000

        cost = round_trip_cost(
            cost_cfg, entry_is_maker=entry_is_maker, exit_is_maker=exit_is_maker,
            hold_minutes=hold_bars * bar_minutes,
        )

        trades.append(RealizedTrade(
            entry_time=entry_time, exit_time=exit_time, direction=direction,
            entry_price=entry_price, exit_price=exit_price, exit_reason=exit_reason,
            hold_bars=hold_bars, gross_pnl_bps=gross_pnl_bps,
            cost_bps=cost.total_bps, net_pnl_bps=gross_pnl_bps - cost.total_bps,
        ))
        blocked_until = exit_time

    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([vars(t) for t in trades])