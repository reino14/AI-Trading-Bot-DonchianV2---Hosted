"""
microstructure/volume_profile.py

LANGKAH 2: hitung Point of Control (POC) dan Value Area dari trade
prints -- bagian paling objektif dari seluruh framework Chris, karena
murni aritmetika atas volume per level harga, tidak ada penilaian
subjektif sama sekali.

DEFINISI (Market Profile, Steidlmayer)
---------------------------------------
POC: level harga dengan volume transaksi TERBANYAK dalam satu sesi.
Value Area: rentang harga tersempit yang menampung `value_area_pct`
    (default 70%) dari total volume sesi -- BUKAN 70% jangkauan harga,
    tapi 70% VOLUME. Caranya: mulai dari bin POC, lalu perluas ke bin
    tetangga (atas atau bawah, mana yang volumenya lebih besar) satu
    per satu sampai volume kumulatif mencapai target.

KEPUTUSAN DESAIN YANG SAYA BUAT SECARA EKSPLISIT (bukan standar tunggal
yang disepakati semua orang -- Dalton punya beberapa varian)
-------------------------------------------------------------
1. Perluasan value area dilakukan SATU BIN per langkah (bandingkan bin
   tetangga bawah vs atas, ambil yang volumenya lebih besar), BUKAN
   dua bin sekaligus seperti beberapa implementasi TPO klasik era
   30-menit bracket. Hasilnya sangat mirip di praktik, tapi ini pilihan
   saya, bukan satu-satunya definisi yang "benar".
2. SESI = satu hari kalender UTC. Ini keputusan penting yang PERLU Anda
   sadari: konsep "sesi" Chris (Asia/London/New York) ada karena bursa
   TradFi punya jam buka-tutup yang jelas. Kripto berdagang 24/7 --
   tidak ada penutupan struktural yang membenarkan batas sesi tertentu.
   UTC harian adalah pilihan netral, BUKAN padanan sesi NY yang Chris
   pakai. Kalau nanti mau meniru pembagian sesi Asia/London/NY, itu
   parameter yang bisa diubah (lihat `session_hour_utc`), tapi jangan
   anggap defaultnya "sama seperti Chris".
3. Bin harga: lebar bin dihitung dinamis dari rentang harga sesi dibagi
   `n_bins` (default 50) -- BUKAN tick size tetap dalam dolar, supaya
   tidak perlu disetel ulang manual saat BTC naik dari $60rb ke $150rb.

BONUS YANG SUDAH DIHITUNG SEKALIAN (untuk langkah 5, footprint/delta)
------------------------------------------------------------------------
Setiap bin JUGA memecah volume jadi buy_volume vs sell_volume, memakai
kolom `is_buyer_maker` dari aggTrades:
    is_buyer_maker=True  -> pembeli adalah maker (order diam) -> pihak
        yang AGRESIF adalah PENJUAL -> ini volume JUAL.
    is_buyer_maker=False -> pembeli adalah taker (mengambil harga) ->
        pihak yang AGRESIF adalah PEMBELI -> ini volume BELI.
Ini pemecahan yang sama persis dibutuhkan untuk delta di langkah 5,
jadi dihitung sekali di sini supaya tidak scan ulang jutaan baris trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SessionProfile:
    session_id: str  # label sesi, mis. "2026-07-01"
    poc_price: float
    vah_price: float  # value area high
    val_price: float  # value area low
    total_volume: float
    total_buy_volume: float
    total_sell_volume: float
    session_high: float
    session_low: float
    n_trades: int


def compute_session_profile(
    trades: pd.DataFrame,
    session_id: str,
    n_bins: int = 50,
    value_area_pct: float = 0.70,
) -> SessionProfile | None:
    """
    trades: DataFrame SATU sesi, WAJIB punya kolom price, quantity,
        is_buyer_maker (persis skema aggTrades futures Binance).

    Return None kalau trades kosong -- caller yang memutuskan mau
    diapakan (skip, bukan crash).
    """
    if trades.empty:
        return None

    low = float(trades["price"].min())
    high = float(trades["price"].max())
    if high <= low:
        # Sesi dengan cuma 1 harga unik (jarang, tapi bisa terjadi di
        # periode sangat sepi) -- tidak ada rentang untuk dibagi bin.
        vol = float(trades["quantity"].sum())
        is_sell = trades["is_buyer_maker"]
        return SessionProfile(
            session_id=session_id, poc_price=low, vah_price=low, val_price=low,
            total_volume=vol,
            total_buy_volume=float(trades.loc[~is_sell, "quantity"].sum()),
            total_sell_volume=float(trades.loc[is_sell, "quantity"].sum()),
            session_high=high, session_low=low, n_trades=len(trades),
        )

    bin_width = (high - low) / n_bins
    # np.clip menjaga harga PERSIS di `high` tidak jatuh ke bin ke-(n_bins)
    # yang tidak ada (di luar index valid).
    bin_idx = np.clip(
        ((trades["price"].to_numpy() - low) / bin_width).astype(int), 0, n_bins - 1
    )

    is_sell_side = trades["is_buyer_maker"].to_numpy()
    qty = trades["quantity"].to_numpy()

    bin_volume = np.zeros(n_bins)
    bin_buy_volume = np.zeros(n_bins)
    bin_sell_volume = np.zeros(n_bins)
    np.add.at(bin_volume, bin_idx, qty)
    np.add.at(bin_buy_volume, bin_idx, np.where(~is_sell_side, qty, 0.0))
    np.add.at(bin_sell_volume, bin_idx, np.where(is_sell_side, qty, 0.0))

    total_volume = float(bin_volume.sum())
    poc_bin = int(np.argmax(bin_volume))

    # --- Perluasan Value Area, satu bin per langkah ---
    target = value_area_pct * total_volume
    lo_bound, hi_bound = poc_bin, poc_bin
    va_volume = bin_volume[poc_bin]

    while va_volume < target and (lo_bound > 0 or hi_bound < n_bins - 1):
        next_lo_vol = bin_volume[lo_bound - 1] if lo_bound > 0 else -1.0
        next_hi_vol = bin_volume[hi_bound + 1] if hi_bound < n_bins - 1 else -1.0
        if next_lo_vol >= next_hi_vol:
            lo_bound -= 1
            va_volume += next_lo_vol
        else:
            hi_bound += 1
            va_volume += next_hi_vol

    def bin_to_price(i: int) -> float:
        return low + (i + 0.5) * bin_width  # titik tengah bin

    return SessionProfile(
        session_id=session_id,
        poc_price=bin_to_price(poc_bin),
        vah_price=bin_to_price(hi_bound),
        val_price=bin_to_price(lo_bound),
        total_volume=total_volume,
        total_buy_volume=float(bin_buy_volume.sum()),
        total_sell_volume=float(bin_sell_volume.sum()),
        session_high=high,
        session_low=low,
        n_trades=len(trades),
    )


def build_profile_history(
    data_dir: Path,
    symbol: str,
    n_bins: int = 50,
    value_area_pct: float = 0.70,
    session_hour_utc: int = 0,
) -> pd.DataFrame:
    """
    Baca SEMUA file parquet bulanan di data_dir/aggTrades/{symbol}/,
    SATU BULAN PADA SATU WAKTU (supaya tidak menahan >1 bulan tick data
    di memori sekaligus -- lihat catatan skala di fetch_tick_data.py),
    hitung profil per sesi, gabungkan jadi satu DataFrame ringkas.

    session_hour_utc: jam UTC yang jadi batas mulai sesi (default 0 =
        tengah malam UTC). Ubah kalau mau meniru batas sesi lain.
    """
    symbol_dir = data_dir / "aggTrades" / symbol
    month_files = sorted(symbol_dir.glob("*.parquet"))
    if not month_files:
        raise FileNotFoundError(
            f"Tidak ada file di {symbol_dir} -- jalankan fetch_tick_data.py dulu."
        )

    all_profiles: list[SessionProfile] = []
    for month_file in month_files:
        trades = pd.read_parquet(month_file, columns=["price", "quantity",
                                                        "is_buyer_maker",
                                                        "transact_time"])
        session_key = (
            trades["transact_time"] - pd.Timedelta(hours=session_hour_utc)
        ).dt.date.astype(str)

        for session_id, session_trades in trades.groupby(session_key):
            profile = compute_session_profile(
                session_trades, session_id, n_bins=n_bins,
                value_area_pct=value_area_pct,
            )
            if profile is not None:
                all_profiles.append(profile)

        del trades  # buang sebelum lanjut ke file bulan berikutnya

    if not all_profiles:
        return pd.DataFrame()

    df = pd.DataFrame([vars(p) for p in all_profiles])
    df = df.sort_values("session_id").reset_index(drop=True)
    return df