"""
microstructure/footprint.py

LANGKAH 5: footprint candle -- OHLC + delta (volume beli agresif minus
jual agresif) + POC per bar PENDEK (default 5 menit), bukan per hari.

SENGAJA MEMAKAI ULANG compute_session_profile() dari volume_profile.py,
cuma unit pengelompokannya diganti dari "hari kalender" jadi "bar 5
menit". Logika binning harga, penghitungan POC, dan pemecahan
buy/sell volume SAMA PERSIS -- sudah teruji di smoke_volume_profile.py,
tidak perlu ditulis ulang atau diuji ulang dari nol di sini.

METRIK TAMBAHAN YANG DIHITUNG DI SINI (di atas yang sudah ada di
SessionProfile): `delta` dan `delta_pct`. Ini dasar untuk mendefinisikan
"absorption" di langkah 6 -- volume besar TAPI delta_pct mendekati nol
berarti pembeli dan penjual agresif kira-kira SEIMBANG meski volume
transaksi tinggi, salah satu tanda paling umum absorption di literatur
order flow (bukan definisi Chris secara spesifik, tapi konsisten dengan
apa yang dia gambarkan).

TIDAK ADA RISIKO LOOK-AHEAD DI SINI (beda dari market_structure.py):
tiap bar footprint HANYA memakai trade yang terjadi DI DALAM rentang
waktu bar itu sendiri -- tidak ada swing tetangga yang perlu dilihat.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.microstructure.volume_profile import compute_session_profile


def compute_footprint_bars(
    trades: pd.DataFrame,
    bar_freq: str = "5min",
    n_bins: int = 8,
    value_area_pct: float = 0.70,
) -> pd.DataFrame:
    """
    trades: WAJIB punya kolom price, quantity, is_buyer_maker,
        transact_time (index atau kolom -- dipakai sebagai kolom di sini).

    Return DataFrame terurut waktu, index = awal tiap bar, kolom:
        open, high, low, close, poc_price, vah_price, val_price,
        volume, buy_volume, sell_volume, delta, delta_pct, n_trades.

    Bar TIDAK dipaksa jadi grid seragam -- bar tanpa transaksi sama
    sekali tidak muncul di hasil (bukan diisi NaN/nol). Kalau nanti
    engine backtest butuh grid seragam, reindex eksplisit di sana,
    supaya "tidak ada transaksi" (celah nyata) tidak tercampur dengan
    "volume nol" (yang berarti sesuatu berbeda).
    """
    bar_key = trades["transact_time"].dt.floor(bar_freq)

    rows = []
    for bar_start, bar_trades in trades.groupby(bar_key):
        profile = compute_session_profile(
            bar_trades, session_id=bar_start, n_bins=n_bins,
            value_area_pct=value_area_pct,
        )
        if profile is None:
            continue

        ordered = bar_trades.sort_values("transact_time")
        delta = profile.total_buy_volume - profile.total_sell_volume
        rows.append({
            "bar_start": bar_start,
            "open": float(ordered["price"].iloc[0]),
            "high": profile.session_high,
            "low": profile.session_low,
            "close": float(ordered["price"].iloc[-1]),
            "poc_price": profile.poc_price,
            "vah_price": profile.vah_price,
            "val_price": profile.val_price,
            "volume": profile.total_volume,
            "buy_volume": profile.total_buy_volume,
            "sell_volume": profile.total_sell_volume,
            "delta": delta,
            "delta_pct": delta / profile.total_volume if profile.total_volume > 0 else 0.0,
            "n_trades": profile.n_trades,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("bar_start").sort_index()
    return df


def build_footprint_history(
    data_dir: Path,
    symbol: str,
    bar_freq: str = "5min",
    n_bins: int = 8,
    value_area_pct: float = 0.70,
) -> pd.DataFrame:
    """
    Sama seperti build_profile_history() di volume_profile.py: proses
    SATU FILE BULANAN pada satu waktu supaya tidak menahan >1 bulan
    tick data di memori sekaligus, gabungkan hasil ringkas per bar.
    """
    symbol_dir = data_dir / "aggTrades" / symbol
    month_files = sorted(symbol_dir.glob("*.parquet"))
    if not month_files:
        raise FileNotFoundError(
            f"Tidak ada file di {symbol_dir} -- jalankan fetch_tick_data.py dulu."
        )

    parts = []
    for month_file in month_files:
        trades = pd.read_parquet(month_file, columns=["price", "quantity",
                                                        "is_buyer_maker",
                                                        "transact_time"])
        bars = compute_footprint_bars(
            trades, bar_freq=bar_freq, n_bins=n_bins, value_area_pct=value_area_pct
        )
        if not bars.empty:
            parts.append(bars)
        del trades

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).sort_index()