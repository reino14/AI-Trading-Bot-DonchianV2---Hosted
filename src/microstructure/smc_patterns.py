"""
microstructure/smc_patterns.py

SMC LANGKAH 2: Fair Value Gap (FVG) dan Order Block (OB) -- dua-duanya
murni pola candle OHLC, TIDAK butuh data baru (beda dari absorption di
proyek Chris yang butuh tick data). Order Block memakai ulang
detect_bos_choch() dari smc_structure.py -- bukan dihitung dari nol.

DEFINISI (interpretasi saya -- SMC punya variasi definisi antar
edukator, terutama soal OB)
------------------------------------------------------------------------
FAIR VALUE GAP (FVG): pola TIGA candle berurutan (t-2, t-1, t). Bullish
    FVG kalau high candle t-2 < low candle t -- ada celah harga yang
    tidak pernah tersentuh candle t-1 di antaranya, dianggap "imbalance"
    yang berpotensi ditutup lagi nanti. Bearish FVG kebalikannya: low
    candle t-2 > high candle t. TIDAK ADA LOOK-AHEAD: FVG di candle t
    cuma pakai candle t-2, t-1, t -- data yang sudah lewat semua.

ORDER BLOCK (OB): candle BERLAWANAN WARNA TERAKHIR sebelum peristiwa
    BOS/CHoCH terjadi -- dianggap zona "smart money" mengakumulasi
    posisi sebelum dorongan yang menembus struktur. OB bullish = candle
    MERAH terakhir sebelum peristiwa *_bullish. OB bearish = candle
    HIJAU terakhir sebelum peristiwa *_bearish. Dicari MUNDUR dari bar
    event (TIDAK termasuk bar event itu sendiri) dalam jendela
    `lookback_candles` -- kalau tidak ketemu candle berlawanan warna
    dalam jendela itu, tidak ada OB untuk event itu (dilewati, bukan
    menebak lebih jauh ke belakang).
"""

from __future__ import annotations

import pandas as pd


def detect_fair_value_gaps(df: pd.DataFrame, min_gap_pct: float = 0.0) -> pd.DataFrame:
    """
    df: WAJIB punya kolom high, low, index datetime UTC urut naik.
    min_gap_pct: buang celah yang lebih kecil dari ini (fraksi, mis.
        0.001 = 0.1%) -- default 0 berarti semua celah, sekecil apa pun.

    Return: DataFrame urut waktu, index = waktu candle KETIGA (candle
        yang mengonfirmasi celah), kolom gap_type, gap_low, gap_high,
        gap_size_pct. Kosong kalau tidak ada celah -- bukan error.
    """
    high = df["high"]
    low = df["low"]
    c1_high = high.shift(2)
    c1_low = low.shift(2)

    bullish_mask = c1_high < low
    bearish_mask = c1_low > high

    bullish_size = (low - c1_high) / c1_high
    bearish_size = (c1_low - high) / high

    rows = []
    for t in df.index[bullish_mask.fillna(False)]:
        size = float(bullish_size.loc[t])
        if size >= min_gap_pct:
            rows.append({"timestamp": t, "gap_type": "bullish",
                         "gap_low": float(c1_high.loc[t]), "gap_high": float(low.loc[t]),
                         "gap_size_pct": size})
    for t in df.index[bearish_mask.fillna(False)]:
        size = float(bearish_size.loc[t])
        if size >= min_gap_pct:
            rows.append({"timestamp": t, "gap_type": "bearish",
                         "gap_low": float(high.loc[t]), "gap_high": float(c1_low.loc[t]),
                         "gap_size_pct": size})

    if not rows:
        return pd.DataFrame(columns=["timestamp", "gap_type", "gap_low", "gap_high", "gap_size_pct"])
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def detect_order_blocks(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    lookback_candles: int = 10,
) -> pd.DataFrame:
    """
    df: WAJIB punya kolom open, high, low, close -- index SAMA dengan
        yang dipakai untuk menghasilkan events_df.
    events_df: hasil detect_bos_choch() -- WAJIB punya kolom bar_pos
        dan event_type.

    Return: DataFrame urut waktu peristiwa, kolom ob_timestamp (waktu
        candle order block itu sendiri), ob_type, ob_high, ob_low,
        related_event_type. Event yang tidak punya candle berlawanan
        warna dalam jendela DILEWATI (tidak masuk hasil), bukan diisi
        nilai kosong/menebak.
    """
    if events_df.empty:
        return pd.DataFrame(columns=[
            "event_timestamp", "ob_timestamp", "ob_type", "ob_high", "ob_low", "related_event_type",
        ]).set_index("event_timestamp")

    is_red = df["close"] < df["open"]
    is_green = df["close"] > df["open"]

    rows = []
    for event_time, ev in events_df.iterrows():
        bar_pos = int(ev["bar_pos"])
        is_bullish_event = str(ev["event_type"]).endswith("bullish")

        window_start = max(0, bar_pos - lookback_candles)
        window = df.iloc[window_start:bar_pos]  # SEBELUM bar event, TIDAK termasuk bar event
        if window.empty:
            continue

        mask = is_red.iloc[window_start:bar_pos] if is_bullish_event else is_green.iloc[window_start:bar_pos]
        candidates = window[mask]
        if candidates.empty:
            continue  # tidak ada candle berlawanan warna dalam jendela -- lewati

        ob_bar = candidates.iloc[-1]  # candle PALING BARU sebelum event
        rows.append({
            "event_timestamp": event_time,
            "ob_timestamp": candidates.index[-1],
            "ob_type": "bullish" if is_bullish_event else "bearish",
            "ob_high": float(ob_bar["high"]),
            "ob_low": float(ob_bar["low"]),
            "related_event_type": ev["event_type"],
        })

    if not rows:
        return pd.DataFrame(columns=[
            "event_timestamp", "ob_timestamp", "ob_type", "ob_high", "ob_low", "related_event_type",
        ]).set_index("event_timestamp")
    return pd.DataFrame(rows).set_index("event_timestamp").sort_index()