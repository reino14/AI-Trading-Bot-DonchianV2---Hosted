"""
microstructure/smc_signal_generator.py

SMC signal generator:
BOS/CHoCH -> Premium/Discount -> Order Block -> Setup.

Baseline backtest:
- Bias:
    BOS/CHoCH bullish -> up
    BOS/CHoCH bearish -> down

- Location:
    bias up   -> wajib discount
    bias down -> wajib premium

- Confirmation:
    close candle entry wajib berada DI DALAM Order Block
    terbaru yang jenisnya cocok.

- Entry:
    close candle setup

- Stop:
    LONG  -> batas bawah bullish OB
    SHORT -> batas atas bearish OB

- Target:
    FIXED 2R
    LONG  -> entry + 2 * risk
    SHORT -> entry - 2 * risk

PENTING:
Ini adalah interpretasi mekanis SMC untuk backtest.
Bukan klaim bahwa ini adalah satu-satunya definisi SMC.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SmcSetup:
    timestamp: pd.Timestamp
    direction: str  # "long" / "short"
    entry_price: float
    stop_price: float
    target_price: float
    zone: str
    ob_low: float
    ob_high: float


def _bias_as_of(
    events_df: pd.DataFrame,
    t: pd.Timestamp,
) -> str | None:
    """
    Ambil bias dari BOS/CHoCH terakhir yang sudah terjadi
    sampai candle t.

    Bullish -> up
    Bearish -> down
    """
    eligible = events_df[events_df.index <= t]

    if eligible.empty:
        return None

    last_type = str(
        eligible.iloc[-1]["event_type"]
    )

    if last_type.endswith("bullish"):
        return "up"

    if last_type.endswith("bearish"):
        return "down"

    return None


def generate_smc_setups(
    df: pd.DataFrame,
    bos_events: pd.DataFrame,
    order_blocks: pd.DataFrame,
    zone_series: pd.Series,
    swing_lookback: int = 3,
) -> pd.DataFrame:
    """
    Generate setup SMC secara bar-by-bar.

    Syarat LONG:
    1. Bias terbaru bullish
    2. Zone = discount
    3. Ada bullish OB sebelum candle sekarang
    4. Close candle berada di dalam OB
    5. Stop di bawah OB
    6. Target = 2R

    Syarat SHORT:
    1. Bias terbaru bearish
    2. Zone = premium
    3. Ada bearish OB sebelum candle sekarang
    4. Close candle berada di dalam OB
    5. Stop di atas OB
    6. Target = 2R

    Parameter swing_lookback dipertahankan di signature agar kompatibel
    dengan script backtest yang sudah ada. Untuk baseline ini target
    tidak lagi memakai swing high/low.
    """

    required_df_columns = {
        "open",
        "high",
        "low",
        "close",
    }

    missing_df = required_df_columns - set(df.columns)

    if missing_df:
        raise ValueError(
            f"df kekurangan kolom: {sorted(missing_df)}"
        )

    required_ob_columns = {
        "ob_type",
        "ob_low",
        "ob_high",
    }

    missing_ob = required_ob_columns - set(order_blocks.columns)

    if missing_ob:
        raise ValueError(
            "order_blocks kekurangan kolom: "
            f"{sorted(missing_ob)}"
        )

    setups: list[SmcSetup] = []

    for t, bar in df.iterrows():

        # ====================================================
        # 1. BIAS
        # ====================================================

        bias = _bias_as_of(
            bos_events,
            t,
        )

        if bias not in ("up", "down"):
            continue

        # ====================================================
        # 2. PREMIUM / DISCOUNT
        # ====================================================

        if t not in zone_series.index:
            continue

        zone = zone_series.loc[t]

        if bias == "up" and zone != "discount":
            continue

        if bias == "down" and zone != "premium":
            continue

        # ====================================================
        # 3. DIRECTION
        # ====================================================

        direction = (
            "long"
            if bias == "up"
            else "short"
        )

        ob_type = (
            "bullish"
            if direction == "long"
            else "bearish"
        )

        # ====================================================
        # 4. CARI OB TERBARU
        # ====================================================

        candidate_obs = order_blocks[
            (order_blocks["ob_type"] == ob_type)
            & (order_blocks.index < t)
        ]

        if candidate_obs.empty:
            continue

        # Penyederhanaan baseline:
        # selalu pakai OB terbaru yang cocok.
        ob = candidate_obs.iloc[-1]

        ob_low = float(ob["ob_low"])
        ob_high = float(ob["ob_high"])

        if ob_low >= ob_high:
            continue

        # ====================================================
        # 5. ENTRY
        # ====================================================

        entry = float(bar["close"])

        # Close HARUS benar-benar berada di dalam OB.
        # Bukan sekadar wick bersinggungan.
        within_ob = (
            ob_low <= entry <= ob_high
        )

        if not within_ob:
            continue

        # ====================================================
        # 6. STOP + 2R TARGET
        # ====================================================

        if direction == "long":

            # Stop di batas bawah bullish OB
            stop = ob_low

            # Stop harus berada di bawah entry
            if stop >= entry:
                continue

            risk = entry - stop

            if risk <= 0:
                continue

            # Fixed 2R
            target = entry + (2.0 * risk)

        else:

            # Stop di batas atas bearish OB
            stop = ob_high

            # Stop harus berada di atas entry
            if stop <= entry:
                continue

            risk = stop - entry

            if risk <= 0:
                continue

            # Fixed 2R
            target = entry - (2.0 * risk)

        # ====================================================
        # 7. SIMPAN SETUP
        # ====================================================

        setups.append(
            SmcSetup(
                timestamp=t,
                direction=direction,
                entry_price=entry,
                stop_price=stop,
                target_price=target,
                zone=str(zone),
                ob_low=ob_low,
                ob_high=ob_high,
            )
        )

    # ========================================================
    # RETURN
    # ========================================================

    columns = [
        "timestamp",
        "direction",
        "entry_price",
        "stop_price",
        "target_price",
        "zone",
        "ob_low",
        "ob_high",
    ]

    if not setups:
        return pd.DataFrame(
            columns=columns
        ).set_index("timestamp")

    return (
        pd.DataFrame(
            [vars(s) for s in setups]
        )
        .set_index("timestamp")
        .sort_index()
    )