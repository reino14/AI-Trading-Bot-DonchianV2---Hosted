import pandas as pd
import numpy as np

from src.microstructure.absorption import (
    calibrate_absorption_thresholds,
    detect_absorption,
)
from src.microstructure.market_structure import build_structure_series
from src.microstructure.signal_generator import generate_setups
from src.microstructure.trade_simulator import simulate_trades
from src.core.cost_model import CRYPTO_PERP


# ============================================================
# SETTINGS
# ============================================================

INITIAL_CAPITAL = 10_000_000       # Rp10 juta
POSITION_FRACTION = 1.0            # 100% modal per trade

FOOTPRINT_PATH = "data/raw/tick/btc_footprint_5min.parquet"
KLINE_1H_PATH = "data/raw/klines/BTCUSDT/1h.parquet"
KLINE_4H_PATH = "data/raw/klines/BTCUSDT/4h.parquet"
DAILY_PROFILE_PATH = "data/raw/tick/btc_daily_profile.parquet"

BAR_MINUTES = 5
MAX_HOLDING_BARS = 288


# ============================================================
# LOAD FOOTPRINT
# ============================================================

print("Load footprint...")
footprint = pd.read_parquet(FOOTPRINT_PATH)
print(f"Footprint = {len(footprint):,} bar")


# ============================================================
# ABSORPTION
# ============================================================

print("Detect absorption...")
th = calibrate_absorption_thresholds(footprint)
footprint["is_absorption"] = detect_absorption(footprint, th)


# ============================================================
# 1H MARKET STRUCTURE
# ============================================================

print("Build 1H structure...")
df_1h = (
    pd.read_parquet(KLINE_1H_PATH)
    .set_index("open_time")
)

df_1h.columns = [c.lower() for c in df_1h.columns]

structure_1h = build_structure_series(
    df_1h,
    lookback=3,
)


# ============================================================
# 4H MARKET STRUCTURE
# ============================================================

print("Build 4H structure...")
df_4h = (
    pd.read_parquet(KLINE_4H_PATH)
    .set_index("open_time")
)

df_4h.columns = [c.lower() for c in df_4h.columns]

structure_4h = build_structure_series(
    df_4h,
    lookback=3,
)


# ============================================================
# DAILY VOLUME PROFILE
# ============================================================

print("Load daily profile...")
daily_profile = pd.read_parquet(
    DAILY_PROFILE_PATH
)


# ============================================================
# BACKTEST
# ============================================================

for target_mode in ["prev_poc", "swing"]:

    print()
    print("=" * 60)
    print(f"TARGET = {target_mode}")
    print("TWO ATTEMPTS = ON")
    print("=" * 60)

    setups = generate_setups(
        footprint,
        structure_1h,
        structure_4h,
        daily_profile,
        target_mode=target_mode,
        require_two_attempts=True,
    )

    print(f"Setups = {len(setups)}")

    trades = simulate_trades(
        setups,
        footprint,
        CRYPTO_PERP,
        bar_minutes=BAR_MINUTES,
        max_holding_bars=MAX_HOLDING_BARS,
    )

    print()
    print("PERFORMANCE")
    print("-" * 60)

    if trades.empty:
        print("No trades.")
        continue

    n = len(trades)

    mean = trades["net_pnl_bps"].mean()

    std = trades["net_pnl_bps"].std(
        ddof=1
    )

    if std > 0 and n > 1:
        t_stat = mean / (
            std / np.sqrt(n)
        )
    else:
        t_stat = np.nan

    win_rate = (
        trades["net_pnl_bps"] > 0
    ).mean() * 100

    total = trades["net_pnl_bps"].sum()

    print(f"Trades             = {n}")
    print(f"Win rate           = {win_rate:.1f}%")
    print(f"Total net PnL      = {total:.2f} bps")
    print(f"Mean               = {mean:.2f} bps/trade")
    print(f"Std                = {std:.2f} bps")
    print(f"t-stat             = {t_stat:.3f}")


    # ========================================================
    # CAPITAL / EQUITY BACKTEST
    # ========================================================

    capital = INITIAL_CAPITAL
    equity_curve = [capital]

    for _, trade in trades.iterrows():

        trade_return = (
            trade["net_pnl_bps"] / 10_000
        )

        position_value = (
            capital * POSITION_FRACTION
        )

        pnl = (
            position_value * trade_return
        )

        capital += pnl

        equity_curve.append(capital)

    net_capital_pnl = (
        capital - INITIAL_CAPITAL
    )

    capital_return_pct = (
        net_capital_pnl
        / INITIAL_CAPITAL
        * 100
    )

    equity = pd.Series(equity_curve)

    running_peak = equity.cummax()

    drawdown_pct = (
        (equity - running_peak)
        / running_peak
        * 100
    )

    max_drawdown_pct = drawdown_pct.min()


    print()
    print("CAPITAL PERFORMANCE")
    print("-" * 60)
    print(
        f"Initial capital    = "
        f"Rp{INITIAL_CAPITAL:,.0f}"
    )
    print(
        f"Final capital      = "
        f"Rp{capital:,.0f}"
    )
    print(
        f"Net PnL            = "
        f"Rp{net_capital_pnl:,.0f}"
    )
    print(
        f"Return             = "
        f"{capital_return_pct:.2f}%"
    )
    print(
        f"Max drawdown       = "
        f"{max_drawdown_pct:.2f}%"
    )
    print(
        f"Position fraction  = "
        f"{POSITION_FRACTION:.0%}"
    )


    # ========================================================
    # LARGEST TRADE
    # ========================================================

    best = trades["net_pnl_bps"].max()

    best_pct_total = (
        best / total
        if total != 0
        else np.nan
    )

    print()
    print("LARGEST TRADE")
    print("-" * 60)
    print(
        f"Best trade         = "
        f"{best:.2f} bps"
    )
    print(
        "Kontribusi trade terbesar "
        f"terhadap total = {best_pct_total:.1%}"
    )


    # ========================================================
    # WORST TRADE
    # ========================================================

    worst = trades["net_pnl_bps"].min()

    print()
    print("WORST TRADE")
    print("-" * 60)
    print(
        f"Worst trade        = "
        f"{worst:.2f} bps"
    )


    # ========================================================
    # EXIT REASONS
    # ========================================================

    print()
    print("Exit reasons:")
    print(
        trades["exit_reason"]
        .value_counts()
    )


    # ========================================================
    # DIRECTIONS
    # ========================================================

    print()
    print("Direction:")
    print(
        trades["direction"]
        .value_counts()
    )


    # ========================================================
    # ALL TRADES SORTED BY NET PNL
    # ========================================================

    print()
    print("Trade results:")

    columns = [
        "direction",
        "exit_reason",
        "hold_bars",
        "gross_pnl_bps",
        "net_pnl_bps",
    ]

    print(
        trades[columns]
        .sort_values(
            "net_pnl_bps",
            ascending=False,
        )
        .to_string(index=True)
    )
