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


# =========================
# CONFIG MODAL
# =========================

INITIAL_CAPITAL = 10_000_000  # Rp10 juta

# 1.0 = 100% modal digunakan setiap trade
# 0.5 = 50%
# 0.1 = 10%
POSITION_FRACTION = 1.0


# =========================
# 1. LOAD FOOTPRINT
# =========================

print("Load footprint...")

footprint = pd.read_parquet(
    "data/raw/tick/btc_footprint_5min.parquet"
)

print(f"Footprint = {len(footprint):,} bar")


# =========================
# 2. ABSORPTION
# =========================

print("Detect absorption...")

th = calibrate_absorption_thresholds(footprint)

footprint["is_absorption"] = detect_absorption(
    footprint,
    th
)


# =========================
# 3. MARKET STRUCTURE 1H
# =========================

print("Build 1H structure...")

df_1h = (
    pd.read_parquet(
        "data/raw/klines/BTCUSDT/1h.parquet"
    )
    .set_index("open_time")
)

df_1h.columns = [c.lower() for c in df_1h.columns]

structure_1h = build_structure_series(
    df_1h,
    lookback=3
)


# =========================
# 4. MARKET STRUCTURE 4H
# =========================

print("Build 4H structure...")

df_4h = (
    pd.read_parquet(
        "data/raw/klines/BTCUSDT/4h.parquet"
    )
    .set_index("open_time")
)

df_4h.columns = [c.lower() for c in df_4h.columns]

structure_4h = build_structure_series(
    df_4h,
    lookback=3
)


# =========================
# 5. DAILY PROFILE
# =========================

print("Load daily profile...")

daily_profile = pd.read_parquet(
    "data/raw/tick/btc_daily_profile.parquet"
)


# =========================
# 6. GENERATE SETUPS
# =========================

print("Generate setups dengan target_mode=swing...")

setups = generate_setups(
    footprint,
    structure_1h,
    structure_4h,
    daily_profile,
    target_mode="swing",
)

print()
print(
    f"{len(setups)} setup "
    f"(target_mode=swing)"
)


# =========================
# 7. SIMULATE TRADES
# =========================

print("Simulate trades...")

trades = simulate_trades(
    setups,
    footprint,
    CRYPTO_PERP,
    bar_minutes=5,
    max_holding_bars=288,
)


# =========================
# 8. RESULT
# =========================

n = len(trades)

if n == 0:
    print("Tidak ada trade.")
    raise SystemExit


mean = trades["net_pnl_bps"].mean()

std = trades["net_pnl_bps"].std()

t_stat = (
    mean / (std / np.sqrt(n))
    if std > 0
    else np.nan
)

win_rate = (
    trades["net_pnl_bps"] > 0
).mean()

total = trades["net_pnl_bps"].sum()


# =========================
# 9. SIMULASI MODAL
# =========================

capital = INITIAL_CAPITAL

equity_curve = [capital]

trade_capitals = []
trade_pnls = []

for _, trade in trades.iterrows():

    capital_before = capital

    # Return trade dalam decimal
    trade_return = trade["net_pnl_bps"] / 10_000

    # Modal yang digunakan
    position_capital = capital * POSITION_FRACTION

    # Profit/loss trade
    pnl = position_capital * trade_return

    # Update modal
    capital += pnl

    trade_capitals.append(capital_before)
    trade_pnls.append(pnl)

    equity_curve.append(capital)


# Masukkan hasil simulasi ke dataframe
trades = trades.copy()

trades["capital_before"] = trade_capitals
trades["capital_pnl"] = trade_pnls
trades["capital_after"] = equity_curve[1:]


# =========================
# 10. RETURN MODAL
# =========================

final_capital = capital

absolute_return = final_capital - INITIAL_CAPITAL

percentage_return = (
    absolute_return / INITIAL_CAPITAL
)


# =========================
# 11. MAX DRAWDOWN
# =========================

equity = pd.Series(equity_curve)

running_max = equity.cummax()

drawdown = (
    equity - running_max
) / running_max

max_drawdown = drawdown.min()


# =========================
# 12. RESULT
# =========================

print()
print("=" * 60)
print("HASIL BACKTEST — SWING TARGET")
print("=" * 60)

print()
print("PERFORMA STRATEGI")
print("-" * 60)

print(f"Jumlah trade       = {n}")
print(f"Win rate           = {win_rate:.1%}")
print(f"Total net PnL      = {total:.1f} bps")
print(f"Mean               = {mean:.2f} bps/trade")
print(f"Std                = {std:.2f} bps")
print(f"t-stat             = {t_stat:.3f}")

print()
print("SIMULASI MODAL")
print("-" * 60)

print(f"Modal awal         = Rp{INITIAL_CAPITAL:,.0f}")
print(f"Position fraction  = {POSITION_FRACTION:.0%}")
print(f"Modal akhir        = Rp{final_capital:,.0f}")
print(f"Profit/Loss        = Rp{absolute_return:,.0f}")
print(f"Return             = {percentage_return:.2%}")
print(f"Max Drawdown       = {max_drawdown:.2%}")

print()
print("Exit reasons:")
print(
    trades["exit_reason"]
    .value_counts()
)

print()
print("Direction:")
print(
    trades["direction"]
    .value_counts()
)

print()
print("Trade results:")
print(
    trades[
        [
            "direction",
            "exit_reason",
            "hold_bars",
            "gross_pnl_bps",
            "net_pnl_bps",
            "capital_pnl",
            "capital_after",
        ]
    ].to_string()
)