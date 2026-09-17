"""
Adaptive SMC Bot - Two Way Version
==================================

Mode:
- long_short : POI bullish -> LONG, POI bearish -> SHORT
- spot_only  : POI bullish -> LONG, POI bearish -> exit/avoid signal only, no short trade

Install:
    pip install pandas numpy matplotlib openpyxl yfinance

Run:
    Default :
        python SMC_bot_adaptive.py --symbol BTC-USD --period 1y --interval 1d --no-plot --market-mode long_short
        python SMC_bot_adaptive.py --symbol BTC-USD --period 1y --interval 1d --no-plot --market-mode spot_only
        python SMC_bot_adaptive.py --synthetic --no-plot --market-mode long_short
        
    Best results with:
        python SMC_bot_adaptive.py --symbol BTC-USD --period 180d --interval 1h --no-plot --market-mode long_short --entry-touch-mode zone_touch --entry-mode balanced --min-score 2
"""

import argparse
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")


@dataclass
class SMCConfig:
    swing_length: int = 3
    liquidity_tolerance_pct: float = 0.002
    min_liquidity_touches: int = 2
    min_disp: float = 0.002
    close_confirm: bool = True
    wick_check: bool = True
    wick_ratio_max: float = 1.2
    use_volume: bool = False
    volume_mult: float = 1.5
    atr_window: int = 14
    atr_mult: float = 1.5
    body_ratio: float = 0.6
    min_poi_score: int = 3

    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200
    slope_lookback: int = 10
    regime_lookback: int = 100
    range_window: int = 20
    adx_window: int = 14
    trending_adx_min: float = 18.0
    sideways_adx_max: float = 16.0
    ema_slope_trend_min: float = 0.0015
    ema_slope_sideways_max: float = 0.0010

    # long_short = bullish POI -> LONG, bearish POI -> SHORT
    # spot_only  = bullish POI -> LONG, bearish POI -> EXIT/AVOID only
    market_mode: str = "long_short"

    # midpoint   = price must touch POI midpoint
    # zone_touch = price only needs to enter POI zone, then confirmation handles entry
    entry_touch_mode: str = "zone_touch"

    # aggressive   = enter immediately when POI zone/midpoint is touched
    # balanced     = require directional candle, but no previous high/low break
    # conservative = require stricter confirmation candle
    entry_mode: str = "balanced"
    balanced_body_ratio_min: float = 0.25
    balanced_require_entry_side: bool = True
    # zone_if_same_candle = use theoretical entry only when confirmation occurs on the touch candle;
    # otherwise use confirmation close. This avoids obvious lookahead while not being too late.
    balanced_entry_price: str = "zone_if_same_candle"

    # Dynamic RR/risk and pruning based on regime_summary findings
    use_dynamic_rr: bool = True
    rr_trending: float = 2.0
    rr_high_volatility: float = 1.8
    rr_sideways: float = 1.5
    rr_sideways_sweep: float = 1.7
    rr_low_volatility: float = 1.3

    use_dynamic_risk: bool = True
    high_vol_risk_multiplier: float = 0.50
    sideways_risk_multiplier: float = 0.70
    low_vol_risk_multiplier: float = 0.40

    prune_weak_setups: bool = True

    rr: float = 2.0
    risk_per_trade: float = 0.01
    initial_equity: float = 10_000.0
    max_holding_bars: int = 40
    zone_buffer_pct: float = 0.10
    fee_rate: float = 0.0004
    slippage_pct: float = 0.0002
    allow_same_bar_exit: bool = True

    require_confirmation: bool = True
    confirmation_lookahead: int = 3
    confirmation_body_ratio_min: float = 0.45
    confirmation_break_entry: bool = True

    normal_risk_multiplier: float = 1.0
    conservative_risk_multiplier: float = 0.70
    reduced_risk_multiplier: float = 0.50


def load_yfinance(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install dulu: pip install yfinance") from exc

    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError("Data kosong. Coba symbol/period/interval lain.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df.columns = [str(c).lower() for c in df.columns]
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ada dari yfinance: {missing}")

    if "volume" not in df.columns:
        df["volume"] = np.nan

    return df[["open", "high", "low", "close", "volume"]].dropna(subset=required)


def make_synthetic_ohlcv(n: int = 700, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r1 = rng.normal(0.0010, 0.0060, n // 4)
    r2 = rng.normal(0.0000, 0.0040, n // 4)
    r3 = rng.normal(-0.0008, 0.0080, n // 4)
    r4 = rng.normal(0.0002, 0.0140, n - 3 * (n // 4))
    regimes = np.concatenate([r1, r2, r3, r4])

    close = [100.0]
    for r in regimes:
        close.append(close[-1] * (1 + r))
    close = np.array(close[1:])

    for ix in [80, 160, 280, 360, 470, 580]:
        if ix < len(close):
            close[ix:] *= rng.choice([1.025, 0.975, 1.035, 0.965])

    open_ = np.r_[close[0], close[:-1] * (1 + rng.normal(0, 0.0015, len(close) - 1))]
    spread = np.maximum(close * rng.uniform(0.002, 0.012, len(close)), 0.1)
    high = np.maximum(open_, close) + spread * rng.uniform(0.2, 1.0, len(close))
    low = np.minimum(open_, close) - spread * rng.uniform(0.2, 1.0, len(close))
    volume = rng.integers(100, 1000, len(close)).astype(float)

    for ix in [80, 160, 280, 360, 470, 580]:
        if ix < len(volume):
            volume[ix] *= rng.integers(2, 6)

    idx = pd.date_range("2024-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Kolom wajib hilang: {missing}")
    if "volume" not in out.columns:
        out["volume"] = np.nan
    out = out.sort_index()
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    timestamp = out.index
    out = out[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
    out.insert(0, "timestamp", timestamp)
    out["_i"] = np.arange(len(out))
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(df).rolling(window, min_periods=1).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, np.nan)
    pv = typical * vol
    return (
        pv.rolling(window, min_periods=1).sum()
        / vol.rolling(window, min_periods=1).sum()
    ).fillna(typical)


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_s = true_range(df).rolling(window, min_periods=1).mean()
    plus_di = (
        100
        * pd.Series(plus_dm, index=df.index).rolling(window, min_periods=1).mean()
        / atr_s.replace(0, np.nan)
    )
    minus_di = (
        100
        * pd.Series(minus_dm, index=df.index).rolling(window, min_periods=1).mean()
        / atr_s.replace(0, np.nan)
    )
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(window, min_periods=1).mean().fillna(0)


def build_features(df: pd.DataFrame, cfg: SMCConfig) -> pd.DataFrame:
    data = ensure_ohlcv(df)
    data["atr"] = atr(data, cfg.atr_window)
    data["atr_pct"] = data["atr"] / data["close"].replace(0, np.nan)
    data["ema_fast"] = ema(data["close"], cfg.ema_fast)
    data["ema_mid"] = ema(data["close"], cfg.ema_mid)
    data["ema_slow"] = ema(data["close"], cfg.ema_slow)
    data["ema_fast_slope"] = data["ema_fast"].pct_change(cfg.slope_lookback)
    data["ema_mid_slope"] = data["ema_mid"].pct_change(cfg.slope_lookback)
    data["rsi_14"] = rsi(data["close"], 14)
    data["vwap_20"] = rolling_vwap(data, 20)
    data["vol_ma_20"] = data["volume"].rolling(20, min_periods=1).mean()
    data["volume_ratio"] = data["volume"] / data["vol_ma_20"].replace(0, np.nan)

    rolling_high = data["high"].rolling(cfg.range_window, min_periods=1).max()
    rolling_low = data["low"].rolling(cfg.range_window, min_periods=1).min()
    data["range_pct"] = (rolling_high - rolling_low) / data["close"].replace(0, np.nan)

    bb_mid = data["close"].rolling(20, min_periods=1).mean()
    bb_std = data["close"].rolling(20, min_periods=1).std(ddof=0)
    data["bb_mid"] = bb_mid
    data["bb_upper"] = bb_mid + 2 * bb_std
    data["bb_lower"] = bb_mid - 2 * bb_std
    data["bb_width_pct"] = (data["bb_upper"] - data["bb_lower"]) / data[
        "close"
    ].replace(0, np.nan)
    data["bb_position"] = (data["close"] - data["bb_lower"]) / (
        data["bb_upper"] - data["bb_lower"]
    ).replace(0, np.nan)
    data["adx"] = adx(data, cfg.adx_window)
    return data


def detect_market_regime(df: pd.DataFrame, cfg: SMCConfig) -> pd.DataFrame:
    data = build_features(df, cfg)
    lb = cfg.regime_lookback
    atr_q70 = data["atr_pct"].rolling(lb, min_periods=20).quantile(0.70)
    atr_q30 = data["atr_pct"].rolling(lb, min_periods=20).quantile(0.30)
    range_q40 = data["range_pct"].rolling(lb, min_periods=20).quantile(0.40)
    bb_q35 = data["bb_width_pct"].rolling(lb, min_periods=20).quantile(0.35)

    data["volatility_state"] = np.select(
        [data["atr_pct"] > atr_q70, data["atr_pct"] < atr_q30],
        ["high_volatility", "low_volatility"],
        default="normal_volatility",
    )

    trending_up = (
        (data["close"] > data["ema_mid"])
        & (data["ema_fast"] > data["ema_mid"])
        & (data["ema_fast_slope"] > cfg.ema_slope_trend_min)
        & (data["adx"] >= cfg.trending_adx_min)
    )
    trending_down = (
        (data["close"] < data["ema_mid"])
        & (data["ema_fast"] < data["ema_mid"])
        & (data["ema_fast_slope"] < -cfg.ema_slope_trend_min)
        & (data["adx"] >= cfg.trending_adx_min)
    )
    sideways = (
        (data["adx"] <= cfg.sideways_adx_max)
        | (data["ema_fast_slope"].abs() <= cfg.ema_slope_sideways_max)
        | (data["range_pct"] <= range_q40)
        | (data["bb_width_pct"] <= bb_q35)
    )

    data["trend_state"] = np.select(
        [trending_up, trending_down, sideways],
        ["trending_up", "trending_down", "sideways"],
        default="unclear",
    )
    data["market_regime"] = data["trend_state"] + "_" + data["volatility_state"]
    return data[
        [
            "timestamp",
            "_i",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "atr",
            "atr_pct",
            "ema_fast",
            "ema_mid",
            "ema_slow",
            "ema_fast_slope",
            "ema_mid_slope",
            "rsi_14",
            "vwap_20",
            "volume_ratio",
            "range_pct",
            "bb_width_pct",
            "bb_position",
            "adx",
            "trend_state",
            "volatility_state",
            "market_regime",
        ]
    ]


def get_regime_at_index(
    regime_df: pd.DataFrame, index_value: int
) -> Optional[Dict[str, Any]]:
    row = regime_df[regime_df["_i"] <= index_value].tail(1)
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def select_setup_by_regime(
    trend_state: str, volatility_state: str, cfg: SMCConfig
) -> Dict[str, Any]:
    rules = {
        "allowed_direction": "none",
        "allowed_setups": [],
        "risk_mode": "no_trade",
        "min_score_boost": 0,
        "notes": "unclear_or_no_trade",
    }

    if trend_state == "trending_up":
        rules = {
            "allowed_direction": "long",
            "allowed_setups": [
                "continuation_ob_retest",
                "pullback_to_demand",
                "liquidity_sweep_continuation",
            ],
            "risk_mode": "normal",
            "min_score_boost": 0,
            "notes": "trend_following_long_bias",
        }
    elif trend_state == "trending_down":
        rules = {
            "allowed_direction": "short",
            "allowed_setups": [
                "continuation_ob_retest",
                "pullback_to_supply",
                "liquidity_sweep_continuation",
            ],
            "risk_mode": "normal",
            "min_score_boost": 0,
            "notes": "trend_following_short_bias",
        }
    elif trend_state == "sideways":
        rules = {
            "allowed_direction": "both",
            "allowed_setups": [
                "liquidity_sweep_reversal",
                "reversal_ob_retest",
                "mean_reversion_poi",
            ],
            "risk_mode": "conservative",
            "min_score_boost": 0,
            "notes": "sideways_mean_reversion_or_sweep_bias",
        }

    if volatility_state == "high_volatility" and rules["allowed_direction"] != "none":
        rules["risk_mode"] = "reduced"
        rules["min_score_boost"] = max(rules["min_score_boost"], 1)
        rules["notes"] += "_high_vol_filter"
        if "liquidity_sweep_reversal" not in rules["allowed_setups"]:
            rules["allowed_setups"].append("liquidity_sweep_reversal")

    elif volatility_state == "low_volatility" and rules["allowed_direction"] != "none":
        if trend_state == "sideways":
            rules["allowed_direction"] = "both"
            rules["allowed_setups"] = ["liquidity_sweep_reversal"]
            rules["risk_mode"] = "reduced"
            rules["min_score_boost"] = max(rules["min_score_boost"], 1)
            rules["notes"] += "_low_vol_only_liquidity_sweep"
        else:
            rules["risk_mode"] = "conservative"
            rules["notes"] += "_low_vol_filter"

    return rules


def risk_multiplier_from_mode(risk_mode: str, cfg: SMCConfig) -> float:
    if risk_mode == "normal":
        return cfg.normal_risk_multiplier
    if risk_mode == "conservative":
        return cfg.conservative_risk_multiplier
    if risk_mode == "reduced":
        return cfg.reduced_risk_multiplier
    return 0.0


def should_prune_setup(
    trend_state: str,
    volatility_state: str,
    setup_type: str,
    direction_label: str,
    score: int,
    cfg: SMCConfig,
) -> Tuple[bool, str]:
    """
    Rule pruning berdasarkan regime_summary sebelumnya.

    Fokus:
    - Sideways low volatility sering choppy dan RR 2.0 sulit tercapai.
    - Trending up low volatility long continuation sempat menjadi kombinasi lemah.
    - Trending down short continuation tetap dipertahankan karena historisnya kuat.
    """
    if not cfg.prune_weak_setups:
        return False, ""

    if volatility_state == "low_volatility" and trend_state == "sideways":
        return True, "pruned_sideways_low_volatility"

    if (
        volatility_state == "low_volatility"
        and trend_state == "trending_up"
        and direction_label == "long"
        and setup_type in ["continuation_ob_retest", "liquidity_sweep_continuation"]
    ):
        return True, "pruned_trending_up_low_vol_long_continuation"

    if volatility_state == "high_volatility" and setup_type == "mean_reversion_poi":
        return True, "pruned_high_vol_mean_reversion_without_sweep"

    if trend_state == "sideways" and score <= 2 and setup_type == "mean_reversion_poi":
        return True, "pruned_low_score_sideways_mean_reversion"

    return False, ""


def dynamic_rr_for_poi(poi: Dict[str, Any], cfg: SMCConfig) -> float:
    if not cfg.use_dynamic_rr:
        return float(cfg.rr)

    trend_state = poi.get("trend_state")
    volatility_state = poi.get("volatility_state")
    setup_type = poi.get("setup_type")

    if volatility_state == "low_volatility":
        return float(cfg.rr_low_volatility)

    if trend_state == "sideways":
        if setup_type == "liquidity_sweep_reversal":
            return float(cfg.rr_sideways_sweep)
        return float(cfg.rr_sideways)

    if volatility_state == "high_volatility":
        return float(cfg.rr_high_volatility)

    return float(cfg.rr_trending)


def dynamic_risk_multiplier_for_poi(
    poi: Dict[str, Any],
    base_multiplier: float,
    cfg: SMCConfig,
) -> float:
    if not cfg.use_dynamic_risk:
        return float(base_multiplier)

    trend_state = poi.get("trend_state")
    volatility_state = poi.get("volatility_state")

    adjusted = float(base_multiplier)

    if volatility_state == "high_volatility":
        adjusted = min(adjusted, cfg.high_vol_risk_multiplier)

    if trend_state == "sideways":
        adjusted = min(adjusted, cfg.sideways_risk_multiplier)

    if volatility_state == "low_volatility":
        adjusted = min(adjusted, cfg.low_vol_risk_multiplier)

    return max(adjusted, 0.0)


def extract_swings(df: pd.DataFrame, cfg: SMCConfig) -> Dict[str, Any]:
    data = ensure_ohlcv(df)
    data["atr"] = atr(data, cfg.atr_window)
    data["vol_ma"] = data["volume"].rolling(20, min_periods=1).mean()
    data["range"] = data["high"] - data["low"]
    data["body"] = (data["close"] - data["open"]).abs()
    data["upper_wick"] = data["high"] - data[["open", "close"]].max(axis=1)
    data["lower_wick"] = data[["open", "close"]].min(axis=1) - data["low"]
    data["swing_high"] = False
    data["swing_low"] = False

    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    n = len(data)
    points = []
    L = cfg.swing_length

    for i in range(L, n - L):
        is_high = highs[i] > highs[i - L : i].max() and highs[i] >= highs[
            i + 1 : i + L + 1
        ].max()
        is_low = lows[i] < lows[i - L : i].min() and lows[i] <= lows[
            i + 1 : i + L + 1
        ].min()
        row = data.iloc[i]
        if is_high:
            data.loc[i, "swing_high"] = True
            points.append(
                {
                    "index": int(i),
                    "confirm_index": int(i + L),
                    "timestamp": row["timestamp"],
                    "type": "high",
                    "level": float(row["high"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"])
                    if pd.notna(row["volume"])
                    else np.nan,
                    "atr": float(row["atr"]),
                }
            )
        if is_low:
            data.loc[i, "swing_low"] = True
            points.append(
                {
                    "index": int(i),
                    "confirm_index": int(i + L),
                    "timestamp": row["timestamp"],
                    "type": "low",
                    "level": float(row["low"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"])
                    if pd.notna(row["volume"])
                    else np.nan,
                    "atr": float(row["atr"]),
                }
            )

    return {"df": data, "points": sorted(points, key=lambda x: x["index"])}


def detect_liquidity(
    df: pd.DataFrame, cfg: SMCConfig, max_span: int = 80
) -> List[Dict[str, Any]]:
    swings = extract_swings(df, cfg)
    points = swings["points"]
    data = swings["df"]
    zones = []

    for swing_type in ["high", "low"]:
        pts = [p for p in points if p["type"] == swing_type]
        for i in range(len(pts)):
            cluster = [pts[i]]
            for j in range(i + 1, len(pts)):
                if pts[j]["index"] - cluster[0]["index"] > max_span:
                    break
                levels = [x["level"] for x in cluster] + [pts[j]["level"]]
                avg = np.mean(levels)
                if max(levels) - min(levels) <= avg * cfg.liquidity_tolerance_pct:
                    cluster.append(pts[j])

            if len(cluster) >= cfg.min_liquidity_touches:
                levels = [x["level"] for x in cluster]
                start_idx = cluster[0]["index"]
                end_idx = cluster[-1]["index"]
                top = float(max(levels))
                bottom = float(min(levels))
                level = float(np.mean(levels))
                search = data.iloc[end_idx + 1 :]
                if swing_type == "high":
                    side = "BSL"
                    hit = search[
                        search["high"] > top * (1 + cfg.liquidity_tolerance_pct / 3)
                    ]
                else:
                    side = "SSL"
                    hit = search[
                        search["low"] < bottom * (1 - cfg.liquidity_tolerance_pct / 3)
                    ]
                swept_index = None if hit.empty else int(hit.iloc[0]["_i"])
                zones.append(
                    {
                        "side": side,
                        "level": level,
                        "top": top,
                        "bottom": bottom,
                        "start_index": int(start_idx),
                        "end_index": int(end_idx),
                        "touches": len(cluster),
                        "touch_indices": ",".join(str(x["index"]) for x in cluster),
                        "swept_index": swept_index,
                    }
                )

    uniq = {}
    for z in zones:
        key = (z["side"], round(z["level"], 5), z["start_index"], z["end_index"])
        uniq[key] = z
    return list(uniq.values())


def wick_break_ok(row: pd.Series, direction: int, cfg: SMCConfig) -> bool:
    body = max(abs(row["close"] - row["open"]), 1e-12)
    upper_wick = row["high"] - max(row["open"], row["close"])
    lower_wick = min(row["open"], row["close"]) - row["low"]
    if direction == 1:
        return upper_wick <= body * cfg.wick_ratio_max
    return lower_wick <= body * cfg.wick_ratio_max


def detect_market_structure(
    swings: Dict[str, Any], cfg: SMCConfig
) -> List[Dict[str, Any]]:
    data = swings["df"]
    points = swings["points"]
    point_ptr = 0
    last_high = None
    last_low = None
    trend = None
    structures = []

    for i in range(len(data)):
        while point_ptr < len(points) and points[point_ptr]["confirm_index"] <= i:
            p = points[point_ptr]
            if p["type"] == "high":
                last_high = {**p, "broken": False}
            else:
                last_low = {**p, "broken": False}
            point_ptr += 1

        row = data.iloc[i]
        break_up_value = row["close"] if cfg.close_confirm else row["high"]
        break_dn_value = row["close"] if cfg.close_confirm else row["low"]

        if last_high is not None and not last_high["broken"] and i > last_high["index"]:
            disp = (break_up_value - last_high["level"]) / max(
                abs(last_high["level"]), 1e-12
            )
            vol_ok = True
            if cfg.use_volume:
                vol_ok = (
                    pd.notna(row["volume"])
                    and pd.notna(row["vol_ma"])
                    and row["volume"] >= row["vol_ma"] * cfg.volume_mult
                )
            wick_ok = True if not cfg.wick_check else wick_break_ok(row, 1, cfg)
            if (
                break_up_value > last_high["level"]
                and disp >= cfg.min_disp
                and wick_ok
                and vol_ok
            ):
                event = "CHOCH_BULLISH" if trend == "bearish" else "BOS_BULLISH"
                structures.append(
                    {
                        "index": int(i),
                        "timestamp": row["timestamp"],
                        "type": event,
                        "direction": 1,
                        "direction_text": "bullish",
                        "level": float(last_high["level"]),
                        "break_price": float(break_up_value),
                        "source_swing_index": int(last_high["index"]),
                        "source_confirm_index": int(last_high["confirm_index"]),
                        "disp_pct": float(disp),
                    }
                )
                trend = "bullish"
                last_high["broken"] = True

        if last_low is not None and not last_low["broken"] and i > last_low["index"]:
            disp = (last_low["level"] - break_dn_value) / max(
                abs(last_low["level"]), 1e-12
            )
            vol_ok = True
            if cfg.use_volume:
                vol_ok = (
                    pd.notna(row["volume"])
                    and pd.notna(row["vol_ma"])
                    and row["volume"] >= row["vol_ma"] * cfg.volume_mult
                )
            wick_ok = True if not cfg.wick_check else wick_break_ok(row, -1, cfg)
            if (
                break_dn_value < last_low["level"]
                and disp >= cfg.min_disp
                and wick_ok
                and vol_ok
            ):
                event = "CHOCH_BEARISH" if trend == "bullish" else "BOS_BEARISH"
                structures.append(
                    {
                        "index": int(i),
                        "timestamp": row["timestamp"],
                        "type": event,
                        "direction": -1,
                        "direction_text": "bearish",
                        "level": float(last_low["level"]),
                        "break_price": float(break_dn_value),
                        "source_swing_index": int(last_low["index"]),
                        "source_confirm_index": int(last_low["confirm_index"]),
                        "disp_pct": float(disp),
                    }
                )
                trend = "bearish"
                last_low["broken"] = True

    return structures


def detect_displacement_candles(
    df: pd.DataFrame, cfg: SMCConfig
) -> List[Dict[str, Any]]:
    data = ensure_ohlcv(df)
    data["atr"] = atr(data, cfg.atr_window)
    data["range"] = data["high"] - data["low"]
    data["body"] = (data["close"] - data["open"]).abs()
    data["body_ratio"] = data["body"] / data["range"].replace(0, np.nan)
    data["vol_ma"] = data["volume"].rolling(20, min_periods=1).mean()
    disps = []

    for _, row in data.iterrows():
        if row["range"] <= 0:
            continue
        if row["range"] < row["atr"] * cfg.atr_mult:
            continue
        if row["body_ratio"] < cfg.body_ratio:
            continue
        if cfg.use_volume:
            if pd.isna(row["volume"]) or pd.isna(row["vol_ma"]):
                continue
            if row["volume"] < row["vol_ma"] * cfg.volume_mult:
                continue
        direction = (
            1 if row["close"] > row["open"] else -1 if row["close"] < row["open"] else 0
        )
        if direction == 0:
            continue
        disps.append(
            {
                "index": int(row["_i"]),
                "timestamp": row["timestamp"],
                "direction": direction,
                "direction_text": "bullish" if direction == 1 else "bearish",
                "type": "DISPLACEMENT_BULLISH"
                if direction == 1
                else "DISPLACEMENT_BEARISH",
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "range": float(row["range"]),
                "atr": float(row["atr"]),
                "body_ratio": float(row["body_ratio"]),
            }
        )
    return disps


def detect_order_blocks(
    df: pd.DataFrame, structures: List[Dict[str, Any]], cfg: SMCConfig
) -> List[Dict[str, Any]]:
    data = ensure_ohlcv(df)
    disps = detect_displacement_candles(df, cfg)
    obs = []
    used = set()

    for s in structures:
        direction = s["direction"]
        start_idx = s["source_swing_index"]
        end_idx = s["index"]
        relevant_disps = [
            d
            for d in disps
            if d["direction"] == direction and start_idx <= d["index"] <= end_idx
        ]
        if relevant_disps:
            scan_end = relevant_disps[-1]["index"] - 1
            displacement_index = relevant_disps[-1]["index"]
        else:
            scan_end = end_idx - 1
            displacement_index = None

        origin = None
        if direction == 1:
            for j in range(scan_end, start_idx - 1, -1):
                row = data.iloc[j]
                if row["close"] < row["open"]:
                    origin = row
                    break
        else:
            for j in range(scan_end, start_idx - 1, -1):
                row = data.iloc[j]
                if row["close"] > row["open"]:
                    origin = row
                    break
        if origin is None:
            continue

        key = (int(origin["_i"]), direction)
        if key in used:
            continue
        used.add(key)

        top = float(origin["high"])
        bottom = float(origin["low"])
        origin_idx = int(origin["_i"])
        mitigated_index = None
        for k in range(end_idx + 1, len(data)):
            row = data.iloc[k]
            if row["low"] <= top and row["high"] >= bottom:
                mitigated_index = int(k)
                break

        obs.append(
            {
                "type": "OB_BULLISH" if direction == 1 else "OB_BEARISH",
                "direction": direction,
                "direction_text": "bullish" if direction == 1 else "bearish",
                "top": top,
                "bottom": bottom,
                "origin_index": origin_idx,
                "break_index": int(end_idx),
                "created_by": s["type"],
                "source_swing_index": int(start_idx),
                "displacement_index": displacement_index,
                "mitigated_index": mitigated_index,
            }
        )
    return obs


def generate_pois(
    order_blocks: List[Dict[str, Any]],
    liquidity: List[Dict[str, Any]],
    proximity_pct: float = 0.002,
) -> List[Dict[str, Any]]:
    pois = []
    for ob in order_blocks:
        score = 2
        reasons = ["order_block"]
        if "CHOCH" in ob["created_by"]:
            score += 1
            reasons.append("created_by_choch")
        else:
            reasons.append("created_by_bos")

        linked_liq = None
        for liq in liquidity:
            zone_mid = (ob["top"] + ob["bottom"]) / 2
            level = liq["level"]
            if abs(zone_mid - level) / max(abs(level), 1e-12) <= proximity_pct:
                score += 1
                linked_liq = liq["side"]
                reasons.append(f"near_{liq['side']}")
                if (
                    liq["swept_index"] is not None
                    and liq["swept_index"] <= ob["break_index"]
                ):
                    score += 1
                    reasons.append("liquidity_swept_before_break")
                break

        pois.append(
            {
                "poi_type": "POI_BULLISH" if ob["direction"] == 1 else "POI_BEARISH",
                "direction": ob["direction"],
                "direction_text": "bullish" if ob["direction"] == 1 else "bearish",
                "top": ob["top"],
                "bottom": ob["bottom"],
                "mid": (ob["top"] + ob["bottom"]) / 2,
                "origin_index": ob["origin_index"],
                "active_from": ob["break_index"],
                "mitigated_index": ob["mitigated_index"],
                "score": score,
                "reasons": ", ".join(reasons),
                "created_by": ob["created_by"],
                "linked_liquidity": linked_liq,
            }
        )
    return sorted(pois, key=lambda x: (-x["score"], x["active_from"]))


def classify_poi_setup(
    poi: Dict[str, Any], regime_row: Optional[Dict[str, Any]] = None
) -> str:
    reasons = poi.get("reasons", "")
    direction = poi.get("direction")
    trend_state = None if regime_row is None else regime_row.get("trend_state")

    if "liquidity_swept_before_break" in reasons:
        if trend_state == "sideways":
            return "liquidity_sweep_reversal"
        if trend_state == "trending_up" and direction == 1:
            return "liquidity_sweep_continuation"
        if trend_state == "trending_down" and direction == -1:
            return "liquidity_sweep_continuation"
        return "liquidity_sweep_reversal"

    if "created_by_choch" in reasons:
        return "reversal_ob_retest"

    if "created_by_bos" in reasons:
        if trend_state == "trending_up" and direction == 1:
            return "continuation_ob_retest"
        if trend_state == "trending_down" and direction == -1:
            return "continuation_ob_retest"
        if trend_state == "sideways":
            return "mean_reversion_poi"
        return "continuation_ob_retest"

    if trend_state == "trending_up" and direction == 1:
        return "pullback_to_demand"
    if trend_state == "trending_down" and direction == -1:
        return "pullback_to_supply"
    return "unclear"


def enrich_and_filter_pois_by_regime(
    pois: List[Dict[str, Any]],
    regime_df: pd.DataFrame,
    cfg: SMCConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_enriched = []
    selected = []

    for poi in pois:
        regime_row = get_regime_at_index(regime_df, poi["active_from"])
        poi_new = dict(poi)
        if regime_row is None:
            poi_new.update(
                {
                    "trend_state": None,
                    "volatility_state": None,
                    "market_regime": None,
                    "setup_type": "unclear",
                    "selected": False,
                    "selection_reason": "no_regime_data",
                    "risk_mode": "no_trade",
                    "risk_multiplier": 0.0,
                    "required_min_score": cfg.min_poi_score,
                    "market_mode": cfg.market_mode,
                    "trade_action": "NO_TRADE",
                    "signal_action": "UNKNOWN",
                    "short_allowed": cfg.market_mode == "long_short",
                }
            )
            all_enriched.append(poi_new)
            continue

        trend_state = regime_row["trend_state"]
        volatility_state = regime_row["volatility_state"]
        market_regime = regime_row["market_regime"]
        setup_type = classify_poi_setup(poi_new, regime_row)
        rules = select_setup_by_regime(trend_state, volatility_state, cfg)

        direction_label = "long" if poi_new["direction"] == 1 else "short"
        trade_action = "LONG" if poi_new["direction"] == 1 else "SHORT"
        signal_action = (
            "BUY_OR_LONG" if poi_new["direction"] == 1 else "SELL_SHORT_OR_EXIT_LONG"
        )
        short_allowed = cfg.market_mode == "long_short"
        required_min_score = cfg.min_poi_score + rules.get("min_score_boost", 0)

        selected_flag = True
        rejection_reasons = []

        if poi_new["score"] < required_min_score:
            selected_flag = False
            rejection_reasons.append(f"score_below_{required_min_score}")

        if rules["allowed_direction"] == "none":
            selected_flag = False
            rejection_reasons.append("regime_no_trade")
        elif rules["allowed_direction"] != "both" and direction_label != rules[
            "allowed_direction"
        ]:
            selected_flag = False
            rejection_reasons.append(f"direction_not_allowed_{direction_label}")

        if setup_type not in rules["allowed_setups"]:
            selected_flag = False
            rejection_reasons.append(f"setup_not_allowed_{setup_type}")

        prune_flag, prune_reason = should_prune_setup(
            trend_state=trend_state,
            volatility_state=volatility_state,
            setup_type=setup_type,
            direction_label=direction_label,
            score=int(poi_new["score"]),
            cfg=cfg,
        )
        if prune_flag:
            selected_flag = False
            rejection_reasons.append(prune_reason)

        if direction_label == "short" and not short_allowed:
            selected_flag = False
            rejection_reasons.append("spot_only_no_short_trade")

        risk_multiplier = risk_multiplier_from_mode(rules["risk_mode"], cfg)
        temp_poi_for_dynamic = {
            **poi_new,
            "trend_state": trend_state,
            "volatility_state": volatility_state,
            "market_regime": market_regime,
            "setup_type": setup_type,
        }
        risk_multiplier = dynamic_risk_multiplier_for_poi(
            temp_poi_for_dynamic,
            risk_multiplier,
            cfg,
        )
        rr_used = dynamic_rr_for_poi(temp_poi_for_dynamic, cfg)

        poi_new.update(
            {
                "trend_state": trend_state,
                "volatility_state": volatility_state,
                "market_regime": market_regime,
                "setup_type": setup_type,
                "selected": bool(selected_flag),
                "selection_reason": "selected"
                if selected_flag
                else ", ".join(rejection_reasons),
                "allowed_direction": rules["allowed_direction"],
                "allowed_setups": ", ".join(rules["allowed_setups"]),
                "risk_mode": rules["risk_mode"],
                "market_mode": cfg.market_mode,
                "trade_action": trade_action if selected_flag else "NO_TRADE",
                "signal_action": signal_action,
                "short_allowed": short_allowed,
                "risk_multiplier": risk_multiplier if selected_flag else 0.0,
                "rr_used": rr_used if selected_flag else 0.0,
                "required_min_score": required_min_score,
                "regime_notes": rules["notes"],
            }
        )
        all_enriched.append(poi_new)
        if selected_flag:
            selected.append(poi_new)

    return all_enriched, sorted(selected, key=lambda x: x["active_from"])


def candle_body_ratio(row: pd.Series) -> float:
    candle_range = float(row["high"] - row["low"])
    if candle_range <= 0:
        return 0.0
    return float(abs(row["close"] - row["open"]) / candle_range)


def balanced_confirmation_candle_ok(
    data: pd.DataFrame,
    idx: int,
    direction: int,
    theoretical_entry: float,
    cfg: SMCConfig,
) -> bool:
    """
    Balanced confirmation:
    - butuh candle searah
    - body candle cukup sehat
    - close berada di sisi candle yang benar
    - opsional: close berada di sisi entry yang benar
    Tidak wajib break previous high/low supaya tidak terlalu ketat.
    """
    if idx <= 0 or idx >= len(data):
        return False

    row = data.iloc[idx]
    body_ratio = candle_body_ratio(row)
    if body_ratio < cfg.balanced_body_ratio_min:
        return False

    candle_mid = (float(row["high"]) + float(row["low"])) / 2

    if direction == 1:
        directional = row["close"] > row["open"]
        close_on_correct_side = row["close"] > candle_mid
        close_entry_side = row["close"] >= theoretical_entry
        if cfg.balanced_require_entry_side:
            return bool(directional and close_on_correct_side and close_entry_side)
        return bool(directional and close_on_correct_side)

    directional = row["close"] < row["open"]
    close_on_correct_side = row["close"] < candle_mid
    close_entry_side = row["close"] <= theoretical_entry
    if cfg.balanced_require_entry_side:
        return bool(directional and close_on_correct_side and close_entry_side)
    return bool(directional and close_on_correct_side)


def confirmation_candle_ok(
    data: pd.DataFrame,
    idx: int,
    direction: int,
    theoretical_entry: float,
    cfg: SMCConfig,
) -> bool:
    if idx <= 0 or idx >= len(data):
        return False
    row = data.iloc[idx]
    if candle_body_ratio(row) < cfg.confirmation_body_ratio_min:
        return False

    if direction == 1:
        bullish = row["close"] > row["open"]
        close_breaks_entry = row["close"] > theoretical_entry
        close_breaks_prev = row["close"] > data.iloc[idx - 1]["high"]
        return bool(
            bullish
            and close_breaks_entry
            and (close_breaks_prev if cfg.confirmation_break_entry else True)
        )

    bearish = row["close"] < row["open"]
    close_breaks_entry = row["close"] < theoretical_entry
    close_breaks_prev = row["close"] < data.iloc[idx - 1]["low"]
    return bool(
        bearish
        and close_breaks_entry
        and (close_breaks_prev if cfg.confirmation_break_entry else True)
    )


def find_confirmed_entry(
    data: pd.DataFrame,
    poi: Dict[str, Any],
    theoretical_entry: float,
    cfg: SMCConfig,
) -> Tuple[Optional[int], Optional[float], str]:
    direction = int(poi["direction"])
    top = float(poi["top"])
    bottom = float(poi["bottom"])

    for touch_idx in range(poi["active_from"] + 1, len(data)):
        touch_row = data.iloc[touch_idx]
        if cfg.entry_touch_mode == "midpoint":
            touched = touch_row["low"] <= theoretical_entry <= touch_row["high"]
        elif cfg.entry_touch_mode == "zone_touch":
            touched = touch_row["low"] <= top and touch_row["high"] >= bottom
        else:
            raise ValueError("entry_touch_mode harus 'midpoint' atau 'zone_touch'")

        if not touched:
            continue

        if (not cfg.require_confirmation) or cfg.entry_mode == "aggressive":
            return (
                int(touch_idx),
                float(theoretical_entry),
                f"{cfg.entry_touch_mode}_aggressive_no_confirmation",
            )

        confirm_end = min(len(data), touch_idx + cfg.confirmation_lookahead + 1)

        if cfg.entry_mode == "balanced":
            for confirm_idx in range(touch_idx, confirm_end):
                if balanced_confirmation_candle_ok(
                    data, confirm_idx, direction, theoretical_entry, cfg
                ):
                    if cfg.balanced_entry_price == "close":
                        entry_price = float(data.iloc[confirm_idx]["close"])
                        entry_label = "balanced_close_entry"
                    elif cfg.balanced_entry_price == "theoretical":
                        entry_price = float(theoretical_entry)
                        entry_label = "balanced_theoretical_entry"
                    elif cfg.balanced_entry_price == "zone_if_same_candle":
                        if confirm_idx == touch_idx:
                            entry_price = float(theoretical_entry)
                            entry_label = "balanced_same_candle_zone_entry"
                        else:
                            entry_price = float(data.iloc[confirm_idx]["close"])
                            entry_label = "balanced_delayed_close_entry"
                    else:
                        raise ValueError("balanced_entry_price harus 'close', 'theoretical', atau 'zone_if_same_candle'")
                    return (
                        int(confirm_idx),
                        entry_price,
                        f"{cfg.entry_touch_mode}_{entry_label}",
                    )
            return None, None, f"{cfg.entry_touch_mode}_touch_without_balanced_confirmation"

        if cfg.entry_mode == "conservative":
            for confirm_idx in range(touch_idx, confirm_end):
                if confirmation_candle_ok(
                    data, confirm_idx, direction, theoretical_entry, cfg
                ):
                    return (
                        int(confirm_idx),
                        float(data.iloc[confirm_idx]["close"]),
                        f"{cfg.entry_touch_mode}_conservative_close_entry",
                    )
            return None, None, f"{cfg.entry_touch_mode}_touch_without_conservative_confirmation"

        raise ValueError("entry_mode harus 'aggressive', 'balanced', atau 'conservative'")

    return None, None, f"{cfg.entry_touch_mode}_not_touched"


def apply_slippage(price: float, direction: int, action: str, cfg: SMCConfig) -> float:
    slip = cfg.slippage_pct
    if slip <= 0:
        return float(price)
    if direction == 1:
        return float(price * (1 + slip)) if action == "entry" else float(price * (1 - slip))
    return float(price * (1 - slip)) if action == "entry" else float(price * (1 + slip))


def calculate_fee(notional: float, cfg: SMCConfig) -> float:
    return abs(notional) * max(cfg.fee_rate, 0.0)


def backtest(
    df: pd.DataFrame,
    pois: List[Dict[str, Any]],
    cfg: SMCConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = ensure_ohlcv(df)
    pois = sorted(pois, key=lambda x: x["active_from"])
    equity = cfg.initial_equity
    trades = []
    equity_curve = [
        {
            "index": 0,
            "timestamp": data.iloc[0]["timestamp"] if len(data) else None,
            "equity": equity,
        }
    ]
    last_exit_idx = -1

    for poi in pois:
        direction = int(poi.get("direction", 0))
        if cfg.market_mode == "spot_only" and direction == -1:
            continue
        if poi["score"] < poi.get("required_min_score", cfg.min_poi_score):
            continue
        if poi.get("risk_multiplier", 1.0) <= 0:
            continue
        if last_exit_idx > poi["active_from"]:
            continue

        top = float(poi["top"])
        bottom = float(poi["bottom"])
        theoretical_entry = (top + bottom) / 2
        zone_h = max(top - bottom, 1e-12)
        buffer = zone_h * cfg.zone_buffer_pct

        entry_idx, raw_entry_price, entry_model = find_confirmed_entry(
            data, poi, theoretical_entry, cfg
        )
        if entry_idx is None or raw_entry_price is None:
            continue

        entry = apply_slippage(raw_entry_price, direction, "entry", cfg)
        rr_used = float(poi.get("rr_used", dynamic_rr_for_poi(poi, cfg)))
        if direction == 1:
            stop = bottom - buffer
            risk_per_unit = entry - stop
            if risk_per_unit <= 0:
                continue
            target = entry + rr_used * risk_per_unit
        else:
            stop = top + buffer
            risk_per_unit = stop - entry
            if risk_per_unit <= 0:
                continue
            target = entry - rr_used * risk_per_unit

        risk_multiplier = float(poi.get("risk_multiplier", 1.0))
        cash_risk = equity * cfg.risk_per_trade * risk_multiplier
        size = cash_risk / risk_per_unit
        entry_fee = calculate_fee(abs(size * entry), cfg)

        max_exit = min(len(data), entry_idx + cfg.max_holding_bars + 1)
        start_j = entry_idx if cfg.allow_same_bar_exit else entry_idx + 1
        exit_idx = None
        raw_exit_price = None
        reason = None

        for j in range(start_j, max_exit):
            row = data.iloc[j]
            if direction == 1:
                hit_sl = row["low"] <= stop
                hit_tp = row["high"] >= target
                if hit_sl and hit_tp:
                    exit_idx, raw_exit_price, reason = (
                        j,
                        stop,
                        "SL_and_TP_same_bar_SL_first",
                    )
                    break
                if hit_sl:
                    exit_idx, raw_exit_price, reason = j, stop, "SL"
                    break
                if hit_tp:
                    exit_idx, raw_exit_price, reason = j, target, "TP"
                    break
            else:
                hit_sl = row["high"] >= stop
                hit_tp = row["low"] <= target
                if hit_sl and hit_tp:
                    exit_idx, raw_exit_price, reason = (
                        j,
                        stop,
                        "SL_and_TP_same_bar_SL_first",
                    )
                    break
                if hit_sl:
                    exit_idx, raw_exit_price, reason = j, stop, "SL"
                    break
                if hit_tp:
                    exit_idx, raw_exit_price, reason = j, target, "TP"
                    break

        if exit_idx is None:
            exit_idx = max_exit - 1
            raw_exit_price = float(data.iloc[exit_idx]["close"])
            reason = "TIME_EXIT"

        exit_price = apply_slippage(float(raw_exit_price), direction, "exit", cfg)
        if direction == 1:
            gross_pnl = (exit_price - entry) * size
            result_r_before_cost = (exit_price - entry) / risk_per_unit
        else:
            gross_pnl = (entry - exit_price) * size
            result_r_before_cost = (entry - exit_price) / risk_per_unit

        exit_fee = calculate_fee(abs(size * exit_price), cfg)
        total_fee = entry_fee + exit_fee
        net_pnl = gross_pnl - total_fee
        result_r_after_cost = net_pnl / cash_risk if cash_risk != 0 else 0.0
        equity += net_pnl
        last_exit_idx = int(exit_idx)

        trades.append(
            {
                "direction": "LONG" if direction == 1 else "SHORT",
                "trade_action": poi.get(
                    "trade_action", "LONG" if direction == 1 else "SHORT"
                ),
                "signal_action": poi.get("signal_action"),
                "market_mode": cfg.market_mode,
                "entry_idx": int(entry_idx),
                "exit_idx": int(exit_idx),
                "entry_timestamp": data.iloc[entry_idx]["timestamp"],
                "exit_timestamp": data.iloc[exit_idx]["timestamp"],
                "entry_price": float(entry),
                "exit_price": float(exit_price),
                "theoretical_entry": float(theoretical_entry),
                "raw_entry_price": float(raw_entry_price),
                "entry_model": entry_model,
                "raw_exit_price": float(raw_exit_price),
                "stop": float(stop),
                "target": float(target),
                "risk_per_unit": float(risk_per_unit),
                "rr_used": float(rr_used),
                "r_before_cost": float(result_r_before_cost),
                "r_result": float(result_r_after_cost),
                "gross_pnl": float(gross_pnl),
                "fee": float(total_fee),
                "pnl_cash": float(net_pnl),
                "equity_after": float(equity),
                "reason": reason,
                "poi_score": int(poi["score"]),
                "required_min_score": int(
                    poi.get("required_min_score", cfg.min_poi_score)
                ),
                "position_size": float(size),
                "cash_risk": float(cash_risk),
                "risk_multiplier": float(risk_multiplier),
                "market_regime": poi.get("market_regime"),
                "trend_state": poi.get("trend_state"),
                "volatility_state": poi.get("volatility_state"),
                "setup_type": poi.get("setup_type"),
                "poi_reasons": poi.get("reasons"),
                "poi_origin_index": int(poi.get("origin_index")),
                "poi_active_from": int(poi.get("active_from")),
            }
        )
        equity_curve.append(
            {
                "index": int(exit_idx),
                "timestamp": data.iloc[exit_idx]["timestamp"],
                "equity": float(equity),
            }
        )

    return pd.DataFrame(trades), pd.DataFrame(equity_curve)


def summarize_backtest(
    trades: pd.DataFrame, equity: pd.DataFrame, cfg: SMCConfig
) -> Dict[str, Any]:
    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_return_pct": 0.0,
            "final_equity": cfg.initial_equity,
            "max_drawdown_pct": 0.0,
            "avg_r": 0.0,
            "profit_factor": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
        }
    wins = trades[trades["r_result"] > 0]
    losses = trades[trades["r_result"] <= 0]
    win_rate = len(wins) / len(trades) * 100
    final_equity = float(equity.iloc[-1]["equity"])
    total_return_pct = (final_equity / cfg.initial_equity - 1) * 100
    eq = equity["equity"]
    peak = eq.cummax()
    max_dd = ((eq - peak) / peak).min() * 100
    gross_profit = trades.loc[trades["pnl_cash"] > 0, "pnl_cash"].sum()
    gross_loss = abs(trades.loc[trades["pnl_cash"] < 0, "pnl_cash"].sum())
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
        if gross_profit > 0
        else 0.0
    )
    return {
        "trades": int(len(trades)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": float(win_rate),
        "total_return_pct": float(total_return_pct),
        "final_equity": float(final_equity),
        "max_drawdown_pct": float(max_dd),
        "avg_r": float(trades["r_result"].mean()),
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else "inf",
        "avg_win_r": float(wins["r_result"].mean()) if not wins.empty else 0.0,
        "avg_loss_r": float(losses["r_result"].mean()) if not losses.empty else 0.0,
    }


def summarize_by_regime(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "market_regime",
                "setup_type",
                "direction",
                "trades",
                "win_rate_pct",
                "avg_r",
                "total_pnl",
            ]
        )
    grouped = (
        trades.groupby(["market_regime", "setup_type", "direction"], dropna=False)
        .agg(
            trades=("r_result", "count"),
            win_rate_pct=("r_result", lambda x: (x > 0).mean() * 100),
            avg_r=("r_result", "mean"),
            total_pnl=("pnl_cash", "sum"),
        )
        .reset_index()
    )
    return grouped.sort_values(["total_pnl", "avg_r"], ascending=False)


def build_llm_summary(
    swings: Dict[str, Any],
    regime_df: pd.DataFrame,
    liquidity: List[Dict[str, Any]],
    structures: List[Dict[str, Any]],
    all_pois: List[Dict[str, Any]],
    selected_pois: List[Dict[str, Any]],
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    cfg: SMCConfig,
    symbol_name: str,
) -> pd.DataFrame:
    data = swings["df"]
    latest = data.iloc[-1]
    latest_regime = regime_df.iloc[-1].to_dict() if not regime_df.empty else {}
    last_structure = structures[-1] if structures else {}
    active_selected = [
        p
        for p in selected_pois
        if p.get("mitigated_index") is None
        or p.get("mitigated_index", 999999999) >= len(data) - 1
    ]
    top_poi = (
        sorted(active_selected, key=lambda x: x.get("score", 0), reverse=True)[0]
        if active_selected
        else {}
    )

    liquidity_with_distance = []
    latest_close = float(latest["close"])
    for liq in liquidity:
        level = liq.get("level")
        if level is not None:
            liquidity_with_distance.append(
                {**liq, "distance_pct": abs(latest_close - level) / latest_close}
            )
    nearest_liq = (
        sorted(liquidity_with_distance, key=lambda x: x["distance_pct"])[0]
        if liquidity_with_distance
        else {}
    )

    return pd.DataFrame(
        [
            {
                "symbol": symbol_name,
                "latest_timestamp": latest["timestamp"],
                "latest_close": latest_close,
                "latest_market_regime": latest_regime.get("market_regime"),
                "latest_trend_state": latest_regime.get("trend_state"),
                "latest_volatility_state": latest_regime.get("volatility_state"),
                "latest_adx": latest_regime.get("adx"),
                "latest_rsi_14": latest_regime.get("rsi_14"),
                "latest_atr_pct": latest_regime.get("atr_pct"),
                "last_structure_type": last_structure.get("type"),
                "last_structure_direction": last_structure.get("direction_text"),
                "top_active_selected_poi_type": top_poi.get("poi_type"),
                "top_active_selected_poi_direction": top_poi.get("direction_text"),
                "top_active_selected_trade_action": top_poi.get("trade_action"),
                "top_active_selected_signal_action": top_poi.get("signal_action"),
                "top_active_selected_poi_setup_type": top_poi.get("setup_type"),
                "top_active_selected_poi_regime": top_poi.get("market_regime"),
                "top_active_selected_poi_bottom": top_poi.get("bottom"),
                "top_active_selected_poi_top": top_poi.get("top"),
                "top_active_selected_poi_score": top_poi.get("score"),
                "nearest_liquidity_side": nearest_liq.get("side"),
                "nearest_liquidity_level": nearest_liq.get("level"),
                "nearest_liquidity_distance_pct": nearest_liq.get("distance_pct"),
                "nearest_liquidity_swept": nearest_liq.get("swept_index") is not None
                if nearest_liq
                else None,
                "all_pois": len(all_pois),
                "selected_pois": len(selected_pois),
                "total_trades": len(trades) if trades is not None else 0,
                "win_rate_pct": (trades["r_result"].gt(0).mean() * 100)
                if trades is not None and not trades.empty
                else None,
                "avg_r": trades["r_result"].mean()
                if trades is not None and not trades.empty
                else None,
                "final_equity": float(equity.iloc[-1]["equity"])
                if equity is not None and not equity.empty
                else cfg.initial_equity,
                "config_market_mode": cfg.market_mode,
                "config_entry_touch_mode": cfg.entry_touch_mode,
                "config_entry_mode": cfg.entry_mode,
                "config_dynamic_rr": cfg.use_dynamic_rr,
                "config_prune_weak_setups": cfg.prune_weak_setups,
                "config_rr": cfg.rr,
                "config_risk_per_trade": cfg.risk_per_trade,
                "llm_instruction": (
                    "Gunakan data ini untuk skenario, bukan prediksi pasti. "
                    "Bearish berarti SHORT hanya jika market_mode=long_short; "
                    "jika spot_only berarti exit/avoid."
                ),
            }
        ]
    )


def safe_sheet_df(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame()

def remove_timezone_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Fix index kalau index datetime pakai timezone
    if isinstance(out.index, pd.DatetimeIndex):
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)

    # Fix semua kolom datetime yang timezone-aware
    for col in out.columns:
        if pd.api.types.is_datetime64tz_dtype(out[col]):
            out[col] = out[col].dt.tz_localize(None)
        elif pd.api.types.is_object_dtype(out[col]):
            out[col] = out[col].apply(
                lambda x: x.tz_localize(None)
                if isinstance(x, pd.Timestamp) and x.tz is not None
                else x
            )

    return out

def export_smc_results_to_excel(
    output_file: str,
    swings: Dict[str, Any],
    regime_df: pd.DataFrame,
    liquidity: List[Dict[str, Any]],
    structures: List[Dict[str, Any]],
    displacement: List[Dict[str, Any]],
    order_blocks: List[Dict[str, Any]],
    all_pois: List[Dict[str, Any]],
    selected_pois: List[Dict[str, Any]],
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    regime_summary: pd.DataFrame,
    llm_summary: pd.DataFrame,
):
    sheets = {
        "candles": swings["df"],
        "market_regime": regime_df,
        "swings": safe_sheet_df(swings["points"]),
        "liquidity": safe_sheet_df(liquidity),
        "structures": safe_sheet_df(structures),
        "displacement": safe_sheet_df(displacement),
        "order_blocks": safe_sheet_df(order_blocks),
        "all_pois": safe_sheet_df(all_pois),
        "selected_pois": safe_sheet_df(selected_pois),
        "trades": safe_sheet_df(trades),
        "equity": safe_sheet_df(equity),
        "regime_summary": safe_sheet_df(regime_summary),
        "llm_summary": safe_sheet_df(llm_summary),
    }
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, sheet_df in sheets.items():
            sheet_df = remove_timezone_from_dataframe(sheet_df)
            sheet_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        for ws in writer.sheets.values():
            ws.freeze_panes = "A2"
            for col_cells in ws.columns:
                letter = col_cells[0].column_letter
                max_len = max(
                    len(str(c.value)) if c.value is not None else 0
                    for c in col_cells
                )
                ws.column_dimensions[letter].width = min(max(max_len + 2, 10), 55)
    print(f"\nExcel exported: {output_file}")


def plot_smc(
    swings: Dict[str, Any],
    structures: List[Dict[str, Any]],
    liquidity: List[Dict[str, Any]],
    order_blocks: List[Dict[str, Any]],
    selected_pois: List[Dict[str, Any]],
    title: str,
):
    data = swings["df"]
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(data["_i"], data["close"], linewidth=1.2, label="Close")
    sh = data[data["swing_high"]]
    sl = data[data["swing_low"]]
    ax.scatter(sh["_i"], sh["high"], marker="^", s=45, label="Swing High")
    ax.scatter(sl["_i"], sl["low"], marker="v", s=45, label="Swing Low")
    x_right = int(data["_i"].max())

    for idx, liq in enumerate(liquidity[:25]):
        ax.hlines(
            liq["level"],
            liq["start_index"],
            liq["end_index"],
            linestyles="dashed",
            linewidth=1.0,
            label="Liquidity" if idx == 0 else None,
        )
        ax.text(liq["end_index"], liq["level"], liq["side"], fontsize=8)

    for ob in order_blocks[:25]:
        width = max(
            5,
            (ob["mitigated_index"] if ob["mitigated_index"] is not None else x_right)
            - ob["origin_index"],
        )
        rect = Rectangle(
            (ob["origin_index"], ob["bottom"]),
            width=width,
            height=ob["top"] - ob["bottom"],
            fill=False,
            linewidth=1.0,
        )
        ax.add_patch(rect)

    for p in selected_pois[:25]:
        mid = (p["top"] + p["bottom"]) / 2
        ax.hlines(mid, p["origin_index"], x_right, linestyles=":", linewidth=1.0)
        ax.text(
            x_right,
            mid,
            f"{p.get('trade_action')} | {p.get('setup_type')}",
            fontsize=7,
        )

    ax.set_title(title)
    ax.set_xlabel("Bar Index")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_equity(equity: pd.DataFrame):
    if equity.empty:
        print("Equity kosong, plot dilewati.")
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(equity["index"], equity["equity"], linewidth=1.5)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Bar Index")
    ax.set_ylabel("Equity")
    plt.tight_layout()
    plt.show()


def run_pipeline(
    df: pd.DataFrame,
    symbol_name: str,
    cfg: SMCConfig,
    output_file: str,
    show_plot: bool = True,
) -> Dict[str, Any]:
    swings = extract_swings(df, cfg)
    liquidity = detect_liquidity(df, cfg)
    structures = detect_market_structure(swings, cfg)
    displacement = detect_displacement_candles(df, cfg)
    order_blocks = detect_order_blocks(df, structures, cfg)
    regime_df = detect_market_regime(df, cfg)
    raw_pois = generate_pois(order_blocks, liquidity)
    all_pois, selected_pois = enrich_and_filter_pois_by_regime(
        raw_pois, regime_df, cfg
    )
    trades, equity = backtest(df, selected_pois, cfg)
    summary = summarize_backtest(trades, equity, cfg)
    regime_summary = summarize_by_regime(trades)
    llm_summary = build_llm_summary(
        swings,
        regime_df,
        liquidity,
        structures,
        all_pois,
        selected_pois,
        trades,
        equity,
        cfg,
        symbol_name,
    )

    export_smc_results_to_excel(
        output_file,
        swings,
        regime_df,
        liquidity,
        structures,
        displacement,
        order_blocks,
        all_pois,
        selected_pois,
        trades,
        equity,
        regime_summary,
        llm_summary,
    )

    print("\n==============================")
    print(f"ADAPTIVE TWO-WAY SMC BOT RESULT: {symbol_name}")
    print("==============================")
    print(f"Market mode          : {cfg.market_mode}")
    print(f"Entry touch mode     : {cfg.entry_touch_mode}")
    print(f"Entry mode           : {cfg.entry_mode}")
    print(f"Dynamic RR           : {cfg.use_dynamic_rr}")
    print(f"Prune weak setups    : {cfg.prune_weak_setups}")
    print(f"Total candles        : {len(df)}")
    print(f"Swing points         : {len(swings['points'])}")
    print(f"Liquidity zones      : {len(liquidity)}")
    print(f"BOS/CHOCH events     : {len(structures)}")
    print(f"Displacement candles : {len(displacement)}")
    print(f"Order blocks         : {len(order_blocks)}")
    print(f"All POI              : {len(all_pois)}")
    print(f"Selected POI         : {len(selected_pois)}")
    print(f"Trades               : {summary['trades']}")
    print(f"Wins / Losses        : {summary['wins']} / {summary['losses']}")
    print(f"Win rate             : {summary['win_rate']:.2f}%")
    print(f"Avg R                : {summary['avg_r']:.2f}R")
    print(f"Profit factor        : {summary['profit_factor']}")
    print(f"Total return         : {summary['total_return_pct']:.2f}%")
    print(f"Final equity         : {summary['final_equity']:.2f}")
    print(f"Max drawdown         : {summary['max_drawdown_pct']:.2f}%")

    latest_regime = regime_df.iloc[-1] if not regime_df.empty else None
    print("\nLatest Market Regime:")
    if latest_regime is not None:
        print(
            f"- {latest_regime['timestamp']} | "
            f"{latest_regime['market_regime']} | "
            f"ADX={latest_regime['adx']:.2f} | "
            f"RSI={latest_regime['rsi_14']:.2f} | "
            f"ATR%={latest_regime['atr_pct']:.4%}"
        )

    print("\nTop Selected POI:")
    if selected_pois:
        for p in sorted(selected_pois, key=lambda x: (-x["score"], x["active_from"]))[
            :10
        ]:
            print(
                f"- {p['poi_type']} | score={p['score']} | "
                f"required={p.get('required_min_score')} | "
                f"setup={p.get('setup_type')} | "
                f"regime={p.get('market_regime')} | "
                f"trade_action={p.get('trade_action')} | "
                f"signal={p.get('signal_action')} | "
                f"risk={p.get('risk_mode')}({p.get('risk_multiplier')}) | "
                f"rr={p.get('rr_used')} | "
                f"zone={p['bottom']:.4f}-{p['top']:.4f} | "
                f"from bar={p['active_from']} | reasons={p['reasons']}"
            )
    else:
        print("- Tidak ada POI yang lolos filter regime/market mode.")

    print("\nRejected POI Snapshot:")
    rejected = [p for p in all_pois if not p.get("selected")]
    if rejected:
        for p in rejected[:10]:
            print(
                f"- {p['poi_type']} | score={p['score']} | "
                f"setup={p.get('setup_type')} | "
                f"regime={p.get('market_regime')} | "
                f"reason={p.get('selection_reason')} | "
                f"signal={p.get('signal_action')}"
            )
    else:
        print("- Tidak ada POI yang ditolak atau tidak ada POI.")

    print("\nRegime Summary:")
    if not regime_summary.empty:
        print(regime_summary.to_string(index=False))
    else:
        print("- Belum ada trade untuk summary per regime.")

    print("\nLLM Summary:")
    print(llm_summary.to_string(index=False))

    if show_plot:
        plot_smc(
            swings,
            structures,
            liquidity,
            order_blocks,
            selected_pois,
            f"Adaptive Two-Way SMC - {symbol_name}",
        )
        plot_equity(equity)

    return {
        "swings": swings,
        "market_regime": regime_df,
        "liquidity": liquidity,
        "structures": structures,
        "displacement": displacement,
        "order_blocks": order_blocks,
        "all_pois": all_pois,
        "selected_pois": selected_pois,
        "trades": trades,
        "equity": equity,
        "summary": summary,
        "regime_summary": regime_summary,
        "llm_summary": llm_summary,
        "output_file": output_file,
    }


def sanitize_filename(text: str) -> str:
    bad_chars = [" ", "|", "/", "\\", ":", "*", "?", '"', "<", ">"]
    result = text
    for ch in bad_chars:
        result = result.replace(ch, "_")
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_")


def main():
    parser = argparse.ArgumentParser(description="Adaptive Two-Way SMC Bot")
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC-USD",
        help="Contoh: BTC-USD, ETH-USD, AAPL, MSFT",
    )
    parser.add_argument(
        "--period", type=str, default="1y", help="Contoh: 6mo, 1y, 2y, 5y, 730d"
    )
    parser.add_argument(
        "--interval", type=str, default="1d", help="Contoh: 1d, 4h, 1h, 30m"
    )
    parser.add_argument("--synthetic", action="store_true", help="Pakai data dummy/sintetik")
    parser.add_argument("--no-plot", action="store_true", help="Jangan tampilkan chart")

    parser.add_argument(
        "--market-mode",
        type=str,
        default="long_short",
        choices=["long_short", "spot_only"],
        help="long_short: bearish=SHORT. spot_only: bearish=exit/avoid only.",
    )
    parser.add_argument(
        "--entry-touch-mode",
        type=str,
        default="zone_touch",
        choices=["midpoint", "zone_touch"],
        help="midpoint atau zone_touch",
    )
    parser.add_argument(
        "--entry-mode",
        type=str,
        default="balanced",
        choices=["aggressive", "balanced", "conservative"],
        help=(
            "aggressive=langsung entry saat zone touch; "
            "balanced=candle searah tanpa wajib break high/low; "
            "conservative=confirmation ketat"
        ),
    )

    parser.add_argument("--swing", type=int, default=3)
    parser.add_argument("--min-disp", type=float, default=0.002)
    parser.add_argument("--use-volume", action="store_true")
    parser.add_argument("--min-score", type=int, default=3)

    parser.add_argument("--ema-fast", type=int, default=20)
    parser.add_argument("--ema-mid", type=int, default=50)
    parser.add_argument("--ema-slow", type=int, default=200)
    parser.add_argument("--trending-adx-min", type=float, default=18.0)
    parser.add_argument("--sideways-adx-max", type=float, default=16.0)

    parser.add_argument("--rr", type=float, default=2.0)
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--max-holding", type=int, default=40)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--no-confirmation", action="store_true", help="Matikan confirmation / pakai entry-mode aggressive")
    parser.add_argument("--confirmation-lookahead", type=int, default=3)
    parser.add_argument("--confirmation-body-ratio", type=float, default=0.45)
    parser.add_argument("--balanced-body-ratio", type=float, default=0.25)
    parser.add_argument(
        "--balanced-entry-price",
        type=str,
        default="zone_if_same_candle",
        choices=["zone_if_same_candle", "close", "theoretical"],
        help="Harga entry untuk balanced mode",
    )
    parser.add_argument("--no-balanced-entry-side", action="store_true", help="Balanced mode tidak wajib close di sisi entry")
    parser.add_argument("--disable-dynamic-rr", action="store_true")
    parser.add_argument("--disable-dynamic-risk", action="store_true")
    parser.add_argument("--disable-pruning", action="store_true")
    parser.add_argument("--rr-trending", type=float, default=2.0)
    parser.add_argument("--rr-high-vol", type=float, default=1.8)
    parser.add_argument("--rr-sideways", type=float, default=1.5)
    parser.add_argument("--rr-sideways-sweep", type=float, default=1.7)
    parser.add_argument("--rr-low-vol", type=float, default=1.3)
    parser.add_argument("--output", type=str, default=None, help="Nama file output .xlsx")

    args = parser.parse_args()

    cfg = SMCConfig(
        swing_length=args.swing,
        min_disp=args.min_disp,
        rr=args.rr,
        risk_per_trade=args.risk,
        use_volume=args.use_volume,
        min_poi_score=args.min_score,
        ema_fast=args.ema_fast,
        ema_mid=args.ema_mid,
        ema_slow=args.ema_slow,
        trending_adx_min=args.trending_adx_min,
        sideways_adx_max=args.sideways_adx_max,
        max_holding_bars=args.max_holding,
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage,
        require_confirmation=not args.no_confirmation,
        confirmation_lookahead=args.confirmation_lookahead,
        confirmation_body_ratio_min=args.confirmation_body_ratio,
        balanced_body_ratio_min=args.balanced_body_ratio,
        balanced_require_entry_side=not args.no_balanced_entry_side,
        balanced_entry_price=args.balanced_entry_price,
        market_mode=args.market_mode,
        entry_touch_mode=args.entry_touch_mode,
        entry_mode="aggressive" if args.no_confirmation else args.entry_mode,
        use_dynamic_rr=not args.disable_dynamic_rr,
        use_dynamic_risk=not args.disable_dynamic_risk,
        prune_weak_setups=not args.disable_pruning,
        rr_trending=args.rr_trending,
        rr_high_volatility=args.rr_high_vol,
        rr_sideways=args.rr_sideways,
        rr_sideways_sweep=args.rr_sideways_sweep,
        rr_low_volatility=args.rr_low_vol,
    )

    if args.synthetic:
        df = make_synthetic_ohlcv()
        symbol_name = "SYNTHETIC"
        default_output = f"adaptive_two_way_smc_SYNTHETIC_{cfg.market_mode}.xlsx"
    else:
        df = load_yfinance(args.symbol, args.period, args.interval)
        symbol_name = f"{args.symbol} | {args.period} | {args.interval}"
        default_output = (
            f"adaptive_two_way_smc_"
            f"{sanitize_filename(args.symbol)}_"
            f"{sanitize_filename(args.period)}_"
            f"{sanitize_filename(args.interval)}_"
            f"{sanitize_filename(cfg.market_mode)}.xlsx"
        )

    output_file = args.output if args.output else default_output
    if not output_file.lower().endswith(".xlsx"):
        output_file += ".xlsx"

    run_pipeline(
        df=df,
        symbol_name=symbol_name,
        cfg=cfg,
        output_file=output_file,
        show_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()