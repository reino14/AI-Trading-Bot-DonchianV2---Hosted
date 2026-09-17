import argparse
import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================

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
    rr: float = 2.0
    risk_per_trade: float = 0.01
    initial_equity: float = 10_000.0
    max_holding_bars: int = 40
    min_poi_score: int = 3


# =========================================================
# DATA LOADER
# =========================================================

def load_yfinance(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install dulu: pip install yfinance")

    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError("Data kosong. Coba symbol/period/interval lain.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df.columns = [str(c).lower() for c in df.columns]

    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Kolom {col} tidak ditemukan dari yfinance.")

    if "volume" not in df.columns:
        df["volume"] = np.nan

    df = df[["open", "high", "low", "close", "volume"]].dropna(
        subset=["open", "high", "low", "close"]
    )

    return df


def make_synthetic_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    regimes = np.concatenate([
        rng.normal(0.0008, 0.006, n // 3),
        rng.normal(-0.0004, 0.008, n // 3),
        rng.normal(0.0012, 0.010, n - 2 * (n // 3)),
    ])

    close = [100.0]
    for r in regimes:
        close.append(close[-1] * (1 + r))
    close = np.array(close[1:])

    for ix in [70, 120, 210, 280, 360, 430]:
        if ix < len(close):
            close[ix:] *= rng.choice([1.03, 0.97, 1.04, 0.96])

    open_ = np.r_[close[0], close[:-1] * (1 + rng.normal(0, 0.0015, len(close) - 1))]
    spread = np.maximum(close * rng.uniform(0.002, 0.01, len(close)), 0.1)

    high = np.maximum(open_, close) + spread * rng.uniform(0.2, 1.0, len(close))
    low = np.minimum(open_, close) - spread * rng.uniform(0.2, 1.0, len(close))
    volume = rng.integers(100, 1000, len(close)).astype(float)

    for ix in [70, 120, 210, 280, 360, 430]:
        if ix < len(volume):
            volume[ix] *= rng.integers(2, 5)

    idx = pd.date_range("2024-01-01", periods=len(close), freq="D")

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=idx)


# =========================================================
# CORE HELPER
# =========================================================

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
    timestamp = out.index

    out = out[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
    out.insert(0, "timestamp", timestamp)
    out["_i"] = np.arange(len(out))

    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(df).rolling(window, min_periods=1).mean()


# =========================================================
# SWING HIGH / LOW
# =========================================================

def extract_swings(df: pd.DataFrame, cfg: SMCConfig) -> Dict:
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
        left_h = highs[i - L:i]
        right_h = highs[i + 1:i + L + 1]
        left_l = lows[i - L:i]
        right_l = lows[i + 1:i + L + 1]

        is_high = highs[i] > left_h.max() and highs[i] >= right_h.max()
        is_low = lows[i] < left_l.min() and lows[i] <= right_l.min()

        if is_high:
            data.loc[i, "swing_high"] = True
            row = data.iloc[i]

            points.append({
                "index": int(i),
                "confirm_index": int(i + L),
                "timestamp": row["timestamp"],
                "type": "high",
                "level": float(row["high"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]) if pd.notna(row["volume"]) else np.nan,
                "atr": float(row["atr"]),
            })

        if is_low:
            data.loc[i, "swing_low"] = True
            row = data.iloc[i]

            points.append({
                "index": int(i),
                "confirm_index": int(i + L),
                "timestamp": row["timestamp"],
                "type": "low",
                "level": float(row["low"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]) if pd.notna(row["volume"]) else np.nan,
                "atr": float(row["atr"]),
            })

    points = sorted(points, key=lambda x: x["index"])

    return {
        "df": data,
        "points": points,
    }


# =========================================================
# LIQUIDITY
# =========================================================

def detect_liquidity(df: pd.DataFrame, cfg: SMCConfig, max_span: int = 80) -> List[Dict]:
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

                search = data.iloc[end_idx + 1:]

                if swing_type == "high":
                    side = "BSL"
                    hit = search[search["high"] > top * (1 + cfg.liquidity_tolerance_pct / 3)]
                else:
                    side = "SSL"
                    hit = search[search["low"] < bottom * (1 - cfg.liquidity_tolerance_pct / 3)]

                swept_index = None if hit.empty else int(hit.iloc[0]["_i"])

                zones.append({
                    "side": side,
                    "level": level,
                    "top": top,
                    "bottom": bottom,
                    "start_index": int(start_idx),
                    "end_index": int(end_idx),
                    "touches": len(cluster),
                    "touch_indices": ",".join(str(x["index"]) for x in cluster),
                    "swept_index": swept_index,
                })

    uniq = {}
    for z in zones:
        key = (z["side"], round(z["level"], 5), z["start_index"], z["end_index"])
        uniq[key] = z

    return list(uniq.values())


# =========================================================
# BOS / CHOCH
# =========================================================

def wick_break_ok(row: pd.Series, direction: int, cfg: SMCConfig) -> bool:
    body = max(abs(row["close"] - row["open"]), 1e-12)
    upper_wick = row["high"] - max(row["open"], row["close"])
    lower_wick = min(row["open"], row["close"]) - row["low"]

    if direction == 1:
        return upper_wick <= body * cfg.wick_ratio_max
    else:
        return lower_wick <= body * cfg.wick_ratio_max


def detect_market_structure(swings: Dict, cfg: SMCConfig) -> List[Dict]:
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
            disp = (break_up_value - last_high["level"]) / max(abs(last_high["level"]), 1e-12)

            vol_ok = True
            if cfg.use_volume:
                vol_ok = (
                    pd.notna(row["volume"])
                    and pd.notna(row["vol_ma"])
                    and row["volume"] >= row["vol_ma"] * cfg.volume_mult
                )

            wick_ok = True if not cfg.wick_check else wick_break_ok(row, 1, cfg)

            if break_up_value > last_high["level"] and disp >= cfg.min_disp and wick_ok and vol_ok:
                event = "CHOCH_BULLISH" if trend == "bearish" else "BOS_BULLISH"

                structures.append({
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
                })

                trend = "bullish"
                last_high["broken"] = True

        if last_low is not None and not last_low["broken"] and i > last_low["index"]:
            disp = (last_low["level"] - break_dn_value) / max(abs(last_low["level"]), 1e-12)

            vol_ok = True
            if cfg.use_volume:
                vol_ok = (
                    pd.notna(row["volume"])
                    and pd.notna(row["vol_ma"])
                    and row["volume"] >= row["vol_ma"] * cfg.volume_mult
                )

            wick_ok = True if not cfg.wick_check else wick_break_ok(row, -1, cfg)

            if break_dn_value < last_low["level"] and disp >= cfg.min_disp and wick_ok and vol_ok:
                event = "CHOCH_BEARISH" if trend == "bullish" else "BOS_BEARISH"

                structures.append({
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
                })

                trend = "bearish"
                last_low["broken"] = True

    return structures


# =========================================================
# DISPLACEMENT
# =========================================================

def detect_displacement_candles(df: pd.DataFrame, cfg: SMCConfig) -> List[Dict]:
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

        direction = 1 if row["close"] > row["open"] else -1 if row["close"] < row["open"] else 0

        if direction == 0:
            continue

        disps.append({
            "index": int(row["_i"]),
            "timestamp": row["timestamp"],
            "direction": direction,
            "direction_text": "bullish" if direction == 1 else "bearish",
            "type": "DISPLACEMENT_BULLISH" if direction == 1 else "DISPLACEMENT_BEARISH",
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "range": float(row["range"]),
            "atr": float(row["atr"]),
            "body_ratio": float(row["body_ratio"]),
        })

    return disps


# =========================================================
# ORDER BLOCK
# =========================================================

def detect_order_blocks(df: pd.DataFrame, structures: List[Dict], cfg: SMCConfig) -> List[Dict]:
    data = ensure_ohlcv(df)
    disps = detect_displacement_candles(df, cfg)

    obs = []
    used = set()

    for s in structures:
        direction = s["direction"]
        start_idx = s["source_swing_index"]
        end_idx = s["index"]

        relevant_disps = [
            d for d in disps
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
            touched = row["low"] <= top and row["high"] >= bottom

            if touched:
                mitigated_index = int(k)
                break

        obs.append({
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
        })

    return obs


# =========================================================
# POI
# =========================================================

def generate_pois(order_blocks: List[Dict], liquidity: List[Dict], proximity_pct: float = 0.002) -> List[Dict]:
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

                if liq["swept_index"] is not None and liq["swept_index"] <= ob["break_index"]:
                    score += 1
                    reasons.append("liquidity_swept_before_break")

                break

        pois.append({
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
        })

    return sorted(pois, key=lambda x: (-x["score"], x["active_from"]))


# =========================================================
# BACKTEST
# =========================================================

def backtest(df: pd.DataFrame, pois: List[Dict], cfg: SMCConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = ensure_ohlcv(df)

    equity = cfg.initial_equity
    trades = []
    equity_curve = [{"index": 0, "equity": equity}]
    last_exit_idx = -1

    for poi in pois:
        if poi["score"] < cfg.min_poi_score:
            continue

        if last_exit_idx > poi["active_from"]:
            continue

        top = poi["top"]
        bottom = poi["bottom"]
        entry = (top + bottom) / 2
        zone_h = max(top - bottom, 1e-12)
        buffer = zone_h * 0.1

        entry_idx = None

        for i in range(poi["active_from"] + 1, len(data)):
            row = data.iloc[i]
            touched = row["low"] <= top and row["high"] >= bottom

            if touched:
                entry_idx = i
                break

        if entry_idx is None:
            continue

        if poi["direction"] == 1:
            stop = bottom - buffer
            risk = entry - stop

            if risk <= 0:
                continue

            target = entry + cfg.rr * risk
        else:
            stop = top + buffer
            risk = stop - entry

            if risk <= 0:
                continue

            target = entry - cfg.rr * risk

        cash_risk = equity * cfg.risk_per_trade
        size = cash_risk / risk

        exit_idx = None
        exit_price = None
        result_r = None
        reason = None

        max_exit = min(len(data), entry_idx + cfg.max_holding_bars + 1)

        for j in range(entry_idx, max_exit):
            row = data.iloc[j]

            if poi["direction"] == 1:
                hit_sl = row["low"] <= stop
                hit_tp = row["high"] >= target

                if hit_sl and hit_tp:
                    exit_idx = j
                    exit_price = stop
                    result_r = -1.0
                    reason = "SL_and_TP_same_bar_SL_first"
                    break
                elif hit_sl:
                    exit_idx = j
                    exit_price = stop
                    result_r = -1.0
                    reason = "SL"
                    break
                elif hit_tp:
                    exit_idx = j
                    exit_price = target
                    result_r = cfg.rr
                    reason = "TP"
                    break

            else:
                hit_sl = row["high"] >= stop
                hit_tp = row["low"] <= target

                if hit_sl and hit_tp:
                    exit_idx = j
                    exit_price = stop
                    result_r = -1.0
                    reason = "SL_and_TP_same_bar_SL_first"
                    break
                elif hit_sl:
                    exit_idx = j
                    exit_price = stop
                    result_r = -1.0
                    reason = "SL"
                    break
                elif hit_tp:
                    exit_idx = j
                    exit_price = target
                    result_r = cfg.rr
                    reason = "TP"
                    break

        if exit_idx is None:
            exit_idx = max_exit - 1
            exit_price = float(data.iloc[exit_idx]["close"])

            if poi["direction"] == 1:
                result_r = (exit_price - entry) / risk
            else:
                result_r = (entry - exit_price) / risk

            reason = "TIME_EXIT"

        pnl_cash = cash_risk * result_r
        equity += pnl_cash
        last_exit_idx = exit_idx

        trades.append({
            "direction": "LONG" if poi["direction"] == 1 else "SHORT",
            "entry_idx": int(entry_idx),
            "exit_idx": int(exit_idx),
            "entry_price": float(entry),
            "exit_price": float(exit_price),
            "stop": float(stop),
            "target": float(target),
            "r_result": float(result_r),
            "pnl_cash": float(pnl_cash),
            "equity_after": float(equity),
            "reason": reason,
            "poi_score": int(poi["score"]),
            "position_size": float(size),
        })

        equity_curve.append({
            "index": int(exit_idx),
            "equity": float(equity),
        })

    return pd.DataFrame(trades), pd.DataFrame(equity_curve)


def summarize_backtest(trades: pd.DataFrame, equity: pd.DataFrame, cfg: SMCConfig) -> Dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0,
            "total_return_pct": 0,
            "final_equity": cfg.initial_equity,
            "max_drawdown_pct": 0,
            "avg_r": 0,
        }

    wins = trades[trades["r_result"] > 0]
    win_rate = len(wins) / len(trades) * 100

    final_equity = float(equity.iloc[-1]["equity"])
    total_return_pct = (final_equity / cfg.initial_equity - 1) * 100

    eq = equity["equity"]
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min() * 100

    return {
        "trades": int(len(trades)),
        "win_rate": float(win_rate),
        "total_return_pct": float(total_return_pct),
        "final_equity": float(final_equity),
        "max_drawdown_pct": float(max_dd),
        "avg_r": float(trades["r_result"].mean()),
    }


# =========================================================
# LLM SUMMARY
# =========================================================

def infer_market_bias(
    last_structure_direction: Optional[str],
    top_poi_direction: Optional[str],
    latest_close: float,
    top_poi_bottom: Optional[float],
    top_poi_top: Optional[float],
) -> str:
    if last_structure_direction is None:
        return "neutral_no_structure"

    near_poi = False

    if top_poi_bottom is not None and top_poi_top is not None:
        zone_mid = (top_poi_bottom + top_poi_top) / 2
        distance_to_poi = abs(latest_close - zone_mid) / latest_close
        near_poi = distance_to_poi <= 0.015

    if last_structure_direction == "bullish" and top_poi_direction == "bullish":
        if near_poi:
            return "bullish_near_demand_poi"
        return "bullish_waiting_for_retrace"

    if last_structure_direction == "bearish" and top_poi_direction == "bearish":
        if near_poi:
            return "bearish_near_supply_poi"
        return "bearish_waiting_for_retrace"

    if last_structure_direction == "bullish" and top_poi_direction == "bearish":
        return "mixed_bullish_structure_but_bearish_poi"

    if last_structure_direction == "bearish" and top_poi_direction == "bullish":
        return "mixed_bearish_structure_but_bullish_poi"

    return "neutral_unclear"


def build_llm_summary(
    df: pd.DataFrame,
    swings: Dict,
    liquidity: List[Dict],
    structures: List[Dict],
    order_blocks: List[Dict],
    pois: List[Dict],
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    cfg: SMCConfig,
    symbol_name: str,
) -> pd.DataFrame:
    data = swings["df"].copy()

    latest = data.iloc[-1]
    latest_close = float(latest["close"])
    latest_timestamp = latest["timestamp"]

    last_structure = structures[-1] if structures else None

    if last_structure:
        last_structure_type = last_structure.get("type")
        last_structure_direction = last_structure.get("direction_text")
        last_structure_level = last_structure.get("level")
        last_structure_break_price = last_structure.get("break_price")
        last_structure_index = last_structure.get("index")
    else:
        last_structure_type = None
        last_structure_direction = None
        last_structure_level = None
        last_structure_break_price = None
        last_structure_index = None

    active_pois = [
        p for p in pois
        if p.get("mitigated_index") is None or p.get("mitigated_index", 999999999) >= len(data) - 1
    ]

    active_pois_sorted = sorted(active_pois, key=lambda x: x.get("score", 0), reverse=True)
    top_poi = active_pois_sorted[0] if active_pois_sorted else None

    if top_poi:
        top_poi_type = top_poi.get("poi_type")
        top_poi_direction = top_poi.get("direction_text")
        top_poi_bottom = top_poi.get("bottom")
        top_poi_top = top_poi.get("top")
        top_poi_mid = top_poi.get("mid")
        top_poi_score = top_poi.get("score")
        top_poi_reasons = top_poi.get("reasons")
    else:
        top_poi_type = None
        top_poi_direction = None
        top_poi_bottom = None
        top_poi_top = None
        top_poi_mid = None
        top_poi_score = None
        top_poi_reasons = None

    liquidity_with_distance = []

    for liq in liquidity:
        level = liq.get("level")
        if level is None:
            continue

        distance_pct = abs(latest_close - level) / latest_close

        liquidity_with_distance.append({
            **liq,
            "distance_pct": distance_pct,
        })

    liquidity_with_distance = sorted(liquidity_with_distance, key=lambda x: x["distance_pct"])
    nearest_liq = liquidity_with_distance[0] if liquidity_with_distance else None

    if nearest_liq:
        nearest_liq_side = nearest_liq.get("side")
        nearest_liq_level = nearest_liq.get("level")
        nearest_liq_distance_pct = nearest_liq.get("distance_pct")
        nearest_liq_swept = nearest_liq.get("swept_index") is not None
    else:
        nearest_liq_side = None
        nearest_liq_level = None
        nearest_liq_distance_pct = None
        nearest_liq_swept = None

    if trades is not None and not trades.empty:
        total_trades = len(trades)
        win_rate = len(trades[trades["r_result"] > 0]) / total_trades * 100
        avg_r = trades["r_result"].mean()
        last_trade_result = trades.iloc[-1]["reason"]
    else:
        total_trades = 0
        win_rate = None
        avg_r = None
        last_trade_result = None

    if equity is not None and not equity.empty:
        final_equity = float(equity.iloc[-1]["equity"])
    else:
        final_equity = None

    market_bias = infer_market_bias(
        last_structure_direction=last_structure_direction,
        top_poi_direction=top_poi_direction,
        latest_close=latest_close,
        top_poi_bottom=top_poi_bottom,
        top_poi_top=top_poi_top,
    )

    row = {
        "symbol": symbol_name,
        "latest_timestamp": latest_timestamp,
        "latest_close": latest_close,
        "market_bias": market_bias,

        "last_structure_type": last_structure_type,
        "last_structure_direction": last_structure_direction,
        "last_structure_level": last_structure_level,
        "last_structure_break_price": last_structure_break_price,
        "last_structure_index": last_structure_index,

        "top_active_poi_type": top_poi_type,
        "top_active_poi_direction": top_poi_direction,
        "top_active_poi_bottom": top_poi_bottom,
        "top_active_poi_top": top_poi_top,
        "top_active_poi_mid": top_poi_mid,
        "top_active_poi_score": top_poi_score,
        "top_active_poi_reasons": top_poi_reasons,

        "nearest_liquidity_side": nearest_liq_side,
        "nearest_liquidity_level": nearest_liq_level,
        "nearest_liquidity_distance_pct": nearest_liq_distance_pct,
        "nearest_liquidity_swept": nearest_liq_swept,

        "total_trades": total_trades,
        "win_rate_pct": win_rate,
        "avg_r": avg_r,
        "final_equity": final_equity,
        "last_trade_result": last_trade_result,

        "config_swing_length": cfg.swing_length,
        "config_min_disp": cfg.min_disp,
        "config_rr": cfg.rr,
        "config_risk_per_trade": cfg.risk_per_trade,
        "config_min_poi_score": cfg.min_poi_score,

        "llm_instruction": (
            "Gunakan data ini untuk membuat analisis skenario, bukan prediksi pasti. "
            "Berikan market bias, alasan, area invalidation, skenario bullish, skenario bearish, "
            "dan level yang perlu diawasi berdasarkan SMC."
        ),
    }

    return pd.DataFrame([row])


# =========================================================
# EXPORT TO EXCEL MULTI-SHEET
# =========================================================

def safe_sheet_df(data, columns_if_empty=None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        if data.empty and columns_if_empty:
            return pd.DataFrame(columns=columns_if_empty)
        return data

    if isinstance(data, list):
        df = pd.DataFrame(data)
        if df.empty and columns_if_empty:
            return pd.DataFrame(columns=columns_if_empty)
        return df

    return pd.DataFrame()


def export_smc_results_to_excel(
    output_file: str,
    df: pd.DataFrame,
    swings: Dict,
    liquidity: List[Dict],
    structures: List[Dict],
    displacement: List[Dict],
    order_blocks: List[Dict],
    pois: List[Dict],
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    llm_summary: pd.DataFrame,
):
    candles = swings["df"].copy()

    swings_df = safe_sheet_df(swings["points"])
    liquidity_df = safe_sheet_df(liquidity)
    structures_df = safe_sheet_df(structures)
    displacement_df = safe_sheet_df(displacement)
    order_blocks_df = safe_sheet_df(order_blocks)
    pois_df = safe_sheet_df(pois)
    trades_df = safe_sheet_df(trades)
    equity_df = safe_sheet_df(equity)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        candles.to_excel(writer, sheet_name="candles", index=False)
        swings_df.to_excel(writer, sheet_name="swings", index=False)
        liquidity_df.to_excel(writer, sheet_name="liquidity", index=False)
        structures_df.to_excel(writer, sheet_name="structures", index=False)
        displacement_df.to_excel(writer, sheet_name="displacement", index=False)
        order_blocks_df.to_excel(writer, sheet_name="order_blocks", index=False)
        pois_df.to_excel(writer, sheet_name="pois", index=False)
        trades_df.to_excel(writer, sheet_name="trades", index=False)
        equity_df.to_excel(writer, sheet_name="equity", index=False)
        llm_summary.to_excel(writer, sheet_name="llm_summary", index=False)

        workbook = writer.book

        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"

            for col_cells in ws.columns:
                max_length = 0
                column_letter = col_cells[0].column_letter

                for cell in col_cells:
                    try:
                        value = str(cell.value) if cell.value is not None else ""
                        max_length = max(max_length, len(value))
                    except Exception:
                        pass

                adjusted_width = min(max(max_length + 2, 10), 45)
                ws.column_dimensions[column_letter].width = adjusted_width

    print(f"\nExcel exported: {output_file}")


# =========================================================
# PLOT
# =========================================================

def plot_smc(
    df: pd.DataFrame,
    swings: Dict,
    structures: List[Dict],
    liquidity: List[Dict],
    order_blocks: List[Dict],
    pois: List[Dict],
    title: str,
    max_annotations: int = 20,
):
    data = swings["df"]

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(data["_i"], data["close"], linewidth=1.2, label="Close")

    sh = data[data["swing_high"]]
    sl = data[data["swing_low"]]

    ax.scatter(sh["_i"], sh["high"], marker="^", s=45, label="Swing High")
    ax.scatter(sl["_i"], sl["low"], marker="v", s=45, label="Swing Low")

    y_min = data["low"].min()
    y_max = data["high"].max()
    y_range = y_max - y_min
    x_right = int(data["_i"].max())

    for idx, liq in enumerate(liquidity[:max_annotations]):
        ax.hlines(
            liq["level"],
            liq["start_index"],
            liq["end_index"],
            linestyles="dashed",
            linewidth=1.0,
            label="Liquidity" if idx == 0 else None,
        )
        ax.text(liq["end_index"], liq["level"], liq["side"], fontsize=8)

    for idx, ob in enumerate(order_blocks[:max_annotations]):
        width = max(
            5,
            (ob["mitigated_index"] if ob["mitigated_index"] is not None else x_right) - ob["origin_index"],
        )

        rect = Rectangle(
            (ob["origin_index"], ob["bottom"]),
            width=width,
            height=ob["top"] - ob["bottom"],
            fill=False,
            linewidth=1.0,
        )

        ax.add_patch(rect)

        ax.text(
            ob["origin_index"],
            ob["top"],
            ob["type"],
            fontsize=7,
            verticalalignment="bottom",
        )

    for idx, s in enumerate(structures[:max_annotations]):
        price = data.iloc[s["index"]]["close"]

        ax.scatter(s["index"], price, s=70, marker="o")

        offset = y_range * 0.04 * s["direction"]
        ax.annotate(
            s["type"],
            xy=(s["index"], price),
            xytext=(s["index"], price + offset),
            arrowprops=dict(arrowstyle="->"),
            fontsize=8,
        )

    for idx, p in enumerate(pois[:max_annotations]):
        mid = (p["top"] + p["bottom"]) / 2
        ax.hlines(
            mid,
            p["origin_index"],
            x_right,
            linestyles=":",
            linewidth=1.0,
            label="POI" if idx == 0 else None,
        )

    ax.set_title(title)
    ax.set_xlabel("Bar Index")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_equity(equity: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(equity["index"], equity["equity"], linewidth=1.5)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Bar Index")
    ax.set_ylabel("Equity")
    plt.tight_layout()
    plt.show()


# =========================================================
# PIPELINE
# =========================================================

def run_pipeline(
    df: pd.DataFrame,
    symbol_name: str,
    cfg: SMCConfig,
    output_file: str,
    show_plot: bool = True,
):
    swings = extract_swings(df, cfg)
    liquidity = detect_liquidity(df, cfg)
    structures = detect_market_structure(swings, cfg)
    displacement = detect_displacement_candles(df, cfg)
    order_blocks = detect_order_blocks(df, structures, cfg)
    pois = generate_pois(order_blocks, liquidity)

    trades, equity = backtest(df, pois, cfg)
    summary = summarize_backtest(trades, equity, cfg)

    llm_summary = build_llm_summary(
        df=df,
        swings=swings,
        liquidity=liquidity,
        structures=structures,
        order_blocks=order_blocks,
        pois=pois,
        trades=trades,
        equity=equity,
        cfg=cfg,
        symbol_name=symbol_name,
    )

    export_smc_results_to_excel(
        output_file=output_file,
        df=df,
        swings=swings,
        liquidity=liquidity,
        structures=structures,
        displacement=displacement,
        order_blocks=order_blocks,
        pois=pois,
        trades=trades,
        equity=equity,
        llm_summary=llm_summary,
    )

    print("\n==============================")
    print(f"SMC BOT RESULT: {symbol_name}")
    print("==============================")
    print(f"Total candles       : {len(df)}")
    print(f"Swing points        : {len(swings['points'])}")
    print(f"Liquidity zones     : {len(liquidity)}")
    print(f"BOS/CHOCH events    : {len(structures)}")
    print(f"Displacement        : {len(displacement)}")
    print(f"Order blocks        : {len(order_blocks)}")
    print(f"POI                 : {len(pois)}")
    print(f"Trades              : {summary['trades']}")
    print(f"Win rate            : {summary['win_rate']:.2f}%")
    print(f"Avg R               : {summary['avg_r']:.2f}R")
    print(f"Total return        : {summary['total_return_pct']:.2f}%")
    print(f"Final equity        : {summary['final_equity']:.2f}")
    print(f"Max drawdown        : {summary['max_drawdown_pct']:.2f}%")

    print("\nLatest BOS/CHOCH:")
    if structures:
        for s in structures[-10:]:
            print(
                f"- {s['timestamp']} | {s['type']} | "
                f"level={s['level']:.4f} | break={s['break_price']:.4f} | disp={s['disp_pct']:.4%}"
            )
    else:
        print("- Tidak ada BOS/CHOCH terdeteksi.")

    print("\nTop POI:")
    if pois:
        for p in pois[:10]:
            print(
                f"- {p['poi_type']} | score={p['score']} | "
                f"zone={p['bottom']:.4f} - {p['top']:.4f} | "
                f"from bar={p['active_from']} | reasons={p['reasons']}"
            )
    else:
        print("- Tidak ada POI terdeteksi.")

    print("\nLLM Summary:")
    print(llm_summary.to_string(index=False))

    if show_plot:
        plot_smc(
            df=df,
            swings=swings,
            structures=structures,
            liquidity=liquidity,
            order_blocks=order_blocks,
            pois=pois,
            title=f"SMC Structure - {symbol_name}",
        )

        plot_equity(equity)

    return {
        "swings": swings,
        "liquidity": liquidity,
        "structures": structures,
        "displacement": displacement,
        "order_blocks": order_blocks,
        "pois": pois,
        "trades": trades,
        "equity": equity,
        "summary": summary,
        "llm_summary": llm_summary,
        "output_file": output_file,
    }


# =========================================================
# MAIN
# =========================================================

def sanitize_filename(text: str) -> str:
    bad_chars = [" ", "|", "/", "\\", ":", "*", "?", '"', "<", ">"]
    result = text

    for ch in bad_chars:
        result = result.replace(ch, "_")

    while "__" in result:
        result = result.replace("__", "_")

    return result.strip("_")


def main():
    parser = argparse.ArgumentParser(
        description="Prototype SMC Bot: Swing + Liquidity + BOS/CHOCH + OB + Backtest + Excel Export"
    )

    parser.add_argument("--symbol", type=str, default="BTC-USD", help="Contoh: BTC-USD, ETH-USD, AAPL, MSFT")
    parser.add_argument("--period", type=str, default="1y", help="Contoh: 6mo, 1y, 2y, 5y")
    parser.add_argument("--interval", type=str, default="1d", help="Contoh: 1d, 4h, 1h, 30m")
    parser.add_argument("--synthetic", action="store_true", help="Pakai data dummy/sintetik")
    parser.add_argument("--no-plot", action="store_true", help="Jangan tampilkan chart")

    parser.add_argument("--swing", type=int, default=3)
    parser.add_argument("--min-disp", type=float, default=0.002)
    parser.add_argument("--rr", type=float, default=2.0)
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--use-volume", action="store_true")
    parser.add_argument("--min-score", type=int, default=3)
    parser.add_argument("--output", type=str, default=None, help="Nama file output .xlsx")

    args = parser.parse_args()

    cfg = SMCConfig(
        swing_length=args.swing,
        min_disp=args.min_disp,
        rr=args.rr,
        risk_per_trade=args.risk,
        use_volume=args.use_volume,
        min_poi_score=args.min_score,
    )

    if args.synthetic:
        df = make_synthetic_ohlcv()
        symbol_name = "SYNTHETIC"
        default_output = "smc_SYNTHETIC.xlsx"
    else:
        df = load_yfinance(args.symbol, args.period, args.interval)
        symbol_name = f"{args.symbol} | {args.period} | {args.interval}"
        default_output = f"smc_{sanitize_filename(args.symbol)}_{sanitize_filename(args.period)}_{sanitize_filename(args.interval)}.xlsx"

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