"""
Binance USDS-M Futures Testnet Demo Trader for Adaptive SMC Bot
===============================================================

This file is an execution layer. It imports your strategy engine from:
    SMC_bot_adaptive_next_balanced.py

Main purpose:
- Fetch BTCUSDT/ETHUSDT futures candles from Binance Futures Testnet/Demo API
- Run your Adaptive SMC strategy engine
- Generate latest actionable LONG/SHORT/WAIT signal
- Open demo/testnet futures position automatically
- Manage exit with local SL/TP monitor using market close orders
- Log every decision/order to CSV

IMPORTANT:
- This is for Binance Futures TESTNET/DEMO first.
- Do not use real API keys until paper/demo has been stable.
- Keep the strategy file and this executor file in the same folder.

Install:
    pip install pandas numpy requests python-dotenv matplotlib openpyxl yfinance

.env example:
    BINANCE_TESTNET_API_KEY=your_testnet_key
    BINANCE_TESTNET_API_SECRET=your_testnet_secret

Dry-run once, no order:
    python demo_trader_binance_futures.py --symbol BTCUSDT --interval 1h --lookback-days 180 --dry-run --once

Run testnet demo auto-trader loop:
    python demo_trader_binance_futures.py --symbol BTCUSDT --interval 1h --lookback-days 180 --testnet --loop --poll-seconds 60

Recommended starting config:
    python demo_trader_binance_futures.py --symbol BTCUSDT --interval 1h --lookback-days 180 --testnet --loop --entry-mode aggressive --risk 0.0025 --leverage 1 --max-trades-per-day 2 --daily-loss-limit-pct 1.0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except ImportError:  # optional dependency fallback
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


USDS_M_TESTNET_BASE_URL = "https://demo-fapi.binance.com"
USDS_M_MAINNET_BASE_URL = "https://fapi.binance.com"


@dataclass
class DemoTraderConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    lookback_days: int = 180
    strategy_file: str = "SMC_bot_adaptive_next_balanced.py"

    testnet: bool = True
    dry_run: bool = True
    loop: bool = False
    poll_seconds: int = 60

    # Strategy mode; for BTC futures paper demo based on your latest result, aggressive is the candidate.
    market_mode: str = "long_short"
    entry_touch_mode: str = "zone_touch"
    entry_mode: str = "aggressive"
    min_score: int = 2
    risk_per_trade: float = 0.0025  # 0.25% recommended for first demo
    leverage: int = 1

    # Execution guardrails
    max_open_positions: int = 1
    max_trades_per_day: int = 2
    daily_loss_limit_pct: float = 1.0
    max_entry_distance_pct: float = 0.0020  # 0.20%; do not chase if candle touched zone but close is too far
    min_sl_distance_pct: float = 0.0010     # 0.10%; avoid tiny stop
    max_sl_distance_pct: float = 0.0300     # 3.00%; avoid too-wide stop
    skip_unclear_regime: bool = True
    cancel_open_orders_before_entry: bool = True

    # Local files
    state_file: str = "demo_trader_state.json"
    log_file: str = "demo_trades_log.csv"

    # API
    recv_window: int = 5000
    request_timeout: int = 20


class BinanceFuturesClient:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str,
        recv_window: int = 5000,
        timeout: int = 20,
    ):
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8") if api_secret else b""
        self.base_url = base_url.rstrip("/")
        self.recv_window = recv_window
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": api_key})

    def _timestamp_ms(self) -> int:
        return int(time.time() * 1000)

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        signed = dict(params)
        signed["timestamp"] = self._timestamp_ms()
        signed["recvWindow"] = self.recv_window
        query = urlencode(signed, doseq=True)
        signature = hmac.new(self.api_secret, query.encode("utf-8"), hashlib.sha256).hexdigest()
        signed["signature"] = signature
        return signed

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        params = params or {}
        url = f"{self.base_url}{path}"
        send_params = self._sign(params) if signed else params
        method = method.upper()

        if method == "GET":
            resp = self.session.get(url, params=send_params, timeout=self.timeout)
        elif method == "POST":
            resp = self.session.post(url, params=send_params, timeout=self.timeout)
        elif method == "DELETE":
            resp = self.session.delete(url, params=send_params, timeout=self.timeout)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        try:
            data = resp.json()
        except Exception:
            data = resp.text

        if not resp.ok:
            raise RuntimeError(f"Binance API error {resp.status_code}: {data}")
        return data

    # Public endpoints
    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1500,
    ) -> List[List[Any]]:
        params: Dict[str, Any] = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        return self._request("GET", "/fapi/v1/klines", params=params, signed=False)

    def get_ticker_price(self, symbol: str) -> float:
        data = self._request("GET", "/fapi/v1/ticker/price", {"symbol": symbol.upper()}, signed=False)
        return float(data["price"])

    def get_exchange_info(self) -> Dict[str, Any]:
        return self._request("GET", "/fapi/v1/exchangeInfo", signed=False)

    # Private endpoints
    def get_balance(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/fapi/v2/balance", signed=True)

    def get_position_risk(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return self._request("GET", "/fapi/v2/positionRisk", params=params, signed=True)

    def change_leverage(self, symbol: str, leverage: int) -> Any:
        return self._request(
            "POST",
            "/fapi/v1/leverage",
            params={"symbol": symbol.upper(), "leverage": int(leverage)},
            signed=True,
        )

    def cancel_all_open_orders(self, symbol: str) -> Any:
        return self._request(
            "DELETE",
            "/fapi/v1/allOpenOrders",
            params={"symbol": symbol.upper()},
            signed=True,
        )

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
        new_client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "MARKET",
            "quantity": format_float(quantity),
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        if new_client_order_id:
            params["newClientOrderId"] = new_client_order_id[:36]
        return self._request("POST", "/fapi/v1/order", params=params, signed=True)


@dataclass
class SymbolFilters:
    tick_size: float
    step_size: float
    min_qty: float
    min_notional: float


def utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    amount = int(interval[:-1])
    if unit == "m":
        return amount * 60_000
    if unit == "h":
        return amount * 60 * 60_000
    if unit == "d":
        return amount * 24 * 60 * 60_000
    if unit == "w":
        return amount * 7 * 24 * 60 * 60_000
    raise ValueError(f"Unsupported interval: {interval}")


def format_float(value: float) -> str:
    # Avoid scientific notation in Binance quantities/prices.
    text = f"{value:.16f}".rstrip("0").rstrip(".")
    return text if text else "0"


def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    d_value = Decimal(str(value))
    d_step = Decimal(str(step))
    return float((d_value / d_step).to_integral_value(rounding=ROUND_DOWN) * d_step)


def round_price_to_tick(price: float, tick: float) -> float:
    return floor_to_step(price, tick)


def load_strategy_module(strategy_file: str):
    path = Path(strategy_file).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")
    spec = importlib.util.spec_from_file_location("smc_strategy_engine", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import strategy file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["smc_strategy_engine"] = module
    spec.loader.exec_module(module)
    return module


def fetch_ohlcv_from_binance(
    client: BinanceFuturesClient,
    symbol: str,
    interval: str,
    lookback_days: int,
) -> pd.DataFrame:
    interval_ms = interval_to_ms(interval)
    end_ms = utc_now_ms()
    start_ms = end_ms - lookback_days * 24 * 60 * 60_000
    all_rows: List[List[Any]] = []
    cursor = start_ms

    while cursor < end_ms:
        rows = client.get_klines(symbol=symbol, interval=interval, start_time=cursor, end_time=end_ms, limit=1500)
        if not rows:
            break
        all_rows.extend(rows)
        last_open_time = int(rows[-1][0])
        next_cursor = last_open_time + interval_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(rows) < 1500:
            break
        time.sleep(0.05)

    if not all_rows:
        raise RuntimeError("No klines returned from Binance.")

    # Deduplicate by open time.
    unique: Dict[int, List[Any]] = {int(row[0]): row for row in all_rows}
    rows_sorted = [unique[k] for k in sorted(unique.keys())]

    df = pd.DataFrame(
        rows_sorted,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    df["close_time_ms"] = pd.to_numeric(df["close_time"], errors="coerce").astype("int64")

    # Remove still-open candle. Use only closed candles for signal generation.
    now = utc_now_ms()
    df = df[df["close_time_ms"] < now].copy()
    df = df.set_index("open_time")[["open", "high", "low", "close", "volume"]].dropna()
    return df


def get_symbol_filters(client: BinanceFuturesClient, symbol: str) -> SymbolFilters:
    info = client.get_exchange_info()
    symbol_info = None
    for item in info.get("symbols", []):
        if item.get("symbol") == symbol.upper():
            symbol_info = item
            break
    if not symbol_info:
        raise RuntimeError(f"Symbol not found in exchangeInfo: {symbol}")

    tick_size = 0.01
    step_size = 0.001
    min_qty = 0.001
    min_notional = 5.0
    for f in symbol_info.get("filters", []):
        if f.get("filterType") == "PRICE_FILTER":
            tick_size = float(f.get("tickSize", tick_size))
        elif f.get("filterType") == "LOT_SIZE":
            step_size = float(f.get("stepSize", step_size))
            min_qty = float(f.get("minQty", min_qty))
        elif f.get("filterType") in {"MIN_NOTIONAL", "NOTIONAL"}:
            min_notional = float(f.get("notional", f.get("minNotional", min_notional)))
    return SymbolFilters(tick_size=tick_size, step_size=step_size, min_qty=min_qty, min_notional=min_notional)


def get_usdt_balance(client: BinanceFuturesClient) -> float:
    balances = client.get_balance()
    for row in balances:
        if row.get("asset") == "USDT":
            # availableBalance is safer for new trade sizing, balance is total wallet.
            return float(row.get("availableBalance", row.get("balance", 0.0)))
    return 0.0


def get_open_position(client: BinanceFuturesClient, symbol: str) -> Optional[Dict[str, Any]]:
    positions = client.get_position_risk(symbol=symbol)
    for p in positions:
        amt = float(p.get("positionAmt", 0.0))
        if abs(amt) > 0:
            return p
    return None


def load_state(path: str) -> Dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return {}
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(path: str, state: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def append_csv_log(path: str, row: Dict[str, Any]) -> None:
    file = Path(path)
    file_exists = file.exists()
    fieldnames = list(row.keys())
    with file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def today_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_today_log_stats(log_file: str) -> Tuple[int, float]:
    file = Path(log_file)
    if not file.exists():
        return 0, 0.0
    today = today_utc_str()
    entries = 0
    pnl = 0.0
    try:
        with file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = str(row.get("timestamp_utc", ""))
                if not ts.startswith(today):
                    continue
                event = row.get("event")
                if event == "ENTRY_ORDER_SENT":
                    entries += 1
                if event == "MANAGED_EXIT_SENT":
                    try:
                        pnl += float(row.get("estimated_pnl_usdt", 0.0))
                    except Exception:
                        pass
    except Exception:
        return 0, 0.0
    return entries, pnl


def build_strategy_config(strategy: Any, cfg: DemoTraderConfig):
    # Use the exact SMCConfig from your strategy engine, then override relevant fields.
    smc_cfg = strategy.SMCConfig(
        market_mode=cfg.market_mode,
        entry_touch_mode=cfg.entry_touch_mode,
        entry_mode=cfg.entry_mode,
        min_poi_score=cfg.min_score,
        risk_per_trade=cfg.risk_per_trade,
        prune_weak_setups=True,
        use_dynamic_rr=True,
        use_dynamic_risk=True,
    )
    return smc_cfg


def run_strategy_signal(
    strategy: Any,
    df: pd.DataFrame,
    smc_cfg: Any,
    cfg: DemoTraderConfig,
) -> Dict[str, Any]:
    swings = strategy.extract_swings(df, smc_cfg)
    liquidity = strategy.detect_liquidity(df, smc_cfg)
    structures = strategy.detect_market_structure(swings, smc_cfg)
    order_blocks = strategy.detect_order_blocks(df, structures, smc_cfg)
    regime_df = strategy.detect_market_regime(df, smc_cfg)
    raw_pois = strategy.generate_pois(order_blocks, liquidity)
    all_pois, selected_pois = strategy.enrich_and_filter_pois_by_regime(raw_pois, regime_df, smc_cfg)

    data = strategy.ensure_ohlcv(df)
    if data.empty:
        return {"action": "WAIT", "reason": "empty_data"}

    latest = data.iloc[-1]
    latest_idx = int(latest["_i"])
    latest_close = float(latest["close"])
    latest_high = float(latest["high"])
    latest_low = float(latest["low"])
    latest_timestamp = latest["timestamp"]

    latest_regime_row = regime_df.iloc[-1].to_dict() if not regime_df.empty else {}
    latest_regime = latest_regime_row.get("market_regime")
    if cfg.skip_unclear_regime and str(latest_regime).startswith("unclear"):
        return {
            "action": "WAIT",
            "reason": "latest_regime_unclear",
            "latest_timestamp": str(latest_timestamp),
            "latest_close": latest_close,
            "latest_regime": latest_regime,
            "selected_pois": len(selected_pois),
        }

    candidates: List[Dict[str, Any]] = []
    for poi in selected_pois:
        # We only act when the latest closed candle is the FIRST mitigation/touch.
        mitigated_index = poi.get("mitigated_index")
        if mitigated_index is None:
            continue
        if int(mitigated_index) != latest_idx:
            continue
        if int(poi.get("active_from", -1)) >= latest_idx:
            continue

        direction = int(poi["direction"])
        top = float(poi["top"])
        bottom = float(poi["bottom"])
        theoretical_entry = (top + bottom) / 2.0

        # Candle touched zone by construction, but market entry at candle close may be too far.
        entry_distance_pct = abs(latest_close - theoretical_entry) / max(abs(theoretical_entry), 1e-12)
        close_inside_zone = bottom <= latest_close <= top
        close_near_entry = entry_distance_pct <= cfg.max_entry_distance_pct
        if not (close_inside_zone or close_near_entry):
            continue

        zone_h = max(top - bottom, 1e-12)
        buffer = zone_h * smc_cfg.zone_buffer_pct
        rr_used = float(poi.get("rr_used", strategy.dynamic_rr_for_poi(poi, smc_cfg)))
        if direction == 1:
            action = "LONG"
            stop = bottom - buffer
            risk_per_unit = latest_close - stop
            target = latest_close + rr_used * risk_per_unit
        else:
            action = "SHORT"
            stop = top + buffer
            risk_per_unit = stop - latest_close
            target = latest_close - rr_used * risk_per_unit

        if risk_per_unit <= 0:
            continue
        sl_distance_pct = risk_per_unit / max(abs(latest_close), 1e-12)
        if sl_distance_pct < cfg.min_sl_distance_pct:
            continue
        if sl_distance_pct > cfg.max_sl_distance_pct:
            continue

        candidate = {
            "action": action,
            "direction": direction,
            "entry_price_estimate": latest_close,
            "theoretical_entry": theoretical_entry,
            "stop": float(stop),
            "target": float(target),
            "rr_used": rr_used,
            "risk_per_unit": float(risk_per_unit),
            "sl_distance_pct": float(sl_distance_pct),
            "entry_distance_pct": float(entry_distance_pct),
            "latest_timestamp": str(latest_timestamp),
            "latest_index": latest_idx,
            "latest_close": latest_close,
            "latest_high": latest_high,
            "latest_low": latest_low,
            "latest_regime": latest_regime,
            "poi_score": int(poi.get("score", 0)),
            "setup_type": poi.get("setup_type"),
            "market_regime": poi.get("market_regime"),
            "risk_multiplier": float(poi.get("risk_multiplier", 1.0)),
            "poi_bottom": bottom,
            "poi_top": top,
            "poi_reasons": poi.get("reasons"),
            "poi_active_from": int(poi.get("active_from", -1)),
            "selected_pois": len(selected_pois),
            "all_pois": len(all_pois),
        }
        candidates.append(candidate)

    if not candidates:
        return {
            "action": "WAIT",
            "reason": "no_actionable_poi_on_latest_closed_candle",
            "latest_timestamp": str(latest_timestamp),
            "latest_close": latest_close,
            "latest_regime": latest_regime,
            "selected_pois": len(selected_pois),
            "all_pois": len(all_pois),
        }

    # Prefer higher score, closer entry, and more recent POI.
    candidates.sort(
        key=lambda x: (
            -x["poi_score"],
            x["entry_distance_pct"],
            -x["poi_active_from"],
        )
    )
    selected = candidates[0]
    selected["reason"] = "actionable_latest_poi"
    return selected


def compute_order_quantity(
    balance_usdt: float,
    entry_price: float,
    stop: float,
    risk_per_trade: float,
    risk_multiplier: float,
    filters: SymbolFilters,
) -> Tuple[float, float]:
    cash_risk = balance_usdt * risk_per_trade * risk_multiplier
    risk_per_unit = abs(entry_price - stop)
    if risk_per_unit <= 0:
        raise ValueError("Invalid risk_per_unit <= 0")
    raw_qty = cash_risk / risk_per_unit
    qty = floor_to_step(raw_qty, filters.step_size)
    if qty < filters.min_qty:
        raise ValueError(f"Quantity below minQty: qty={qty}, minQty={filters.min_qty}")
    notional = qty * entry_price
    if notional < filters.min_notional:
        raise ValueError(f"Notional below minNotional: notional={notional}, minNotional={filters.min_notional}")
    return qty, cash_risk


def close_position_if_sl_tp_hit(
    client: BinanceFuturesClient,
    cfg: DemoTraderConfig,
    state: Dict[str, Any],
    dry_run: bool,
) -> Optional[Dict[str, Any]]:
    position = get_open_position(client, cfg.symbol)
    if not position:
        return None

    position_amt = float(position.get("positionAmt", 0.0))
    if abs(position_amt) <= 0:
        return None

    mark_price = float(position.get("markPrice", 0.0))
    entry_price = float(position.get("entryPrice", 0.0))
    side = "LONG" if position_amt > 0 else "SHORT"

    active = state.get("active_trade", {})
    if not active:
        return {
            "event": "POSITION_OPEN_BUT_NO_LOCAL_STATE",
            "position_amt": position_amt,
            "side": side,
            "mark_price": mark_price,
        }

    stop = float(active.get("stop"))
    target = float(active.get("target"))

    exit_reason = None
    if side == "LONG":
        if mark_price <= stop:
            exit_reason = "SL"
        elif mark_price >= target:
            exit_reason = "TP"
    else:
        if mark_price >= stop:
            exit_reason = "SL"
        elif mark_price <= target:
            exit_reason = "TP"

    if not exit_reason:
        return None

    close_side = "SELL" if side == "LONG" else "BUY"
    qty = abs(position_amt)
    estimated_pnl = (mark_price - entry_price) * qty if side == "LONG" else (entry_price - mark_price) * qty

    order_response: Any = {"dry_run": True, "message": "No order sent"}
    if not dry_run:
        order_response = client.place_market_order(
            symbol=cfg.symbol,
            side=close_side,
            quantity=qty,
            reduce_only=True,
            new_client_order_id=f"SMC_EXIT_{int(time.time())}",
        )

    result = {
        "event": "MANAGED_EXIT_SENT",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": cfg.symbol,
        "side": side,
        "close_side": close_side,
        "quantity": qty,
        "mark_price": mark_price,
        "entry_price": entry_price,
        "stop": stop,
        "target": target,
        "exit_reason": exit_reason,
        "estimated_pnl_usdt": estimated_pnl,
        "dry_run": dry_run,
        "order_response": json.dumps(order_response, default=str),
    }
    append_csv_log(cfg.log_file, result)

    # Clear local active trade after exit order is sent.
    state["active_trade"] = {}
    save_state(cfg.state_file, state)
    return result


def execute_entry_if_allowed(
    client: BinanceFuturesClient,
    cfg: DemoTraderConfig,
    signal: Dict[str, Any],
    dry_run: bool,
) -> Dict[str, Any]:
    if signal.get("action") not in {"LONG", "SHORT"}:
        return {"event": "NO_ENTRY", "reason": signal.get("reason", "WAIT"), **signal}

    state = load_state(cfg.state_file)
    candle_key = str(signal.get("latest_timestamp"))
    if state.get("last_entry_candle") == candle_key:
        return {"event": "NO_ENTRY", "reason": "duplicate_signal_same_candle", **signal}

    entries_today, pnl_today = load_today_log_stats(cfg.log_file)
    if entries_today >= cfg.max_trades_per_day:
        return {"event": "NO_ENTRY", "reason": "max_trades_per_day_reached", "entries_today": entries_today, **signal}

    # daily_loss_limit_pct is based on current available balance approximation.
    balance = get_usdt_balance(client)
    daily_loss_limit_cash = -abs(balance * cfg.daily_loss_limit_pct / 100.0)
    if pnl_today <= daily_loss_limit_cash:
        return {
            "event": "NO_ENTRY",
            "reason": "daily_loss_limit_reached",
            "pnl_today": pnl_today,
            "daily_loss_limit_cash": daily_loss_limit_cash,
            **signal,
        }

    open_position = get_open_position(client, cfg.symbol)
    if open_position:
        return {"event": "NO_ENTRY", "reason": "position_already_open", "position": json.dumps(open_position), **signal}

    filters = get_symbol_filters(client, cfg.symbol)
    entry_price = float(signal["entry_price_estimate"])
    stop = round_price_to_tick(float(signal["stop"]), filters.tick_size)
    target = round_price_to_tick(float(signal["target"]), filters.tick_size)
    risk_multiplier = float(signal.get("risk_multiplier", 1.0))
    qty, cash_risk = compute_order_quantity(
        balance_usdt=balance,
        entry_price=entry_price,
        stop=stop,
        risk_per_trade=cfg.risk_per_trade,
        risk_multiplier=risk_multiplier,
        filters=filters,
    )

    if cfg.cancel_open_orders_before_entry and not dry_run:
        try:
            client.cancel_all_open_orders(cfg.symbol)
        except Exception as exc:
            print(f"WARNING: cancel_all_open_orders failed: {exc}")

    side = "BUY" if signal["action"] == "LONG" else "SELL"
    order_response: Any = {"dry_run": True, "message": "No order sent"}
    if not dry_run:
        client.change_leverage(cfg.symbol, cfg.leverage)
        order_response = client.place_market_order(
            symbol=cfg.symbol,
            side=side,
            quantity=qty,
            reduce_only=False,
            new_client_order_id=f"SMC_ENTRY_{int(time.time())}",
        )

    entry_log = {
        "event": "ENTRY_ORDER_SENT",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": cfg.symbol,
        "action": signal["action"],
        "side": side,
        "quantity": qty,
        "estimated_entry": entry_price,
        "stop": stop,
        "target": target,
        "rr_used": signal.get("rr_used"),
        "cash_risk": cash_risk,
        "risk_multiplier": risk_multiplier,
        "balance_usdt": balance,
        "latest_timestamp": signal.get("latest_timestamp"),
        "latest_regime": signal.get("latest_regime"),
        "setup_type": signal.get("setup_type"),
        "poi_score": signal.get("poi_score"),
        "poi_reasons": signal.get("poi_reasons"),
        "entry_distance_pct": signal.get("entry_distance_pct"),
        "sl_distance_pct": signal.get("sl_distance_pct"),
        "dry_run": dry_run,
        "order_response": json.dumps(order_response, default=str),
    }
    append_csv_log(cfg.log_file, entry_log)

    state["last_entry_candle"] = candle_key
    state["active_trade"] = {
        "symbol": cfg.symbol,
        "action": signal["action"],
        "side": side,
        "quantity": qty,
        "estimated_entry": entry_price,
        "stop": stop,
        "target": target,
        "cash_risk": cash_risk,
        "latest_timestamp": signal.get("latest_timestamp"),
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_state(cfg.state_file, state)

    return entry_log


def run_once(client: BinanceFuturesClient, strategy: Any, cfg: DemoTraderConfig) -> Dict[str, Any]:
    state = load_state(cfg.state_file)

    # First manage any existing position.
    exit_result = close_position_if_sl_tp_hit(client, cfg, state, dry_run=cfg.dry_run)
    if exit_result:
        print(json.dumps(exit_result, indent=2, default=str))
        return exit_result

    # Then evaluate new entry only if no position.
    df = fetch_ohlcv_from_binance(client, cfg.symbol, cfg.interval, cfg.lookback_days)
    smc_cfg = build_strategy_config(strategy, cfg)
    signal = run_strategy_signal(strategy, df, smc_cfg, cfg)

    print("\n===== LATEST SIGNAL =====")
    print(json.dumps(signal, indent=2, default=str))

    if signal.get("action") not in {"LONG", "SHORT"}:
        log_row = {
            "event": "SIGNAL_WAIT",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": cfg.symbol,
            "reason": signal.get("reason"),
            "latest_timestamp": signal.get("latest_timestamp"),
            "latest_close": signal.get("latest_close"),
            "latest_regime": signal.get("latest_regime"),
            "selected_pois": signal.get("selected_pois"),
            "all_pois": signal.get("all_pois"),
            "dry_run": cfg.dry_run,
        }
        append_csv_log(cfg.log_file, log_row)
        return log_row

    result = execute_entry_if_allowed(client, cfg, signal, dry_run=cfg.dry_run)
    print("\n===== EXECUTION RESULT =====")
    print(json.dumps(result, indent=2, default=str))
    return result


def parse_args() -> DemoTraderConfig:
    parser = argparse.ArgumentParser(description="Binance Futures Testnet Demo Trader for Adaptive SMC Bot")

    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--strategy-file", type=str, default="SMC_bot_adaptive_next_balanced.py")

    parser.add_argument("--testnet", action="store_true", default=True)
    parser.add_argument("--mainnet", action="store_true", help="DANGER: use real Binance Futures endpoint. Avoid until demo is stable.")
    parser.add_argument("--dry-run", action="store_true", help="Do not send orders; only print/log decisions.")
    parser.add_argument("--live-orders", action="store_true", help="Send orders to selected endpoint. Use only with testnet first.")
    parser.add_argument("--once", action="store_true", help="Run once and exit.")
    parser.add_argument("--loop", action="store_true", help="Run continuously.")
    parser.add_argument("--poll-seconds", type=int, default=60)

    parser.add_argument("--market-mode", type=str, default="long_short", choices=["long_short", "spot_only"])
    parser.add_argument("--entry-touch-mode", type=str, default="zone_touch", choices=["midpoint", "zone_touch"])
    parser.add_argument("--entry-mode", type=str, default="aggressive", choices=["aggressive", "balanced", "conservative"])
    parser.add_argument("--min-score", type=int, default=2)
    parser.add_argument("--risk", type=float, default=0.0025, help="Base risk per trade, e.g. 0.0025 = 0.25%")
    parser.add_argument("--leverage", type=int, default=1)

    parser.add_argument("--max-trades-per-day", type=int, default=2)
    parser.add_argument("--daily-loss-limit-pct", type=float, default=1.0)
    parser.add_argument("--max-entry-distance-pct", type=float, default=0.0020)
    parser.add_argument("--min-sl-distance-pct", type=float, default=0.0010)
    parser.add_argument("--max-sl-distance-pct", type=float, default=0.0300)
    parser.add_argument("--allow-unclear-regime", action="store_true")

    parser.add_argument("--state-file", type=str, default="demo_trader_state.json")
    parser.add_argument("--log-file", type=str, default="demo_trades_log.csv")

    args = parser.parse_args()

    dry_run = True
    if args.live_orders:
        dry_run = False
    if args.dry_run:
        dry_run = True

    testnet = not args.mainnet

    return DemoTraderConfig(
        symbol=args.symbol.upper(),
        interval=args.interval,
        lookback_days=args.lookback_days,
        strategy_file=args.strategy_file,
        testnet=testnet,
        dry_run=dry_run,
        loop=args.loop,
        poll_seconds=args.poll_seconds,
        market_mode=args.market_mode,
        entry_touch_mode=args.entry_touch_mode,
        entry_mode=args.entry_mode,
        min_score=args.min_score,
        risk_per_trade=args.risk,
        leverage=args.leverage,
        max_trades_per_day=args.max_trades_per_day,
        daily_loss_limit_pct=args.daily_loss_limit_pct,
        max_entry_distance_pct=args.max_entry_distance_pct,
        min_sl_distance_pct=args.min_sl_distance_pct,
        max_sl_distance_pct=args.max_sl_distance_pct,
        skip_unclear_regime=not args.allow_unclear_regime,
        state_file=args.state_file,
        log_file=args.log_file,
    )


def main() -> None:
    load_dotenv()
    cfg = parse_args()

    base_url = USDS_M_TESTNET_BASE_URL if cfg.testnet else USDS_M_MAINNET_BASE_URL
    api_key = os.getenv("BINANCE_TESTNET_API_KEY" if cfg.testnet else "BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_TESTNET_API_SECRET" if cfg.testnet else "BINANCE_API_SECRET", "")

    if not api_key or not api_secret:
        raise RuntimeError(
            "API key/secret not found. Create .env with BINANCE_TESTNET_API_KEY and "
            "BINANCE_TESTNET_API_SECRET for testnet."
        )

    if not cfg.testnet and not cfg.dry_run:
        raise RuntimeError("Mainnet live orders are blocked in this script. Use testnet first.")

    strategy = load_strategy_module(cfg.strategy_file)
    client = BinanceFuturesClient(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url,
        recv_window=cfg.recv_window,
        timeout=cfg.request_timeout,
    )

    print("\n===== DEMO TRADER CONFIG =====")
    print(json.dumps({**cfg.__dict__, "base_url": base_url, "api_key_loaded": bool(api_key)}, indent=2, default=str))

    if not cfg.loop:
        run_once(client, strategy, cfg)
        return

    while True:
        try:
            run_once(client, strategy, cfg)
        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as exc:
            print(f"ERROR: {exc}")
            append_csv_log(
                cfg.log_file,
                {
                    "event": "ERROR",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "symbol": cfg.symbol,
                    "error": str(exc),
                    "dry_run": cfg.dry_run,
                },
            )
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
