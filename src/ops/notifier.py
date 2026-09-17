"""
src/ops/notifier.py

Telegram notifier for sending alerts and order approval requests.
Supports both simple messages and interactive approval buttons.
"""

import os
import requests
from dataclasses import dataclass
from typing import Any


@dataclass
class TelegramConfig:
    bot_token: str
    chat_id: str | int
    parse_mode: str = "HTML"
    timeout: int = 10


class TelegramNotifier:
    """
    Thin wrapper around Telegram Bot API.
    All methods are synchronous (requests) for simplicity.
    For high-volume async, use aiohttp instead.
    """
    
    def __init__(self, config: TelegramConfig):
        self.config = config
        self.base_url = f"https://api.telegram.org/bot{config.bot_token}"
    
    def _post(self, method: str, data: dict) -> dict:
        url = f"{self.base_url}/{method}"
        try:
            resp = requests.post(url, json=data, timeout=self.config.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            # Don't raise - notification failure shouldn't stop trading
            print(f"[Telegram] Error sending {method}: {e}")
            return {"ok": False, "error": str(e)}
    
    def send_message(
        self,
        text: str,
        parse_mode: str | None = None,
        disable_web_page_preview: bool = True,
        reply_markup: dict | None = None,
    ) -> dict:
        """Send a simple text message."""
        data = {
            "chat_id": self.config.chat_id,
            "text": text,
            "parse_mode": parse_mode or self.config.parse_mode,
            "disable_web_page_preview": disable_web_page_preview,
        }
        if reply_markup:
            data["reply_markup"] = reply_markup
        return self._post("sendMessage", data)
    
    def send_approval_request(self, order_info: dict) -> dict:
        """
        Send an order approval request with inline buttons.
        
        order_info should contain:
        - approval_id
        - symbol
        - side
        - amount
        - price
        - reduce_only
        - strategy_name
        - signal_price
        - current_position
        - target_position
        - position_change_reason
        - estimated_cost_bps
        - estimated_cost_pct
        - account_balance_usdt (optional)
        - position_size_pct (optional)
        - time_remaining_str
        """
        # Position emoji map
        pos_emoji = {1: "📈 LONG", 0: "➖ FLAT", -1: "📉 SHORT"}
        side_emoji = {"buy": "🟢 BUY", "sell": "🔴 SELL"}
        
        curr_pos = pos_emoji.get(order_info["current_position"], str(order_info["current_position"]))
        tgt_pos = pos_emoji.get(order_info["target_position"], str(order_info["target_position"]))
        
        text = (
            f"⚠️ <b>ORDER APPROVAL REQUEST</b>\n\n"
            f"<b>Approval ID:</b> <code>{order_info['approval_id']}</code>\n"
            f"<b>Strategy:</b> {order_info['strategy_name']}\n"
            f"<b>Action:</b> {side_emoji.get(order_info['side'], order_info['side'])} "
            f"{order_info['amount']} {order_info['symbol']} @ {order_info['price']:,.2f}\n"
            f"<b>Reduce Only:</b> {'Yes' if order_info['reduce_only'] else 'No'}\n\n"
            f"<b>Position Change:</b> {curr_pos} → {tgt_pos}\n"
            f"<b>Reason:</b> {order_info['position_change_reason']}\n"
            f"<b>Signal Price:</b> {order_info['signal_price']:,.2f}\n\n"
            f"<b>Est. Cost:</b> {order_info['estimated_cost_bps']:.1f} bps "
            f"({order_info['estimated_cost_pct']:.3f}%)\n"
        )
        
        if order_info.get("account_balance_usdt"):
            text += f"<b>Account Balance:</b> ${order_info['account_balance_usdt']:,.2f}\n"
        if order_info.get("position_size_pct"):
            text += f"<b>Position Size:</b> {order_info['position_size_pct']:.1f}% of equity\n"
        
        text += f"\n⏱ <b>Expires in:</b> {order_info['time_remaining_str']}\n"
        text += f"\n<i>Reply with approval ID to approve/reject, or use buttons below.</i>"
        
        # Inline keyboard with Approve/Reject buttons
        reply_markup = {
            "inline_keyboard": [
                [
                    {"text": "✅ APPROVE", "callback_data": f"approve:{order_info['approval_id']}"},
                    {"text": "❌ REJECT", "callback_data": f"reject:{order_info['approval_id']}"},
                ],
                [
                    {"text": "📋 View All Pending", "callback_data": "list_pending"},
                ]
            ]
        }
        
        return self.send_message(text, reply_markup=reply_markup)
    
    def send_approval_result(
        self,
        approval_id: str,
        approved: bool,
        execution_result: dict | None = None,
    ) -> dict:
        """Send confirmation after approval/rejection."""
        if approved:
            text = f"✅ <b>APPROVED</b> — {approval_id}"
            if execution_result:
                status = execution_result.get("status", "unknown")
                filled = execution_result.get("filled_amount", 0)
                avg_price = execution_result.get("average_price")
                text += f"\n<b>Status:</b> {status}"
                if filled:
                    text += f"\n<b>Filled:</b> {filled}"
                if avg_price:
                    text += f"\n<b>Avg Price:</b> {avg_price:,.2f}"
        else:
            text = f"❌ <b>REJECTED</b> — {approval_id}"
        
        return self.send_message(text)
    
    def send_alert(self, title: str, message: str, level: str = "info") -> dict:
        """Send a general alert message."""
        level_emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "🚨",
            "success": "✅",
        }
        emoji = level_emoji.get(level, "ℹ️")
        text = f"{emoji} <b>{title}</b>\n\n{message}"
        return self.send_message(text)
    
    def send_position_update(
        self,
        symbol: str,
        position: int,
        entry_price: float | None,
        current_price: float,
        unrealized_pnl: float,
        unrealized_pnl_pct: float,
    ) -> dict:
        """Send position update."""
        pos_emoji = {1: "📈 LONG", 0: "➖ FLAT", -1: "📉 SHORT"}
        pnl_emoji = "🟢" if unrealized_pnl >= 0 else "🔴"
        
        text = (
            f"{pos_emoji.get(position, '❓')} <b>Position Update</b>\n\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Position:</b> {pos_emoji.get(position, position)}\n"
        )
        
        if entry_price:
            text += f"<b>Entry:</b> {entry_price:,.2f}\n"
        text += f"<b>Current:</b> {current_price:,.2f}\n"
        text += f"{pnl_emoji} <b>Unrealized P&L:</b> {unrealized_pnl:+,.2f} ({unrealized_pnl_pct:+.2f}%)\n"
        
        return self.send_message(text)
    
    def send_session_summary(
        self,
        session_hours: float,
        total_trades: int,
        win_rate: float,
        total_pnl: float,
        total_pnl_pct: float,
        max_drawdown: float,
    ) -> dict:
        """Send end-of-session summary."""
        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
        
        text = (
            f"📊 <b>Session Summary ({session_hours:.1f}h)</b>\n\n"
            f"<b>Total Trades:</b> {total_trades}\n"
            f"<b>Win Rate:</b> {win_rate:.1f}%\n"
            f"{pnl_emoji} <b>Total P&L:</b> {total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)\n"
            f"🔴 <b>Max Drawdown:</b> {max_drawdown:.2f}%\n"
        )
        return self.send_message(text)


def create_notifier_from_env() -> TelegramNotifier | None:
    """
    Create TelegramNotifier from environment variables.
    Returns None if not configured.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        return None
    
    return TelegramNotifier(TelegramConfig(
        bot_token=bot_token,
        chat_id=chat_id,
    ))


if __name__ == "__main__":
    # Test with env vars
    notifier = create_notifier_from_env()
    if notifier:
        print("Testing Telegram notifier...")
        result = notifier.send_alert("Test Alert", "This is a test message from the trading bot.", "info")
        print(f"Result: {result}")
        
        # Test approval request
        result = notifier.send_approval_request({
            "approval_id": "appr-test123",
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "amount": 0.01,
            "price": 100_000,
            "reduce_only": False,
            "strategy_name": "DonchianBreakout",
            "signal_price": 99_950,
            "current_position": 0,
            "target_position": 1,
            "position_change_reason": "open_long",
            "estimated_cost_bps": 10.5,
            "estimated_cost_pct": 0.105,
            "account_balance_usdt": 10_000,
            "position_size_pct": 10.0,
            "time_remaining_str": "4m 30s",
        })
        print(f"Approval request result: {result}")
    else:
        print("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID not set in environment")
        print("Set them in config/secrets.env to test")