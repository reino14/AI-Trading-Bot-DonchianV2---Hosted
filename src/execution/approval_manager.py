"""
src/execution/approval_manager.py

Manages the approval workflow: receives orders from strategy/runner,
queues them for approval, notifies human, handles approve/reject/timeout,
and executes approved orders via OrderManager.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.execution.approval_queue import (
    ApprovalQueue,
    ApprovalStatus,
    PendingOrder,
    create_pending_order,
)
from src.execution.broker import Broker, OrderResult, OrderStatus
from src.execution.order_manager import OrderManager
from src.ops.notifier import TelegramNotifier, create_notifier_from_env
from src.strategy.base import Position, Strategy


class ApprovalAction(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    EXPIRE = "expire"


@dataclass
class ApprovalResult:
    """Result of an approval decision."""
    approval_id: str
    action: ApprovalAction
    success: bool
    message: str
    execution_result: OrderResult | None = None


class ApprovalManager:
    """
    Orchestrates the human-in-the-loop approval workflow.
    
    Flow:
    1. Runner/Strategy calls submit_for_approval() with order details
    2. Creates PendingOrder with full context, adds to queue
    3. Sends Telegram notification with Approve/Reject buttons
    4. Waits for human decision (or timeout)
    5. If approved: executes via OrderManager, marks EXECUTED
    6. If rejected/expired: marks REJECTED/EXPIRED, no execution
    """
    
    def __init__(
        self,
        order_manager: OrderManager,
        broker: Broker,
        approval_queue: ApprovalQueue | None = None,
        notifier: TelegramNotifier | None = None,
        default_expiry_seconds: float = 300,  # 5 minutes
        auto_expire_check_interval: float = 30,  # check every 30s
    ):
        self.order_manager = order_manager
        self.broker = broker
        self.approval_queue = approval_queue or ApprovalQueue()
        self.notifier = notifier or create_notifier_from_env()
        self.default_expiry_seconds = default_expiry_seconds
        self.auto_expire_check_interval = auto_expire_check_interval
        self._last_expire_check = 0.0
    
    def submit_for_approval(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        reduce_only: bool,
        strategy: Strategy,
        signal_price: float,
        current_position: int,
        target_position: int,
        account_balance_usdt: float | None = None,
        position_size_pct: float | None = None,
        expiry_seconds: float | None = None,
    ) -> PendingOrder:
        """
        Submit an order for human approval.
        Returns the PendingOrder (with approval_id) immediately.
        Does NOT execute the order.
        """
        from src.execution.broker import round_trip_cost
        from src.core.cost_model import CRYPTO_PERP
        
        # Estimate cost
        cost = round_trip_cost(CRYPTO_PERP, entry_is_maker=False, exit_is_maker=False)
        estimated_cost_bps = cost.total_bps
        
        # Determine position change reason
        reason_map = {
            (Position.FLAT, Position.LONG): "open_long",
            (Position.FLAT, Position.SHORT): "open_short",
            (Position.LONG, Position.FLAT): "close_long",
            (Position.SHORT, Position.FLAT): "close_short",
            (Position.LONG, Position.SHORT): "flip_long_to_short",
            (Position.SHORT, Position.LONG): "flip_short_to_long",
        }
        reason = reason_map.get((current_position, target_position), "unknown")
        
        pending = create_pending_order(
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            reduce_only=reduce_only,
            strategy_name=strategy.name,
            signal_price=signal_price,
            current_position=current_position,
            target_position=target_position,
            position_change_reason=reason,
            estimated_cost_bps=estimated_cost_bps,
            account_balance_usdt=account_balance_usdt,
            position_size_pct=position_size_pct,
            expiry_seconds=expiry_seconds or self.default_expiry_seconds,
        )
        
        # Add to queue
        self.approval_queue.add(pending)
        
        # Send notification
        self._notify_pending(pending)
        
        print(f"[ApprovalManager] Submitted for approval: {pending.approval_id}")
        print(f"  {side} {amount} {symbol} @ {price} (reduce_only={reduce_only})")
        print(f"  Expires in: {pending.time_remaining_str}")
        
        return pending
    
    def _notify_pending(self, order: PendingOrder) -> None:
        if self.notifier:
            self.notifier.send_approval_request({
                "approval_id": order.approval_id,
                "symbol": order.symbol,
                "side": order.side,
                "amount": order.amount,
                "price": order.price,
                "reduce_only": order.reduce_only,
                "strategy_name": order.strategy_name,
                "signal_price": order.signal_price,
                "current_position": order.current_position,
                "target_position": order.target_position,
                "position_change_reason": order.position_change_reason,
                "estimated_cost_bps": order.estimated_cost_bps,
                "estimated_cost_pct": order.estimated_cost_pct,
                "account_balance_usdt": order.account_balance_usdt,
                "position_size_pct": order.position_size_pct,
                "time_remaining_str": order.time_remaining_str,
            })
        else:
            print(f"[ApprovalManager] No notifier configured - approval request NOT sent to Telegram")
            print(f"  Check pending via CLI: python -m scripts.approve_order list")
    
    def check_and_expire(self) -> list[PendingOrder]:
        """Check for expired pending orders and mark them. Returns newly expired orders."""
        now = time.time()
        if now - self._last_expire_check < self.auto_expire_check_interval:
            return []
        self._last_expire_check = now
        
        pending = self.approval_queue.get_pending()  # This auto-expires
        expired = [o for o in pending if o.is_expired]
        
        for order in expired:
            self.approval_queue.update_status(
                order.approval_id, 
                ApprovalStatus.EXPIRED, 
                "timeout"
            )
            print(f"[ApprovalManager] Order expired: {order.approval_id}")
            if self.notifier:
                self.notifier.send_alert(
                    "Order Expired",
                    f"Approval request {order.approval_id} expired without response.\n"
                    f"Order: {order.side} {order.amount} {order.symbol} @ {order.price}",
                    "warning"
                )
        
        return expired
    
    def process_approval(self, approval_id: str, action: ApprovalAction) -> ApprovalResult:
        """
        Process a human approval decision.
        Call this when human clicks Approve/Reject button or uses CLI.
        """
        order = self.approval_queue.get_by_id(approval_id)
        if not order:
            return ApprovalResult(
                approval_id=approval_id,
                action=action,
                success=False,
                message=f"Approval ID not found: {approval_id}"
            )
        
        if order.status != ApprovalStatus.PENDING:
            return ApprovalResult(
                approval_id=approval_id,
                action=action,
                success=False,
                message=f"Order already {order.status.value} (resolved by {order.resolved_by})"
            )
        
        if action == ApprovalAction.APPROVE:
            return self._execute_approved(order)
        elif action == ApprovalAction.REJECT:
            return self._reject_order(order)
        else:
            return ApprovalResult(
                approval_id=approval_id,
                action=action,
                success=False,
                message=f"Unknown action: {action}"
            )
    
    def _execute_approved(self, order: PendingOrder) -> ApprovalResult:
        """Execute an approved order via OrderManager."""
        print(f"[ApprovalManager] Executing approved order: {order.approval_id}")
        
        try:
            # Submit via OrderManager (which handles write-ahead log, etc.)
            result = self.order_manager.submit_limit_order(
                symbol=order.symbol,
                side=order.side,
                amount=order.amount,
                price=order.price,
                reduce_only=order.reduce_only,
            )
            
            # Mark as executed in queue
            exec_result = {
                "client_order_id": result.client_order_id,
                "exchange_order_id": result.exchange_order_id,
                "status": result.status.value,
                "filled_amount": result.filled_amount,
                "average_price": result.average_price,
                "fee": result.fee,
            }
            self.approval_queue.mark_executed(order.approval_id, exec_result)
            
            # Notify
            if self.notifier:
                self.notifier.send_approval_result(
                    order.approval_id, 
                    True, 
                    exec_result
                )
            
            print(f"[ApprovalManager] Order executed: {result.client_order_id}, status={result.status.value}")
            
            return ApprovalResult(
                approval_id=order.approval_id,
                action=ApprovalAction.APPROVE,
                success=True,
                message=f"Order executed: {result.status.value}",
                execution_result=result,
            )
            
        except Exception as e:
            # Mark as rejected due to execution failure
            self.approval_queue.update_status(order.approval_id, ApprovalStatus.REJECTED, "system")
            
            if self.notifier:
                self.notifier.send_approval_result(
                    order.approval_id, 
                    False
                )
            
            print(f"[ApprovalManager] Execution failed: {e}")
            
            return ApprovalResult(
                approval_id=order.approval_id,
                action=ApprovalAction.APPROVE,
                success=False,
                message=f"Execution failed: {e}",
            )
    
    def _reject_order(self, order: PendingOrder) -> ApprovalResult:
        """Mark order as rejected (no execution)."""
        self.approval_queue.update_status(order.approval_id, ApprovalStatus.REJECTED, "human")
        
        if self.notifier:
            self.notifier.send_approval_result(order.approval_id, False)
        
        print(f"[ApprovalManager] Order rejected: {order.approval_id}")
        
        return ApprovalResult(
            approval_id=order.approval_id,
            action=ApprovalAction.REJECT,
            success=True,
            message="Order rejected by human",
        )
    
    def get_pending_orders(self) -> list[PendingOrder]:
        """Get all currently pending orders (auto-expires old ones)."""
        self.check_and_expire()
        return self.approval_queue.get_pending()
    
    def get_history(self, limit: int = 50) -> list[PendingOrder]:
        """Get recent approval history."""
        return self.approval_queue.get_history(limit)
    
    def handle_telegram_callback(self, callback_data: str) -> ApprovalResult | None:
        """
        Handle Telegram callback query data.
        Expected format: "approve:appr-xxx" or "reject:appr-xxx" or "list_pending"
        """
        if callback_data == "list_pending":
            pending = self.get_pending_orders()
            if not pending:
                return ApprovalResult("", ApprovalAction.APPROVE, True, "No pending orders")
            # Send list as message
            if self.notifier:
                text = "📋 <b>Pending Orders:</b>\n\n"
                for o in pending:
                    pos_map = {1: "LONG", 0: "FLAT", -1: "SHORT"}
                    text += (
                        f"• <code>{o.approval_id}</code>: "
                        f"{o.side.upper()} {o.amount} {o.symbol} @ {o.price:,.2f} "
                        f"({pos_map.get(o.current_position,'?')}→{pos_map.get(o.target_position,'?')}) "
                        f"⏱ {o.time_remaining_str}\n"
                    )
                self.notifier.send_message(text)
            return None
        
        try:
            action_str, approval_id = callback_data.split(":", 1)
            action = ApprovalAction(action_str)
            return self.process_approval(approval_id, action)
        except Exception as e:
            print(f"[ApprovalManager] Invalid callback data: {callback_data} - {e}")
            return None


def create_approval_manager(
    order_manager: OrderManager,
    broker: Broker,
    default_expiry_seconds: float = 300,
) -> ApprovalManager:
    """Factory to create ApprovalManager with default queue and notifier from env."""
    return ApprovalManager(
        order_manager=order_manager,
        broker=broker,
        approval_queue=ApprovalQueue(),
        notifier=create_notifier_from_env(),
        default_expiry_seconds=default_expiry_seconds,
    )


if __name__ == "__main__":
    # Smoke test with MockBroker
    import tempfile
    from src.execution.broker import MockBroker
    from src.execution.order_manager import OrderManager
    from src.strategy.donchian_breakout import DonchianBreakoutStrategy
    
    with tempfile.TemporaryDirectory() as tmp:
        # Setup
        broker = MockBroker(fill_immediately=True)
        om = OrderManager(broker, log_path=Path(tmp) / "order_intents.jsonl")
        am = ApprovalManager(
            order_manager=om,
            broker=broker,
            approval_queue=ApprovalQueue(Path(tmp) / "approval_queue.jsonl"),
            notifier=None,  # No notifier for test
            default_expiry_seconds=60,
        )
        
        strat = DonchianBreakoutStrategy()
        
        # Submit for approval
        pending = am.submit_for_approval(
            symbol="BTC/USDT:USDT",
            side="buy",
            amount=0.01,
            price=100_000,
            reduce_only=False,
            strategy=strat,
            signal_price=99_950,
            current_position=Position.FLAT,
            target_position=Position.LONG,
            account_balance_usdt=10_000,
            position_size_pct=10.0,
        )
        
        print(f"\nPending orders: {len(am.get_pending_orders())}")
        
        # Approve
        result = am.process_approval(pending.approval_id, ApprovalAction.APPROVE)
        print(f"Approve result: {result.success} - {result.message}")
        if result.execution_result:
            print(f"  Executed: {result.execution_result.client_order_id}, status={result.execution_result.status.value}")
        
        # Check history
        history = am.get_history()
        print(f"\nHistory ({len(history)}):")
        for h in history:
            print(f"  {h.approval_id}: {h.status.value} ({h.resolved_by})")
    
    print("\n✅ ApprovalManager smoke test passed")