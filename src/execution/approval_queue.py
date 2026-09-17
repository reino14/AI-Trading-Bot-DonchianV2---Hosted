"""
execution/approval_queue.py

Persistent queue for orders awaiting human approval (Human-in-the-Loop).

Each pending order is stored as a JSON line with:
- Unique approval_id (for human to reference when approving/rejecting)
- Full order details (symbol, side, amount, price, reduce_only, etc.)
- Strategy context (strategy name, signal price, signal time, position change)
- Metadata (created_at, expires_at, status)

Status flow: PENDING -> APPROVED/REJECTED/EXPIRED
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


DEFAULT_QUEUE_PATH = Path("data/approval_queue.jsonl")


class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTED = "executed"  # approved + successfully sent to exchange


@dataclass
class PendingOrder:
    """
    One order waiting for human approval.
    """
    approval_id: str
    # Order details
    symbol: str
    side: str  # "buy" or "sell"
    amount: float
    price: float
    reduce_only: bool
    client_order_id: str  # The client_order_id that WILL be used if approved
    
    # Strategy context (for human decision making)
    strategy_name: str
    signal_price: float
    signal_time: float  # unix timestamp
    current_position: int  # Position.LONG/FLAT/SHORT
    target_position: int   # Position.LONG/FLAT/SHORT
    position_change_reason: str  # e.g., "close_long", "open_short", "flip_long_to_short"
    
    # Risk context
    estimated_cost_bps: float
    estimated_cost_pct: float
    account_balance_usdt: float | None = None
    position_size_pct: float | None = None
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0  # 0 = no expiry
    status: ApprovalStatus = ApprovalStatus.PENDING
    resolved_at: float = 0.0
    resolved_by: str = ""  # "human", "timeout", "system"
    execution_result: dict | None = None  # Filled after execution
    
    def __post_init__(self):
        if self.expires_at == 0.0:
            # Default 5 minutes expiry
            self.expires_at = self.created_at + 300
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at == 0:
            return False
        return time.time() > self.expires_at and self.status == ApprovalStatus.PENDING
    
    @property
    def time_remaining_sec(self) -> float:
        if self.expires_at == 0:
            return float('inf')
        return max(0, self.expires_at - time.time())
    
    @property
    def time_remaining_str(self) -> str:
        sec = self.time_remaining_sec
        if sec == float('inf'):
            return "no expiry"
        if sec < 60:
            return f"{sec:.0f}s"
        if sec < 3600:
            return f"{sec/60:.1f}m"
        return f"{sec/3600:.1f}h"
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> "PendingOrder":
        d = d.copy()
        d["status"] = ApprovalStatus(d["status"])
        return cls(**d)


class ApprovalQueue:
    """
    Append-only JSONL queue for pending approvals.
    Thread-safe enough for single-process use (one writer at a time).
    """
    
    def __init__(self, queue_path: Path = DEFAULT_QUEUE_PATH):
        self.queue_path = Path(queue_path)
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _read_all(self) -> list[PendingOrder]:
        if not self.queue_path.exists():
            return []
        orders = []
        with open(self.queue_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        orders.append(PendingOrder.from_dict(json.loads(line)))
                    except Exception:
                        continue
        return orders
    
    def _write_all(self, orders: list[PendingOrder]) -> None:
        with open(self.queue_path, "w") as f:
            for o in orders:
                f.write(json.dumps(o.to_dict()) + "\n")
    
    def add(self, order: PendingOrder) -> None:
        """Add a new pending order to the queue."""
        orders = self._read_all()
        orders.append(order)
        self._write_all(orders)
    
    def get_pending(self) -> list[PendingOrder]:
        """Get all orders still in PENDING status (auto-expire old ones)."""
        orders = self._read_all()
        now = time.time()
        changed = False
        for o in orders:
            if o.status == ApprovalStatus.PENDING and o.expires_at > 0 and now > o.expires_at:
                o.status = ApprovalStatus.EXPIRED
                o.resolved_at = now
                o.resolved_by = "timeout"
                changed = True
        if changed:
            self._write_all(orders)
        return [o for o in orders if o.status == ApprovalStatus.PENDING]
    
    def get_by_id(self, approval_id: str) -> PendingOrder | None:
        for o in self._read_all():
            if o.approval_id == approval_id:
                return o
        return None
    
    def update_status(
        self,
        approval_id: str,
        status: ApprovalStatus,
        resolved_by: str = "human",
        execution_result: dict | None = None
    ) -> bool:
        """Update status of an order. Returns True if found and updated."""
        orders = self._read_all()
        for o in orders:
            if o.approval_id == approval_id:
                if o.status != ApprovalStatus.PENDING:
                    return False  # Already resolved
                o.status = status
                o.resolved_at = time.time()
                o.resolved_by = resolved_by
                if execution_result:
                    o.execution_result = execution_result
                self._write_all(orders)
                return True
        return False
    
    def approve(self, approval_id: str) -> bool:
        return self.update_status(approval_id, ApprovalStatus.APPROVED, "human")
    
    def reject(self, approval_id: str) -> bool:
        return self.update_status(approval_id, ApprovalStatus.REJECTED, "human")
    
    def mark_executed(self, approval_id: str, execution_result: dict) -> bool:
        return self.update_status(
            approval_id, 
            ApprovalStatus.EXECUTED, 
            "system", 
            execution_result
        )
    
    def get_history(self, limit: int = 100) -> list[PendingOrder]:
        """Get recent orders (all statuses), newest first."""
        orders = self._read_all()
        orders.sort(key=lambda x: x.created_at, reverse=True)
        return orders[:limit]
    
    def cleanup_old(self, max_age_hours: float = 24) -> int:
        """Remove resolved orders older than max_age_hours. Returns count removed."""
        orders = self._read_all()
        cutoff = time.time() - max_age_hours * 3600
        original_len = len(orders)
        orders = [
            o for o in orders 
            if o.status == ApprovalStatus.PENDING or o.resolved_at > cutoff
        ]
        if len(orders) != original_len:
            self._write_all(orders)
        return original_len - len(orders)


def create_pending_order(
    symbol: str,
    side: str,
    amount: float,
    price: float,
    reduce_only: bool,
    strategy_name: str,
    signal_price: float,
    current_position: int,
    target_position: int,
    position_change_reason: str,
    estimated_cost_bps: float,
    account_balance_usdt: float | None = None,
    position_size_pct: float | None = None,
    expiry_seconds: float = 300,
) -> PendingOrder:
    """
    Factory function to create a PendingOrder with all context filled in.
    """
    from src.strategy.base import Position
    
    # Human-readable reason
    reason_map = {
        (Position.FLAT, Position.LONG): "open_long",
        (Position.FLAT, Position.SHORT): "open_short",
        (Position.LONG, Position.FLAT): "close_long",
        (Position.SHORT, Position.FLAT): "close_short",
        (Position.LONG, Position.SHORT): "flip_long_to_short",
        (Position.SHORT, Position.LONG): "flip_short_to_long",
    }
    reason = reason_map.get((current_position, target_position), position_change_reason)
    
    return PendingOrder(
        approval_id=f"appr-{uuid.uuid4().hex[:12]}",
        symbol=symbol,
        side=side,
        amount=amount,
        price=price,
        reduce_only=reduce_only,
        client_order_id=f"bot-{uuid.uuid4().hex[:16]}",
        strategy_name=strategy_name,
        signal_price=signal_price,
        signal_time=time.time(),
        current_position=current_position,
        target_position=target_position,
        position_change_reason=reason,
        estimated_cost_bps=estimated_cost_bps,
        estimated_cost_pct=estimated_cost_bps / 100.0,
        account_balance_usdt=account_balance_usdt,
        position_size_pct=position_size_pct,
        expires_at=time.time() + expiry_seconds,
    )


if __name__ == "__main__":
    # Smoke test
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp:
        q = ApprovalQueue(Path(tmp) / "test_queue.jsonl")
        
        # Add test order
        order = create_pending_order(
            symbol="BTC/USDT:USDT",
            side="buy",
            amount=0.01,
            price=100_000,
            reduce_only=False,
            strategy_name="DonchianBreakout",
            signal_price=100_000,
            current_position=0,
            target_position=1,
            position_change_reason="open_long",
            estimated_cost_bps=10.5,
            account_balance_usdt=10_000,
            position_size_pct=10.0,
            expiry_seconds=60,
        )
        q.add(order)
        print(f"Added: {order.approval_id}")
        
        # Get pending
        pending = q.get_pending()
        print(f"Pending: {len(pending)}")
        print(f"  {pending[0].approval_id} - {pending[0].side} {pending[0].amount} {pending[0].symbol} @ {pending[0].price}")
        print(f"  Expires in: {pending[0].time_remaining_str}")
        
        # Approve
        q.approve(order.approval_id)
        updated = q.get_by_id(order.approval_id)
        print(f"After approve: {updated.status.value}")
        
        # Test expire
        order2 = create_pending_order(
            symbol="ETH/USDT:USDT",
            side="sell",
            amount=0.1,
            price=3_000,
            reduce_only=True,
            strategy_name="Test",
            signal_price=3_000,
            current_position=1,
            target_position=0,
            position_change_reason="close_long",
            estimated_cost_bps=8.0,
            expiry_seconds=1,  # Expires in 1 second
        )
        q.add(order2)
        time.sleep(1.5)
        pending = q.get_pending()
        print(f"After expiry, pending: {len(pending)} (should be 0)")
    
    print("\n✅ ApprovalQueue smoke test passed")