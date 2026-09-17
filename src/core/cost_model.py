"""
Model biaya transaksi.
(Salinan dari src/core/cost_model.py milik Nero, untuk uji adapter.)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CostConfig:
    maker_bps: float = 2.0
    taker_bps: float = 5.0
    spread_bps: float = 1.0
    slippage_bps: float = 2.0
    slippage_is_measured: bool = False
    funding_bps_per_8h: float = 1.0
    sell_regulatory_bps: float = 0.0


@dataclass(frozen=True)
class CostBreakdown:
    entry_fee_bps: float
    exit_fee_bps: float
    spread_bps: float
    slippage_bps: float
    funding_bps: float
    regulatory_bps: float = 0.0

    @property
    def total_bps(self) -> float:
        return (
            self.entry_fee_bps
            + self.exit_fee_bps
            + self.spread_bps
            + self.slippage_bps
            + self.funding_bps
            + self.regulatory_bps
        )

    @property
    def total_pct(self) -> float:
        return self.total_bps / 100.0

    def __str__(self) -> str:
        return (
            f"fee masuk {self.entry_fee_bps:.2f} + fee keluar {self.exit_fee_bps:.2f} "
            f"+ spread {self.spread_bps:.2f} + slippage {self.slippage_bps:.2f} "
            f"+ funding {self.funding_bps:.2f} + regulator {self.regulatory_bps:.2f} "
            f"= {self.total_bps:.2f} bps "
            f"({self.total_pct:.3f}%)"
        )


def round_trip_cost(
    cfg: CostConfig,
    entry_is_maker: bool = False,
    exit_is_maker: bool = False,
    hold_minutes: float = 5.0,
) -> CostBreakdown:
    entry_fee = cfg.maker_bps if entry_is_maker else cfg.taker_bps
    exit_fee = cfg.maker_bps if exit_is_maker else cfg.taker_bps

    spread = 0.0
    if not entry_is_maker:
        spread += cfg.spread_bps / 2
    if not exit_is_maker:
        spread += cfg.spread_bps / 2

    slippage = 0.0
    if not entry_is_maker:
        slippage += cfg.slippage_bps
    if not exit_is_maker:
        slippage += cfg.slippage_bps

    funding = cfg.funding_bps_per_8h * (hold_minutes / (8 * 60))

    return CostBreakdown(
        entry_fee_bps=entry_fee,
        exit_fee_bps=exit_fee,
        spread_bps=spread,
        slippage_bps=slippage,
        funding_bps=funding,
        regulatory_bps=cfg.sell_regulatory_bps,
    )


def breakeven_move_bps(cfg: CostConfig, **kwargs) -> float:
    return round_trip_cost(cfg, **kwargs).total_bps


CRYPTO_PERP = CostConfig(
    maker_bps=2.0,
    taker_bps=5.0,
    spread_bps=1.0,
    slippage_bps=2.0,
    funding_bps_per_8h=1.0,
    sell_regulatory_bps=0.0,
)

CRYPTO_SPOT = CostConfig(
    maker_bps=10.0,
    taker_bps=10.0,
    spread_bps=1.0,
    slippage_bps=2.0,
    slippage_is_measured=False,
    funding_bps_per_8h=0.0,
    sell_regulatory_bps=0.0,
)

US_EQUITY = CostConfig(
    maker_bps=0.0,
    taker_bps=0.0,
    spread_bps=0.55,
    slippage_bps=1.0,
    funding_bps_per_8h=0.0,
    sell_regulatory_bps=0.3,
)

PRESETS = {"crypto_perp": CRYPTO_PERP, "crypto_spot": CRYPTO_SPOT, "us_equity": US_EQUITY}