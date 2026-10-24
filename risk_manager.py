from dataclasses import dataclass
from typing import Dict, List, Optional
import math


@dataclass
class RiskConfig:
    # Maximum number of open positions across the entire account
    max_positions: int = 3

    # Risk no more than 1% of account equity on a single trade
    risk_per_trade: float = 0.01

    # Maximum portion of account equity invested in one position
    max_position_fraction: float = 0.50

    # Keep some cash available rather than investing every dollar
    cash_reserve_fraction: float = 0.05

    # Ignore orders too small to be worthwhile
    min_order_value: float = 1.00

    # Allow fractional shares
    fractional_shares: bool = True


@dataclass
class TradePlan:
    approved: bool
    symbol: str
    quantity: float
    estimated_cost: float
    stop_price: float
    risk_amount: float
    reason: str


def _positive_number(value) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def calculate_position_size(
    account_equity: float,
    available_cash: float,
    entry_price: float,
    stop_price: float,
    open_positions: int,
    config: Optional[RiskConfig] = None,
) -> TradePlan:
    """
    Calculate a position size without submitting an order.

    This function is for risk planning only. It does not place trades.
    """

    if config is None:
        config = RiskConfig()

    def reject(reason: str) -> TradePlan:
        return TradePlan(
            approved=False,
            symbol="",
            quantity=0.0,
            estimated_cost=0.0,
            stop_price=0.0,
            risk_amount=0.0,
            reason=reason,
        )

    if not _positive_number(account_equity):
        return reject("Account equity must be greater than zero.")

    if not _positive_number(available_cash):
        return reject("Available cash must be greater than zero.")

    if not _positive_number(entry_price):
        return reject("Entry price must be greater than zero.")

    if not _positive_number(stop_price):
        return reject("Stop price must be greater than zero.")

    if stop_price >= entry_price:
        return reject(
            "For a long position, the stop price must be below entry price."
        )

    if open_positions < 0:
        return reject("Open position count cannot be negative.")

    if open_positions >= config.max_positions:
        return reject(
            f"Position limit reached: maximum {config.max_positions} open positions."
        )

    equity = float(account_equity)
    cash = float(available_cash)
    entry = float(entry_price)
    stop = float(stop_price)

    risk_per_share = entry - stop

    if risk_per_share <= 0:
        return reject("Invalid risk per share.")

    # Dollar risk allowed for this trade
    risk_budget = equity * config.risk_per_trade

    # Position size based on the maximum acceptable loss at the stop
    quantity_by_risk = risk_budget / risk_per_share

    # Do not invest more than the allowed fraction of account equity
    position_budget = equity * config.max_position_fraction

    # Preserve a portion of available cash
    spendable_cash = cash * (1.0 - config.cash_reserve_fraction)

    # Respect both the position limit and available funds
    maximum_cost = min(position_budget, spendable_cash)

    quantity_by_capital = maximum_cost / entry

    # The final quantity must satisfy every constraint
    quantity = min(quantity_by_risk, quantity_by_capital)

    if not config.fractional_shares:
        quantity = math.floor(quantity)

    if not math.isfinite(quantity) or quantity <= 0:
        return reject("Calculated share quantity is zero or invalid.")

    estimated_cost = quantity * entry
    estimated_risk = quantity * risk_per_share

    if estimated_cost < config.min_order_value:
        return reject(
            f"Order value ${estimated_cost:.2f} is below the "
            f"${config.min_order_value:.2f} minimum."
        )

    if estimated_cost > cash:
        return reject("Order would exceed available cash.")

    return TradePlan(
        approved=True,
        symbol="",
        quantity=round(quantity, 8),
        estimated_cost=round(estimated_cost, 2),
        stop_price=round(stop, 4),
        risk_amount=round(estimated_risk, 2),
        reason="Trade passes the basic position-sizing checks.",
    )


def check_position_limit(
    open_positions: int,
    config: Optional[RiskConfig] = None,
) -> bool:
    """
    Return True only when another position may be considered.
    """

    if config is None:
        config = RiskConfig()

    return 0 <= open_positions < config.max_positions


def calculate_affordable_quantity(
    available_cash: float,
    stock_price: float,
    fractional_shares: bool = True,
) -> float:
    """
    Estimate the maximum quantity affordable with available cash.

    This is a simple affordability calculation, not a trade recommendation.
    """

    if not _positive_number(available_cash):
        return 0.0

    if not _positive_number(stock_price):
        return 0.0

    quantity = float(available_cash) / float(stock_price)

    if not fractional_shares:
        quantity = math.floor(quantity)

    return round(max(quantity, 0.0), 8)


def validate_portfolio(
    account_equity: float,
    available_cash: float,
    open_positions: int,
    config: Optional[RiskConfig] = None,
) -> List[str]:
    """
    Return a list of portfolio warnings.

    An empty list means these basic checks found no issues.
    """

    if config is None:
        config = RiskConfig()

    warnings = []

    if not _positive_number(account_equity):
        warnings.append("Account equity is zero, negative, or invalid.")

    if not _positive_number(available_cash):
        warnings.append("Available cash is zero, negative, or invalid.")

    if open_positions < 0:
        warnings.append("Open position count cannot be negative.")

    if open_positions > config.max_positions:
        warnings.append(
            f"Portfolio has {open_positions} positions, exceeding "
            f"the limit of {config.max_positions}."
        )

    if (
        _positive_number(account_equity)
        and _positive_number(available_cash)
        and available_cash > account_equity
    ):
        warnings.append(
            "Available cash exceeds reported account equity; verify account data."
        )

    return warnings
