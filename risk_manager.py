from dataclasses import dataclass
from typing import List, Optional
import math


@dataclass
class RiskConfig:
    # Maximum number of open positions across the account.
    max_positions: int = 3

    # Maximum planned loss per trade as a fraction of account equity.
    risk_per_trade: float = 0.01

    # Maximum position value as a fraction of account equity.
    max_position_fraction: float = 0.50

    # Preserve this fraction of available cash.
    cash_reserve_fraction: float = 0.05

    # Reject orders below this estimated dollar value.
    min_order_value: float = 1.00

    # Permit fractional shares when supported by the broker.
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


def _finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _positive_number(value) -> bool:
    return _finite_number(value) and float(value) > 0


def _valid_config(config: RiskConfig) -> bool:
    if not isinstance(config.max_positions, int):
        return False

    if isinstance(config.max_positions, bool) or config.max_positions < 1:
        return False

    if not _positive_number(config.risk_per_trade):
        return False

    if config.risk_per_trade > 1:
        return False

    if not _positive_number(config.max_position_fraction):
        return False

    if config.max_position_fraction > 1:
        return False

    if not _finite_number(config.cash_reserve_fraction):
        return False

    if not 0 <= config.cash_reserve_fraction < 1:
        return False

    if not _finite_number(config.min_order_value):
        return False

    if config.min_order_value < 0:
        return False

    if not isinstance(config.fractional_shares, bool):
        return False

    return True


def _rejected_plan(reason: str) -> TradePlan:
    return TradePlan(
        approved=False,
        symbol="",
        quantity=0.0,
        estimated_cost=0.0,
        stop_price=0.0,
        risk_amount=0.0,
        reason=reason,
    )


def calculate_position_size(
    account_equity: float,
    available_cash: float,
    entry_price: float,
    stop_price: float,
    open_positions: int,
    config: Optional[RiskConfig] = None,
    symbol: str = "",
) -> TradePlan:
    """
    Calculate a proposed long-position size without submitting an order.

    The caller remains responsible for checking live account information,
    broker restrictions, existing orders, and execution costs.

    This function does not place trades.
    """

    if config is None:
        config = RiskConfig()

    if not _valid_config(config):
        return _rejected_plan("Risk configuration is invalid.")

    if not _positive_number(account_equity):
        return _rejected_plan("Account equity must be greater than zero.")

    if not _positive_number(available_cash):
        return _rejected_plan("Available cash must be greater than zero.")

    if not _positive_number(entry_price):
        return _rejected_plan("Entry price must be greater than zero.")

    if not _positive_number(stop_price):
        return _rejected_plan("Stop price must be greater than zero.")

    if (
        not isinstance(open_positions, int)
        or isinstance(open_positions, bool)
        or open_positions < 0
    ):
        return _rejected_plan(
            "Open position count must be a nonnegative integer."
        )

    if open_positions >= config.max_positions:
        return _rejected_plan(
            f"Position limit reached: maximum "
            f"{config.max_positions} open positions."
        )

    equity = float(account_equity)
    cash = float(available_cash)
    entry = float(entry_price)
    stop = float(stop_price)

    if cash > equity:
        return _rejected_plan(
            "Available cash exceeds account equity; verify account data."
        )

    if stop >= entry:
        return _rejected_plan(
            "For a long position, the stop price must be below entry price."
        )

    risk_per_share = entry - stop

    if not _positive_number(risk_per_share):
        return _rejected_plan("Risk per share is invalid.")

    # Maximum planned dollar loss at the stop price.
    risk_budget = equity * config.risk_per_trade

    # Maximum quantity allowed by the planned loss.
    quantity_by_risk = risk_budget / risk_per_share

    # Maximum dollar investment in one position.
    position_budget = equity * config.max_position_fraction

    # Preserve the configured cash reserve.
    spendable_cash = cash * (1.0 - config.cash_reserve_fraction)

    maximum_cost = min(position_budget, spendable_cash)

    if not _positive_number(maximum_cost):
        return _rejected_plan("No spendable cash is available.")

    quantity_by_capital = maximum_cost / entry

    quantity = min(quantity_by_risk, quantity_by_capital)

    if not config.fractional_shares:
        quantity = math.floor(quantity)

    if not _positive_number(quantity):
        return _rejected_plan(
            "Calculated share quantity is zero or invalid."
        )

    estimated_cost = quantity * entry
    estimated_risk = quantity * risk_per_share

    if not _finite_number(estimated_cost) or not _finite_number(
        estimated_risk
    ):
        return _rejected_plan("Calculated trade values are invalid.")

    # Allow only a tiny floating-point tolerance in the checks.
    tolerance = 1e-8

    if estimated_cost > cash + tolerance:
        return _rejected_plan("Order would exceed available cash.")

    if estimated_cost > position_budget + tolerance:
        return _rejected_plan(
            "Order would exceed the maximum position budget."
        )

    if estimated_risk > risk_budget + tolerance:
        return _rejected_plan(
            "Order would exceed the planned trade-risk budget."
        )

    if estimated_cost < config.min_order_value:
        return _rejected_plan(
            f"Order value ${estimated_cost:.2f} is below the "
            f"${config.min_order_value:.2f} minimum."
        )

    return TradePlan(
        approved=True,
        symbol=str(symbol).strip().upper(),
        quantity=round(quantity, 8),
        estimated_cost=round(estimated_cost, 2),
        stop_price=round(stop, 4),
        risk_amount=round(estimated_risk, 2),
        reason=(
            "Trade passes position-count, risk-budget, "
            "position-value, and cash checks."
        ),
    )


def check_position_limit(
    open_positions: int,
    config: Optional[RiskConfig] = None,
) -> bool:
    """Return True only if another position may be considered."""

    if config is None:
        config = RiskConfig()

    if not _valid_config(config):
        return False

    if (
        not isinstance(open_positions, int)
        or isinstance(open_positions, bool)
        or open_positions < 0
    ):
        return False

    return open_positions < config.max_positions


def calculate_affordable_quantity(
    available_cash: float,
    stock_price: float,
    fractional_shares: bool = True,
) -> float:
    """
    Calculate affordability using available cash alone.

    This helper does not apply the reserve or risk limits.
    """

    if not _positive_number(available_cash):
        return 0.0

    if not _positive_number(stock_price):
        return 0.0

    if not isinstance(fractional_shares, bool):
        return 0.0

    quantity = float(available_cash) / float(stock_price)

    if not fractional_shares:
        quantity = math.floor(quantity)

    if not _finite_number(quantity) or quantity <= 0:
        return 0.0

    return round(quantity, 8)


def validate_portfolio(
    account_equity: float,
    available_cash: float,
    open_positions: int,
    config: Optional[RiskConfig] = None,
) -> List[str]:
    """
    Return warnings for basic portfolio consistency checks.

    An empty list means these checks found no issues. It does not
    guarantee that the portfolio is safe or correctly reported.
    """

    if config is None:
        config = RiskConfig()

    warnings = []

    if not _valid_config(config):
        warnings.append("Risk configuration is invalid.")

    if not _positive_number(account_equity):
        warnings.append(
            "Account equity is zero, negative, or invalid."
        )

    if not _finite_number(available_cash) or float(available_cash) < 0:
        warnings.append(
            "Available cash is negative or invalid."
        )

    if (
        not isinstance(open_positions, int)
        or isinstance(open_positions, bool)
        or open_positions < 0
    ):
        warnings.append(
            "Open position count must be a nonnegative integer."
        )
    elif _valid_config(config) and open_positions > config.max_positions:
        warnings.append(
            f"Portfolio has {open_positions} positions, exceeding "
            f"the limit of {config.max_positions}."
        )

    if (
        _positive_number(account_equity)
        and _finite_number(available_cash)
        and float(available_cash) > float(account_equity)
    ):
        warnings.append(
            "Available cash exceeds reported account equity; "
            "verify account data."
        )

    return warnings
