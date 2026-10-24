"""
Strategy Selector
-----------------
Ranks independent trading strategies using recorded performance.

This module does not submit orders and does not guarantee future returns.
Strategies with insufficient data should remain in research/testing.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math


@dataclass
class StrategyPerformance:
    strategy_name: str
    total_trades: int
    win_rate: float
    profit_factor: float
    total_return_pct: float
    max_drawdown_pct: float
    benchmark_return_pct: float = 0.0


@dataclass
class StrategyDecision:
    strategy_name: str
    eligible_for_paper_testing: bool
    score: float
    reason: str


@dataclass
class SelectorConfig:
    # Require enough trades before judging a strategy
    minimum_trades: int = 30

    # Require a positive profit factor
    minimum_profit_factor: float = 1.05

    # Require positive returns in the tested period
    minimum_return_pct: float = 0.0

    # Reject strategies with excessive historical drawdown
    maximum_drawdown_pct: float = 20.0

    # Require the strategy to outperform its benchmark
    require_benchmark_outperformance: bool = True


def _is_finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def evaluate_strategy(
    performance: StrategyPerformance,
    config: Optional[SelectorConfig] = None,
) -> StrategyDecision:
    """
    Evaluate one strategy for eligibility for further paper testing.

    Eligibility is not permission for live trading.
    """

    if config is None:
        config = SelectorConfig()

    name = performance.strategy_name

    def reject(reason: str) -> StrategyDecision:
        return StrategyDecision(
            strategy_name=name,
            eligible_for_paper_testing=False,
            score=0.0,
            reason=reason,
        )

    numeric_values = [
        performance.win_rate,
        performance.profit_factor,
        performance.total_return_pct,
        performance.max_drawdown_pct,
        performance.benchmark_return_pct,
    ]

    if not all(_is_finite(value) for value in numeric_values):
        return reject("Performance data contains invalid numeric values.")

    if performance.total_trades < config.minimum_trades:
        return reject(
            f"Insufficient trade history: {performance.total_trades} trades; "
            f"at least {config.minimum_trades} required."
        )

    if performance.total_trades < 0:
        return reject("Trade count cannot be negative.")

    if not 0 <= performance.win_rate <= 100:
        return reject("Win rate must be between 0 and 100.")

    if performance.profit_factor < config.minimum_profit_factor:
        return reject(
            f"Profit factor {performance.profit_factor:.2f} is below "
            f"the required {config.minimum_profit_factor:.2f}."
        )

    if performance.total_return_pct <= config.minimum_return_pct:
        return reject(
            f"Return {performance.total_return_pct:.2f}% does not exceed "
            f"the required {config.minimum_return_pct:.2f}%."
        )

    if performance.max_drawdown_pct < 0:
        return reject("Maximum drawdown must be zero or positive.")

    if performance.max_drawdown_pct > config.maximum_drawdown_pct:
        return reject(
            f"Maximum drawdown {performance.max_drawdown_pct:.2f}% exceeds "
            f"the allowed {config.maximum_drawdown_pct:.2f}%."
        )

    if (
        config.require_benchmark_outperformance
        and performance.total_return_pct <= performance.benchmark_return_pct
    ):
        return reject(
            "Strategy did not outperform its benchmark during the test period."
        )

    # A ranking score is for comparison, not a forecast of future performance.
    score = (
        performance.profit_factor * 30.0
        + performance.total_return_pct * 1.0
        + performance.win_rate * 0.10
        - performance.max_drawdown_pct * 0.50
        + min(performance.total_trades, 200) * 0.05
    )

    return StrategyDecision(
        strategy_name=name,
        eligible_for_paper_testing=True,
        score=round(score, 3),
        reason="Passed the configured historical screening checks.",
    )


def rank_strategies(
    performances: List[StrategyPerformance],
    config: Optional[SelectorConfig] = None,
) -> List[StrategyDecision]:
    """
    Evaluate and rank strategies from highest to lowest score.

    Strategies that fail screening remain in the output so their reasons
    can be reviewed.
    """

    if config is None:
        config = SelectorConfig()

    decisions = [
        evaluate_strategy(performance, config)
        for performance in performances
    ]

    decisions.sort(
        key=lambda decision: (
            decision.eligible_for_paper_testing,
            decision.score,
        ),
        reverse=True,
    )

    return decisions


def select_strategies_for_paper_testing(
    performances: List[StrategyPerformance],
    config: Optional[SelectorConfig] = None,
) -> List[str]:
    """
    Return names of strategies that pass the historical screening rules.

    This does not start paper trading or place broker orders.
    """

    decisions = rank_strategies(performances, config)

    return [
        decision.strategy_name
        for decision in decisions
        if decision.eligible_for_paper_testing
    ]


def format_strategy_report(
    performances: List[StrategyPerformance],
    config: Optional[SelectorConfig] = None,
) -> str:
    """
    Build a readable report showing every strategy and its screening result.
    """

    decisions = rank_strategies(performances, config)

    if not decisions:
        return "No strategy performance records were provided."

    lines = [
        "MULTI-STRATEGY PERFORMANCE SCREEN",
        "=" * 68,
        "Historical screening only; not a live-trading authorization.",
        "",
    ]

    performance_by_name: Dict[str, StrategyPerformance] = {
        performance.strategy_name: performance
        for performance in performances
    }

    for position, decision in enumerate(decisions, start=1):
        performance = performance_by_name[decision.strategy_name]

        status = (
            "PAPER-TEST CANDIDATE"
            if decision.eligible_for_paper_testing
            else "NOT QUALIFIED"
        )

        lines.extend(
            [
                f"{position}. {decision.strategy_name}",
                f"   Status: {status}",
                f"   Trades: {performance.total_trades}",
                f"   Win rate: {performance.win_rate:.2f}%",
                f"   Profit factor: {performance.profit_factor:.2f}",
                f"   Return: {performance.total_return_pct:.2f}%",
                f"   Benchmark return: {performance.benchmark_return_pct:.2f}%",
                f"   Maximum drawdown: {performance.max_drawdown_pct:.2f}%",
                f"   Score: {decision.score:.3f}",
                f"   Reason: {decision.reason}",
                "",
            ]
        )

    return "\n".join(lines)
