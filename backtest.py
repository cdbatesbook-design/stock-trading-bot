#!/usr/bin/env python3
"""
Penny-stock strategy backtester.

Historical simulation only. Does not place live orders.

Strategy:
- Penny-stock candidates priced $0.10-$5.00.
- Minimum current bar volume: 1 share.
- Minimum average volume: 1 share.
- Buy when price is rising and EMA trend is positive.
- Sell on the first confirmed closing-price decline,
  a stop-loss, or a profit target.
- Maximum three simultaneous positions.
- Track realized profit separately from available trading cash.
- Never label estimated slippage as actual slippage.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

STARTING_CAPITAL = 10_000.00

MIN_PRICE = 0.10
MAX_PRICE = 5.00

MIN_BAR_VOLUME = 1
MIN_AVERAGE_VOLUME = 1

MAX_POSITIONS = 3
MAX_ALLOCATION_PER_STOCK = 0.30

RISK_PER_TRADE = 0.005
MAX_TOTAL_RISK = 0.015

# These are TEST PARAMETERS, not guaranteed profitable values.
STOP_LOSSES = [0.003, 0.005, 0.01]
PROFIT_TARGETS = [0.001, 0.0025, 0.005, 0.0075]

# Actual costs are unknown until broker execution records exist.
# Keep estimated slippage and fees at zero rather than misrepresenting
# estimates as actual costs. Results are explicitly marked accordingly.
ESTIMATED_SLIPPAGE_RATE = 0.0
ESTIMATED_FEE_RATE = 0.0

# Penny-stock candidates. These are examples, not recommendations.
DEFAULT_TICKERS = [
    "SNDL",
    "ACB",
    "TLRY",
    "PLUG",
    "DNA",
    "OPEN",
    "CHPT",
    "BBAI",
]

PERIOD = os.getenv("BACKTEST_PERIOD", "2y")
INTERVAL = os.getenv("BACKTEST_INTERVAL", "1d")

OUTPUT_DIR = Path(".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

logger = logging.getLogger("penny_backtest")


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Position:
    ticker: str
    entry_time: str
    entry_price: float
    shares: int
    entry_cost: float
    stop_loss: float
    profit_target: float
    risk_dollars: float


@dataclass
class Trade:
    ticker: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    shares: int
    gross_pnl: float
    estimated_costs: float
    net_pnl_before_unknown_real_costs: float
    exit_reason: str


# ============================================================
# DATA DOWNLOAD
# ============================================================

def normalize_yfinance_data(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize downloaded OHLCV data to standard column names."""

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        # yfinance can return either ticker-first or field-first columns.
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        elif ticker in data.columns.get_level_values(0):
            data = data.xs(ticker, axis=1, level=0)
        else:
            data.columns = data.columns.get_level_values(0)

    data = data.copy()
    data.columns = [str(c).strip().title() for c in data.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]

    if not all(c in data.columns for c in required):
        logger.warning("%s: required OHLCV columns missing", ticker)
        return pd.DataFrame()

    data = data[required].copy()

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=required)
    data = data[(data["Close"] > 0) & (data["Open"] > 0)]
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]

    return data


def download_data(ticker: str) -> pd.DataFrame:
    """Download historical data for one ticker."""

    logger.info("Downloading %s", ticker)

    try:
        raw = yf.download(
            ticker,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        data = normalize_yfinance_data(raw, ticker)

        if data.empty:
            logger.warning("%s: no usable historical data", ticker)
            return data

        data["EMA_FAST"] = data["Close"].ewm(
            span=5, adjust=False
        ).mean()

        data["EMA_SLOW"] = data["Close"].ewm(
            span=13, adjust=False
        ).mean()

        data["PREVIOUS_CLOSE"] = data["Close"].shift(1)

        data["AVERAGE_VOLUME"] = data["Volume"].rolling(
            window=20, min_periods=1
        ).mean()

        data["MOMENTUM"] = data["Close"].pct_change(periods=3)

        data["BUY_SIGNAL"] = (
            (data["Close"] >= MIN_PRICE)
            & (data["Close"] <= MAX_PRICE)
            & (data["Volume"] >= MIN_BAR_VOLUME)
            & (data["AVERAGE_VOLUME"] >= MIN_AVERAGE_VOLUME)
            & (data["Close"] > data["EMA_FAST"])
            & (data["EMA_FAST"] > data["EMA_SLOW"])
            & (data["MOMENTUM"] > 0)
        )

        # First confirmed decline: closing below the previous close
        # or below the fast EMA, or the trend EMA turning negative.
        data["SELL_SIGNAL"] = (
            (data["Close"] < data["PREVIOUS_CLOSE"])
            | (data["Close"] < data["EMA_FAST"])
            | (data["EMA_FAST"] < data["EMA_SLOW"])
        )

        return data

    except Exception as exc:
        logger.exception("%s: download failed: %s", ticker, exc)
        return pd.DataFrame()


# ============================================================
# ACCOUNTING
# ============================================================

def apply_costs(price: float, side: str) -> float:
    """
    Apply explicitly configured cost assumptions.

    These defaults are zero because actual slippage and broker costs
    have not been established. Zero is NOT a claim that real costs are zero.
    """

    if side == "buy":
        return price * (1.0 + ESTIMATED_SLIPPAGE_RATE)

    return price * (1.0 - ESTIMATED_SLIPPAGE_RATE)


def estimate_fee(notional: float) -> float:
    """Calculate the configured fee estimate."""

    return max(0.0, notional) * ESTIMATED_FEE_RATE


def calculate_position_size(
    cash: float,
    price: float,
    total_open_risk: float,
    stop_loss_pct: float,
) -> int:
    """Size an integer-share position within the configured risk limits."""

    if cash <= 0 or price <= 0 or stop_loss_pct <= 0:
        return 0

    risk_budget = min(
        STARTING_CAPITAL * RISK_PER_TRADE,
        STARTING_CAPITAL * MAX_TOTAL_RISK - total_open_risk,
    )

    if risk_budget <= 0:
        return 0

    risk_shares = math.floor(risk_budget / (price * stop_loss_pct))

    allocation_limit = cash * MAX_ALLOCATION_PER_STOCK
    allocation_shares = math.floor(allocation_limit / price)

    affordable_shares = math.floor(cash / price)

    return max(
        0,
        min(risk_shares, allocation_shares, affordable_shares),
    )


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(
    market_data: dict[str, pd.DataFrame],
    stop_loss_pct: float,
    profit_target_pct: float,
    starting_capital: float = STARTING_CAPITAL,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Simulate trades using daily bars.

    Signals are calculated from completed bars and executed at the
    following bar's open. Stop/target triggers use the available bar
    high/low. If both stop and target are touched in one bar, the
    conservative assumption is that the stop executes first.
    """

    cash = float(starting_capital)
    realized_pnl = 0.0
    total_realized_profit = 0.0

    positions: dict[str, Position] = {}
    trades: list[dict] = []
    equity_records: list[dict] = []

    all_dates = sorted(
        set().union(*(set(df.index) for df in market_data.values()))
    )

    if not all_dates:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {"error": "No historical data available"},
        )

    for i, current_date in enumerate(all_dates):
        if i == 0:
            continue

        previous_date = all_dates[i - 1]

        # -------------------------
        # PROCESS EXISTING POSITIONS
        # -------------------------
        for ticker in list(positions):
            data = market_data.get(ticker)

            if data is None or current_date not in data.index:
                continue

            bar = data.loc[current_date]
            position = positions[ticker]

            exit_price = None
            exit_reason = None

            # Stop/target checks are based on the current bar.
            # If both levels are touched, assume the stop happened first.
            if float(bar["Low"]) <= position.stop_loss:
                exit_price = position.stop_loss
                exit_reason = "stop_loss"

            elif float(bar["High"]) >= position.profit_target:
                exit_price = position.profit_target
                exit_reason = "profit_target"

            # Signal from the preceding completed bar, executed at
            # the current bar's open.
            elif previous_date in data.index:
                previous_bar = data.loc[previous_date]

                if bool(previous_bar.get("SELL_SIGNAL", False)):
                    exit_price = float(bar["Open"])
                    exit_reason = "decline_signal"

            if exit_price is None:
                continue

            executed_exit = apply_costs(exit_price, "sell")
            proceeds = position.shares * executed_exit
            fee = estimate_fee(proceeds)

            pnl = proceeds - fee - position.entry_cost

            # Entry cost was removed from cash at purchase.
            # Return the sale proceeds to cash.
            cash += proceeds - fee
            realized_pnl += pnl

            if pnl > 0:
                total_realized_profit += pnl

            trades.append(
                {
                    "ticker": ticker,
                    "entry_time": position.entry_time,
                    "exit_time": str(current_date),
                    "entry_price": position.entry_price,
                    "exit_price": executed_exit,
                    "shares": position.shares,
                    "gross_pnl": (
                        position.shares
                        * (executed_exit - position.entry_price)
                    ),
                    "entry_cost": position.entry_cost,
                    "exit_fee_estimate": fee,
                    "net_pnl_before_unknown_real_costs": pnl,
                    "exit_reason": exit_reason,
                    "slippage_status": "unknown_actual_slippage",
                    "fee_status": "estimated_fee_rate_is_configured",
                }
            )

            del positions[ticker]

        # -------------------------
        # OPEN NEW POSITIONS
        # -------------------------
        if len(positions) < MAX_POSITIONS:
            for ticker, data in market_data.items():
                if len(positions) >= MAX_POSITIONS:
                    break

                if ticker in positions:
                    continue

                if current_date not in data.index:
                    continue

                if previous_date not in data.index:
                    continue

                previous_bar = data.loc[previous_date]
                current_bar = data.loc[current_date]

                if not bool(previous_bar.get("BUY_SIGNAL", False)):
                    continue

                open_price = float(current_bar["Open"])

                if not MIN_PRICE <= open_price <= MAX_PRICE:
                    continue

                if float(previous_bar["Volume"]) < MIN_BAR_VOLUME:
                    continue

                if (
                    float(previous_bar["AVERAGE_VOLUME"])
                    < MIN_AVERAGE_VOLUME
                ):
                    continue

                total_open_risk = sum(
                    p.risk_dollars for p in positions.values()
                )

                shares = calculate_position_size(
                    cash=cash,
                    price=open_price,
                    total_open_risk=total_open_risk,
                    stop_loss_pct=stop_loss_pct,
                )

                if shares <= 0:
                    continue

                executed_entry = apply_costs(open_price, "buy")
                notional = shares * executed_entry
                entry_fee = estimate_fee(notional)
                total_entry_cost = notional + entry_fee

                if total_entry_cost > cash:
                    continue

                cash -= total_entry_cost

                positions[ticker] = Position(
                    ticker=ticker,
                    entry_time=str(current_date),
                    entry_price=executed_entry,
                    shares=shares,
                    entry_cost=total_entry_cost,
                    stop_loss=executed_entry * (1.0 - stop_loss_pct),
                    profit_target=executed_entry * (
                        1.0 + profit_target_pct
                    ),
                    risk_dollars=shares * executed_entry * stop_loss_pct,
                )

        # -------------------------
        # RECORD EQUITY
        # -------------------------
        open_market_value = 0.0

        for ticker, position in positions.items():
            data = market_data[ticker]

            if current_date in data.index:
                mark_price = float(data.loc[current_date]["Close"])
            else:
                mark_price = position.entry_price

            open_market_value += position.shares * mark_price

        equity_records.append(
            {
                "date": str(current_date),
                "cash": cash,
                "open_positions": len(positions),
                "open_market_value": open_market_value,
                "equity": cash + open_market_value,
                "realized_pnl": realized_pnl,
                "positive_trade_profit_total": total_realized_profit,
            }
        )

    # -------------------------
    # CLOSE REMAINING POSITIONS
    # -------------------------
    if all_dates:
        last_date = all_dates[-1]

        for ticker in list(positions):
            position = positions[ticker]
            data = market_data[ticker]

            if last_date in data.index:
                raw_exit = float(data.loc[last_date]["Close"])
            else:
                raw_exit = position.entry_price

            executed_exit = apply_costs(raw_exit, "sell")
            proceeds = position.shares * executed_exit
            fee = estimate_fee(proceeds)
            pnl = proceeds - fee - position.entry_cost

            cash += proceeds - fee
            realized_pnl += pnl

            if pnl > 0:
                total_realized_profit += pnl

            trades.append(
                {
                    "ticker": ticker,
                    "entry_time": position.entry_time,
                    "exit_time": str(last_date),
                    "entry_price": position.entry_price,
                    "exit_price": executed_exit,
                    "shares": position.shares,
                    "gross_pnl": (
                        position.shares
                        * (executed_exit - position.entry_price)
                    ),
                    "entry_cost": position.entry_cost,
                    "exit_fee_estimate": fee,
                    "net_pnl_before_unknown_real_costs": pnl,
                    "exit_reason": "end_of_test",
                    "slippage_status": "unknown_actual_slippage",
                    "fee_status": "estimated_fee_rate_is_configured",
                }
            )

            del positions[ticker]

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_records)

    final_equity = cash
    total_return = (
        (final_equity / starting_capital) - 1.0
        if starting_capital > 0
        else 0.0
    )

    if not trades_df.empty:
        winners = trades_df[
            trades_df["net_pnl_before_unknown_real_costs"] > 0
        ]
        losers = trades_df[
            trades_df["net_pnl_before_unknown_real_costs"] < 0
        ]

        win_rate = len(winners) / len(trades_df)
        gross_wins = winners[
            "net_pnl_before_unknown_real_costs"
        ].sum()
        gross_losses = abs(
            losers["net_pnl_before_unknown_real_costs"].sum()
        )

        profit_factor = (
            float(gross_wins / gross_losses)
            if gross_losses > 0
            else None
        )
    else:
        win_rate = 0.0
        profit_factor = None

    summary = {
        "starting_capital": starting_capital,
        "ending_equity": final_equity,
        "net_pnl_before_unknown_real_costs": (
            final_equity - starting_capital
        ),
        "total_return_pct": total_return * 100.0,
        "trade_count": len(trades_df),
        "win_rate_pct": win_rate * 100.0,
        "profit_factor": profit_factor,
        "max_simultaneous_positions": MAX_POSITIONS,
        "minimum_bar_volume": MIN_BAR_VOLUME,
        "minimum_average_volume": MIN_AVERAGE_VOLUME,
        "min_price": MIN_PRICE,
        "max_price": MAX_PRICE,
        "stop_loss_pct": stop_loss_pct * 100.0,
        "profit_target_pct": profit_target_pct * 100.0,
        "actual_slippage": "UNKNOWN - requires actual broker fills",
        "actual_fees": "UNKNOWN - requires broker fee information",
        "live_orders_enabled": False,
        "warning": (
            "Historical simulation only. Results exclude unknown real "
            "slippage and fees and do not establish future profitability."
        ),
    }

    return trades_df, equity_df, summary


# ============================================================
# SELF-TEST
# ============================================================

def self_test() -> None:
    """Run basic configuration and position-sizing checks."""

    assert MAX_POSITIONS == 3
    assert MIN_BAR_VOLUME == 1
    assert MIN_AVERAGE_VOLUME == 1
    assert MIN_PRICE == 0.10
    assert MAX_PRICE == 5.00

    shares = calculate_position_size(
        cash=STARTING_CAPITAL,
        price=1.00,
        total_open_risk=0.0,
        stop_loss_pct=0.01,
    )

    assert shares > 0
    assert shares * 1.00 <= STARTING_CAPITAL * MAX_ALLOCATION_PER_STOCK

    print("SELF-TEST PASSED")
    print("Minimum bar volume:", MIN_BAR_VOLUME)
    print("Minimum average volume:", MIN_AVERAGE_VOLUME)
    print("Maximum positions:", MAX_POSITIONS)
    print("Live trading: DISABLED")


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    if "--self-test" in sys.argv:
        self_test()
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tickers = os.getenv(
        "BACKTEST_TICKERS",
        ",".join(DEFAULT_TICKERS),
    )

    ticker_list = [
        ticker.strip().upper()
        for ticker in tickers.split(",")
        if ticker.strip()
    ]

    market_data: dict[str, pd.DataFrame] = {}

    for ticker in ticker_list:
        data = download_data(ticker)

        if not data.empty:
            market_data[ticker] = data

    if not market_data:
        logger.error("No usable historical data. Backtest stopped.")
        return 1

    all_results = []

    for stop_pct in STOP_LOSSES:
        for target_pct in PROFIT_TARGETS:
            logger.info(
                "Testing stop=%.3f%% target=%.3f%%",
                stop_pct * 100,
                target_pct * 100,
            )

            trades, equity, summary = run_backtest(
                market_data=market_data,
                stop_loss_pct=stop_pct,
                profit_target_pct=target_pct,
            )

            summary["stop_loss_fraction"] = stop_pct
            summary["profit_target_fraction"] = target_pct
            all_results.append(summary)

            label = (
                f"stop_{stop_pct:.4f}_target_{target_pct:.4f}"
                .replace(".", "_")
            )

            trades.to_csv(
                OUTPUT_DIR / f"trades_{label}.csv",
                index=False,
            )

            equity.to_csv(
                OUTPUT_DIR / f"equity_{label}.csv",
                index=False,
            )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        OUTPUT_DIR / "strategy_results.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "research_report.json",
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(all_results, report_file, indent=2, allow_nan=False)

    print("\nBACKTEST SUMMARY")
    print("================")

    if not results_df.empty:
        columns = [
            "stop_loss_pct",
            "profit_target_pct",
            "trade_count",
            "win_rate_pct",
            "total_return_pct",
            "actual_slippage",
        ]

        print(results_df[columns].to_string(index=False))

    print("\nOutput files:")
    print("- strategy_results.csv")
    print("- research_report.json")
    print("- trades_*.csv")
    print("- equity_*.csv")

    print("\nActual slippage: UNKNOWN")
    print("Actual fees: UNKNOWN")
    print("Live trading: DISABLED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
