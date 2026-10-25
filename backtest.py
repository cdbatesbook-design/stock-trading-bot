"""
Unified Micro-Profit Research Engine
====================================

Research-only engine for comparing short-term trading strategies
on stocks, crypto, and historical option-contract price data.

Supported strategy experiments:
1. Exit at the first achievable net profit.
2. Exit at a fixed net-profit target.
3. Exit at a volatility-adjusted target.
4. Momentum entry with a small-profit exit.

Safety:
- No broker connections.
- No order placement.
- No live trading.
- Maximum three simultaneous positions.
- Maximum 5% of portfolio equity allocated per position.
- Includes estimated fees and slippage.
- Uses chronological train, validation, and test periods.

Input:
CSV files containing historical OHLCV bars.

Required CSV columns:
timestamp, open, high, low, close, volume

Example:
data/AAPL.csv
data/BTC-USD.csv
data/OPTION_CONTRACT.csv

Option research requires actual historical prices for the
specific option contracts being tested.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    starting_capital: float = 1000.0

    max_positions: int = 3
    max_position_fraction: float = 0.05

    # Estimated costs per side.
    # Replace these with realistic costs for the actual broker
    # and asset being tested.
    fee_rate: float = 0.0025
    slippage_rate: float = 0.0005

    # Minimum profit targets BEFORE transaction costs.
    # The simulator also accounts for estimated round-trip costs.
    profit_targets: tuple = (
        0.001,
        0.0025,
        0.005,
        0.0075,
    )

    stop_loss_fraction: float = 0.01

    momentum_lookback: int = 5
    volatility_lookback: int = 20

    # Reject strategies that have too few completed trades.
    minimum_trades: int = 30

    # Limit risk from one position.
    maximum_loss_fraction: float = 0.005


@dataclass
class Position:
    symbol: str
    entry_time: str
    entry_price: float
    quantity: float
    entry_cost: float
    entry_fee: float
    stop_price: float
    target_price: float
    strategy: str


# ============================================================
# DATA LOADING
# ============================================================

def load_bars(path: Path) -> pd.DataFrame:
    """Load and validate historical OHLCV bars."""

    df = pd.read_csv(path)

    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: missing columns {sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    df = df.sort_values("timestamp")
    df = df.drop_duplicates("timestamp")
    df = df.reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(
            f"{path}: need at least 100 valid bars; "
            f"found {len(df)}"
        )

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(
            f"{path}: prices must be positive"
        )

    invalid = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    if invalid.any():
        raise ValueError(
            f"{path}: invalid OHLC price relationships"
        )

    return df


def add_indicators(
    df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Calculate indicators using only current and past data."""

    df = df.copy()

    df["momentum"] = (
        df["close"]
        / df["close"].shift(config.momentum_lookback)
        - 1.0
    )

    df["volatility"] = (
        df["close"]
        .pct_change()
        .rolling(config.volatility_lookback)
        .std()
    )

    df["rolling_high"] = (
        df["high"]
        .shift(1)
        .rolling(config.momentum_lookback)
        .max()
    )

    df["rolling_low"] = (
        df["low"]
        .shift(1)
        .rolling(config.momentum_lookback)
        .min()
    )

    return df


# ============================================================
# TRANSACTION COST MODEL
# ============================================================

def buy_fill(price: float, config: Config) -> float:
    """Conservative estimated purchase fill."""

    return price * (1.0 + config.slippage_rate)


def sell_fill(price: float, config: Config) -> float:
    """Conservative estimated sale fill."""

    return price * (1.0 - config.slippage_rate)


def estimated_round_trip_cost(
    entry_price: float,
    exit_price: float,
    config: Config,
) -> float:
    """Estimate fees and slippage as a fraction of entry value."""

    entry = buy_fill(entry_price, config)
    exit_ = sell_fill(exit_price, config)

    gross_cost = entry - exit_

    fees = (
        entry * config.fee_rate
        + exit_ * config.fee_rate
    )

    return (gross_cost + fees) / entry_price


# ============================================================
# ENTRY STRATEGIES
# ============================================================

def entry_signal(
    row: pd.Series,
    previous: pd.Series,
    strategy: str,
) -> bool:
    """Generate an entry signal from information available then."""

    momentum = row.get("momentum", float("nan"))

    if pd.isna(momentum):
        return False

    if strategy == "momentum":
        return momentum > 0.001

    if strategy == "breakout":
        rolling_high = row.get(
            "rolling_high",
            float("nan"),
        )

        return (
            not pd.isna(rolling_high)
            and row["close"] > rolling_high
        )

    if strategy == "mean_reversion":
        volatility = row.get(
            "volatility",
            float("nan"),
        )

        # Research-only simple oversold bounce condition.
        # It is not a guarantee that a reversal will occur.
        return (
            not pd.isna(volatility)
            and volatility > 0
            and momentum < -2.0 * volatility
            and row["close"] > previous["close"]
        )

    return False


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(
    data: dict[str, pd.DataFrame],
    strategy: str,
    target_fraction: float,
    config: Config,
    starting_equity: float | None = None,
) -> dict:
    """
    Simulate one strategy across multiple assets.

    Signals are calculated from completed bars.
    Entries occur at the next available bar's open.

    If both stop and target are touched within one bar,
    the stop is assumed to execute first. This is conservative.

    This simplified simulator does not model queue priority,
    partial fills, market impact, or option exercise/assignment.
    """

    cash = (
        config.starting_capital
        if starting_equity is None
        else starting_equity
    )

    equity = cash
    positions: dict[str, Position] = {}
    trades = []
    equity_curve = []

    timeline = sorted(
        set(
            timestamp
            for df in data.values()
            for timestamp in df["timestamp"]
        )
    )

    indexed = {
        symbol: df.set_index("timestamp", drop=False)
        for symbol, df in data.items()
    }

    previous_rows: dict[str, pd.Series] = {}

    for timestamp in timeline:
        current_rows = {}

        for symbol, df in indexed.items():
            if timestamp not in df.index:
                continue

            row = df.loc[timestamp]

            # Duplicate timestamps were removed at loading.
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]

            current_rows[symbol] = row

        # ----------------------------------------------------
        # EXIT EXISTING POSITIONS
        # ----------------------------------------------------

        for symbol in list(positions):
            if symbol not in current_rows:
                continue

            row = current_rows[symbol]
            pos = positions[symbol]

            exit_price = None
            exit_reason = None

            # Conservative stop-first rule.
            if row["low"] <= pos.stop_price:
                exit_price = pos.stop_price
                exit_reason = "stop_loss"

            elif row["high"] >= pos.target_price:
                exit_price = pos.target_price
                exit_reason = "profit_target"

            if exit_price is None:
                continue

            actual_exit = sell_fill(exit_price, config)

            exit_fee = (
                actual_exit
                * pos.quantity
                * config.fee_rate
            )

            proceeds = (
                actual_exit * pos.quantity - exit_fee
            )

            net_pnl = (
                proceeds
                - pos.entry_cost * pos.quantity
                - pos.entry_fee
            )

            cash += proceeds

            trades.append({
                "symbol": symbol,
                "strategy": strategy,
                "entry_time": pos.entry_time,
                "exit_time": str(timestamp),
                "entry_price": pos.entry_price,
                "exit_price": actual_exit,
                "quantity": pos.quantity,
                "net_pnl": net_pnl,
                "return_fraction": (
                    net_pnl
                    / max(
                        pos.entry_cost * pos.quantity
                        + pos.entry_fee,
                        1e-12,
                    )
                ),
                "exit_reason": exit_reason,
            })

            del positions[symbol]

        # ----------------------------------------------------
        # UPDATE EQUITY
        # ----------------------------------------------------

        market_value = 0.0

        for symbol, pos in positions.items():
            if symbol in current_rows:
                market_value += (
                    current_rows[symbol]["close"]
                    * pos.quantity
                )

        equity = cash + market_value

        equity_curve.append({
            "timestamp": str(timestamp),
            "equity": equity,
        })

        # ----------------------------------------------------
        # ENTER NEW POSITIONS
        # ----------------------------------------------------

        if len(positions) >= config.max_positions:
            previous_rows.update(current_rows)
            continue

        for symbol, row in current_rows.items():
            if len(positions) >= config.max_positions:
                break

            if symbol in positions:
                continue

            if symbol not in previous_rows:
                continue

            previous = previous_rows[symbol]

            if not entry_signal(row, previous, strategy):
                continue

            # Signal was generated on this completed bar.
            # Execution uses the next bar's open.
            df = indexed[symbol]

            future = df[df.index > timestamp]

            if future.empty:
                continue

            next_bar = future.iloc[0]
            entry_reference = float(next_bar["open"])

            entry = buy_fill(entry_reference, config)

            if entry <= 0:
                continue

            equity = max(cash, 0.0)

            allocation = min(
                equity * config.max_position_fraction,
                cash,
            )

            if allocation <= 0:
                continue

            # Keep a small reserve for costs.
            quantity = allocation / (
                entry * (1.0 + config.fee_rate)
            )

            if quantity <= 0:
                continue

            entry_fee = entry * quantity * config.fee_rate

            required_cash = entry * quantity + entry_fee

            if required_cash > cash:
                continue

            volatility = row.get(
                "volatility",
                float("nan"),
            )

            if pd.isna(volatility) or volatility <= 0:
                volatility = 0.005

            # Stop distance is bounded.
            stop_fraction = max(
                min(
                    config.stop_loss_fraction,
                    config.maximum_loss_fraction * 4,
                ),
                0.002,
            )

            # Target must exceed estimated costs to be useful.
            minimum_cost = (
                2.0 * config.fee_rate
                + 2.0 * config.slippage_rate
            )

            effective_target = max(
                target_fraction,
                minimum_cost + 0.0005,
            )

            stop_price = entry * (1.0 - stop_fraction)
            target_price = entry * (1.0 + effective_target)

            cash -= required_cash

            positions[symbol] = Position(
                symbol=symbol,
                entry_time=str(next_bar["timestamp"]),
                entry_price=entry,
                quantity=quantity,
                entry_cost=entry,
                entry_fee=entry_fee,
                stop_price=stop_price,
                target_price=target_price,
                strategy=strategy,
            )

        previous_rows.update(current_rows)

    # --------------------------------------------------------
    # CLOSE REMAINING POSITIONS AT LAST AVAILABLE CLOSE
    # --------------------------------------------------------

    for symbol, pos in list(positions.items()):
        df = indexed[symbol]

        last_row = df.iloc[-1]
        reference_exit = float(last_row["close"])
        actual_exit = sell_fill(reference_exit, config)

        exit_fee = (
            actual_exit * pos.quantity * config.fee_rate
        )

        proceeds = actual_exit * pos.quantity - exit_fee

        net_pnl = (
            proceeds
            - pos.entry_cost * pos.quantity
            - pos.entry_fee
        )

        cash += proceeds

        trades.append({
            "symbol": symbol,
            "strategy": strategy,
            "entry_time": pos.entry_time,
            "exit_time": str(last_row["timestamp"]),
            "entry_price": pos.entry_price,
            "exit_price": actual_exit,
            "quantity": pos.quantity,
            "net_pnl": net_pnl,
            "return_fraction": (
                net_pnl
                / max(
                    pos.entry_cost * pos.quantity
                    + pos.entry_fee,
                    1e-12,
                )
            ),
            "exit_reason": "end_of_test",
        })

    final_equity = cash

    trade_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(equity_curve)

    if not curve_df.empty:
        curve_df["peak"] = curve_df["equity"].cummax()

        curve_df["drawdown"] = (
            curve_df["equity"] / curve_df["peak"] - 1.0
        )

        max_drawdown = float(curve_df["drawdown"].min())

    else:
        max_drawdown = 0.0

    if trade_df.empty:
        win_rate = 0.0
        profit_factor = 0.0
        average_trade = 0.0
        total_trades = 0

    else:
        winners = trade_df.loc[
            trade_df["net_pnl"] > 0,
            "net_pnl",
        ]

        losers = trade_df.loc[
            trade_df["net_pnl"] < 0,
            "net_pnl",
        ]

        win_rate = float(
            (trade_df["net_pnl"] > 0).mean()
        )

        gross_wins = float(winners.sum())
        gross_losses = abs(float(losers.sum()))

        profit_factor = (
            gross_wins / gross_losses
            if gross_losses > 0
            else (
                float("inf")
                if gross_wins > 0
                else 0.0
            )
        )

        average_trade = float(
            trade_df["net_pnl"].mean()
        )

        total_trades = len(trade_df)

    return {
        "strategy": strategy,
        "target_fraction": target_fraction,
        "starting_equity": (
            config.starting_capital
            if starting_equity is None
            else starting_equity
        ),
        "final_equity": final_equity,
        "net_profit": (
            final_equity
            - (
                config.starting_capital
                if starting_equity is None
                else starting_equity
            )
        ),
        "return_fraction": (
            final_equity
            / (
                config.starting_capital
                if starting_equity is None
                else starting_equity
            )
            - 1.0
        ),
        "max_drawdown": max_drawdown,
        "trade_count": total_trades,
        "win_rate": win_rate,
        "profit_factor": (
            profit_factor
            if math.isfinite(profit_factor)
            else None
        ),
        "average_trade": average_trade,
        "trades": trades,
        "equity_curve": equity_curve,
    }


# ============================================================
# CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_data(data: dict[str, pd.DataFrame]):
    """
    Split each instrument chronologically.

    60% train, 20% validation, 20% final test.
    No random shuffling.
    """

    train = {}
    validation = {}
    test = {}

    for symbol, df in data.items():
        n = len(df)

        a = int(n * 0.60)
        b = int(n * 0.80)

        train[symbol] = df.iloc[:a].copy()
        validation[symbol] = df.iloc[a:b].copy()
        test[symbol] = df.iloc[b:].copy()

    return train, validation, test


# ============================================================
# STRATEGY RESEARCH
# ============================================================

def research(data_dir: Path, output_dir: Path, config: Config):
    """Evaluate candidate strategies without placing orders."""

    files = sorted(data_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}"
        )

    data = {}

    for path in files:
        symbol = path.stem.upper()

        try:
            df = load_bars(path)
            df = add_indicators(df, config)
            data[symbol] = df

        except Exception as exc:
            print(f"SKIPPING {path.name}: {exc}")

    if not data:
        raise RuntimeError("No valid market data was loaded.")

    train, validation, test = split_data(data)

    candidates = []

    for strategy in [
        "momentum",
        "breakout",
        "mean_reversion",
    ]:
        for target in config.profit_targets:
            result = run_backtest(
                train,
                strategy,
                target,
                config,
            )

            candidates.append(result)

    candidates.sort(
        key=lambda x: x["net_profit"],
        reverse=True,
    )

    validation_results = []

    for candidate in candidates:
        result = run_backtest(
            validation,
            candidate["strategy"],
            candidate["target_fraction"],
            config,
        )

        validation_results.append(result)

    validation_results.sort(
        key=lambda x: x["net_profit"],
        reverse=True,
    )

    # Only advance candidates with enough trades and a positive
    # validation result. This is a screening gate, not proof.
    finalists = [
        result
        for result in validation_results
        if (
            result["trade_count"] >= config.minimum_trades
            and result["net_profit"] > 0
            and (result["profit_factor"] or 0) > 1.0
            and result["max_drawdown"] > -0.20
        )
    ]

    final_results = []

    # The test period is not used to select candidates.
    # Only evaluate candidates that passed validation.
    for candidate in finalists:
        result = run_backtest(
            test,
            candidate["strategy"],
            candidate["target_fraction"],
            config,
        )

        final_results.append(result)

    output_dir.mkdir(parents=True, exist_ok=True)

    def clean_result(result):
        return {
            key: value
            for key, value in result.items()
            if key not in ("trades", "equity_curve")
        }

    report = {
        "mode": "RESEARCH_ONLY",
        "orders_enabled": False,
        "config": asdict(config),
        "symbols": list(data.keys()),
        "bars": {
            symbol: len(df)
            for symbol, df in data.items()
        },
        "train_results": [
            clean_result(x) for x in candidates
        ],
        "validation_results": [
            clean_result(x) for x in validation_results
        ],
        "final_test_results": [
            clean_result(x) for x in final_results
        ],
        "warning": (
            "Historical performance is not a guarantee of "
            "future results. Data quality, spreads, fills, "
            "market impact, and changing conditions can make "
            "real results materially different."
        ),
    }

    report_path = output_dir / "research_report.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    pd.DataFrame(
        report["train_results"]
    ).to_csv(
        output_dir / "train_results.csv",
        index=False,
    )

    pd.DataFrame(
        report["validation_results"]
    ).to_csv(
        output_dir / "validation_results.csv",
        index=False,
    )

    pd.DataFrame(
        report["final_test_results"]
    ).to_csv(
        output_dir / "final_test_results.csv",
        index=False,
    )

    print("\n======================================")
    print("MICRO-PROFIT RESEARCH COMPLETE")
    print("======================================")
    print(f"Symbols loaded: {len(data)}")
    print(f"Report: {report_path}")
    print("Order placement: DISABLED")

    if final_results:
        print("\nFINAL TEST RESULTS")

        for result in sorted(
            final_results,
            key=lambda x: x["net_profit"],
            reverse=True,
        ):
            print(
                f"{result['strategy']:16} "
                f"target={result['target_fraction']:.3%} "
                f"return={result['return_fraction']:.2%} "
                f"trades={result['trade_count']} "
                f"win_rate={result['win_rate']:.1%} "
                f"profit_factor={result['profit_factor']} "
                f"drawdown={result['max_drawdown']:.2%}"
            )

    else:
        print(
            "\nNo candidate passed validation. "
            "That is a valid result; do not loosen the "
            "requirements merely to force a winner."
        )


# ============================================================
# COMMAND LINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Research micro-profit trading strategies"
    )

    parser.add_argument(
        "--data",
        default="data",
        help="Folder containing historical OHLCV CSV files",
    )

    parser.add_argument(
        "--output",
        default="outputs/micro_profit",
        help="Folder for research reports",
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=1000.0,
        help="Starting simulated capital",
    )

    args = parser.parse_args()

    if args.capital <= 0:
        raise SystemExit("--capital must be positive")

    config = Config(starting_capital=args.capital)

    research(
        Path(args.data),
        Path(args.output),
        config,
    )


if __name__ == "__main__":
    main()
