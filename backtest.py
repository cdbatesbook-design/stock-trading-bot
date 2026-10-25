import argparse
import json
import logging
import math
import time

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SYMBOLS = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMD",
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
]

STARTING_CAPITAL = 1000.00

MAX_POSITIONS = 3

# Maximum percentage of account equity allocated per position.
MAX_ALLOCATION = 0.30

# Estimated transaction costs.
# These are assumptions, not guaranteed real-world costs.
FEE_RATE = 0.0025
SLIPPAGE_RATE = 0.0005

STOP_LOSS = 0.005

PROFIT_TARGETS = [
    0.001,
    0.0025,
    0.005,
    0.0075,
]

STRATEGIES = [
    "momentum",
    "breakout",
    "mean_reversion",
]

MINIMUM_BARS = 150

OUTPUT_DIR = Path("backtest_results")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# DOWNLOAD HISTORICAL DATA
# ============================================================

def download_data(symbol, period, interval):
    """
    Download historical market data directly.
    No local CSV files are required.
    """

    logger.info(
        "Downloading %s: period=%s interval=%s",
        symbol,
        period,
        interval,
    )

    for attempt in range(3):
        try:
            data = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=30,
            )

            if data is None or data.empty:
                logger.warning(
                    "No historical data returned for %s",
                    symbol,
                )
                return None

            # Handle yfinance MultiIndex columns.
            if isinstance(data.columns, pd.MultiIndex):
                cleaned = {}

                for column in data.columns:
                    parts = [
                        str(part)
                        for part in column
                        if str(part).lower() in {
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                        }
                    ]

                    if parts:
                        name = parts[0].lower()

                        if name not in cleaned:
                            cleaned[name] = column

                if not all(
                    name in cleaned
                    for name in ["open", "high", "low", "close"]
                ):
                    logger.warning(
                        "Unexpected columns for %s: %s",
                        symbol,
                        list(data.columns),
                    )
                    return None

                data = pd.DataFrame({
                    name: data[column]
                    for name, column in cleaned.items()
                })

            data.columns = [
                str(column).lower()
                for column in data.columns
            ]

            required = ["open", "high", "low", "close"]

            if not all(
                column in data.columns
                for column in required
            ):
                logger.warning(
                    "Required price columns missing for %s",
                    symbol,
                )
                return None

            if "volume" not in data.columns:
                data["volume"] = 0.0

            for column in required + ["volume"]:
                data[column] = pd.to_numeric(
                    data[column],
                    errors="coerce",
                )

            data = data.dropna(
                subset=required
            )

            data = data[
                (data["close"] > 0)
                & (data["open"] > 0)
            ]

            data = data[
                ~data.index.duplicated(keep="last")
            ]

            data = data.sort_index()

            if len(data) < MINIMUM_BARS:
                logger.warning(
                    "%s has only %s usable bars; need at least %s",
                    symbol,
                    len(data),
                    MINIMUM_BARS,
                )
                return None

            logger.info(
                "%s: downloaded %s usable bars",
                symbol,
                len(data),
            )

            return data

        except Exception as exc:
            logger.warning(
                "Download attempt %s failed for %s: %s",
                attempt + 1,
                symbol,
                exc,
            )

            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    return None


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(data):
    """
    Calculate indicators using current and historical bars only.
    """

    df = data.copy()

    close = df["close"]

    df["sma_10"] = close.rolling(10).mean()
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()

    df["ema_9"] = close.ewm(
        span=9,
        adjust=False,
    ).mean()

    df["ema_21"] = close.ewm(
        span=21,
        adjust=False,
    ).mean()

    df["momentum"] = close.pct_change(5)

    df["volatility"] = (
        close.pct_change()
        .rolling(20)
        .std()
    )

    df["rolling_high"] = (
        df["high"]
        .rolling(20)
        .max()
        .shift(1)
    )

    df["rolling_low"] = (
        df["low"]
        .rolling(20)
        .min()
        .shift(1)
    )

    df["mean_20"] = close.rolling(20).mean()

    df["std_20"] = close.rolling(20).std()

    df["zscore"] = (
        (close - df["mean_20"])
        / df["std_20"].replace(0, np.nan)
    )

    return df


# ============================================================
# ENTRY SIGNALS
# ============================================================

def get_signal(row, strategy):
    """
    Return True when the historical bar satisfies a strategy.
    """

    values = [
        row.get("close"),
        row.get("sma_20"),
        row.get("sma_50"),
    ]

    if any(
        pd.isna(value)
        for value in values
    ):
        return False

    close = float(row["close"])
    sma20 = float(row["sma_20"])
    sma50 = float(row["sma_50"])

    if strategy == "momentum":
        momentum = row.get("momentum")

        if pd.isna(momentum):
            return False

        return (
            close > sma20
            and sma20 > sma50
            and float(momentum) > 0
        )

    if strategy == "breakout":
        rolling_high = row.get("rolling_high")

        if pd.isna(rolling_high):
            return False

        return (
            close > float(rolling_high)
            and close > sma50
        )

    if strategy == "mean_reversion":
        zscore = row.get("zscore")

        if pd.isna(zscore):
            return False

        return (
            float(zscore) < -1.5
            and close > 0
        )

    return False


# ============================================================
# PORTFOLIO BACKTEST
# ============================================================

def run_backtest(
    market_data,
    strategy,
    target_fraction,
    starting_capital,
):
    """
    Simulate a portfolio using historical OHLC data.

    Signals use the previous completed bar.
    Entries occur at the next bar's open.
    Stops and targets are checked against subsequent
    price bars.

    If a bar hits both stop and target, the stop is
    assumed to have occurred first.

    This conservative assumption does not eliminate
    all historical simulation limitations.
    """

    cash = float(starting_capital)

    positions = {}

    trades = []

    equity_history = []

    timestamps = sorted(
        set().union(
            *[
                set(df.index)
                for df in market_data.values()
            ]
        )
    )

    if not timestamps:
        return {
            "strategy": strategy,
            "target": target_fraction,
            "starting_capital": starting_capital,
            "ending_equity": starting_capital,
            "return_pct": 0.0,
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "max_drawdown_pct": 0.0,
            "net_profit": 0.0,
        }

    row_locations = {}

    for symbol, df in market_data.items():
        row_locations[symbol] = {
            timestamp: index
            for index, timestamp in enumerate(df.index)
        }

    for timestamp in timestamps:

        current_rows = {}

        for symbol, df in market_data.items():
            if timestamp in row_locations[symbol]:
                index = row_locations[symbol][timestamp]

                current_rows[symbol] = (
                    df.iloc[index],
                    index,
                )

        # ----------------------------------------------------
        # 1. PROCESS EXITS
        # ----------------------------------------------------

        for symbol in list(positions.keys()):

            if symbol not in current_rows:
                continue

            row, bar_index = current_rows[symbol]

            position = positions[symbol]

            # Do not evaluate the entry bar's high or low.
            # This avoids assuming an intrabar exit can be
            # determined when entry occurred at that same open.
            if timestamp <= position["entry_time"]:
                continue

            high = float(row["high"])
            low = float(row["low"])

            stop_price = position["stop_price"]
            target_price = position["target_price"]

            exit_price = None
            exit_reason = None

            if low <= stop_price:
                exit_price = stop_price
                exit_reason = "stop_loss"

            elif high >= target_price:
                exit_price = target_price
                exit_reason = "take_profit"

            if exit_price is None:
                continue

            # Apply adverse exit slippage.
            execution_price = (
                exit_price * (1 - SLIPPAGE_RATE)
            )

            quantity = position["quantity"]

            gross_proceeds = (
                execution_price * quantity
            )

            exit_fee = gross_proceeds * FEE_RATE

            cash += gross_proceeds - exit_fee

            net_pnl = (
                gross_proceeds
                - exit_fee
                - position["entry_cost"]
                - position["entry_fee"]
            )

            trades.append({
                "symbol": symbol,
                "strategy": strategy,
                "target_pct": target_fraction * 100,
                "entry_time": str(position["entry_time"]),
                "exit_time": str(timestamp),
                "entry_price": position["entry_price"],
                "exit_price": execution_price,
                "quantity": quantity,
                "net_pnl": net_pnl,
                "exit_reason": exit_reason,
            })

            del positions[symbol]

        # ----------------------------------------------------
        # 2. CALCULATE CURRENT EQUITY
        # ----------------------------------------------------

        equity = cash

        for symbol, position in positions.items():
            if symbol in current_rows:
                row, _ = current_rows[symbol]

                equity += (
                    float(row["close"])
                    * position["quantity"]
                )

        # ----------------------------------------------------
        # 3. OPEN NEW POSITIONS
        # ----------------------------------------------------

        for symbol, (row, bar_index) in current_rows.items():

            if symbol in positions:
                continue

            if len(positions) >= MAX_POSITIONS:
                break

            df = market_data[symbol]

            # A previous completed bar is required.
            if bar_index < 1:
                continue

            previous_row = df.iloc[bar_index - 1]

            if not get_signal(previous_row, strategy):
                continue

            raw_entry = float(row["open"])

            if (
                not math.isfinite(raw_entry)
                or raw_entry <= 0
            ):
                continue

            # Adverse entry slippage.
            entry_price = (
                raw_entry * (1 + SLIPPAGE_RATE)
            )

            # Limit each position to a portion of equity.
            allocation = min(
                equity * MAX_ALLOCATION,
                cash / (1 + FEE_RATE),
            )

            if allocation <= 0:
                continue

            quantity = allocation / entry_price

            if quantity <= 0:
                continue

            entry_cost = entry_price * quantity

            entry_fee = entry_cost * FEE_RATE

            total_cost = entry_cost + entry_fee

            if total_cost > cash:
                continue

            cash -= total_cost

            positions[symbol] = {
                "entry_time": timestamp,
                "entry_price": entry_price,
                "quantity": quantity,
                "entry_cost": entry_cost,
                "entry_fee": entry_fee,
                "stop_price": (
                    entry_price * (1 - STOP_LOSS)
                ),
                "target_price": (
                    entry_price * (1 + target_fraction)
                ),
            }

        # ----------------------------------------------------
        # 4. RECORD EQUITY
        # ----------------------------------------------------

        equity = cash

        for symbol, position in positions.items():
            if symbol in current_rows:
                row, _ = current_rows[symbol]

                equity += (
                    float(row["close"])
                    * position["quantity"]
                )

        equity_history.append({
            "timestamp": str(timestamp),
            "equity": equity,
        })

    # --------------------------------------------------------
    # 5. LIQUIDATE REMAINING POSITIONS AT LAST AVAILABLE CLOSE
    # --------------------------------------------------------

    for symbol in list(positions.keys()):

        position = positions[symbol]

        df = market_data[symbol]

        final_row = df.iloc[-1]

        raw_exit = float(final_row["close"])

        execution_price = (
            raw_exit * (1 - SLIPPAGE_RATE)
        )

        quantity = position["quantity"]

        proceeds = execution_price * quantity

        exit_fee = proceeds * FEE_RATE

        cash += proceeds - exit_fee

        net_pnl = (
            proceeds
            - exit_fee
            - position["entry_cost"]
            - position["entry_fee"]
        )

        trades.append({
            "symbol": symbol,
            "strategy": strategy,
            "target_pct": target_fraction * 100,
            "entry_time": str(position["entry_time"]),
            "exit_time": str(df.index[-1]),
            "entry_price": position["entry_price"],
            "exit_price": execution_price,
            "quantity": quantity,
            "net_pnl": net_pnl,
            "exit_reason": "end_of_test",
        })

        del positions[symbol]

    # --------------------------------------------------------
    # 6. CALCULATE PERFORMANCE
    # --------------------------------------------------------

    net_profit = cash - starting_capital

    return_pct = (
        net_profit / starting_capital
    ) * 100

    trade_df = pd.DataFrame(trades)

    if not trade_df.empty:

        wins = trade_df[
            trade_df["net_pnl"] > 0
        ]

        losses = trade_df[
            trade_df["net_pnl"] < 0
        ]

        win_rate = (
            len(wins) / len(trade_df)
        ) * 100

        gross_wins = wins["net_pnl"].sum()

        gross_losses = abs(
            losses["net_pnl"].sum()
        )

        if gross_losses > 0:
            profit_factor = (
                gross_wins / gross_losses
            )
        elif gross_wins > 0:
            profit_factor = None
        else:
            profit_factor = 0.0

    else:
        win_rate = 0.0
        profit_factor = 0.0

    equity_df = pd.DataFrame(equity_history)

    if not equity_df.empty:

        curve = equity_df["equity"].astype(float)

        peaks = curve.cummax()

        drawdowns = (
            (peaks - curve)
            / peaks.replace(0, np.nan)
        ) * 100

        max_drawdown = float(
            drawdowns.max()
        )

        if not math.isfinite(max_drawdown):
            max_drawdown = 0.0

    else:
        max_drawdown = 0.0

    return {
        "strategy": strategy,
        "target": target_fraction,
        "target_pct": target_fraction * 100,
        "starting_capital": starting_capital,
        "ending_equity": cash,
        "return_pct": return_pct,
        "trades": len(trades),
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown,
        "net_profit": net_profit,
        "trade_details": trades,
    }


# ============================================================
# TIME-ORDERED TRAINING / VALIDATION / TEST
# ============================================================

def split_data(data, start_fraction, end_fraction):
    """
    Split data chronologically, never randomly.
    """

    result = {}

    for symbol, df in data.items():

        size = len(df)

        start = int(size * start_fraction)
        end = int(size * end_fraction)

        part = df.iloc[start:end].copy()

        if len(part) >= 30:
            result[symbol] = part

    return result


# ============================================================
# RESEARCH ENGINE
# ============================================================

def research(
    symbols,
    period,
    interval,
    capital,
    output_dir,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    downloaded = {}

    for symbol in symbols:

        df = download_data(
            symbol,
            period,
            interval,
        )

        if df is None:
            continue

        downloaded[symbol] = add_indicators(df)

    if not downloaded:
        raise RuntimeError(
            "No usable market data was downloaded. "
            "Check the GitHub Actions logs, network access, "
            "Yahoo Finance availability, and yfinance version."
        )

    logger.info(
        "Researching %s instruments",
        len(downloaded),
    )

    # Chronological splits.
    training_data = split_data(
        downloaded,
        0.0,
        0.60,
    )

    validation_data = split_data(
        downloaded,
        0.60,
        0.80,
    )

    test_data = split_data(
        downloaded,
        0.80,
        1.0,
    )

    if not training_data:
        raise RuntimeError(
            "Not enough data for training."
        )

    if not validation_data:
        raise RuntimeError(
            "Not enough data for validation."
        )

    if not test_data:
        raise RuntimeError(
            "Not enough data for final testing."
        )

    results = []

    combinations = list(
        product(
            STRATEGIES,
            PROFIT_TARGETS,
        )
    )

    for strategy, target in combinations:

        logger.info(
            "Testing strategy=%s target=%.3f%%",
            strategy,
            target * 100,
        )

        training_result = run_backtest(
            training_data,
            strategy,
            target,
            capital,
        )

        validation_result = run_backtest(
            validation_data,
            strategy,
            target,
            capital,
        )

        test_result = run_backtest(
            test_data,
            strategy,
            target,
            capital,
        )

        # Strategy selection must be based on training and
        # validation, not on the final holdout test.
        acceptable = (
            training_result["trades"] >= 10
            and validation_result["trades"] >= 5
            and training_result["return_pct"] > 0
            and validation_result["return_pct"] > 0
            and validation_result["max_drawdown_pct"] < 20
            and (
                validation_result["profit_factor"] is None
                or validation_result["profit_factor"] > 1
            )
        )

        results.append({
            "strategy": strategy,
            "target_pct": target * 100,
            "training_return_pct": (
                training_result["return_pct"]
            ),
            "training_trades": (
                training_result["trades"]
            ),
            "validation_return_pct": (
                validation_result["return_pct"]
            ),
            "validation_trades": (
                validation_result["trades"]
            ),
            "validation_win_rate_pct": (
                validation_result["win_rate_pct"]
            ),
            "validation_profit_factor": (
                validation_result["profit_factor"]
            ),
            "validation_max_drawdown_pct": (
                validation_result["max_drawdown_pct"]
            ),
            "final_test_return_pct": (
                test_result["return_pct"]
            ),
            "final_test_net_profit": (
                test_result["net_profit"]
            ),
            "final_test_trades": (
                test_result["trades"]
            ),
            "final_test_win_rate_pct": (
                test_result["win_rate_pct"]
            ),
            "final_test_profit_factor": (
                test_result["profit_factor"]
            ),
            "final_test_max_drawdown_pct": (
                test_result["max_drawdown_pct"]
            ),
            "passes_training_validation_gates": (
                acceptable
            ),
            "test_trade_details": (
                test_result["trade_details"]
            ),
        })

    results_df = pd.DataFrame(results)

    # Keep the final test separate from the selection decision.
    results_df = results_df.sort_values(
        by=[
            "passes_training_validation_gates",
            "validation_return_pct",
        ],
        ascending=[False, False],
    )

    results_path = (
        output_dir / "strategy_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # Export detailed trades from the holdout test.
    all_trades = []

    for result in results:
        for trade in result["test_trade_details"]:
            all_trades.append(trade)

    trades_path = (
        output_dir / "test_trades.csv"
    )

    pd.DataFrame(all_trades).to_csv(
        trades_path,
        index=False,
    )

    # Export a machine-readable report.
    report = {
        "symbols_downloaded": list(downloaded.keys()),
        "period": period,
        "interval": interval,
        "starting_capital": capital,
        "max_simultaneous_positions": MAX_POSITIONS,
        "max_position_allocation_pct": (
            MAX_ALLOCATION * 100
        ),
        "estimated_fee_per_side_pct": (
            FEE_RATE * 100
        ),
        "estimated_slippage_per_side_pct": (
            SLIPPAGE_RATE * 100
        ),
        "stop_loss_pct": STOP_LOSS * 100,
        "strategy_combinations_tested": len(results),
        "results": results,
    }

    # Avoid JSON errors from NaN or Infinity.
    def clean_json(value):
        if isinstance(value, dict):
            return {
                key: clean_json(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                clean_json(item)
                for item in value
            ]

        if isinstance(value, (np.integer,)):
            return int(value)

        if isinstance(value, (np.floating, float)):
            value = float(value)

            if not math.isfinite(value):
                return None

            return value

        if isinstance(value, (np.bool_,)):
            return bool(value)

        return value

    report_path = (
        output_dir / "research_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            clean_json(report),
            file,
            indent=2,
            allow_nan=False,
        )

    print("\n" + "=" * 65)
    print("MICRO-PROFIT RESEARCH RESULTS")
    print("=" * 65)

    display_columns = [
        "strategy",
        "target_pct",
        "validation_return_pct",
        "validation_trades",
        "final_test_return_pct",
        "final_test_trades",
        "passes_training_validation_gates",
    ]

    print(
        results_df[display_columns].to_string(
            index=False
        )
    )

    print("\nFiles created:")
    print(f"Results: {results_path}")
    print(f"Trades:  {trades_path}")
    print(f"Report:  {report_path}")

    print(
        "\nIMPORTANT: These are historical simulations, "
        "not guarantees of future performance."
    )

    print(
        "The final test results must not be used repeatedly "
        "to tune strategies, or the holdout test loses value."
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Historical micro-profit strategy research engine"
        )
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Symbols to research",
    )

    parser.add_argument(
        "--period",
        default="1y",
        help="Yahoo Finance history period, e.g. 6mo or 1y",
    )

    parser.add_argument(
        "--interval",
        default="1h",
        help="Yahoo Finance interval, e.g. 1h or 1d",
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=STARTING_CAPITAL,
        help="Starting portfolio capital",
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Output directory",
    )

    args = parser.parse_args()

    if args.capital <= 0:
        parser.error("--capital must be greater than zero")

    research(
        symbols=args.symbols,
        period=args.period,
        interval=args.interval,
        capital=args.capital,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
