#!/usr/bin/env python3
"""Historical-only stock strategy backtest. Does not place live orders."""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

SYMBOLS = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA",
    "AMZN", "META", "GOOGL", "TSLA", "AMD", "PLTR",
]

LOOKBACK_CALENDAR_DAYS = int(os.getenv("BACKTEST_DAYS", "180"))

# Never hold more than three positions simultaneously.
MAX_POSITIONS = 3

# Compare the strategy against these account sizes.
CAPITAL_SCENARIOS = [500.0, 1000.0, 2500.0, 5000.0]

POSITION_ALLOCATION_FRACTION = 1.0 / MAX_POSITIONS

# Estimated slippage per side, expressed in basis points.
SLIPPAGE_BPS_PER_SIDE = 5.0

COMMISSION_PER_SHARE = 0.0
FRACTIONAL_SHARES = True

TOP_TRAIN_CANDIDATES = 10
MIN_TRAIN_TRADES = 5
MIN_VALIDATION_TRADES = 3
MAX_VALIDATION_DRAWDOWN = 0.15

DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")
API_BASE = "https://data.alpaca.markets/v2/stocks/bars"

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger("historical_backtest")

PARAMETER_GRID = {
    "fast_ema": [5, 9],
    "slow_ema": [13, 21],
    "breakout_window": [10],
    "volume_multiplier": [1.2, 1.5],
    "stop_loss": [0.0075, 0.01],
    "take_profit": [0.01, 0.015],
    "max_hold_bars": [30, 60],
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def parameter_grid() -> list[dict[str, Any]]:
    keys = list(PARAMETER_GRID)

    combinations = itertools.product(
        *(PARAMETER_GRID[key] for key in keys)
    )

    result = [
        dict(zip(keys, values))
        for values in combinations
    ]

    # Only use a fast EMA that is shorter than the slow EMA.
    return [
        params
        for params in result
        if params["fast_ema"] < params["slow_ema"]
    ]


# ============================================================
# ALPACA HISTORICAL DATA
# ============================================================

def alpaca_headers() -> dict[str, str]:
    key = os.getenv("APCA_API_KEY_ID", "").strip()
    secret = os.getenv("APCA_API_SECRET_KEY", "").strip()

    if not key or not secret:
        raise RuntimeError(
            "Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY. "
            "Add both as GitHub Actions repository secrets."
        )

    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }


def download_bars(
    symbols: list[str],
    start: datetime,
    end: datetime,
) -> dict[str, pd.DataFrame]:
    """Download historical one-minute stock bars from Alpaca."""

    headers = alpaca_headers()

    params: dict[str, Any] = {
        "symbols": ",".join(symbols),
        "timeframe": "1Min",
        "start": start.astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "end": end.astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "limit": 10000,
        "adjustment": "split",
        "feed": DATA_FEED,
        "sort": "asc",
    }

    collected: dict[str, list[dict[str, Any]]] = {
        symbol: [] for symbol in symbols
    }

    page_token = None
    page_count = 0

    with requests.Session() as session:
        while True:
            query = dict(params)

            if page_token:
                query["page_token"] = page_token

            response = session.get(
                API_BASE,
                headers=headers,
                params=query,
                timeout=90,
            )

            if response.status_code == 429:
                log.warning(
                    "Alpaca rate limit reached. Waiting 3 seconds."
                )
                time.sleep(3)
                continue

            if response.status_code >= 500:
                log.warning(
                    "Alpaca server error %s. Retrying in 3 seconds.",
                    response.status_code,
                )
                time.sleep(3)
                continue

            if not response.ok:
                raise RuntimeError(
                    "Alpaca data request failed "
                    f"({response.status_code}): "
                    f"{response.text[:1000]}"
                )

            payload = response.json()
            bars = payload.get("bars", {}) or {}

            for symbol in symbols:
                collected[symbol].extend(
                    bars.get(symbol, []) or []
                )

            page_count += 1
            page_token = payload.get("next_page_token")

            total_bars = sum(
                len(items) for items in collected.values()
            )

            log.info(
                "Downloaded page %s; bars collected: %s",
                page_count,
                total_bars,
            )

            if not page_token:
                break

            if page_count >= 500:
                raise RuntimeError(
                    "Stopped after 500 pages to prevent "
                    "an endless pagination loop."
                )

    result: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        rows = collected[symbol]

        if not rows:
            log.warning("No historical bars returned for %s.", symbol)
            continue

        frame = pd.DataFrame(rows).rename(
            columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
        )

        required = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        missing = [
            column for column in required
            if column not in frame.columns
        ]

        if missing:
            log.warning(
                "%s is missing columns: %s",
                symbol,
                missing,
            )
            continue

        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            utc=True,
            errors="coerce",
        )

        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
        ):
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

        frame = frame.dropna(subset=required)

        frame = (
            frame.sort_values("timestamp")
            .drop_duplicates("timestamp")
            .reset_index(drop=True)
        )

        if len(frame) < 100:
            log.warning(
                "Only %s bars for %s; skipping symbol.",
                len(frame),
                symbol,
            )
            continue

        frame["symbol"] = symbol
        result[symbol] = frame

        log.info(
            "Loaded %s: %s bars, from %s through %s",
            symbol,
            len(frame),
            frame["timestamp"].min(),
            frame["timestamp"].max(),
        )

    if not result:
        raise RuntimeError(
            "No usable historical bars downloaded. "
            "Check Alpaca credentials, data permissions, "
            "date range, and feed settings."
        )

    return result


# ============================================================
# STRATEGY INDICATORS
# ============================================================

def add_signals(
    data: dict[str, pd.DataFrame],
    params: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Calculate indicators without using future information."""

    prepared: dict[str, pd.DataFrame] = {}

    for symbol, source in data.items():
        frame = source.copy()

        frame["ema_fast"] = frame["close"].ewm(
            span=int(params["fast_ema"]),
            adjust=False,
        ).mean()

        frame["ema_slow"] = frame["close"].ewm(
            span=int(params["slow_ema"]),
            adjust=False,
        ).mean()

        window = int(params["breakout_window"])

        frame["prior_high"] = (
            frame["high"]
            .rolling(window)
            .max()
            .shift(1)
        )

        frame["prior_volume_avg"] = (
            frame["volume"]
            .rolling(20)
            .mean()
            .shift(1)
        )

        frame["entry_signal"] = (
            (frame["close"] > frame["prior_high"])
            & (frame["ema_fast"] > frame["ema_slow"])
            & (
                frame["volume"]
                >= frame["prior_volume_avg"]
                * float(params["volume_multiplier"])
            )
        ).fillna(False)

        # Enter at the next bar's open, not at the signal bar's close.
        frame["enter_now"] = (
            frame["entry_signal"]
            .shift(1)
            .fillna(False)
            .astype(bool)
        )

        prepared[symbol] = frame

    return prepared


# ============================================================
# POSITION MODEL
# ============================================================

@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_time: pd.Timestamp
    entry_index: int
    stop_price: float
    target_price: float
    entry_cost: float


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_simulation(
    data: dict[str, pd.DataFrame],
    params: dict[str, Any],
    starting_capital: float,
) -> dict[str, Any]:
    """Simulate long positions, limiting exposure to three stocks."""

    if starting_capital <= 0:
        raise ValueError("Starting capital must be greater than zero.")

    if not data:
        return {
            "starting_capital": starting_capital,
            "ending_equity": starting_capital,
            "return_pct": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "net_profit": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "trades_detail": [],
        }

    events: dict[
        pd.Timestamp,
        list[tuple[str, int, pd.Series]]
    ] = {}

    for symbol, frame in data.items():
        for index, row in enumerate(
            frame.itertuples(index=False)
        ):
            timestamp = getattr(row, "timestamp")
            series = pd.Series(row._asdict())

            events.setdefault(timestamp, []).append(
                (symbol, index, series)
            )

    cash = float(starting_capital)

    positions: dict[str, Position] = {}
    closed_trades: list[dict[str, Any]] = []
    equity_curve: list[tuple[pd.Timestamp, float]] = []
    last_prices: dict[str, float] = {}

    slippage = SLIPPAGE_BPS_PER_SIDE / 10000.0

    def mark_equity() -> float:
        market_value = sum(
            position.qty
            * last_prices.get(symbol, position.entry_price)
            for symbol, position in positions.items()
        )
        return cash + market_value

    def close_position(
        symbol: str,
        timestamp: pd.Timestamp,
        raw_price: float,
        reason: str,
    ) -> None:
        nonlocal cash

        position = positions.pop(symbol)

        exit_price = max(
            0.0001,
            raw_price * (1.0 - slippage),
        )

        proceeds = (
            position.qty * exit_price
            - position.qty * COMMISSION_PER_SHARE
        )

        cash += proceeds
        pnl = proceeds - position.entry_cost

        closed_trades.append(
            {
                "symbol": symbol,
                "entry_time": position.entry_time,
                "exit_time": timestamp,
                "qty": position.qty,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "reason": reason,
            }
        )

    for timestamp in sorted(events):
        current = {
            symbol: (index, row)
            for symbol, index, row in events[timestamp]
        }

        for symbol, (_, row) in current.items():
            last_prices[symbol] = safe_float(
                row.get("close"),
                last_prices.get(symbol, 0.0),
            )

        # Manage open positions before considering new entries.
        # If both stop and target are hit during one bar,
        # conservatively assume the stop was hit first.
        for symbol in list(positions):
            if symbol not in current:
                continue

            index, row = current[symbol]
            position = positions[symbol]

            open_price = safe_float(
                row.get("open"),
                position.entry_price,
            )
            low = safe_float(row.get("low"), open_price)
            high = safe_float(row.get("high"), open_price)
            close = safe_float(row.get("close"), open_price)

            held_bars = index - position.entry_index

            if open_price <= position.stop_price:
                close_position(
                    symbol, timestamp, open_price, "stop_gap"
                )

            elif low <= position.stop_price:
                close_position(
                    symbol, timestamp,
                    position.stop_price, "stop_loss"
                )

            elif open_price >= position.target_price:
                close_position(
                    symbol, timestamp, open_price, "target_gap"
                )

            elif high >= position.target_price:
                close_position(
                    symbol, timestamp,
                    position.target_price, "take_profit"
                )

            elif held_bars >= int(params["max_hold_bars"]):
                close_position(
                    symbol, timestamp, close, "max_hold"
                )

            elif (
                safe_float(row.get("ema_fast"))
                <= safe_float(row.get("ema_slow"))
            ):
                close_position(
                    symbol, timestamp, close, "ema_exit"
                )

        # New positions use the previous completed bar's signal.
        for symbol, (_, row) in current.items():
            if symbol in positions:
                continue

            if len(positions) >= MAX_POSITIONS:
                break

            if not bool(row.get("enter_now", False)):
                continue

            price = safe_float(row.get("open"))

            if price <= 0 or cash <= 1:
                continue

            equity = mark_equity()

            allocation = min(
                equity * POSITION_ALLOCATION_FRACTION,
                cash,
            )

            entry_price = price * (1.0 + slippage)

            if FRACTIONAL_SHARES:
                quantity = allocation / entry_price
            else:
                quantity = math.floor(allocation / entry_price)

            if quantity <= 0:
                continue

            cost = (
                quantity * entry_price
                + quantity * COMMISSION_PER_SHARE
            )

            # Protect against rounding exceeding available cash.
            while cost > cash and quantity > 0:
                if FRACTIONAL_SHARES:
                    quantity *= 0.999
                else:
                    quantity -= 1

                cost = (
                    quantity * entry_price
                    + quantity * COMMISSION_PER_SHARE
                )

            if quantity <= 1e-10 or cost > cash:
                continue

            cash -= cost

            positions[symbol] = Position(
                symbol=symbol,
                qty=quantity,
                entry_price=entry_price,
                entry_time=timestamp,
                entry_index=current[symbol][0],
                stop_price=(
                    entry_price
                    * (1.0 - float(params["stop_loss"]))
                ),
                target_price=(
                    entry_price
                    * (1.0 + float(params["take_profit"]))
                ),
                entry_cost=cost,
            )

        equity_curve.append((timestamp, mark_equity()))

    # Close any remaining positions using each stock's last close.
    for symbol in list(positions):
        last_bar = data[symbol].iloc[-1]

        close_position(
            symbol,
            last_bar["timestamp"],
            safe_float(last_bar["close"]),
            "end_of_data",
        )

    final_equity = cash

    pnls = [
        safe_float(trade["pnl"])
        for trade in closed_trades
    ]

    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [-pnl for pnl in pnls if pnl < 0]

    gross_profit = sum(wins)
    gross_loss = sum(losses)

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = 99.0
    else:
        profit_factor = 0.0

    curve = pd.Series(
        [value for _, value in equity_curve] + [final_equity],
        dtype=float,
    )

    running_peak = curve.cummax()

    drawdowns = (
        (running_peak - curve)
        / running_peak.replace(0, float("nan"))
    )

    max_drawdown = safe_float(drawdowns.max())

    return {
        "starting_capital": float(starting_capital),
        "ending_equity": float(final_equity),
        "net_profit": float(final_equity - starting_capital),
        "return_pct": (
            float(final_equity / starting_capital - 1.0)
            if starting_capital
            else 0.0
        ),
        "trades": len(closed_trades),
        "win_rate": (
            len(wins) / len(pnls)
            if pnls
            else 0.0
        ),
        "profit_factor": float(profit_factor),
        "max_drawdown_pct": float(max_drawdown),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "trades_detail": closed_trades,
    }


# ============================================================
# VALIDATION AND REPORTING
# ============================================================

def validation_passes(result: dict[str, Any]) -> bool:
    return (
        int(result.get("trades", 0)) >= MIN_VALIDATION_TRADES
        and safe_float(result.get("return_pct")) > 0
        and safe_float(result.get("profit_factor")) > 1.0
        and abs(
            safe_float(result.get("max_drawdown_pct"))
        ) <= MAX_VALIDATION_DRAWDOWN
    )


def save_trades(
    path: Path,
    trades: list[dict[str, Any]],
) -> None:
    frame = pd.DataFrame(trades)

    if not frame.empty:
        for column in ("entry_time", "exit_time"):
            frame[column] = pd.to_datetime(
                frame[column],
                utc=True,
                errors="coerce",
            ).astype(str)

    frame.to_csv(path, index=False)


# ============================================================
# STRATEGY OPTIMIZATION
# ============================================================

def optimize_strategy(
    train: dict[str, pd.DataFrame],
    validation: dict[str, pd.DataFrame],
) -> dict[str, Any] | None:

    candidates = parameter_grid()

    log.info("=" * 65)
    log.info("DIAGNOSTIC STRATEGY OPTIMIZATION")
    log.info("Candidates to evaluate: %s", len(candidates))
    log.info("Maximum simultaneous positions: %s", MAX_POSITIONS)
    log.info("Capital scenarios: %s", CAPITAL_SCENARIOS)
    log.info("=" * 65)

    training_results: list[dict[str, Any]] = []

    for index, params in enumerate(candidates, start=1):
        try:
            signals = add_signals(train, params)

            result = run_simulation(
                signals,
                params,
                starting_capital=5000.0,
            )

            trades = int(result["trades"])
            ret = safe_float(result["return_pct"])
            profit_factor = safe_float(result["profit_factor"])
            drawdown = abs(
                safe_float(result["max_drawdown_pct"])
            )

            score = (
                ret
                - 0.75 * drawdown
                + 0.02 * min(profit_factor, 3.0)
            )

            training_results.append(
                {
                    "params": params,
                    "diagnostic_score": score,
                    "train_return": ret,
                    "train_trades": trades,
                    "train_win_rate": safe_float(
                        result["win_rate"]
                    ),
                    "train_profit_factor": profit_factor,
                    "train_drawdown": drawdown,
                }
            )

        except Exception:
            log.exception(
                "Training candidate %s failed: %s",
                index,
                params,
            )

        if index % 5 == 0 or index == len(candidates):
            log.info(
                "Training progress: %s/%s",
                index,
                len(candidates),
            )

    if not training_results:
        raise RuntimeError(
            "All training candidates failed. "
            "Review the exception logs above."
        )

    training_results.sort(
        key=lambda item: (
            item["diagnostic_score"],
            item["train_trades"],
        ),
        reverse=True,
    )

    train_rows = [
        {
            **item,
            "params": json.dumps(
                item["params"],
                sort_keys=True,
            ),
        }
        for item in training_results
    ]

    pd.DataFrame(train_rows).to_csv(
        OUTPUT_DIR / "training_results.csv",
        index=False,
    )

    log.info(
        "Saved %s training rows.",
        len(train_rows),
    )

    log.info(
        "TOP TRAINING CANDIDATES — RANKING IS NOT PROOF OF PROFIT"
    )

    for rank, item in enumerate(training_results[:10], start=1):
        log.info(
            "Rank %s | return %.2f%% | trades %s | "
            "win %.1f%% | PF %.3f | DD %.2f%% | "
            "score %.5f | %s",
            rank,
            100 * item["train_return"],
            item["train_trades"],
            100 * item["train_win_rate"],
            item["train_profit_factor"],
            100 * item["train_drawdown"],
            item["diagnostic_score"],
            item["params"],
        )

    finalists = [
        item
        for item in training_results
        if item["train_trades"] >= MIN_TRAIN_TRADES
    ][:TOP_TRAIN_CANDIDATES]

    if not finalists:
        log.error(
            "No training candidate generated at least %s trades. "
            "Training CSV was saved; nothing will be promoted.",
            MIN_TRAIN_TRADES,
        )
        return None

    validation_results: list[dict[str, Any]] = []

    for index, item in enumerate(finalists, start=1):
        params = item["params"]

        try:
            result = run_simulation(
                add_signals(validation, params),
                params,
                starting_capital=5000.0,
            )

            passed = validation_passes(result)

            row = {
                "params": params,
                "passed_validation": passed,
                "validation_return": result["return_pct"],
                "validation_net_profit": result["net_profit"],
                "validation_trades": result["trades"],
                "validation_win_rate": result["win_rate"],
                "validation_profit_factor": result["profit_factor"],
                "validation_drawdown": result["max_drawdown_pct"],
                "training_score": item["diagnostic_score"],
                "training_return": item["train_return"],
                "training_trades": item["train_trades"],
                "training_profit_factor": item["train_profit_factor"],
            }

            validation_results.append(row)

            log.info(
                "Validation %s/%s | pass=%s | return=%.2f%% | "
                "trades=%s | PF=%.3f | DD=%.2f%%",
                index,
                len(finalists),
                passed,
                100 * row["validation_return"],
                row["validation_trades"],
                row["validation_profit_factor"],
                100 * row["validation_drawdown"],
            )

        except Exception:
            log.exception(
                "Validation candidate failed: %s",
                params,
            )

    if validation_results:
        validation_rows = [
            {
                **row,
                "params": json.dumps(
                    row["params"],
                    sort_keys=True,
                ),
            }
            for row in validation_results
        ]

        pd.DataFrame(validation_rows).to_csv(
            OUTPUT_DIR / "validation_results.csv",
            index=False,
        )

        log.info("Saved validation results.")

    accepted = [
        row
        for row in validation_results
        if row["passed_validation"]
    ]

    log.info(
        "Candidates passing validation: %s of %s",
        len(accepted),
        len(validation_results),
    )

    if not accepted:
        log.warning(
            "NO STRATEGY PASSED VALIDATION. "
            "No candidate will be promoted. "
            "Diagnostic CSV files remain available."
        )
        return None

    accepted.sort(
        key=lambda row: (
            row["validation_return"]
            - 0.75 * abs(row["validation_drawdown"])
        ),
        reverse=True,
    )

    selected = accepted[0]
    selected_params = selected["params"]

    selected_file = OUTPUT_DIR / "selected_strategy.json"

    selected_file.write_text(
        json.dumps(
            {
                "params": selected_params,
                "validation_metrics": {
                    key: value
                    for key, value in selected.items()
                    if key != "params"
                },
                "note": (
                    "Historical research only. "
                    "Past performance does not guarantee future results."
                ),
            },
            indent=2,
            default=str,
        )
    )

    return selected


# ============================================================
# MAIN PROGRAM
# ============================================================

def main() -> int:
    log.info(
        "Historical backtest starting. NO LIVE ORDERS WILL BE SUBMITTED."
    )

    log.info("Symbols: %s", ", ".join(SYMBOLS))

    end = datetime.now(timezone.utc).replace(
        second=0,
        microsecond=0,
    )

    start = end - timedelta(
        days=LOOKBACK_CALENDAR_DAYS
    )

    log.info(
        "Downloading one-minute bars from %s to %s UTC; feed=%s",
        start.isoformat(),
        end.isoformat(),
        DATA_FEED,
    )

    raw_data = download_bars(
        SYMBOLS,
        start,
        end,
    )

    all_timestamps = sorted(
        {
            stamp
            for frame in raw_data.values()
            for stamp in frame["timestamp"].tolist()
        }
    )

    if len(all_timestamps) < 1000:
        raise RuntimeError(
            f"Only {len(all_timestamps)} distinct timestamps found. "
            "There is not enough history for this test."
        )

    # Chronological splits: 60% training, 20% validation, 20% final test.
    count = len(all_timestamps)

    train_cut = all_timestamps[
        max(1, int(count * 0.60)) - 1
    ]

    validation_cut = all_timestamps[
        max(2, int(count * 0.80)) - 1
    ]

    train = {
        symbol: frame[
            frame["timestamp"] <= train_cut
        ].copy().reset_index(drop=True)
        for symbol, frame in raw_data.items()
    }

    validation = {
        symbol: frame[
            (frame["timestamp"] > train_cut)
            & (frame["timestamp"] <= validation_cut)
        ].copy().reset_index(drop=True)
        for symbol, frame in raw_data.items()
    }

    test = {
        symbol: frame[
            frame["timestamp"] > validation_cut
        ].copy().reset_index(drop=True)
        for symbol, frame in raw_data.items()
    }

    # Remove symbols without enough data in each period.
    train = {
        symbol: frame
        for symbol, frame in train.items()
        if len(frame) >= 100
    }

    validation = {
        symbol: frame
        for symbol, frame in validation.items()
        if len(frame) >= 30 and symbol in train
    }

    test = {
        symbol: frame
        for symbol, frame in test.items()
        if len(frame) >= 30
        and symbol in train
        and symbol in validation
    }

    log.info(
        "Split boundaries: train through %s; validation through %s; "
        "final test through %s",
        train_cut,
        validation_cut,
        max(all_timestamps),
    )

    log.info(
        "Usable symbols: training=%s, validation=%s, final test=%s",
        len(train),
        len(validation),
        len(test),
    )

    if not train or not validation or not test:
        raise RuntimeError(
            "One or more chronological data splits are empty. "
            "Increase BACKTEST_DAYS or check the downloaded data."
        )

    selected = optimize_strategy(
        train,
        validation,
    )

    if selected is None:
        log.warning(
            "Final holdout test skipped because no strategy passed "
            "validation. Training and validation reports were saved."
        )

    else:
        params = selected["params"]

        log.info(
            "Running final holdout test on the validation-approved strategy."
        )

        capital_rows = []

        for capital in CAPITAL_SCENARIOS:
            result = run_simulation(
                add_signals(test, params),
                params,
                starting_capital=capital,
            )

            row = {
                key: value
                for key, value in result.items()
                if key != "trades_detail"
            }

            row["params"] = json.dumps(
                params,
                sort_keys=True,
            )

            capital_rows.append(row)

            log.info(
                "FINAL TEST | capital=$%.2f | ending=$%.2f | "
                "net=$%.2f | return=%.2f%% | trades=%s | "
                "win=%.1f%% | PF=%.3f | max DD=%.2f%%",
                capital,
                result["ending_equity"],
                result["net_profit"],
                100 * result["return_pct"],
                result["trades"],
                100 * result["win_rate"],
                result["profit_factor"],
                100 * result["max_drawdown_pct"],
            )

            save_trades(
                OUTPUT_DIR / f"final_test_trades_{int(capital)}.csv",
                result["trades_detail"],
            )

        pd.DataFrame(capital_rows).to_csv(
            OUTPUT_DIR / "capital_scenarios.csv",
            index=False,
        )

        log.info("Capital scenario results saved.")

    summary = {
        "symbols_requested": SYMBOLS,
        "symbols_downloaded": list(raw_data),
        "bars_downloaded": {
            symbol: len(frame)
            for symbol, frame in raw_data.items()
        },
        "train_cut_utc": str(train_cut),
        "validation_cut_utc": str(validation_cut),
        "test_end_utc": str(max(all_timestamps)),
        "selected_strategy": (
            selected["params"] if selected else None
        ),
        "validation_passed": selected is not None,
        "max_positions": MAX_POSITIONS,
        "fractional_shares": FRACTIONAL_SHARES,
        "capital_scenarios": CAPITAL_SCENARIOS,
        "live_orders_submitted": False,
    }

    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
    )

    log.info("Run summary saved.")
    log.info(
        "Backtest finished. Results directory: %s",
        OUTPUT_DIR.resolve(),
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log.exception("BACKTEST FAILED: %s", exc)
        sys.exit(1)
