#!/usr/bin/env python3
"""Historical-only stock strategy backtester.

Downloads historical Alpaca data, evaluates trading strategies,
saves CSV reports, and tests multiple starting capital amounts.

NO LIVE BROKERAGE ORDERS ARE SUBMITTED.
"""

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

MAX_POSITIONS = 3

CAPITAL_SCENARIOS = [
    500.0,
    1000.0,
    2500.0,
    5000.0,
]

POSITION_ALLOCATION_FRACTION = 1.0 / MAX_POSITIONS

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


# ============================================================
# STRATEGY PARAMETERS
# ============================================================

PARAMETER_GRID = {
    "fast_ema": [5, 9],
    "slow_ema": [13, 21],
    "breakout_window": [10],
    "volume_multiplier": [1.2, 1.5],
    "stop_loss": [0.0075, 0.01],
    "take_profit": [0.01, 0.015],
    "max_hold_bars": [30, 60],
}


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to a finite float safely."""
    try:
        number = float(value)

        if math.isfinite(number):
            return number

        return default

    except (TypeError, ValueError):
        return default


def parameter_grid() -> list[dict[str, Any]]:
    """Create all valid parameter combinations."""
    keys = list(PARAMETER_GRID)

    combinations = itertools.product(
        *(PARAMETER_GRID[key] for key in keys)
    )

    candidates = [
        dict(zip(keys, values))
        for values in combinations
    ]

    # Fast EMA must be shorter than slow EMA.
    candidates = [
        params
        for params in candidates
        if params["fast_ema"] < params["slow_ema"]
    ]

    log.info(
        "Generated %s valid strategy combinations.",
        len(candidates),
    )

    return candidates


# ============================================================
# ALPACA HISTORICAL DATA
# ============================================================

def alpaca_headers() -> dict[str, str]:
    """Read Alpaca credentials from environment variables."""
    key = os.getenv("APCA_API_KEY_ID", "").strip()
    secret = os.getenv("APCA_API_SECRET_KEY", "").strip()

    if not key or not secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Configure "
            "APCA_API_KEY_ID and APCA_API_SECRET_KEY "
            "under GitHub Settings > Secrets and variables "
            "> Actions."
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
    """Download paginated historical one-minute bars."""
    headers = alpaca_headers()

    params = {
        "symbols": ",".join(symbols),
        "timeframe": "1Min",
        "start": start.astimezone(timezone.utc).isoformat()
        .replace("+00:00", "Z"),
        "end": end.astimezone(timezone.utc).isoformat()
        .replace("+00:00", "Z"),
        "limit": 10000,
        "adjustment": "split",
        "feed": DATA_FEED,
        "sort": "asc",
    }

    collected = {symbol: [] for symbol in symbols}

    page_token = None
    pages = 0

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
                    "Alpaca rate limit reached. "
                    "Waiting before retrying."
                )
                time.sleep(3)
                continue

            if response.status_code >= 500:
                log.warning(
                    "Alpaca server error %s. Retrying.",
                    response.status_code,
                )
                time.sleep(3)
                continue

            if not response.ok:
                raise RuntimeError(
                    "Alpaca historical data request failed: "
                    f"HTTP {response.status_code}: "
                    f"{response.text[:1000]}"
                )

            payload = response.json()
            bars = payload.get("bars", {}) or {}

            for symbol in symbols:
                collected[symbol].extend(
                    bars.get(symbol, []) or []
                )

            pages += 1

            log.info(
                "Downloaded page %s. Total bars: %s",
                pages,
                sum(len(values) for values in collected.values()),
            )

            page_token = payload.get("next_page_token")

            if not page_token:
                break

            if pages >= 500:
                raise RuntimeError(
                    "Stopped after 500 data pages to prevent "
                    "an infinite pagination loop."
                )

    result = {}

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

        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            utc=True,
            errors="coerce",
        )

        for column in [
            "open", "high", "low", "close", "volume"
        ]:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

        frame = frame.dropna(
            subset=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

        frame = (
            frame.sort_values("timestamp")
            .drop_duplicates("timestamp")
            .reset_index(drop=True)
        )

        if len(frame) < 100:
            log.warning(
                "Skipping %s: only %s bars available.",
                symbol,
                len(frame),
            )
            continue

        frame["symbol"] = symbol

        result[symbol] = frame

        log.info(
            "Loaded %s: %s bars, %s through %s",
            symbol,
            len(frame),
            frame["timestamp"].min(),
            frame["timestamp"].max(),
        )

    if not result:
        raise RuntimeError(
            "No usable historical bars downloaded. "
            "Check credentials, Alpaca data access, and feed."
        )

    return result


# ============================================================
# STRATEGY SIGNALS
# ============================================================

def add_signals(
    data: dict[str, pd.DataFrame],
    params: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Calculate indicators using completed historical bars."""
    prepared = {}

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

        # Only use a signal from a previously completed bar.
        frame["enter_now"] = (
            frame["entry_signal"]
            .shift(1)
            .fillna(False)
            .astype(bool)
        )

        prepared[symbol] = frame

    return prepared


# ============================================================
# SIMULATION
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


def run_simulation(
    data: dict[str, pd.DataFrame],
    params: dict[str, Any],
    starting_capital: float,
) -> dict[str, Any]:
    """Simulate long-only trading with a maximum of three positions."""
    if not data:
        raise RuntimeError("Cannot simulate with an empty dataset.")

    events = {}

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

    positions = {}
    closed_trades = []
    equity_curve = []
    last_prices = {}

    slippage = SLIPPAGE_BPS_PER_SIDE / 10000.0

    def mark_equity() -> float:
        return cash + sum(
            position.qty
            * last_prices.get(symbol, position.entry_price)
            for symbol, position in positions.items()
        )

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

        # Manage existing positions first.
        # If stop and target are both hit in one bar,
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

            held = index - position.entry_index

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

            elif held >= int(params["max_hold_bars"]):
                close_position(
                    symbol, timestamp, close, "max_hold"
                )

            elif safe_float(row.get("ema_fast")) <= safe_float(
                row.get("ema_slow")
            ):
                close_position(
                    symbol, timestamp, close, "ema_exit"
                )

        # Enter positions at the current bar's open.
        # Signals were generated from the preceding completed bar.
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
                qty = allocation / entry_price
            else:
                qty = math.floor(allocation / entry_price)

            if qty <= 0:
                continue

            cost = (
                qty * entry_price
                + qty * COMMISSION_PER_SHARE
            )

            while cost > cash and qty > 0:
                if FRACTIONAL_SHARES:
                    qty *= 0.999
                else:
                    qty -= 1

                cost = (
                    qty * entry_price
                    + qty * COMMISSION_PER_SHARE
                )

            if qty <= 1e-10 or cost > cash:
                continue

            cash -= cost

            positions[symbol] = Position(
                symbol=symbol,
                qty=qty,
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

        equity_curve.append(
            (timestamp, mark_equity())
        )

    # Close remaining positions at the final available close.
    for symbol in list(positions):
        last = data[symbol].iloc[-1]

        close_position(
            symbol,
            last["timestamp"],
            safe_float(last["close"]),
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
        [value for _, value in equity_curve]
        + [final_equity],
        dtype=float,
    )

    running_peak = curve.cummax()

    drawdowns = (
        running_peak - curve
    ) / running_peak.replace(0, float("nan"))

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
# VALIDATION
# ============================================================

def validation_passes(
    result: dict[str, Any],
) -> bool:
    """Require positive validation results and controlled drawdown."""
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
        for column in ["entry_time", "exit_time"]:
            frame[column] = pd.to_datetime(
                frame[column],
                utc=True,
                errors="coerce",
            ).astype(str)

    frame.to_csv(path, index=False)


# ============================================================
# OPTIMIZATION
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

    training_results = []

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
            pf = safe_float(result["profit_factor"])
            drawdown = abs(
                safe_float(result["max_drawdown_pct"])
            )

            score = (
                ret
                - 0.75 * drawdown
                + 0.02 * min(pf, 3.0)
            )

            training_results.append(
                {
                    "pa
