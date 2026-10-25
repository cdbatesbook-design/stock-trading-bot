#!/usr/bin/env python3
"""Research-only stock/crypto backtester. It never places live orders."""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("backtest")

DEFAULT_SYMBOLS = [
    "SPY", "QQQ", "AAPL", "MSFT",
    "NVDA", "AMD", "BTC-USD", "ETH-USD"
]


def normalize_ohlcv(raw, symbol="unknown"):
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    wanted = {"open", "high", "low", "close", "volume"}

    if isinstance(df.columns, pd.MultiIndex):
        cols = {}

        for col in df.columns:
            for part in col:
                name = str(part).lower().strip()

                if name in wanted and name not in cols:
                    cols[name] = col

        df = pd.DataFrame(
            {k: df[v] for k, v in cols.items()},
            index=df.index
        )
    else:
        df.columns = [
            str(c).lower().strip()
            for c in df.columns
        ]
        df = df.loc[:, ~df.columns.duplicated()]

    if not {"open", "high", "low", "close"}.issubset(df.columns):
        LOG.warning(
            "Skipping %s: missing OHLC columns",
            symbol
        )
        return pd.DataFrame()

    if "volume" not in df:
        df["volume"] = 0.0

    df.index = pd.to_datetime(
        df.index,
        utc=True,
        errors="coerce"
    )

    df = df.loc[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["open", "high", "low", "close"]
    )

    df = df[
        (df[["open", "high", "low", "close"]] > 0).all(axis=1)
    ]

    df = df[
        (df.high >= df[["open", "close", "low"]].max(axis=1))
        & (df.low <= df[["open", "close", "high"]].min(axis=1))
    ]

    return df


def download_data(symbols, period, interval):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "Install yfinance by adding it to requirements.txt."
        ) from exc

    result = {}

    for symbol in symbols:
        try:
            LOG.info("Downloading %s", symbol)

            raw = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="column"
            )

            df = normalize_ohlcv(raw, symbol)

            if len(df) >= 160:
                result[symbol] = df
            else:
                LOG.warning(
                    "Skipping %s: only %d usable bars",
                    symbol,
                    len(df)
                )

        except Exception as exc:
            LOG.exception(
                "Download failed for %s: %s",
                symbol,
                exc
            )

    if not result:
        raise RuntimeError(
            "No usable market data downloaded. "
            "Check network, symbols, period, and interval."
        )

    return result


def add_indicators(df):
    x = df.copy()
    close = x.close

    x["sma20"] = close.rolling(20).mean()
    x["sma50"] = close.rolling(50).mean()
    x["momentum10"] = close.pct_change(10)

    x["prior_high20"] = (
        x.high.rolling(20).max().shift(1)
    )

    std = (
        close.rolling(20)
        .std(ddof=0)
        .replace(0, np.nan)
    )

    x["zscore20"] = (
        close - close.rolling(20).mean()
    ) / std

    x["signal_momentum"] = (
        (close > x.sma20)
        & (x.sma20 > x.sma50)
        & (x.momentum10 > 0)
    )

    x["signal_breakout"] = (
        (close > x.prior_high20)
        & (close > x.sma50)
    )

    x["signal_mean_reversion"] = (
        (x.zscore20 < -1.5)
        & (close > close.shift(1))
    )

    for col in [
        "signal_momentum",
        "signal_breakout",
        "signal_mean_reversion"
    ]:
        x[col] = x[col].fillna(False).astype(bool)

    return x


def split_data(data):
    output = {
        "train": {},
        "validation": {},
        "test": {}
    }

    data = {
        symbol: frame.sort_index()
        for symbol, frame in data.items()
        if not frame.empty
    }

    if not data:
        return output

    start = min(
        frame.index.min()
        for frame in data.values()
    )

    end = max(
        frame.index.max()
        for frame in data.values()
    )

    span = end - start

    if span <= pd.Timedelta(0):
        return output

    train_end = start + span * 0.60
    validation_end = start + span * 0.80

    for symbol, frame in data.items():
        parts = {
            "train": frame[frame.index < train_end],

            "validation": frame[
                (frame.index >= train_end)
                & (frame.index < validation_end)
            ],

            "test": frame[frame.index >= validation_end]
        }

        if (
            len(parts["train"]) >= 60
            and len(parts["validation"]) >= 20
            and len(parts["test"]) >= 20
        ):
            for name, part in parts.items():
                output[name][symbol] = part.copy()
        else:
            LOG.warning(
                "Skipping %s: not enough bars in all splits",
                symbol
            )

    return output


def signal_strength(row, strategy):
    if strategy == "momentum":
        value = row.momentum10
        return float(value) if pd.notna(value) else 0.0

    if strategy == "breakout":
        high = row.prior_high20
        close = row.close

        if pd.notna(high) and high > 0:
            return float(close / high - 1)

        return 0.0

    if strategy == "mean_reversion":
        value = row.zscore20
        return -float(value) if pd.notna(value) else 0.0

    return 0.0


def run_portfolio(
    data,
    strategy,
    target_pct,
    starting_cash=10000.0,
    fee_rate=0.0025,
    slippage_rate=0.0005,
    stop_pct=0.005,
    max_positions=3,
    max_allocation=0.30,
    risk_per_trade=0.005,
    max_total_risk=0.015
):
    """
    Simulated portfolio backtest.

    Positive trade profits are segregated and withdrawn at day-end.
    Profits are not reinvested into trading capital.
    No live orders are placed.
    """

    signal_col = "signal_" + strategy

    frames = {
        symbol: frame.sort_index()
        for symbol, frame in data.items()
        if len(frame) and signal_col in frame
    }

    if not frames:
        return {
            "trades": 0,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "total_wealth_max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "ending_equity": starting_cash,
            "total_withdrawals": 0.0,
            "total_wealth_including_withdrawals": starting_cash,
            "segregated_profit_not_yet_transferred": 0.0,
            "max_observed_positions": 0
        }, [], []

    timeline = sorted(
        set().union(
            *(set(frame.index) for frame in frames.values())
        )
    )

    cash = float(starting_cash)
    reserve = 0.0
    withdrawn = 0.0
    day_pnl = 0.0

    positions = {}
    signals = {symbol: False for symbol in frames}
    strengths = {symbol: 0.0 for symbol in frames}
    last_close = {}

    trades = []
    equity_rows = []
    withdrawals = []

    max_open_positions = 0

    def close_pos(symbol, raw_exit, timestamp, reason):
        nonlocal cash, reserve, day_pnl

        position = positions.pop(symbol)

        fill = max(
            1e-6,
            raw_exit * (1 - slippage_rate)
        )

        proceeds = (
            position["qty"]
            * fill
            * (1 - fee_rate)
        )

        pnl = proceeds - position["entry_cost"]

        day_pnl += pnl
        reserve += max(0.0, pnl)

        # Keep profits out of trading capital.
        cash += proceeds - max(0.0, pnl)

        trades.append({
            "symbol": symbol,
            "entry_time": str(position["entry_time"]),
            "exit_time": str(timestamp),
            "entry_price": position["entry_fill"],
            "exit_price": fill,
            "quantity": position["qty"],
            "entry_cost": position["entry_cost"],
            "exit_proceeds": proceeds,
            "pnl": pnl,
            "return_pct": (
                pnl / position["entry_cost"] * 100
                if position["entry_cost"]
                else 0.0
            ),
            "reason": reason,
            "strategy": strategy,
            "target_pct": target_pct
        })

    def exit_check(symbol, bar, timestamp, opening_only=False):
        if symbol not in positions:
            return

        position = positions[symbol]

        opening = float(bar.open)
        low = float(bar.low)
        high = float(bar.high)

        stop = position["stop"]
        target = position["target"]

        if opening <= stop:
            close_pos(
                symbol,
                min(opening, stop),
                timestamp,
                "stop_gap"
            )
            return

        if opening >= target:
            close_pos(
                symbol,
                target,
                timestamp,
                "target_gap"
            )
            return

        if opening_only:
            return

        # Conservative assumption: if stop and target both touch
        # within the same candle, assume the stop happens first.
        if low <= stop:
            close_pos(symbol, stop, timestamp, "stop")
        elif high >= target:
            close_pos(symbol, target, timestamp, "target")

    def sweep(day):
        nonlocal reserve, withdrawn, day_pnl

        amount = max(0.0, reserve)

        if amount:
            reserve = 0.0
            withdrawn += amount

            withdrawals.append({
                "date": day,
                "amount": amount,
                "daily_realized_pnl": day_pnl,
                "cumulative_withdrawals": withdrawn
            })

        day_pnl = 0.0

    for index, timestamp in enumerate(timeline):
        bars = {
            symbol: frame.loc[timestamp]
            for symbol, frame in frames.items()
            if timestamp in frame.index
        }

        exited = set()

        # Process gap exits before new entries.
        for symbol in list(positions):
            if symbol in bars:
                exit_check(
                    symbol,
                    bars[symbol],
                    timestamp,
                    opening_only=True
                )

                if symbol not in positions:
                    exited.add(symbol)

        candidates = [
            symbol
            for symbol in bars
            if (
                symbol not in positions
                and symbol not in exited
                and signals.get(symbol, False)
            )
        ]

        candidates.sort(
            key=lambda symbol: (
                -strengths.get(symbol, 0.0),
                symbol
            )
        )

        for symbol in candidates:
            if len(positions) >= max_positions:
                break

            bar = bars[symbol]
            opening = float(bar.open)

            if not math.isfinite(opening) or opening <= 0:
                continue

            fill = opening * (1 + slippage_rate)
            cost_share = fill * (1 + fee_rate)

            stop = fill * (1 - stop_pct)

            target = (
                cost_share * (1 + target_pct)
                / ((1 - slippage_rate) * (1 - fee_rate))
            )

            loss_share = max(
                1e-6,
                cost_share
                - stop
                * (1 - slippage_rate)
                * (1 - fee_rate)
            )

            equity_for_sizing = cash

            for open_symbol, position in positions.items():
                mark = (
                    float(bars[open_symbol].open)
                    if open_symbol in bars
                    else position["last_close"]
                )

                equity_for_sizing += (
                    position["qty"]
                    * mark
                    * (1 - slippage_rate)
                    * (1 - fee_rate)
                )

            used_risk = sum(
                position["qty"] * position["risk_share"]
                for position in positions.values()
            )

            risk_cash = min(
                equity_for_sizing * risk_per_trade,
                max(
                    0.0,
                    equity_for_sizing * max_total_risk - used_risk
                )
            )

            quantity = (
                min(
                    risk_cash / loss_share,
                    equity_for_sizing * max_allocation / cost_share,
                    cash / cost_share
                )
                if risk_cash > 0
                else 0.0
            )

            if not math.isfinite(quantity) or quantity <= 0:
                continue

            cost = quantity * cost_share

            if cost > cash + 1e-8:
                continue

            cash -= cost

            positions[symbol] = {
                "qty": quantity,
                "entry_fill": fill,
                "entry_cost": cost,
                "entry_time": timestamp,
                "stop": stop,
                "target": target,
                "risk_share": loss_share,
                "last_close": opening
            }

            last_close[symbol] = opening

            max_open_positions = max(
                max_open_positions,
                len(positions)
            )

        # Intrabar exits.
        for symbol in list(positions):
            if symbol in bars:
                exit_check(
                    symbol,
                    bars[symbol],
                    timestamp,
                    opening_only=False
                )

                if symbol not in positions:
                    exited.add(symbol)

        for symbol, bar in bars.items():
            signals[symbol] = bool(bar[signal_col])

            strengths[symbol] = signal_strength(
                bar,
                strategy
            )

            last_close[symbol] = float(bar.close)

            if symbol in positions:
                positions[symbol]["last_close"] = float(bar.close)

        equity = cash + sum(
            position["qty"]
            * last_close.get(
                symbol,
                position["last_close"]
            )
            * (1 - slippage_rate)
            * (1 - fee_rate)
            for symbol, position in positions.items()
        )

        equity_rows.append({
            "timestamp": str(timestamp),
            "equity": equity,
            "cash": cash,
            "open_positions": len(positions),
            "daily_withdrawal": 0.0,
            "cumulative_withdrawals": withdrawn,
            "segregated_profit": reserve,
            "total_wealth": equity + withdrawn + reserve
        })

        is_last_bar = index == len(timeline) - 1

        is_last_bar_of_day = (
            not is_last_bar
            and timeline[index + 1].date() != timestamp.date()
        )

        if is_last_bar or is_last_bar_of_day:
            previous_withdrawals = withdrawn

            sweep(str(timestamp.date()))

            amount_withdrawn = withdrawn - previous_withdrawals

            equity_rows[-1]["daily_withdrawal"] = amount_withdrawn
            equity_rows[-1]["cumulative_withdrawals"] = withdrawn
            equity_rows[-1]["segregated_profit"] = reserve
            equity_rows[-1]["total_wealth"] = (
                equity + withdrawn + reserve
            )

    if timeline:
        timestamp = timeline[-1]
        previous_withdrawals = withdrawn

        # Close remaining positions at the last available close.
        for symbol in list(positions):
            close_pos(
                symbol,
                last_close.get(
                    symbol,
                    positions[symbol]["last_close"]
                ),
                timestamp,
                "end_of_test"
            )

        sweep(str(timestamp.date()))

        final_sweep = withdrawn - previous_withdrawals
        final_equity = cash

        if equity_rows:
            last_row = equity_rows[-1]

            last_row.update({
                "equity": final_equity,
                "cash": cash,
                "open_positions": 0,
                "daily_withdrawal": (
                    last_row["daily_withdrawal"] + final_sweep
                ),
                "cumulative_withdrawals": withdrawn,
                "segregated_profit": reserve,
                "total_wealth": final_equity + withdrawn + reserve
            })
    else:
        final_equity = cash

    # Calculate drawdown for both the trading account and total wealth.
    peak_equity = float(starting_cash)
    max_drawdown = 0.0

    peak_wealth = float(starting_cash)
    max_wealth_drawdown = 0.0

    for row in equity_rows:
        current_equity = float(row["equity"])

        current_wealth = float(
            row.get(
                "total_wealth",
                current_equity + row.get("cumulative_withdrawals", 0.0)
            )
        )

        peak_equity = max(peak_equity, current_equity)
        peak_wealth = max(peak_wealth, current_wealth)

        if peak_equity > 0:
            max_drawdown = max(
                max_drawdown,
                (peak_equity - current_equity) / peak_equity
            )

        if peak_wealth > 0:
            max_wealth_drawdown = max(
                max_wealth_drawdown,
                (peak_wealth - current_wealth) / peak_wealth
            )

    pnls = [float(trade["pnl"]) for trade in trades]

    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    if gross_loss:
        profit_factor = gross_profit / gross_loss
    elif gross_profit:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    total_wealth = final_equity + reserve + withdrawn

    metrics = {
        "trades": len(trades),
        "return_pct": (
            (total_wealth / starting_cash - 1) * 100
        ),
        "max_drawdown_pct": max_drawdown * 100,
        "total_wealth_max_drawdown_pct": max_wealth_drawdown * 100,
        "win_rate_pct": (
            len(wins) / len(trades) * 100
            if trades
            else 0.0
        ),
        "profit_factor": profit_factor,
        "ending_equity": final_equity,
        "total_withdrawals": withdrawn,
        "segregated_profit_not_yet_transferred": reserve,
        "total_wealth_including_withdrawals": total_wealth,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "max_observed_positions": max_open_positions
    }

    return metrics, trades, equity_rows


def run_self_test():
    """Offline deterministic checks; no downloads or live orders."""

    index = pd.date_range(
        "2024-01-01",
        periods=260,
        freq="h",
        tz="UTC"
    )

    close = pd.Series(
        np.linspace(20, 50, len(index)),
        index=index
    )

    frame = pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1000.0
    }, index=index)

    frame = add_indicators(frame)
    frame["signal_momentum"] = False

    frame.iloc[
        60,
        frame.columns.get_loc("signal_momentum")
    ] = True

    metrics, trades, equity = run_portfolio(
        {"TEST": frame},
        "momentum",
        0.005
    )

    assert metrics["trades"] >= 1, (
        "Synthetic test should produce a trade"
    )

    assert metrics["max_observed_positions"] <= 3, (
        "Position limit exceeded"
    )

    assert all(
        math.isfinite(float(row["equity"]))
        for row in equity
    ), "Non-finite equity"

    assert (
        metrics["segregated_profit_not_yet_transferred"] == 0
    ), "Profit reserve should be swept"

    assert math.isclose(
        metrics["total_wealth_including_withdrawals"],
        metrics["ending_equity"]
        + metrics["total_withdrawals"]
        + metrics["segregated_profit_not_yet_transferred"],
        rel_tol=1e-9,
        abs_tol=1e-7
    ), "Wealth accounting does not reconcile"

    assert "total_wealth_max_drawdown_pct" in metrics, (
        "Total-wealth drawdown missing"
    )

    # Mixed winning and losing trade scenario.
    index2 = pd.date_range(
        "2025-01-01",
        periods=4,
        freq="h",
        tz="UTC"
    )

    def make_symbol(winner):
        frame = pd.DataFrame({
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [
                100.1,
                100.1,
                102.0 if winner else 100.1,
                100.1
            ],
            "low": [
                99.9,
                99.9,
                100.0 if winner else 98.0,
                99.9
            ],
            "close": [100.0, 100.0, 100.5, 100.0],
            "volume": [1000.0] * 4
        }, index=index2)

        frame["signal_momentum"] = False
        frame.loc[index2[1], "signal_momentum"] = True
        frame["momentum10"] = 0.1
        frame["prior_high20"] = 100.0
        frame["zscore20"] = 0.0

        return frame

    mixed_metrics, mixed_trades, mixed_equity = run_portfolio(
        {
            "WIN": make_symbol(True),
            "LOSS": make_symbol(False)
        },
        "momentum",
        0.005,
        starting_cash=10000.0,
        fee_rate=0.0001,
        slippage_rate=0.0001,
        stop_pct=0.005,
        max_positions=3,
        max_allocation=0.30,
        risk_per_trade=0.005,
        max_total_risk=0.015
    )

    assert any(
        trade["pnl"] > 0 for trade in mixed_trades
    ), "Mixed test did not create a winner"

    assert any(
        trade["pnl"] < 0 for trade in mixed_trades
    ), "Mixed test did not create a loser"

    assert (
        mixed_metrics["segregated_profit_not_yet_transferred"] == 0
    ), "Mixed-day reserve was not swept"

    assert math.isclose(
        mixed_metrics["total_wealth_including_withdrawals"],
        mixed_metrics["ending_equity"]
        + mixed_metrics["total_withdrawals"],
        rel_tol=1e-9,
        abs_tol=1e-7
    ), "Mixed-day wealth accounting does not reconcile"

    assert mixed_metrics["max_observed_positions"] <= 3

    # Positive fees and slippage must make a flat round trip lose money.
    entry_cost = 100 * 1.0005 * 1.0025
    exit_proceeds = 100 * 0.9995 * 0.9975

    assert exit_proceeds < entry_cost, (
        "Costs should make a flat round trip negative"
    )

    print(
        "SELF-TEST PASSED "
        "(including mixed win/loss and accounting checks)"
    )


def research(args):
    raw = download_data(
        args.symbols,
        args.period,
        args.interval
    )

    data = {
        symbol: add_indicators(frame)
        for symbol, frame in raw.items()
    }

    splits = split_data(data)

    if (
        not splits["train"]
        or not splits["validation"]
        or not splits["test"]
    ):
        raise RuntimeError(
            "Not enough data for train/validation/test splits."
        )

    targets = [
        float(value)
        for value in args.targets.split(",")
        if value.strip()
    ]

    if not targets or any(
        value <= 0 or value > 0.10
        for value in targets
    ):
        raise ValueError(
            "Targets must be > 0 and <= 0.10; "
            "example: 0.001,0.0025,0.005"
        )

    rows = []
    all_trades = []
    all_equity = []

    for strategy in [
        "momentum",
        "breakout",
        "mean_reversion"
    ]:
        for target in targets:
            runs = [
                run_portfolio(
                    splits[split_name],
                    strategy,
                    target,
                    args.starting_cash,
                    args.fee,
                    args.slippage,
                    args.stop,
                    args.max_positions,
                    args.max_allocation,
                    args.risk_per_trade,
                    args.max_total_risk
                )
                for split_name in [
                    "train",
                    "validation",
                    "test"
                ]
            ]

            metrics = [run[0] for run in runs]

            train_metrics = metrics[0]
            validation_metrics = metrics[1]
            test_metrics = metrics[2]

            eligible = (
                train_metrics["trades"] >= args.min_train_trades
                and validation_metrics["trades"]
                >= args.min_validation_trades
                and train_metrics["return_pct"] > 0
                and validation_metrics["return_pct"] > 0
                and validation_metrics["max_drawdown_pct"]
                < args.max_validation_drawdown
                and validation_metrics["profit_factor"] > 1
            )

            row = {
                "strategy": strategy,
                "target_pct": target,
                "eligible_from_validation": bool(eligible)
            }

            for name, result_metrics in zip(
                ["train", "validation", "test"],
                metrics
            ):
                row.update({
                    f"{name}_{key}": value
                    for key, value in result_metrics.items()
                })

            rows.append(row)

            all_trades.extend([
                {
                    **trade,
                    "strategy": strategy,
                    "target_pct": target
                }
                for trade in runs[2][1]
            ])

            all_equity.extend([
                {
                    **equity_row,
                    "strategy": strategy,
                    "target_pct": target
                }
                for equity_row in runs[2][2]
            ])

            LOG.info(
                "%s target %.4f validation %.2f%% test %.2f%% eligible=%s",
                strategy,
                target,
                validation_metrics["return_pct"],
                test_metrics["return_pct"],
                eligible
            )

    results = pd.DataFrame(rows).sort_values(
        [
            "eligible_from_validation",
            "validation_return_pct",
            "validation_profit_factor"
        ],
        ascending=[False, False, False],
        na_position="last"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output_dir / "strategy_results.csv",
        index=False
    )

    pd.DataFrame(all_trades).to_csv(
        output_dir / "test_trades.csv",
        index=False
    )

    pd.DataFrame(all_equity).to_csv(
        output_dir / "test_equity.csv",
        index=False
    )

    report = {
        "research_only": True,
        "live_trading_enabled": False,
        "symbols_downloaded": list(data),
        "period": args.period,
        "interval": args.interval,
        "profit_policy": (
            "Sweep all positive trade P&L out at day-end; "
            "do not reinvest. This is simulated, not a real "
            "brokerage transfer."
        ),
        "assumptions": {
            "starting_cash_per_run": args.starting_cash,
            "fee_per_side": args.fee,
            "slippage_per_side": args.slippage,
            "stop_loss_pct": args.stop,
            "targets": targets,
            "max_positions": args.max_positions,
            "max_allocation": args.max_allocation,
            "risk_per_trade": args.risk_per_trade,
            "max_total_risk": args.max_total_risk,
            "same_bar_stop_target_rule": "stop first"
        },
        "eligible_candidates": int(
            results["eligible_from_validation"].sum()
        )
    }

    report_path = output_dir / "research_report.json"

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
            allow_nan=False
        ),
        encoding="utf-8"
    )

    columns = [
        "strategy",
        "target_pct",
        "eligible_from_validation",
        "train_return_pct",
        "validation_return_pct",
        "validation_trades",
        "validation_profit_factor",
        "validation_max_drawdown_pct",
        "test_return_pct",
        "test_trades"
    ]

    print("\nFinished. Files saved in:", output_dir.resolve())
    print(results[columns].head(15).to_string(index=False))

    print(
        "\nRESEARCH ONLY: Results do not prove future profitability. "
        "No live orders are placed."
    )

    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Research-only stock/crypto backtest; no live trading."
        )
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS
    )

    parser.add_argument("--period", default="1y")
    parser.add_argument("--interval", default="1h")

    parser.add_argument(
        "--starting-cash",
        type=float,
        default=10000.0
    )

    parser.add_argument("--fee", type=float, default=0.0025)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--stop", type=float, default=0.005)

    parser.add_argument(
        "--targets",
        default="0.001,0.0025,0.005,0.0075"
    )

    # Your maximum of three simultaneous positions is retained.
    parser.add_argument("--max-positions", type=int, default=3)

    parser.add_argument(
        "--max-allocation",
        type=float,
        default=0.30
    )

    parser.add_argument(
        "--risk-per-trade",
        type=float,
        default=0.005
    )

    parser.add_argument(
        "--max-total-risk",
        type=float,
        default=0.015
    )

    parser.add_argument(
        "--min-train-trades",
        type=int,
        default=10
    )

    parser.add_argument(
        "--min-validation-trades",
        type=int,
        default=5
    )

    parser.add_argument(
        "--max-validation-drawdown",
        type=float,
        default=20.0
    )

    parser.add_argument(
        "--output-dir",
        default="backtest_output"
    )

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    args = parse_args()

    if args.self_test:
        run_self_test()
        return 0

    if args.starting_cash <= 0:
        raise ValueError("starting-cash must be > 0")

    if not (
        0 <= args.fee < 1
        and 0 <= args.slippage < 1
    ):
        raise ValueError(
            "fee/slippage must be between 0 and 1"
        )

    if not (
        0 < args.stop < 1
        and 0 < args.max_allocation <= 1
    ):
        raise ValueError(
            "stop/max-allocation out of range"
        )

    if not (
        0 < args.risk_per_trade
        <= args.max_total_risk
        <= 1
    ):
        raise ValueError("risk settings invalid")

    if args.max_positions < 1:
        raise ValueError(
            "max-positions must be at least 1"
        )

    if not (
        args.min_train_trades > 0
        and args.min_validation_trades > 0
    ):
        raise ValueError(
            "minimum trade counts must be positive"
        )

    if not (
        0 < args.max_validation_drawdown <= 100
    ):
        raise ValueError(
            "max-validation-drawdown must be > 0 and <= 100"
        )

    return research(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        LOG.exception("Backtest stopped: %s", exc)
        sys.exit(1)
