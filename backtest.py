import os
import time
import json
import logging
from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo
from itertools import product

import requests
import pandas as pd

# ============================================================
# STOCK TRADING BOT V2 — AUTOMATED STRATEGY RESEARCH
# Research/backtesting only. Does NOT place trades.
# ============================================================

DATA_URL = "https://data.alpaca.markets/v2"
TRADING_URL = "https://paper-api.alpaca.markets"

BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "90"))
SCREEN_LOOKBACK_DAYS = int(os.getenv("SCREEN_LOOKBACK_DAYS", "45"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "50"))

MAX_POSITIONS = 3
POSITION_ALLOCATION = 0.05
STARTING_CAPITALS = [500, 1000, 2500, 5000]

MIN_STOCK_PRICE = 5.0
MIN_AVG_DOLLAR_VOLUME = 20_000_000
DATA_FEED = os.getenv("DATA_FEED", "iex")

ALLOW_FRACTIONAL_SHARES = (
    os.getenv("ALLOW_FRACTIONAL_SHARES", "true").lower() == "true"
)

SLIPPAGE_BPS_PER_SIDE = 5
COMMISSION_PER_SHARE = 0.0

STOP_LOSSES = [0.005, 0.0075, 0.01]
TARGET_RETURNS = [0.005, 0.0075, 0.01, 0.015]
EMA_PAIRS = [(5, 13), (9, 21), (12, 26)]
BREAKOUT_LOOKBACKS = [10, 20]
VOLUME_MULTIPLIERS = [1.2, 1.5]
MAX_HOLD_MINUTES_OPTIONS = [30, 60, 120]

TOP_CANDIDATES_TO_VALIDATE = 8
MIN_TRAIN_TRADES = 30

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

OUT_DIR = os.getenv("BACKTEST_OUTPUT_DIR", ".")
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger("backtest_v2")

SESSION_OPEN = dt_time(9, 30)
SESSION_CLOSE = dt_time(16, 0)


# ============================================================
# API ACCESS
# ============================================================

def get_credentials():
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret = (
        os.getenv("APCA_API_SECRET_KEY")
        or os.getenv("ALPACA_API_SECRET_KEY")
        or os.getenv("ALPACA_SECRET_KEY")
    )

    if not key or not secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Configure APCA_API_KEY_ID "
            "and APCA_API_SECRET_KEY in GitHub repository secrets."
        )

    return key, secret


API_KEY, API_SECRET = get_credentials()
SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
})


def api_get(url, params=None, attempts=5):
    last_error = None

    for attempt in range(attempts):
        try:
            response = SESSION.get(url, params=params, timeout=45)

            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(min(2 ** attempt, 20))
                continue

            response.raise_for_status()
            return response

        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(min(2 ** attempt, 20))

    raise RuntimeError(f"API request failed: {url}; {last_error}")


def fetch_active_symbols():
    url = f"{TRADING_URL}/v2/assets"
    response = api_get(
        url,
        {"status": "active", "asset_class": "us_equity"},
    )

    symbols = []

    for asset in response.json():
        symbol = asset.get("symbol")

        if (
            symbol
            and asset.get("tradable")
            and asset.get("status") == "active"
            and asset.get("class") in (None, "us_equity")
            and "/" not in symbol
            and "." not in symbol
        ):
            symbols.append(symbol)

    log.info("Found %s active tradable symbols", len(symbols))
    return sorted(set(symbols))


def fetch_bars(symbol, timeframe, start, end, limit=10000):
    rows = []
    token = None
    url = f"{DATA_URL}/stocks/{symbol}/bars"

    while True:
        params = {
            "timeframe": timeframe,
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "limit": limit,
            "adjustment": "raw",
            "feed": DATA_FEED,
            "sort": "asc",
        }

        if token:
            params["page_token"] = token

        payload = api_get(url, params).json()
        rows.extend(payload.get("bars", []))
        token = payload.get("next_page_token")

        if not token:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).rename(columns={
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "n": "trade_count",
        "vw": "vwap",
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


# ============================================================
# HISTORICAL STOCK SCREEN
# ============================================================

def get_screened_universe(symbols, screen_start, test_start, screen_end):
    ranked = []

    for i, symbol in enumerate(symbols, 1):
        try:
            daily = fetch_bars(symbol, "1Day", screen_start, screen_end)

            if daily.empty:
                continue

            daily = daily[daily["timestamp"] < pd.Timestamp(test_start)]

            if len(daily) < 20:
                continue

            last_close = float(daily.iloc[-1]["close"])
            recent = daily.tail(20)

            avg_dollar_volume = float(
                (recent["close"] * recent["volume"]).mean()
            )

            if (
                last_close >= MIN_STOCK_PRICE
                and avg_dollar_volume >= MIN_AVG_DOLLAR_VOLUME
            ):
                ranked.append((symbol, avg_dollar_volume, last_close))

        except Exception as exc:
            log.warning("Screen failed for %s: %s", symbol, exc)

        if i % 100 == 0:
            log.info("Screened %s/%s stocks", i, len(symbols))

    ranked.sort(key=lambda x: x[1], reverse=True)
    selected = ranked[:MAX_SYMBOLS]

    log.info("Selected %s stocks", len(selected))
    return [x[0] for x in selected], selected


# ============================================================
# INTRADAY PREPARATION AND SIGNALS
# ============================================================

def prepare_intraday(df):
    if df.empty:
        return df

    out = df.copy()
    out["timestamp_et"] = out["timestamp"].dt.tz_convert(ET)
    out["date"] = out["timestamp_et"].dt.date
    out["clock"] = out["timestamp_et"].dt.time

    out = out[
        (out["clock"] >= SESSION_OPEN)
        & (out["clock"] < SESSION_CLOSE)
        & (out["timestamp_et"].dt.weekday < 5)
    ].copy()

    return out.sort_values("timestamp").reset_index(drop=True)


def add_signals(df, fast_ema, slow_ema, breakout_lookback, volume_multiplier):
    if df.empty:
        return df

    chunks = []

    for _, day in df.groupby("date", sort=True):
        day = day.copy().sort_values("timestamp")

        day["ema_fast"] = day["close"].ewm(
            span=fast_ema, adjust=False
        ).mean()

        day["ema_slow"] = day["close"].ewm(
            span=slow_ema, adjust=False
        ).mean()

        typical = (day["high"] + day["low"] + day["close"]) / 3
        day["cum_dollar_volume"] = (typical * day["volume"]).cumsum()
        day["cum_volume"] = day["volume"].cumsum().replace(0, float("nan"))
        day["session_vwap"] = day["cum_dollar_volume"] / day["cum_volume"]

        day["prior_high"] = (
            day["high"].shift(1).rolling(breakout_lookback).max()
        )

        day["prior_avg_volume"] = (
            day["volume"].shift(1).rolling(20).mean()
        )

        day["signal"] = (
            (day["ema_fast"] > day["ema_slow"])
            & (day["close"] > day["session_vwap"])
            & (day["close"] > day["prior_high"])
            & (day["volume"] >= volume_multiplier * day["prior_avg_volume"])
        )

        day["next_timestamp"] = day["timestamp"].shift(-1)
        day["next_open"] = day["open"].shift(-1)
        chunks.append(day)

    return (
        pd.concat(chunks, ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

def run_simulation(symbol_frames, start, end, capital, params):
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    cash = float(capital)
    positions = {}
    pending = []
    trades = []
    equity_rows = []

    max_positions_seen = 0
    peak_equity = float(capital)
    max_drawdown = 0.0
    slip = SLIPPAGE_BPS_PER_SIDE / 10000.0

    timeline = {}

    for symbol, df in symbol_frames.items():
        phase = df[
            (df["timestamp"] >= start_ts)
            & (df["timestamp"] <= end_ts)
        ]

        for row in phase.to_dict("records"):
            timeline.setdefault(row["timestamp"], {})[symbol] = row

    if not timeline:
        return {
            "start_capital": capital,
            "end_equity": capital,
            "return_pct": 0.0,
            "trades": [],
            "trade_count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "max_positions": 0,
            "equity_curve": [],
            "avg_trade_pnl": 0.0,
        }

    def mark_equity(rows):
        value = cash

        for symbol, pos in positions.items():
            row = rows.get(symbol)
            price = (
                float(row["close"])
                if row
                else float(pos["last_price"])
            )
            value += pos["shares"] * price

        return value

    def close_position(symbol, raw_price, timestamp, reason):
        nonlocal cash

        pos = positions.pop(symbol)
        exit_price = max(0.01, float(raw_price) * (1 - slip))
        shares = pos["shares"]

        gross_pnl = shares * (exit_price - pos["entry_price"])
        commission = COMMISSION_PER_SHARE * shares * 2
        pnl = gross_pnl - commission

        cash += shares * exit_price - COMMISSION_PER_SHARE * shares

        trades.append({
            "symbol": symbol,
            "entry_time": pos["entry_time"].isoformat(),
            "exit_time": timestamp.isoformat(),
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "shares": shares,
            "pnl": pnl,
            "return_pct": (
                (exit_price / pos["entry_price"] - 1) * 100
            ),
            "holding_minutes": (
                timestamp - pos["entry_time"]
            ).total_seconds() / 60,
            "exit_reason": reason,
            "target_pct": params["target"] * 100,
            "stop_pct": params["stop"] * 100,
            "fast_ema": params["fast_ema"],
            "slow_ema": params["slow_ema"],
            "breakout_lookback": params["breakout"],
            "volume_multiplier": params["volume_multiplier"],
            "max_hold_minutes": params["max_hold"],
        })

    for timestamp in sorted(timeline):
        rows = timeline[timestamp]

        for symbol, pos in list(positions.items()):
            if symbol in rows:
                pos["last_price"] = float(rows[symbol]["close"])

        ready = []
        still_pending = []

        for order in pending:
            if timestamp < order["execute_at"]:
                still_pending.append(order)
            elif timestamp == order["execute_at"]:
                if order["symbol"] in rows:
                    ready.append((order["symbol"], order, rows[order["symbol"]]))

        pending = still_pending

        ready.sort(key=lambda item: item[1]["strength"], reverse=True)

        for symbol, order, row in ready:
            if symbol in positions or len(positions) >= MAX_POSITIONS:
                continue

            equity_now = mark_equity(rows)
            budget = min(cash, equity_now * POSITION_ALLOCATION)
            fill = float(row["open"]) * (1 + slip)

            if fill <= 0 or budget <= 0:
                continue

            if ALLOW_FRACTIONAL_SHARES:
                shares = budget / fill
            else:
                shares = int(budget // fill)

            if shares <= 0:
                continue

            cost = shares * fill + COMMISSION_PER_SHARE * shares

            if cost > cash + 1e-8:
                continue

            cash -= cost

            positions[symbol] = {
                "shares": shares,
                "entry_price": fill,
                "entry_time": timestamp,
                "stop_price": fill * (1 - params["stop"]),
                "target_price": fill * (1 + params["target"]),
                "last_price": float(row["open"]),
            }

            max_positions_seen = max(
                max_positions_seen, len(positions)
            )

        for symbol, pos in list(positions.items()):
            row = rows.get(symbol)

            if not row:
                continue

            low = float(row["low"])
            high = float(row["high"])
            open_price = float(row["open"])

            # Conservative assumption: stop wins if stop and target
            # are both touched within the same bar.
            if low <= pos["stop_price"]:
                raw_exit = min(open_price, pos["stop_price"])
                close_position(symbol, raw_exit, timestamp, "stop_loss")
                continue

            if high >= pos["target_price"]:
                close_position(
                    symbol,
                    pos["target_price"],
                    timestamp,
                    "take_profit",
                )
                continue

            held = (
                timestamp - pos["entry_time"]
            ).total_seconds() / 60

            if held >= params["max_hold"]:
                close_position(
                    symbol,
                    float(row["close"]),
                    timestamp,
                    "max_hold",
                )
                continue

            if timestamp.tz_convert(ET).time() >= dt_time(15, 59):
                close_position(
                    symbol,
                    float(row["close"]),
                    timestamp,
                    "session_close",
                )

        for symbol, row in rows.items():
            if not bool(row.get("signal", False)):
                continue

            next_ts = row.get("next_timestamp")

            if pd.isna(next_ts) or next_ts <= timestamp:
                continue

            if next_ts.tz_convert(ET).date() != timestamp.tz_convert(ET).date():
                continue

            if next_ts > end_ts:
                continue

            if symbol in positions:
                continue

            if any(order["symbol"] == symbol for order in pending):
                continue

            prior_high = row.get("prior_high")

            if pd.isna(prior_high) or prior_high <= 0:
                continue

            strength = float(row["close"]) / float(prior_high) - 1

            pending.append({
                "symbol": symbol,
                "execute_at": next_ts,
                "strength": strength,
            })

        equity = mark_equity(rows)
        peak_equity = max(peak_equity, equity)

        drawdown = (
            (peak_equity - equity) / peak_equity
            if peak_equity > 0
            else 0
        )

        max_drawdown = max(max_drawdown, drawdown)

        equity_rows.append({
            "timestamp": timestamp.isoformat(),
            "equity": equity,
            "cash": cash,
            "open_positions": len(positions),
        })

    final_ts = max(timeline)
    final_rows = timeline[final_ts]

    for symbol in list(positions):
        row = final_rows.get(symbol)

        raw_price = (
            float(row["close"])
            if row
            else float(positions[symbol]["last_price"])
        )

        close_position(
            symbol,
            raw_price,
            final_ts,
            "phase_end_liquidation",
        )

    end_equity = cash
    equity_rows.append({
        "timestamp": final_ts.isoformat(),
        "equity": end_equity,
        "cash": cash,
        "open_positions": 0,
    })

    pnls = [trade["pnl"] for trade in trades]
    winners = [pnl for pnl in pnls if pnl > 0]
    losers = [pnl for pnl in pnls if pnl < 0]

    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (float("inf") if gross_profit > 0 else 0.0)
    )

    return {
        "start_capital": capital,
        "end_equity": end_equity,
        "return_pct": (end_equity / capital - 1) * 100,
        "trades": trades,
        "trade_count": len(trades),
        "win_rate": len(winners) / len(trades) * 100 if trades else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown * 100,
        "max_positions": max_positions_seen,
        "equity_curve": equity_rows,
        "avg_trade_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
    }


# ============================================================
# STRATEGY RESEARCH
# ============================================================

def score_result(result):
    if result["trade_count"] < MIN_TRAIN_TRADES:
        return -1e9

    pf = result["profit_factor"]

    if pf == float("inf"):
        pf = 5.0

    pf = min(pf, 5.0)

    return (
        result["return_pct"]
        + (pf - 1.0) * 2.0
        - result["max_drawdown_pct"] * 0.5
    )


def params_grid():
    combinations = product(
        EMA_PAIRS,
        BREAKOUT_LOOKBACKS,
        VOLUME_MULTIPLIERS,
        STOP_LOSSES,
        TARGET_RETURNS,
        MAX_HOLD_MINUTES_OPTIONS,
    )

    for ema_pair, breakout, vol_mult, stop, target, max_hold in combinations:
        fast, slow = ema_pair

        if fast >= slow:
            continue

        yield {
            "fast_ema": fast,
            "slow_ema": slow,
            "breakout": breakout,
            "volume_multiplier": vol_mult,
            "stop": stop,
            "target": target,
            "max_hold": max_hold,
        }


# ============================================================
# MAIN RESEARCH LOOP
# ============================================================

def main():
    now
