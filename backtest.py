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
# CONFIGURATION
# ============================================================

DATA_URL = "https://data.alpaca.markets/v2"

API_KEY = (
    os.getenv("ALPACA_API_KEY")
    or os.getenv("ALPACA_API_KEY_ID")
    or os.getenv("APCA_API_KEY_ID")
)

API_SECRET = (
    os.getenv("ALPACA_SECRET_KEY")
    or os.getenv("ALPACA_API_SECRET_KEY")
    or os.getenv("APCA_API_SECRET_KEY")
)

DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")

ET = ZoneInfo("America/New_York")

BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "180"))

MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "12"))

# Your required limit.
MAX_POSITIONS = 3

# Initial target allocation per position.
POSITION_ALLOCATION = 0.05

MIN_STOCK_PRICE = 5.00

# Conservative estimated slippage on each side of a trade.
SLIPPAGE_BPS = 5.0

# Commission per share on each side.
COMMISSION_PER_SHARE = 0.0

ALLOW_FRACTIONAL = (
    os.getenv("ALLOW_FRACTIONAL", "true").lower()
    in ("1", "true", "yes")
)

STARTING_CAPITALS = [500, 1000, 2500, 5000]

# Data split proportions:
# 60% training, 20% validation, 20% final test.
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20

# A candidate needs sufficient training and validation trades.
MIN_TRAIN_TRADES = 30
MIN_VALIDATION_TRADES = 10

# Validation acceptance requirements.
MIN_VALIDATION_RETURN = 0.0
MIN_VALIDATION_PROFIT_FACTOR = 1.0
MAX_VALIDATION_DRAWDOWN = 0.10

# Only the best training candidates are checked on validation.
TOP_TRAIN_CANDIDATES = 15

OUTPUT_DIR = os.getenv("OUTPUT_DIR", ".")

DEFAULT_SYMBOLS = (
    "SPY,QQQ,IWM,AAPL,MSFT,NVDA,AMZN,META,"
    "GOOGL,TSLA,AMD,PLTR"
)

SYMBOLS = [
    symbol.strip().upper()
    for symbol in os.getenv(
        "SYMBOLS",
        DEFAULT_SYMBOLS
    ).split(",")
    if symbol.strip()
][:MAX_SYMBOLS]


# ============================================================
# SMALLER STRATEGY SEARCH
# ============================================================

FAST_EMAS = [5, 9]
SLOW_EMAS = [13, 21]

BREAKOUT_PERIODS = [10]

VOLUME_MULTIPLIERS = [1.2, 1.5]

STOP_PCTS = [0.0075, 0.01]

TARGET_PCTS = [0.01, 0.015]

MAX_HOLD_BARS = [30, 60]


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

log = logging.getLogger("trading_research")


# ============================================================
# VALIDATION
# ============================================================

def validate_configuration():

    if not API_KEY or not API_SECRET:
        raise RuntimeError(
            "Missing Alpaca credentials. Configure "
            "ALPACA_API_KEY and ALPACA_SECRET_KEY "
            "in GitHub Secrets."
        )

    if not SYMBOLS:
        raise RuntimeError(
            "No symbols configured."
        )

    if MAX_POSITIONS != 3:
        raise RuntimeError(
            "MAX_POSITIONS must equal 3."
        )

    if POSITION_ALLOCATION <= 0:
        raise RuntimeError(
            "POSITION_ALLOCATION must be positive."
        )

    if POSITION_ALLOCATION > 1:
        raise RuntimeError(
            "POSITION_ALLOCATION cannot exceed 1."
        )

    if BACKTEST_DAYS < 60:
        raise RuntimeError(
            "BACKTEST_DAYS must be at least 60 "
            "for this three-way research split."
        )

    if SLIPPAGE_BPS < 0:
        raise RuntimeError(
            "SLIPPAGE_BPS cannot be negative."
        )

    if COMMISSION_PER_SHARE < 0:
        raise RuntimeError(
            "COMMISSION_PER_SHARE cannot be negative."
        )

    if not STARTING_CAPITALS:
        raise RuntimeError(
            "No starting capital values configured."
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    log.info(
        "Configuration validated. "
        "No live orders will be submitted."
    )


# ============================================================
# ALPACA API
# ============================================================

def alpaca_headers():

    return {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
    }


def fetch_bars(
    symbol,
    start,
    end,
    timeframe="1Min"
):
    """
    Download historical bars using pagination.

    Uses split-adjusted data to reduce distortions from
    stock splits.

    Retries temporary failures and reports permanent errors.
    """

    url = (
        f"{DATA_URL}/stocks/"
        f"{symbol}/bars"
    )

    params = {
        "timeframe": timeframe,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": 10000,
        "adjustment": "split",
        "feed": DATA_FEED,
        "sort": "asc",
    }

    all_rows = []
    page_token = None

    while True:

        if page_token:
            params["page_token"] = page_token
        else:
            params.pop(
                "page_token",
                None
            )

        response = None

        for attempt in range(4):

            try:

                response = requests.get(
                    url,
                    headers=alpaca_headers(),
                    params=params,
                    timeout=60,
                )

                if response.status_code == 429:

                    wait_seconds = 3 * (
                        attempt + 1
                    )

                    log.warning(
                        "%s rate limited. "
                        "Waiting %s seconds.",
                        symbol,
                        wait_seconds,
                    )

                    time.sleep(wait_seconds)
                    continue

                if response.status_code >= 500:

                    time.sleep(
                        2 * (attempt + 1)
                    )

                    continue

                response.raise_for_status()
                break

            except requests.RequestException as exc:

                if attempt == 3:

                    raise RuntimeError(
                        f"Download failed for {symbol}: {exc}"
                    ) from exc

                time.sleep(
                    2 * (attempt + 1)
                )

        if response is None:

            raise RuntimeError(
                f"No response received for {symbol}."
            )

        if not response.ok:

            raise RuntimeError(
                f"Alpaca returned HTTP "
                f"{response.status_code} for {symbol}: "
                f"{response.text[:300]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid JSON returned for {symbol}."
            ) from exc

        rows = payload.get(
            "bars",
            []
        )

        all_rows.extend(rows)

        page_token = payload.get(
            "next_page_token"
        )

        if not page_token:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    required = {
        "t",
        "o",
        "h",
        "l",
        "c",
        "v",
    }

    if not required.issubset(df.columns):

        raise RuntimeError(
            f"Unexpected data columns for {symbol}: "
            f"{list(df.columns)}"
        )

    df = df.rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["symbol"] = symbol

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

    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] >= 0)
        & (df["high"] >= df["low"])
    ].copy()

    df = df.drop_duplicates(
        subset=[
            "timestamp",
            "symbol",
        ]
    )

    return df.sort_values(
        "timestamp"
    ).reset_index(drop=True)


# ============================================================
# DOWNLOAD THE FIXED WATCHLIST
# ============================================================

def download_universe(start, end):

    frames = []

    for symbol in SYMBOLS:

        try:

            log.info(
                "Downloading %s: %s through %s",
                symbol,
                start.date(),
                end.date(),
            )

            df = fetch_bars(
                symbol,
                start,
                end,
            )

            if df.empty:

                log.warning(
                    "No data returned for %s.",
                    symbol,
                )

                continue

            frames.append(df)

            log.info(
                "%s downloaded bars: %s",
                symbol,
                len(df),
            )

        except Exception as exc:

            log.error(
                "Could not download %s: %s",
                symbol,
                exc,
            )

    if not frames:

        raise RuntimeError(
            "No market data was downloaded. "
            "Check API credentials, data permissions, "
            "network access, and the watchlist."
        )

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    return data.sort_values(
        ["timestamp", "symbol"]
    ).reset_index(drop=True)


# ============================================================
# MARKET SESSION PREPARATION
# ============================================================

def prepare_intraday(data):

    df = data.copy()

    df["timestamp_et"] = (
        df["timestamp"].dt.tz_convert(ET)
    )

    df["session_date"] = (
        df["timestamp_et"].dt.date
    )

    df["session_time"] = (
        df["timestamp_et"].dt.time
    )

    # Regular U.S. stock market session.
    # Exchange holidays and early closes are handled
    # using the last available bar for each symbol.
    df = df[
        (df["timestamp_et"].dt.dayofweek < 5)
        & (
            df["session_time"]
            >= dt_time(9, 30)
        )
        & (
            df["session_time"]
            < dt_time(16, 0)
        )
    ].copy()

    df = df.sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)

    df["is_session_last_bar"] = (
        df.groupby(
            ["symbol", "session_date"]
        )["timestamp"].transform("max")
        == df["timestamp"]
    )

    return df


# ============================================================
# SIGNAL GENERATION
# ============================================================

def add_signals(data, params):

    (
        fast,
        slow,
        breakout,
        volume_mult,
        stop_pct,
        target_pct,
        max_hold,
    ) = params

    df = data.copy()

    df = df.sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)

    groups = df.groupby(
        ["symbol", "session_date"],
        sort=False,
    )

    # Moving averages use only the current session's bars.
    df["ema_fast"] = groups["close"].transform(
        lambda s: s.ewm(
            span=fast,
            adjust=False,
        ).mean()
    )

    df["ema_slow"] = groups["close"].transform(
        lambda s: s.ewm(
            span=slow,
            adjust=False,
        ).mean()
    )

    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3.0

    df["_dollar_volume"] = (
        typical_price * df["volume"]
    )

    grouped_dollars = df.groupby(
        ["symbol", "session_date"],
        sort=False,
    )["_dollar_volume"]

    grouped_volume = df.groupby(
        ["symbol", "session_date"],
        sort=False,
    )["volume"]

    cumulative_dollars = (
        grouped_dollars.cumsum()
    )

    cumulative_volume = (
        grouped_volume.cumsum()
    )

    df["session_vwap"] = (
        cumulative_dollars
        / cumulative_volume.replace(
            0,
            float("nan"),
        )
    )

    # Exclude the current bar from the breakout threshold.
    df["prior_high"] = groups["high"].transform(
        lambda s: s.shift(1).rolling(
            breakout,
            min_periods=breakout,
        ).max()
    )

    # Exclude current volume from the average.
    df["prior_volume_avg"] = groups["volume"].transform(
        lambda s: s.shift(1).rolling(
            20,
            min_periods=20,
        ).mean()
    )

    df["signal"] = (
        (df["ema_fast"] > df["ema_slow"])
        & (df["close"] > df["session_vwap"])
        & (df["close"] > df["prior_high"])
        & (
            df["volume"]
            >= volume_mult * df["prior_volume_avg"]
        )
        & (df["close"] >= MIN_STOCK_PRICE)
        & (~df["is_session_last_bar"])
    )

    # Enter at the next bar's open.
    df["entry_signal"] = (
        df.groupby(
            ["symbol", "session_date"]
        )["signal"]
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    df = df.drop(
        columns=["_dollar_volume"]
    )

    return df


# ============================================================
# EXECUTION COSTS AND POSITION SIZING
# ============================================================

def apply_slippage(price, side):

    rate = SLIPPAGE_BPS / 10000.0

    if side == "buy":
        return price * (1.0 + rate)

    return price * (1.0 - rate)


def calculate_quantity(
    allocation,
    price,
    cash,
):

    if (
        allocation <= 0
        or price <= 0
        or cash <= 0
    ):
        return 0.0

    budget = min(
        allocation,
        cash,
    )

    unit_cost = (
        price + COMMISSION_PER_SHARE
    )

    if unit_cost <= 0:
        return 0.0

    qty = budget / unit_cost

    if not ALLOW_FRACTIONAL:
        qty = float(int(qty))

    # Correct for floating-point rounding.
    if ALLOW_FRACTIONAL:

        qty = round(qty, 6)

        while (
            qty > 0
            and qty * unit_cost > cash + 1e-8
        ):
            qty = round(
                qty - 0.000001,
                6,
            )

    return max(
        0.0,
        qty,
    )


def max_drawdown(equity_values):

    if not equity_values:
        return 0.0

    series = pd.Series(
        equity_values,
        dtype=float,
    )

    peaks = series.cummax().replace(
        0,
        float("nan"),
    )

    drawdowns = (
        series / peaks - 1.0
    )

    return float(
        drawdowns.min()
    )


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_simulation(
    data,
    params,
    starting_capital,
):

    if data.empty:

        return {
            "starting_capital": starting_capital,
            "ending_equity": starting_capital,
            "return_pct": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "trade_log": [],
            "equity_curve": [],
        }

    (
        fast,
        slow,
        breakout,
        volume_mult,
        stop_pct,
        target_pct,
        max_hold,
    ) = params

    df = data.sort_values(
        ["timestamp", "symbol"]
    ).copy()

    cash = float(starting_capital)

    positions = {}
    trade_log = []
    equity_curve = []

    # Keep a last mark for any position whose symbol
    # has no bar at the current market timestamp.
    for timestamp, group in df.groupby(
        "timestamp",
        sort=True,
    ):

        rows = {
            row["symbol"]: row
            for _, row in group.iterrows()
        }

        exited_this_bar = set()

        # ----------------------------------------------------
        # A. MANAGE EXISTING POSITIONS
        # ----------------------------------------------------

        for symbol in list(positions.keys()):

            if symbol not in rows:
                continue

            row = rows[symbol]
            pos = positions[symbol]

            pos["bars_held"] += 1

            low = float(row["low"])
            high = float(row["high"])
            close = float(row["close"])

            exit_reason = None
            raw_exit = None

            # Conservative intrabar ordering:
            # assume the stop occurs first if both levels hit.
            if low <= pos["stop_price"]:

                raw_exit = pos["stop_price"]
                exit_reason = "stop"

            elif high >= pos["target_price"]:

                raw_exit = pos["target_price"]
                exit_reason = "target"

            elif pos["bars_held"] >= max_hold:

                raw_exit = close
                exit_reason = "max_hold"

            elif bool(row["is_session_last_bar"]):

                raw_exit = close
                exit_reason = "session_close"

            if exit_reason is None:
                pos["last_mark"] = close
                continue

            exit_fill = apply_slippage(
                float(raw_exit),
                "sell",
            )

            exit_fee = (
                pos["qty"] * COMMISSION_PER_SHARE
            )

            proceeds = (
                pos["qty"] * exit_fill
                - exit_fee
            )

            cash += proceeds

            net_pnl = (
                proceeds - pos["entry_cost"]
            )

            trade_log.append({
                "symbol": symbol,
                "entry_time": pos["entry_time"],
                "exit_time": timestamp,
                "entry_price": pos["entry_price"],
                "exit_price": exit_fill,
                "quantity": pos["qty"],
                "net_pnl": net_pnl,
                "return_pct": (
                    net_pnl / pos["entry_cost"]
                    if pos["entry_cost"] > 0
                    else 0.0
                ),
                "bars_held": pos["bars_held"],
                "exit_reason": exit_reason,
            })

            del positions[symbol]
            exited_this_bar.add(symbol)

        # ----------------------------------------------------
        # B. OPEN NEW POSITIONS
        # ----------------------------------------------------

        for symbol, row in rows.items():

            if not bool(row["entry_signal"]):
                continue

            if symbol in positions:
                continue

            # Avoid closing and reopening the same symbol
            # on the same timestamp.
            if symbol in exited_this_bar:
                continue

            if len(positions) >= MAX_POSITIONS:
                break

            raw_entry = float(row["open"])

            if (
                raw_entry < MIN_STOCK_PRICE
                or raw_entry <= 0
            ):
                continue

            # Mark existing holdings to estimate equity.
            marked_equity = cash

            for held_symbol, held_pos in positions.items():

                if held_symbol in rows:

                    mark = float(
                        rows[held_symbol]["close"]
                    )

                else:

                    mark = held_pos["last_mark"]

                marked_equity += (
                    held_pos["qty"] * mark
                )

            allocation = (
                marked_equity
                * POSITION_ALLOCATION
            )

            entry_fill = apply_slippage(
                raw_entry,
                "buy",
            )

            qty = calculate_quantity(
                allocation,
                entry_fill,
                cash,
            )

            if qty <= 0:
                continue

            entry_fee = (
                qty * COMMISSION_PER_SHARE
            )

            entry_cost = (
                qty * entry_fill + entry_fee
            )

            if entry_cost > cash + 1e-8:
                continue

            cash -= entry_cost

            positions[symbol] = {
                "entry_time": timestamp,
                "entry_price": entry_fill,
                "entry_cost": entry_cost,
                "qty": qty,
                "stop_price": (
                    entry_fill * (1.0 - stop_pct)
                ),
                "target_price": (
                    entry_fill * (1.0 + target_pct)
                ),
                "bars_held": 0,
                "last_mark": float(row["close"]),
            }

            # Since entry is modeled at the current bar open,
            # test the bar's range conservatively. OHLC data
            # cannot establish the exact intrabar price sequence.
            pos = positions[symbol]

            low = float(row["low"])
            high = float(row["high"])

            exit_reason = None
            raw_exit = None

            if low <= pos["stop_price"]:

                raw_exit = pos["stop_price"]
                exit_reason = "stop_entry_bar"

            elif high >= pos["target_price"]:

                raw_exit = pos["target_price"]
                exit_reason = "target_entry_bar"

            elif bool(row["is_session_last_bar"]):

                raw_exit = float(row["close"])
                exit_reason = "session_close"

            if exit_reason is not None:

                exit_fill = apply_slippage(
                    float(raw_exit),
                    "sell",
                )

                exit_fee = (
                    qty * COMMISSION_PER_SHARE
                )

                proceeds = (
                    qty * exit_fill - exit_fee
                )

                cash += proceeds

                net_pnl = (
                    proceeds - entry_cost
                )

                trade_log.append({
                    "symbol": symbol,
                    "entry_time": timestamp,
                    "exit_time": timestamp,
                    "entry_price": entry_fill,
                    "exit_price": exit_fill,
                    "quantity": qty,
                    "net_pnl": net_pnl,
                    "return_pct": (
                        net_pnl / entry_cost
                        if entry_cost > 0
                        else 0.0
                    ),
                    "bars_held": 0,
                    "exit_reason": exit_reason,
                })

                del positions[symbol]
                exited_this_bar.add(symbol)

        # ----------------------------------------------------
        # C. UPDATE EQUITY
        # ----------------------------------------------------

        equity = cash

        for symbol, pos in positions.items():

            if symbol in rows:

                pos["last_mark"] = float(
                    rows[symbol]["close"]
                )

            equity += (
                pos["qty"] * pos["last_mark"]
            )

        equity_curve.append({
            "timestamp": timestamp,
            "equity": equity,
            "cash": cash,
            "open_positions": len(positions),
        })

    # --------------------------------------------------------
    # D. SAFETY CLOSE
    # --------------------------------------------------------

    # No position should normally remain after the session-close
    # logic. This is a fallback, not a substitute for valid data.
    for symbol, pos in list(positions.items()):

        exit_fill = apply_slippage(
            pos["last_mark"],
            "sell",
        )

        exit_fee = (
            pos["qty"] * COMMISSION_PER_SHARE
        )

        proceeds = (
            pos["qty"] * exit_fill - exit_fee
        )

        cash += proceeds

        net_pnl = (
            proceeds - pos["entry_cost"]
        )

        trade_log.append({
            "symbol": symbol,
            "entry_time": pos["entry_time"],
            "exit_time": df["timestamp"].max(),
            "entry_price": pos["entry_price"],
            "exit_price": exit_fill,
            "quantity": pos["qty"],
            "net_pnl": net_pnl,
            "return_pct": (
                net_pnl / pos["entry_cost"]
                if pos["entry_cost"] > 0
                else 0.0
            ),
            "bars_held": pos["bars_held"],
            "exit_reason": "safety_close",
        })

        del positions[symbol]

    trades_df = pd.DataFrame(
        trade_log
    )

    if trades_df.empty:

        wins = 0
        losses = 0
        win_rate = 0.0
        profit_factor = 0.0

    else:

        wins = int(
            (trades_df["net_pnl"] > 0).sum()
        )

        losses = int(
            (trades_df["net_pnl"] < 0).sum()
        )

        win_rate = (
            wins / len(trades_df)
        )

        gross_profit = float(
            trades_df.loc[
                trades_df["net_pnl"] > 0,
                "net_pnl",
            ].sum()
        )

        gross_loss = abs(
            float(
                trades_df.loc[
                    trades_df["net_pnl"] < 0,
                    "net_pnl",
                ].sum()
            )
        )

        if gross_loss > 0:

            profit_factor = (
                gross_profit / gross_loss
            )

        elif gross_profit > 0:

            profit_factor = float("inf")

        else:

            profit_factor = 0.0

    curve = pd.DataFrame(
        equity_curve
    )

    ending_equity = (
        float(curve["equity"].iloc[-1])
        if not curve.empty
        else float(cash)
    )

    return {
        "starting_capital": float(starting_capital),
        "ending_equity": ending_equity,
        "return_pct": (
            ending_equity / starting_capital - 1.0
        ),
        "trades": len(trades_df),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown(
            curve["equity"].tolist()
        ),
        "trade_log": trade_log,
        "equity_curve": equity_curve,
    }


# ============================================================
# STRATEGY SCORING
# ============================================================

def safe_float(value):

    try:

        result = float(value)

        if pd.isna(result):
            return 0.0

        return result

    except (TypeError, ValueError):

        return 0.0


def score_result(result):

    trades = int(
        result["trades"]
    )

    ret = safe_float(
        result["return_pct"]
    )

    pf = safe_float(
        result["profit_factor"]
    )

    drawdown = abs(
        safe_float(
            result["max_drawdown_pct"]
        )
    )

    # Reject strategies with too few trades.
    if trades < MIN_TRAIN_TRADES:
        return -1000.0

    # Reject unprofitable strategies from top ranking.
    if ret <= 0:
        return -100.0 + ret

    if pf <= 1.0:
        return -100.0 + ret

    # Do not let an extreme PF dominate the score.
    capped_pf = min(
        pf,
        3.0,
    )

    return (
        ret
        + 0.10 * (capped_pf - 1.0)
        - 0.75 * drawdown
    )


def validation_passes(result):

    if result["trades"] < MIN_VALIDATION_TRADES:
        return False

    if result["return_pct"] <= MIN_VALIDATION_RETURN:
        return False

    if result["profit_factor"] <= MIN_VALIDATION_PROFIT_FACTOR:
        return False

    if abs(result["max_drawdown_pct"]) > MAX_VALIDATION_DRAWDOWN:
        return False

    return True


# ============================================================
# CHRONOLOGICAL DATA SPLITS
# ============================================================

def split_by_trading_day(data):

    dates = sorted(
        data["session_date"].dropna().unique()
    )

    if len(dates) < 60:

        raise RuntimeError(
            f"Only {len(dates)} trading sessions were found. "
            "At least 60 sessions are required."
        )

    n = len(dates)

    train_end = int(
        n * TRAIN_FRACTION
    )

    validation_end = int(
        n * (
            TRAIN_FRACTION
            + VALIDATION_FRACTION
        )
    )

    train_dates = set(
        dates[:train_end]
    )

    validation_dates = set(
        dates[train_end:validation_end]
    )

    test_dates = set(
        dates[validation_end:]
    )

    train = data[
        data["session_date"].isin(train_dates)
    ].copy()

    validation = data[
        data["session_date"].isin(validation_dates)
    ].copy()

    test = data[
        data["session_date"].isin(test_dates)
    ].copy()

    if (
        train.empty
        or validation.empty
        or test.empty
    ):

        raise RuntimeError(
            "One or more chronological data splits are empty."
        )

    log.info(
        "Training: %s to %s; %s sessions",
        min(train_dates),
        max(train_dates),
        len(train_dates),
    )

    log.info(
        "Validation: %s to %s; %s sessions",
        min(validation_dates),
        max(validation_dates),
        len(validation_dates),
    )

    log.info(
        "Final test: %s to %s; %s sessions",
        min(test_dates),
        max(test_dates),
        len(test_dates),
    )

    return train, validation, test


# ============================================================
# PARAMETER GRID
# ============================================================

def parameter_grid():

    combinations = []

    for values in product(
        FAST_EMAS,
        SLOW_EMAS,
        BREAKOUT_PERIODS,
        VOLUME_MULTIPLIERS,
        STOP_PCTS,
        TARGET_PCTS,
        MAX_HOLD_BARS,
    ):

        (
            fast,
            slow,
            breakout,
            volume_mult,
            stop,
            target,
            hold,
        ) = values

        if fast >= slow:
            continue

        combinations.append(values)

    return combinations


# ============================================================
# TRAINING AND VALIDATION
# ============================================================

def optimize_strategy(
    train,
    validation,
):

    candidates = parameter_grid()

    log.info(
        "Testing %s parameter combinations.",
        len(candidates),
    )

    training_results = []

    for index, params in enumerate(
        candidates,
        start=1,
    ):

        try:

            signals = add_signals(
                train,
                params,
            )

            result = run_simulation(
                signals,
                params,
                starting_capital=5000,
            )

            training_results.append({
                "params": params,
                "score": score_result(result),
                "train_return": result["return_pct"],
                "train_trades": result["trades"],
                "train_win_rate": result["win_rate"],
                "train_profit_factor": result["profit_factor"],
                "train_drawdown": result["max_drawdown_pct"],
            })

        except Exception as exc:

            log.exception(
                "Training candidate failed: %s; %s",
                params,
                exc,
            )

        if index % 10 == 0:

            log.info(
                "Training progress: %s/%s",
                index,
                len(candidates),
            )

    if not training_results:

        raise RuntimeError(
            "No training candidates completed."
        )

    training_results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    training_output = []

    for item in training_results:

        row = dict(item)

        row["params"] = json.dumps(
            row["params"]
        )

        training_output.append(row)

    pd.DataFrame(
        training_output
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "training_results.csv",
        ),
        index=False,
    )

    # Do not promote candidates that failed training scoring.
    finalists = [
        item
        for item in training_results
        if item["score"] > -100.0
    ][:TOP_TRAIN_CANDIDATES]

    if not finalists:

        raise RuntimeError(
            "NO STRATEGY PASSED TRAINING. "
            "No strategy will be promoted."
        )

    validation_results = []

    for index, item in enumerate(
        finalists,
        start=1,
    ):

        params = item["params"]

        try:

            signals = add_signals(
                validation,
                params,
            )

            result = run_simulation(
                signals,
                params,
                starting_capital=5000,
            )

            passed = validation_passes(
                result
            )

            validation_results.append({
                "params": params,
                "passed_validation": passed,
                "validation_return": result["return_pct"],
                "validation_trades": result["trades"],
                "validation_win_rate": result["win_rate"],
                "validation_profit_factor": result["profit_factor"],
                "validation_drawdown": result["max_drawdown_pct"],
                "training_score": item["score"],
                "training_return": item["train_return"],
                "training_trades": item["train_trades"],
                "training_profit_factor": item["train_profit_factor"],
            })

        except Exception as exc:

            log.exception(
                "Validation candidate failed: %s; %s",
                params,
                exc,
            )

        log.info(
            "Validation progress: %s/%s",
            index,
            len(finalists),
        )

    if not validation_results:

        raise RuntimeError(
            "No validation results were produced."
        )

    validation_output = []

    for item in validation_results:

        row = dict(item)

        row["params"] = json.dumps(
            row["params"]
        )

        validation_output.append(row)

    pd.DataFrame(
        validation_output
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "validation_results.csv",
        ),
        index=False,
    )

    accepted = [
        item
        for item in validation_results
        if item["passed_validation"]
    ]

    if not accepted:

        raise RuntimeError(
            "NO STRATEGY PASSED VALIDATION. "
            "No final strategy will be selected. "
            "Review validation_results.csv."
        )

    # Choose only among strategies that passed all validation rules.
    accepted.sort(
        key=lambda item: (
            item["validation_return"]
            - 0.75 * abs(item["validation_drawdown"])
        ),
        reverse=True,
    )

    selected = accepted[0]

    log.info(
        "Strategy passed validation: %s",
        selected["params"],
    )

    log.info(
        "Validation return: %.2f%%",
        100 * selected["validation_return"],
    )

    log.info(
        "Validation profit factor: %.3f",
        safe_float(
            selected["validation_profit_factor"]
        ),
    )

    return selected


# ============================================================
# FINAL UNTOUCHED TEST
# ============================================================

def run_final_test(
    test,
    selected,
):

    params = selected["params"]

    test_signals = add_signals(
        test,
        params,
    )

    summary_rows = []
    all_trades = []
    all_curves = []

    for capital in STARTING_CAPITALS:

        result = run_simulation(
            test_signals,
            params,
            starting_capital=capital,
        )

        summary_rows.append({
            "starting_capital": capital,
            "ending_equity": result["ending_equity"],
            "net_profit_loss": (
                result["ending_equity"] - capital
            ),
            "return_pct": result["return_pct"],
            "trades": result["trades"],
            "wins": result["wins"],
            "losses": result["losses"],
            "win_rate": result["win_rate"],
            "profit_factor": result["profit_factor"],
            "max_drawdown_pct": result["max_drawdown_pct"],
            "max_positions": MAX_POSITIONS,
            "position_allocation_pct": POSITION_ALLOCATION,
            "slippage_bps_per_side": SLIPPAGE_BPS,
            "fractional_shares_enabled": ALLOW_FRACTIONAL,
        })

        for trade in result["trade_log"]:

            trade_row = dict(trade)

            trade_row["starting_capital"] = capital

            all_trades.append(
                trade_row
            )

        for point in result["equity_curve"]:

            curve_row = dict(point)

            curve_row["starting_capital"] = capital

            all_curves.append(
                curve_row
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "final_test_summary.csv",
        ),
        index=False,
    )

    pd.DataFrame(
        all_trades
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "final_test_trades.csv",
        ),
        index=False,
    )

    pd.DataFrame(
        all_curves
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "final_test_equity_curves.csv",
        ),
        index=False,
    )

    return summary


# ============================================================
# EXPERIMENT HISTORY
# ============================================================

def save_experiment(
    selected,
    summary,
    start,
    end,
):

    path = os.path.join(
        OUTPUT_DIR,
        "experiment_history.csv",
    )

    record = {
        "run_time_utc": datetime.now().isoformat(),
        "data_start": start.isoformat(),
        "data_end": end.isoformat(),
        "symbols": ",".join(SYMBOLS),
        "feed": DATA_FEED,
        "parameters": json.dumps(
            selected["params"]
        ),
        "validation_return": selected["validation_return"],
        "validation_trades": selected["validation_trades"],
        "validation_profit_factor": selected["validation_profit_factor"],
        "validation_drawdown": selected["validation_drawdown"],
        "max_positions": MAX_POSITIONS,
        "position_allocation": POSITION_ALLOCATION,
        "slippage_bps_per_side": SLIPPAGE_BPS,
        "backtest_days": BACKTEST_DAYS,
    }

    for _, row in summary.iterrows():

        capital = int(
            row["starting_capital"]
        )

        record[
            f"test_return_{capital}"
        ] = row["return_pct"]

        record[
            f"test_trades_{capital}"
        ] = row["trades"]

        record[
            f"test_profit_factor_{capital}"
        ] = row["profit_factor"]

        record[
            f"test_drawdown_{capital}"
        ] = row["max_drawdown_pct"]

    new_df = pd.DataFrame(
        [record]
    )

    if os.path.exists(path):

        old_df = pd.read_csv(path)

        new_df = pd.concat(
            [old_df, new_df],
            ignore_index=True,
        )

    new_df.to_csv(
        path,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    validate_configuration()

    end = datetime.now(
        ET
    ).replace(
        second=0,
        microsecond=0,
    )

    start = end - timedelta(
        days=BACKTEST_DAYS
    )

    log.info("=" * 60)
    log.info("AUTOMATED STRATEGY RESEARCH")
    log.info("=" * 60)

    log.info(
        "Watchlist: %s",
        ", ".join(SYMBOLS),
    )

    log.info(
        "Maximum simultaneous positions: %s",
        MAX_POSITIONS,
    )

    log.info(
        "Starting account sizes: %s",
        STARTING_CAPITALS,
    )

    log.info(
        "Historical data feed: %s",
        DATA_FEED,
    )

    raw_data = download_universe(
        start,
        end,
    )

    intraday = prepare_intraday(
        raw_data
    )

    if intraday.empty:

        raise RuntimeError(
            "No regular-session data remains."
        )

    symbols_found = sorted(
        intraday["symbol"].unique()
    )

    log.info(
        "Symbols with usable data: %s",
        ", ".join(symbols_found),
    )

    # IMPORTANT:
    # There is deliberately no liquidity filter using the entire
    # historical period. Such a filter would let future information
    # influence the selection of symbols in the final test period.

    train, validation, test = split_by_trading_day(
        intraday
    )

    selected = optimize_strategy(
        train,
        validation,
    )

    log.info(
        "A strategy passed validation. "
        "Running the untouched final test."
    )

    summary = run_final_test(
        test,
        selected,
    )

    save_experiment(
        selected,
        summary,
        start,
        end,
    )

    log.info("=" * 60)
    log.info("FINAL OUT-OF-SAMPLE RESULTS")
    log.info("=" * 60)

    for _, row in summary.iterrows():

        log.info(
            "$%s account | Ending equity: $%.2f | "
            "Profit/Loss: $%.2f | Return: %.2f%% | "
            "Trades: %s | Win rate: %.2f%% | "
            "Profit factor: %.3f | Drawdown: %.2f%%",
            int(row["starting_capital"]),
            row["ending_equity"],
            row["net_profit_loss"],
            100 * row["return_pct"],
            int(row["trades"]),
            100 * row["win_rate"],
            safe_float(row["profit_factor"]),
            100 * row["max_drawdown_pct"],
        )

    log.info("=" * 60)
    log.info("RESEARCH COMPLETED")
    log.info("=" * 60)

    log.info(
        "Output directory: %s",
        os.path.abspath(OUTPUT_DIR),
    )

    log.info(
        "Reports: training_results.csv, "
        "validation_results.csv, "
        "final_test_summary.csv, "
        "final_test_trades.csv, "
        "final_test_equity_curves.csv, "
        "experiment_history.csv"
    )


if __name__ == "__main__":
    main()
