import os
import sys
import time
import math
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
import pandas as pd


# ============================================================
# BACKTEST CONFIGURATION
# ============================================================

ALPACA_DATA_URL = "https://data.alpaca.markets/v2"
ALPACA_TRADING_URL = "https://paper-api.alpaca.markets"

# Historical test settings
BACKTEST_DAYS = 90
MAX_SYMBOLS = 50
IN_SAMPLE_PERCENT = 0.70

# Portfolio settings
STARTING_CAPITALS = [500, 1000, 2500, 5000]
MAX_POSITIONS = 3
POSITION_ALLOCATION = 0.05

# Strategy settings
TARGET_RETURNS = [0.005, 0.0075, 0.01, 0.015]
STOP_LOSS = 0.01
MAX_HOLD_MINUTES = 120
VOLUME_MULTIPLIER = 1.2
LOOKBACK_BARS = 20

# Stock screening
MIN_STOCK_PRICE = 5.00
MIN_AVG_DOLLAR_VOLUME = 20_000_000

# Trading assumptions
SLIPPAGE_BPS_PER_SIDE = 5
DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")

# Time settings
MARKET_TZ = ZoneInfo("America/New_York")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log = logging.getLogger("backtest")


# ============================================================
# ALPACA AUTHENTICATION
# ============================================================

API_KEY = (
    os.getenv("APCA_API_KEY_ID")
    or os.getenv("ALPACA_API_KEY")
)

API_SECRET = (
    os.getenv("APCA_API_SECRET_KEY")
    or os.getenv("ALPACA_SECRET_KEY")
)

if not API_KEY or not API_SECRET:
    raise RuntimeError(
        "Missing Alpaca API credentials. Configure "
        "APCA_API_KEY_ID and APCA_API_SECRET_KEY "
        "in your GitHub Actions secrets."
    )

SESSION = requests.Session()

SESSION.headers.update({
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Accept": "application/json",
})


# ============================================================
# HTTP HELPERS
# ============================================================

def api_get(url, params=None, retries=5):
    """
    GET request with retries and exponential backoff.
    Never prints API credentials.
    """
    last_error = None

    for attempt in range(retries):
        try:
            response = SESSION.get(
                url,
                params=params,
                timeout=60,
            )

            if response.status_code == 429:
                wait = min(2 ** attempt, 30)
                log.warning(
                    "Rate limit reached. Waiting %s seconds.",
                    wait,
                )
                time.sleep(wait)
                continue

            if response.status_code >= 500:
                wait = min(2 ** attempt, 30)
                log.warning(
                    "Alpaca server error %s. Retrying in %s seconds.",
                    response.status_code,
                    wait,
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException as exc:
            last_error = exc
            wait = min(2 ** attempt, 30)

            log.warning(
                "Request failed, attempt %s/%s: %s",
                attempt + 1,
                retries,
                exc,
            )

            time.sleep(wait)

    raise RuntimeError(
        f"Alpaca request failed after {retries} attempts: "
        f"{last_error}"
    )


# ============================================================
# TIME HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def to_utc_datetime(value):
    """
    Parse timestamps consistently as timezone-aware UTC datetimes.
    """
    parsed = pd.Timestamp(value)

    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")

    return parsed.to_pydatetime()


def is_regular_session(timestamp):
    """
    Regular US equity session, 9:30 AM through 4:00 PM ET.

    This checks clock time only. It does not independently
    verify exchange holidays or early-close dates.
    """
    local_time = to_utc_datetime(timestamp).astimezone(MARKET_TZ)

    if local_time.weekday() >= 5:
        return False

    minute = local_time.hour * 60 + local_time.minute

    return 570 <= minute < 960


def session_date(timestamp):
    return to_utc_datetime(timestamp).astimezone(MARKET_TZ).date()


# ============================================================
# FETCH STOCK UNIVERSE
# ============================================================

def fetch_assets():
    """
    Fetch active US equities from Alpaca.
    """
    url = f"{ALPACA_TRADING_URL}/v2/assets"

    payload = api_get(
        url,
        params={
            "status": "active",
            "asset_class": "us_equity",
        },
    )

    assets = []

    for asset in payload:
        symbol = asset.get("symbol", "")

        if not symbol:
            continue

        if asset.get("tradable") is not True:
            continue

        # Avoid symbols that commonly represent nonstandard
        # share classes or instruments in this simple backtest.
        if any(char in symbol for char in "^$/"):
            continue

        if len(symbol) > 6:
            continue

        assets.append(symbol)

    assets = sorted(set(assets))

    log.info("Active tradable symbols found: %s", len(assets))

    return assets


# ============================================================
# FETCH HISTORICAL BARS WITH PAGINATION
# ============================================================

def fetch_bars(
    symbols,
    timeframe,
    start,
    end,
    limit=10000,
):
    """
    Download historical Alpaca bars, following page tokens.

    Returns:
        Dictionary mapping symbol to a list of bar dictionaries.
    """
    if not symbols:
        return {}

    output = {symbol: [] for symbol in symbols}

    url = f"{ALPACA_DATA_URL}/stocks/bars"

    for batch_start in range(0, len(symbols), 50):
        batch = symbols[batch_start:batch_start + 50]

        params = {
            "symbols": ",".join(batch),
            "timeframe": timeframe,
            "start": to_utc_datetime(start).isoformat(),
            "end": to_utc_datetime(end).isoformat(),
            "limit": limit,
            "adjustment": "raw",
            "feed": DATA_FEED,
            "sort": "asc",
        }

        page_token = None
        page_count = 0

        while True:
            request_params = dict(params)

            if page_token:
                request_params["page_token"] = page_token

            data = api_get(url, params=request_params)

            bars_by_symbol = data.get("bars", {})

            for symbol, bars in bars_by_symbol.items():
                if symbol in output:
                    output[symbol].extend(bars)

            page_token = data.get("next_page_token")
            page_count += 1

            if not page_token:
                break

            if page_count > 1000:
                raise RuntimeError(
                    "Stopped pagination after 1,000 pages. "
                    "Check the requested date range."
                )

        log.info(
            "Downloaded %s batch symbols; timeframe=%s",
            len(batch),
            timeframe,
        )

    # Remove duplicate timestamps and sort each symbol.
    for symbol in output:
        deduplicated = {}

        for bar in output[symbol]:
            timestamp = bar.get("t")

            if timestamp:
                deduplicated[timestamp] = bar

        output[symbol] = sorted(
            deduplicated.values(),
            key=lambda bar: bar["t"],
        )

    return output


# ============================================================
# DAILY LIQUIDITY SCREEN
# ============================================================

def screen_symbols(assets, start, end):
    """
    Select symbols using daily price and dollar-volume data.

    This is a current-universe screen, so the resulting backtest
    can have survivorship bias. It should not be interpreted as
    a historical point-in-time stock-universe simulation.
    """
    log.info("Fetching daily bars for screening.")

    daily_bars = fetch_bars(
        assets,
        timeframe="1Day",
        start=start,
        end=end,
    )

    candidates = []

    for symbol, rows in daily_bars.items():
        if len(rows) < 10:
            continue

        frame = pd.DataFrame(rows)

        try:
            frame["close"] = pd.to_numeric(frame["c"])
            frame["volume"] = pd.to_numeric(frame["v"])

            frame = frame.dropna(
                subset=["close", "volume"]
            )

            if frame.empty:
                continue

            latest_price = float(frame["close"].iloc[-1])

            recent = frame.tail(20)

            dollar_volume = (
                recent["close"] * recent["volume"]
            )

            avg_dollar_volume = float(
                dollar_volume.mean()
            )

            if latest_price < MIN_STOCK_PRICE:
                continue

            if avg_dollar_volume < MIN_AVG_DOLLAR_VOLUME:
                continue

            candidates.append({
                "symbol": symbol,
                "price": latest_price,
                "avg_dollar_volume": avg_dollar_volume,
            })

        except (ValueError, TypeError, KeyError):
            continue

    candidates.sort(
        key=lambda item: item["avg_dollar_volume"],
        reverse=True,
    )

    selected = [
        item["symbol"]
        for item in candidates[:MAX_SYMBOLS]
    ]

    log.info(
        "Screening complete: %s symbols passed; selecting %s.",
        len(candidates),
        len(selected),
    )

    return selected


# ============================================================
# PREPARE MINUTE BARS
# ============================================================

def prepare_minute_bars(raw_bars):
    """
    Convert Alpaca minute bars into clean regular-session bars.
    """
    prepared = {}

    for symbol, rows in raw_bars.items():
        cleaned = []

        for row in rows:
            try:
                timestamp = to_utc_datetime(row["t"])

                if not is_regular_session(timestamp):
                    continue

                open_price = float(row["o"])
                high_price = float(row["h"])
                low_price = float(row["l"])
                close_price = float(row["c"])
                volume = float(row["v"])

                if min(
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                ) <= 0:
                    continue

                if high_price < low_price:
                    continue

                if volume < 0:
                    continue

                cleaned.append({
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                })

            except (ValueError, TypeError, KeyError):
                continue

        cleaned.sort(key=lambda bar: bar["timestamp"])

        # Ensure one bar per timestamp.
        deduplicated = {}

        for bar in cleaned:
            deduplicated[bar["timestamp"]] = bar

        prepared[symbol] = list(
            sorted(
                deduplicated.values(),
                key=lambda bar: bar["timestamp"],
            )
        )

    return prepared


# ============================================================
# BUILD STRATEGY SIGNALS
# ============================================================

def build_signal_flags(rows):
    """
    Build intraday momentum signals.

    Entry setup:
      1. EMA 9 > EMA 21.
      2. Close above session VWAP.
      3. Close breaks above the prior 20-bar high.
      4. Volume is at least 1.2 times prior 20-bar average.

    Indicators reset each trading session.

    The signal is generated after a bar closes. The simulated
    entry, if accepted, occurs at the next bar's open.
    """
    if not rows:
        return []

    frame = pd.DataFrame(rows).copy()

    frame["signal"] = False

    frame["session"] = [
        session_date(timestamp)
        for timestamp in frame["timestamp"]
    ]

    for _, group in frame.groupby("session", sort=True):
        indexes = group.index

        close = group["close"]
        high = group["high"]
        volume = group["volume"]

        ema9 = close.ewm(
            span=9,
            adjust=False,
            min_periods=9,
        ).mean()

        ema21 = close.ewm(
            span=21,
            adjust=False,
            min_periods=21,
        ).mean()

        typical_price = (
            group["high"]
            + group["low"]
            + group["close"]
        ) / 3.0

        cumulative_volume = volume.cumsum()

        cumulative_price_volume = (
            typical_price * volume
        ).cumsum()

        vwap = (
            cumulative_price_volume
            / cumulative_volume.replace(0, float("nan"))
        )

        # Shifted values ensure that the current bar is not
        # included in the breakout threshold or volume average.
        prior_high = high.shift(1).rolling(
            LOOKBACK_BARS,
            min_periods=LOOKBACK_BARS,
        ).max()

        prior_volume_average = volume.shift(1).rolling(
            LOOKBACK_BARS,
            min_periods=LOOKBACK_BARS,
        ).mean()

        signal = (
            (ema9 > ema21)
            & (close > vwap)
            & (close > prior_high)
            & (
                volume
                >= prior_volume_average * VOLUME_MULTIPLIER
            )
        )

        frame.loc[indexes, "signal"] = (
            signal.fillna(False).astype(bool)
        ).to_numpy()

    return frame.to_dict("records")


# ============================================================
# SIMULATION HELPERS
# ============================================================

def apply_entry_slippage(price):
    return price * (
        1 + SLIPPAGE_BPS_PER_SIDE / 10000.0
    )


def apply_exit_slippage(price):
    return price * (
        1 - SLIPPAGE_BPS_PER_SIDE / 10000.0
    )


def safe_float(value, default=0.0):
    try:
        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def trade_record(
    symbol,
    entry_time,
    exit_time,
    entry_price,
    exit_price,
    shares,
    reason,
):
    entry_value = shares * entry_price
    exit_value = shares * exit_price
    pnl = exit_value - entry_value

    return {
        "symbol": symbol,
        "entry_time": entry_time.isoformat(),
        "exit_time": exit_time.isoformat(),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "shares": shares,
        "entry_value": entry_value,
        "exit_value": exit_value,
        "pnl": pnl,
        "return_pct": (
            pnl / entry_value * 100
            if entry_value > 0
            else 0.0
        ),
        "reason": reason,
    }


# ============================================================
# PORTFOLIO BACKTEST
# ============================================================

def simulate_backtest(
    bars_by_symbol,
    starting_capital,
    target_return,
    phase_start,
    phase_end,
):
    """
    Event-driven simulation with a maximum of three positions.

    Assumptions:
      - Signals execute no earlier than the next minute-bar open.
      - Entry and exit slippage are both applied.
      - A stop is assumed to trigger first if stop and target
        are both touched within the same bar.
      - No leverage is used.
      - Fractional shares are permitted in the simulation.
      - The maximum of MAX_POSITIONS applies to open positions.
      - Positions are liquidated using the final available
        in-phase close when the phase ends.
      - This simulation does not submit orders to Alpaca.
    """
    phase_start = to_utc_datetime(phase_start)
    phase_end = to_utc_datetime(phase_end)

    if phase_end <= phase_start:
        raise ValueError(
            "phase_end must be later than phase_start."
        )

    # Build each symbol's signal list and timestamp lookup.
    bars = {}
    signal_lookup = {}
    timestamp_lookup = {}

    for symbol, rows in bars_by_symbol.items():
        prepared = []

        for row in rows:
            timestamp = to_utc_datetime(row["timestamp"])

            if timestamp < phase_start:
                continue

            if timestamp >= phase_end:
                continue

            prepared.append({
                **row,
                "timestamp": timestamp,
            })

        prepared.sort(key=lambda row: row["timestamp"])

        if not prepared:
            continue

        bars[symbol] = prepared

        for index, row in enumerate(prepared):
            timestamp = row["timestamp"]

            timestamp_lookup.setdefault(
                timestamp,
                {},
            )[symbol] = row

            if row.get("signal", False):
                signal_lookup.setdefault(
                    timestamp,
                    [],
                ).append({
                    "symbol": symbol,
                    "index": index,
                })

    timeline = sorted(timestamp_lookup.keys())

    if not timeline:
        return {
            "starting_capital": starting_capital,
            "ending_equity": starting_capital,
            "return_pct": 0.0,
            "trades": [],
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "win_rate_pct": 0.0,
            "max_positions": 0,
        }

    cash = float(starting_capital)

    # Open positions keyed by symbol.
    positions = {}

    # Pending signals are entered at the next available bar open.
    pending_entries = []

    # Latest observed price per symbol.
    current_prices = {}

    trades = []
    equity_curve = []

    max_positions_seen = 0

    def mark_equity():
        market_value = 0.0

        for symbol, position in positions.items():
            price = current_prices.get(
                symbol,
                position["entry_price"],
            )

            market_value += (
                position["shares"] * price
            )

        return cash + market_value

    def close_position(
        symbol,
        timestamp,
        raw_exit_price,
        reason,
    ):
        nonlocal cash

        position = positions.get(symbol)

        if position is None:
            return

        exit_price = apply_exit_slippage(
            raw_exit_price
        )

        shares = position["shares"]

        cash += shares * exit_price

        trades.append(
            trade_record(
                symbol=symbol,
                entry_time=position["entry_time"],
                exit_time=timestamp,
                entry_price=position["entry_price"],
                exit_price=exit_price,
                shares=shares,
                reason=reason,
            )
        )

        del positions[symbol]

    for timestamp in timeline:
        bars_at_time = timestamp_lookup[timestamp]

        # ----------------------------------------------------
        # 1. Update prices for the current timestamp first.
        # ----------------------------------------------------
        for symbol, row in bars_at_time.items():
            current_prices[symbol] = row["close"]

        # ----------------------------------------------------
        # 2. Execute pending entries at current bar OPEN.
        # ----------------------------------------------------
        candidates = pending_entries
        pending_entries = []

        # Deterministic ordering: process symbols alphabetically.
        candidates.sort(
            key=lambda item: item["symbol"]
        )

        for candidate in candidates:
            symbol = candidate["symbol"]

            if symbol not in bars_at_time:
                # The symbol has no bar at this timestamp.
                # Do not enter at a later, unrelated timestamp.
                continue

            if symbol in positions:
                continue

            if len(positions) >= MAX_POSITIONS:
                continue

            row = bars_at_time[symbol]

            if not row.get("next_bar_entry_allowed", True):
                continue

            raw_entry = safe_float(row["open"])

            if raw_entry <= 0:
                continue

            entry_price = apply_entry_slippage(
                raw_entry
            )

            equity_now = mark_equity()

            allocation = min(
                equity_now * POSITION_ALLOCATION,
                cash,
            )

            if allocation <= 0:
                continue

            shares = allocation / entry_price

            if shares <= 0 or not math.isfinite(shares):
                continue

            cost = shares * entry_price

            # Numerical safety: never spend more cash than exists.
            if cost > cash + 1e-8:
                continue

            cash -= cost

            positions[symbol] = {
                "entry_time": timestamp,
                "entry_price": entry_price,
                "shares": shares,
                "stop_price": entry_price * (1 - STOP_LOSS),
                "target_price": entry_price * (
                    1 + target_return
                ),
                "entry_index": candidate["index"],
            }

            max_positions_seen = max(
                max_positions_seen,
                len(positions),
            )

        # ----------------------------------------------------
        # 3. Process stops, targets, and time-based exits.
        # ----------------------------------------------------
        for symbol in list(positions.keys()):
            row = bars_at_time.get(symbol)

            if row is None:
                continue

            position = positions[symbol]

            # Do not evaluate the entry bar against its stop
            # and target as though the trade existed beforehand.
            if timestamp <= position["entry_time"]:
                continue

            bar_open = row["open"]
            bar_high = row["high"]
            bar_low = row["low"]
            bar_close = row["close"]

            stop_price = position["stop_price"]
            target_price = position["target_price"]

            exit_price = None
            exit_reason = None

            # Gap-aware stop loss.
            if bar_open <= stop_price:
                exit_price = bar_open
                exit_reason = "STOP_GAP"

            # If stop and target are both touched in one bar,
            # assume the less favorable outcome: stop first.
            elif (
                bar_low <= stop_price
                and bar_high >= target_price
            ):
                exit_price = stop_price
                exit_reason = "STOP_AND_TARGET_SAME_BAR"

            elif bar_low <= stop_price:
                exit_price = stop_price
                exit_reason = "STOP_LOSS"

            # Gap-aware profit target.
            elif bar_open >= target_price:
                exit_price = bar_open
                exit_reason = "TARGET_GAP"

            elif bar_high >= target_price:
                exit_price = target_price
                exit_reason = "TAKE_PROFIT"

            else:
                elapsed_minutes = (
                    timestamp - position["entry_time"]
                ).total_seconds() / 60.0

                # Close at the session's final regular-session bar.
                local = timestamp.astimezone(MARKET_TZ)

                is_session_close = (
                    local.hour == 15
                    and local.minute == 59
                )

                if is_session_close:
                    exit_price = bar_close
                    exit_reason = "SESSION_CLOSE"

                elif elapsed_minutes >= MAX_HOLD_MINUTES:
                    exit_price = bar_close
                    exit_reason = "TIME_LIMIT"

            if exit_price is not None:
                close_position(
                    symbol=symbol,
                    timestamp=timestamp,
                    raw_exit_price=exit_price,
                    reason=exit_reason,
                )

        # ----------------------------------------------------
        # 4. Schedule signals for the next bar only.
        # ----------------------------------------------------
        for signal in signal_lookup.get(timestamp, []):
            symbol = signal["symbol"]
            signal_index = signal["index"]

            if symbol in positions:
                continue

            if len(positions) >= MAX_POSITIONS:
                continue

            rows = bars[symbol]

            next_index = signal_index + 1

            if next_index >= len(rows):
                continue

            next_row = rows[next_index]

            next_timestamp = next_row["timestamp"]

            # Signals cannot use bars beyond this test phase.
            if next_timestamp >= phase_end:
                continue

            # Require the next bar to be exactly one minute later.
            if (
                next_timestamp - timestamp
            ).total_seconds() != 60:
                continue

            # Do not carry an entry into a different session.
            if session_date(next_timestamp) != session_date(timestamp):
                continue

            # The entry is only executed when this exact next
            # timestamp is processed and the symbol has a bar.
            pending_entries.append({
                "symbol": symbol,
                "index": next_index,
            })

        # ----------------------------------------------------
        # 5. Record marked-to-market equity.
        # ----------------------------------------------------
        equity_curve.append({
            "timestamp": timestamp,
            "equity": mark_equity(),
            "cash": cash,
            "open_positions": len(positions),
        })

    # --------------------------------------------------------
    # 6. Liquidate positions at the last available in-phase
    #    close. No bars after phase_end are used.
    # --------------------------------------------------------
    if timeline:
        final_timestamp = timeline[-1]
        final_rows = timestamp_lookup[final_timestamp]

        for symbol in list(positions.keys()):
            row = final_rows.get(symbol)

            if row is not None:
                final_price = row["close"]
            else:
                final_price = current_prices.get(
                    symbol,
                    positions[symbol]["entry_price"],
                )

            close_position(
                symbol=symbol,
                timestamp=final_timestamp,
                raw_exit_price=final_price,
                reason="PHASE_END",
            )

    ending_equity = cash

    # --------------------------------------------------------
    # 7. Calculate performance statistics.
    # --------------------------------------------------------
    total_return = (
        (ending_equity / starting_capital) - 1
    ) * 100 if starting_capital > 0 else 0.0

    pnl_values = [
        trade["pnl"]
        for trade in trades
    ]

    winning_trades = [
        pnl for pnl in pnl_values if pnl > 0
    ]

    losing_trades = [
        pnl for pnl in pnl_values if pnl < 0
    ]

    gross_profit = sum(winning_trades)
    gross_loss = abs(sum(losing_trades))

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (
            float("inf")
            if gross_profit > 0
            else 0.0
        )
    )

    win_rate = (
        len(winning_trades) / len(trades) * 100
        if trades
        else 0.0
    )

    # Drawdown is calculated from the recorded marked equity
    # curve and includes open-position marks during the test.
    equity_values = [
        point["equity"]
        for point in equity_curve
    ]

    if equity_values:
        running_peak = starting_capital
        max_drawdown = 0.0

        for value in equity_values:
            running_peak = max(running_peak, value)

            if running_peak > 0:
                drawdown = (
                    (running_peak - value)
                    / running_peak
                )

                max_drawdown = max(
                    max_drawdown,
                    drawdown,
                )
    else:
        max_drawdown = 0.0

    return {
        "starting_capital": starting_capital,
        "ending_equity": ending_equity,
        "return_pct": total_return,
        "trades": trades,
        "trade_count": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate_pct": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown * 100,
        "max_positions": max_positions_seen,
        "equity_curve": equity_curve,
    }


# ============================================================
# SPY BENCHMARK
# ============================================================

def calculate_spy_benchmark(
    start,
    end,
):
    """
    Fetch SPY daily bars for a simple benchmark.

    The benchmark is an approximate daily buy-and-hold comparison.
    Its dates and valuation points will not perfectly match an
    intraday portfolio's exact execution timestamps.
    """
    rows = fetch_bars(
        ["SPY"],
        timeframe="1Day",
        start=start,
        end=end,
    ).get("SPY", [])

    if len(rows) < 2:
        return None

    closes = []

    for row in rows:
        try:
            close = float(row["c"])

            if close > 0:
                closes.append(close)

        except (ValueError, TypeError, KeyError):
            continue

    if len(closes) < 2:
        return None

    return (closes[-1] / closes[0] - 1) * 100


# ============================================================
# REPORTING
# ============================================================

def format_money(value):
    return f"${value:,.2f}"


def print_result(label, result):
    pf = result["profit_factor"]

    if math.isinf(pf):
        pf_display = "INF"
    else:
        pf_display = f"{pf:.3f}"

    log.info("")
    log.info("=" * 72)
    log.info("%s", label)
    log.info("=" * 72)

    log.info(
        "Starting capital: %s",
        format_money(result["starting_capital"]),
    )

    log.info(
        "Ending equity:    %s",
        format_money(result["ending_equity"]),
    )

    log.info(
        "Net return:       %.2f%%",
        result["return_pct"],
    )

    log.info(
        "Closed trades:    %s",
        result["trade_count"],
    )

    log.info(
        "Winning trades:   %s",
        result["winning_trades"],
    )

    log.info(
        "Losing trades:    %s",
        result["losing_trades"],
    )

    log.info(
        "Win rate:         %.2f%%",
        result["win_rate_pct"],
    )

    log.info(
        "Profit factor:    %s",
        pf_display,
    )

    log.info(
        "Max drawdown:     %.2f%%",
        result["max_drawdown_pct"],
    )

    log.info(
        "Max open positions observed: %s / %s",
        result["max_positions"],
        MAX_POSITIONS,
    )


def save_trades_csv(filename, trades):
    if not trades:
        log.info("No trades to save to %s.", filename)
        return

    pd.DataFrame(trades).to_csv(
        filename,
        index=False,
    )

    log.info("Saved trade details to %s.", filename)


def save_equity_csv(filename, equity_curve):
    if not equity_curve:
        return

    pd.DataFrame(equity_curve).to_csv(
        filename,
        index=False,
    )

    log.info("Saved equity curve to %s.", filename)


# ============================================================
# MAIN
# ============================================================

def main():
    log.info("=" * 72)
    log.info("STOCK STRATEGY HISTORICAL BACKTEST")
    log.info("=" * 72)

    log.info("Mode: historical simulation only")
    log.info("No orders will be submitted.")
    log.info("Maximum simultaneous positions: %s", MAX_POSITIONS)
    log.info("Data feed: %s", DATA_FEED)
    log.info("Backtest days: %s", BACKTEST_DAYS)
    log.info("Maximum symbols: %s", MAX_SYMBOLS)
    log.info("Target returns: %s", TARGET_RETURNS)
    log.info("Stop loss: %.2f%%", STOP_LOSS * 100)
    log.info("Position allocation: %.2f%%", POSITION_ALLOCATION * 100)

    end = utc_now()

    # Extra calendar days are requested to allow for weekends
    # and market holidays. The effective test interval still
    # uses the requested number of calendar days.
    start = end - timedelta(days=BACKTEST_DAYS)

    split = start + (
        end - start
    ) * IN_SAMPLE_PERCENT

    # Historical warmup for the daily screening process.
    screening_start = start - timedelta(days=45)

    log.info("Historical data start: %s", start.isoformat())
    log.info("Historical data end:   %s", end.isoformat())
    log.info("In-sample cutoff:      %s", split.isoformat())

    # --------------------------------------------------------
    # Discover symbols.
    # --------------------------------------------------------
    assets = fetch_assets()

    if not assets:
        raise RuntimeError(
            "No active tradable stock symbols were returned."
        )

    selected_symbols = screen_symbols(
        assets=assets,
        start=screening_start,
        end=end,
    )

    if not selected_symbols:
        raise RuntimeError(
            "No stocks passed the liquidity screen. "
            "Check Alpaca permissions, data feed, and date range."
        )

    log.info(
        "Selected symbols: %s",
        ", ".join(selected_symbols),
    )

    # --------------------------------------------------------
    # Fetch minute bars.
    # --------------------------------------------------------
    log.info("Downloading historical minute bars.")

    raw_minute_bars = fetch_bars(
        symbols=selected_symbols,
        timeframe="1Min",
        start=start,
        end=end,
    )

    minute_bars = prepare_minute_bars(
        raw_minute_bars
    )

    # --------------------------------------------------------
    # Build signals.
    # --------------------------------------------------------
    bars_by_symbol = {}

    for symbol, rows in minute_bars.items():
        if len(rows) < LOOKBACK_BARS + 30:
            log.info(
                "Skipping %s: insufficient minute bars (%s).",
                symbol,
                len(rows),
            )
            continue

        bars_by_symbol[symbol] = build_signal_flags(rows)

    if not bars_by_symbol:
        raise RuntimeError(
            "No symbols have enough minute bars to backtest."
        )

    total_bars = sum(
        len(rows)
        for rows in bars_by_symbol.values()
    )

    total_signals = sum(
        sum(bool(row.get("signal")) for row in rows)
        for rows in bars_by_symbol.values()
    )

    log.info("Symbols with minute bars: %s", len(bars_by_symbol))
    log.info("Total minute bars:        %s", total_bars)
    log.info("Total generated signals:  %s", total_signals)

    # --------------------------------------------------------
    # Approximate SPY benchmark.
    # --------------------------------------------------------
    benchmark = calculate_spy_benchmark(
        start=start,
        end=end,
    )

    if benchmark is not None:
        log.info(
            "Approximate SPY buy-and-hold return: %.2f%%",
            benchmark,
        )
    else:
        log.warning(
            "SPY benchmark could not be calculated."
        )

    # --------------------------------------------------------
    # Run targets separately.
    # --------------------------------------------------------
    summary_rows = []

    for target in TARGET_RETURNS:
        log.info("")
        log.info("#" * 72)
        log.info(
            "TARGET RETURN: %.2f%%",
            target * 100,
        )
        log.info("#" * 72)

        for phase_name, phase_start, phase_end in [
            ("IN_SAMPLE", start, split),
            ("OUT_OF_SAMPLE", split, end),
        ]:
            for capital in STARTING_CAPITALS:
                log.info(
                    "Running %s | target=%.2f%% | capital=%s",
                    phase_name,
                    target * 100,
                    format_money(capital),
                )

                result = simulate_backtest(
                    bars_by_symbol=bars_by_symbol,
                    starting_capital=capital,
                    target_return=target,
                    phase_start=phase_start,
                    phase_end=phase_end,
                )

                label = (
                    f"{phase_name} | "
                    f"Target {target * 100:.2f}% | "
                    f"Capital {format_money(capital)}"
                )

                print_result(label, result)

                summary_rows.append({
                    "phase": phase_name,
                    "target_pct": target * 100,
                    "starting_capital": capital,
                    "ending_equity": result["ending_equity"],
                    "return_pct": result["return_pct"],
                    "trade_count": result["trade_count"],
                    "win_rate_pct": result["win_rate_pct"],
                    "profit_factor": result["profit_factor"],
                    "max_drawdown_pct": result["max_drawdown_pct"],
                    "max_positions": result["max_positions"],
                })

                safe_target = str(target).replace(".", "_")

                filename_base = (
                    f"{phase_name.lower()}_"
                    f"target_{safe_target}_"
                    f"capital_{capital}"
                )

                save_trades_csv(
                    f"trades_{filename_base}.csv",
                    result["trades"],
                )

                save_equity_csv(
                    f"equity_{filename_base}.csv",
                    result["equity_curve"],
                )

    # --------------------------------------------------------
    # Save comparison summary.
    # --------------------------------------------------------
    summary_frame = pd.DataFrame(summary_rows)

    summary_frame.to_csv(
        "backtest_summary.csv",
        index=False,
    )

    log.info("")
    log.info("=" * 72)
    log.info("BACKTEST SUMMARY")
    log.info("=" * 72)

    if not summary_frame.empty:
        log.info(
            "\n%s",
            summary_frame.to_string(index=False),
        )

    log.info("")
    log.info("Saved comparison results to backtest_summary.csv")
    log.info("Backtest completed.")
    log.info("No orders were submitted.")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        log.error("Backtest interrupted by user.")
        raise

    except Exception:
        log.exception("Backtest failed.")
        raise
