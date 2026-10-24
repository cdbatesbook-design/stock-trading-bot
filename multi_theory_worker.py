"""
Multi-Theory Trading Research Worker

Evaluates independent day-trading and swing-trading strategies.
Calculates hypothetical position sizes and records candidates.

SAFETY:
- No order submission.
- Paper account only.
- Maximum of three existing or proposed positions.
- Fractional shares supported by the risk calculator.
- Strategies are not approved for live trading.
"""

import logging
import os
import time

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from supabase import create_client
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from strategies import evaluate_all
from risk_manager import calculate_position_size, RiskConfig


# ============================================================
# SAFETY AND CONFIGURATION
# ============================================================

PAPER_MODE = True

MAX_POSITIONS = 3

SCAN_INTERVAL_SECONDS = 300

EASTERN = ZoneInfo("America/New_York")

WATCHLIST = [
    "EDAC",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AMD",
    "AVGO",
    "NFLX",
    "JPM",
    "BAC",
    "WMT",
    "COST",
    "DIS",
    "KO",
    "PEP",
    "XOM",
    "F",
    "INTC",
]

RISK_CONFIG = RiskConfig(
    max_positions=MAX_POSITIONS,
    risk_per_trade=0.01,
    max_position_fraction=0.50,
    cash_reserve_fraction=0.05,
    min_order_value=1.00,
    fractional_shares=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

def required_env(name):
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}"
        )

    return value


def get_alpaca_credentials():
    """
    Accept either the existing APCA_* names or the ALPACA_* names.
    This avoids requiring duplicate secrets.
    """

    api_key = (
        os.getenv("APCA_API_KEY_ID")
        or os.getenv("ALPACA_API_KEY")
    )

    api_secret = (
        os.getenv("APCA_API_SECRET_KEY")
        or os.getenv("ALPACA_SECRET_KEY")
    )

    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Set APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY, or ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY."
        )

    return api_key, api_secret


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# CLIENT INITIALIZATION
# ============================================================

def init_clients():

    supabase_url = required_env("SUPABASE_URL")
    supabase_key = required_env("SUPABASE_SERVICE_ROLE_KEY")

    api_key, api_secret = get_alpaca_credentials()

    supabase = create_client(
        supabase_url,
        supabase_key,
    )

    # Hard-coded paper mode.
    # No order-submission methods are used in this worker.
    trading = TradingClient(
        api_key,
        api_secret,
        paper=True,
    )

    data_client = StockHistoricalDataClient(
        api_key,
        api_secret,
    )

    return supabase, trading, data_client


# ============================================================
# DATABASE LOGGING
# ============================================================

def log_db(supabase, level, message):

    message = str(message)[:1000]

    logging.log(
        getattr(logging, level.upper(), logging.INFO),
        message,
    )

    try:
        supabase.table("bot_logs").insert(
            {
                "level": level.upper(),
                "message": message,
                "created_at": utc_now(),
            }
        ).execute()

    except Exception as exc:
        logging.warning(
            "Could not write bot_logs row: %s",
            exc,
        )


def set_status(supabase, message, scan=False):

    payload = {
        "id": 1,
        "status_message": str(message)[:1000],
        "last_heartbeat": utc_now(),
    }

    if scan:
        payload["last_scan"] = utc_now()

    try:
        supabase.table("bot_status").upsert(
            payload,
            on_conflict="id",
        ).execute()

    except Exception as exc:
        logging.error(
            "Could not update bot_status: %s",
            exc,
        )


def get_config(supabase):

    result = (
        supabase.table("bot_status")
        .select("*")
        .eq("id", 1)
        .limit(1)
        .execute()
    )

    rows = result.data or []

    if not rows:
        raise RuntimeError(
            "The bot_status table has no row with id=1. "
            "The scanner cannot determine whether it is enabled."
        )

    return rows[0]


# ============================================================
# MARKET DATA
# ============================================================

def extract_symbol_bars(raw, symbol):

    empty = pd.DataFrame(
        columns=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    )

    if raw is None or raw.empty:
        return empty

    frame = raw.copy()

    if isinstance(frame.index, pd.MultiIndex):

        try:
            frame = frame.xs(
                symbol,
                level="symbol",
            )

        except (KeyError, TypeError):

            try:
                frame = frame.xs(
                    symbol,
                    level=0,
                )

            except (KeyError, TypeError):
                return empty

    if not isinstance(frame.index, pd.DatetimeIndex):
        return empty

    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")

    frame.index = frame.index.tz_convert(EASTERN)

    frame.columns = [
        str(column).title()
        for column in frame.columns
    ]

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if not all(
        column in frame.columns
        for column in required_columns
    ):
        return empty

    frame = frame[required_columns].copy()

    frame = frame.sort_index()

    frame = frame[
        ~frame.index.duplicated(keep="last")
    ]

    return frame


def fetch_bars(
    data_client,
    symbols,
    timeframe,
    start,
    end,
):

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end,
        limit=10000,
        feed=DataFeed.IEX,
    )

    raw = data_client.get_stock_bars(request).df

    return {
        symbol: extract_symbol_bars(raw, symbol)
        for symbol in symbols
    }


def completed_daily_bars(frame):

    if frame.empty:
        return frame

    today = datetime.now(EASTERN).date()

    # Do not evaluate today's unfinished daily candle.
    return frame[
        frame.index.date < today
    ]


def regular_session_bars(frame):

    if frame.empty:
        return frame

    regular = frame.between_time(
        "09:30",
        "16:00",
        inclusive="both",
    )

    # Exclude the newest bar because it may still be forming.
    if len(regular) > 0:
        regular = regular.iloc[:-1]

    return regular


# ============================================================
# ACCOUNT SNAPSHOT
# ============================================================

def account_snapshot(trading):

    account = trading.get_account()

    equity = float(
        getattr(account, "equity", 0) or 0
    )

    cash = float(
        getattr(account, "cash", 0) or 0
    )

    buying_power = float(
        getattr(account, "buying_power", 0) or 0
    )

    positions = trading.get_all_positions()

    held = {
        str(position.symbol): position
        for position in positions
    }

    return equity, cash, buying_power, held


# ============================================================
# STRATEGY SCANNING
# ============================================================

def scan_once(
    supabase,
    trading,
    data_client,
):

    config = get_config(supabase)

    if config.get("enabled") is not True:

        set_status(
            supabase,
            "Scanner paused: bot_status.enabled is not true.",
            scan=True,
        )

        return

    if str(config.get("mode", "paper")).lower() != "paper":

        set_status(
            supabase,
            "SAFETY STOP: mode must be paper.",
            scan=True,
        )

        log_db(
            supabase,
            "ERROR",
            "Safety stop: scanner requires paper mode.",
        )

        return

    if not PAPER_MODE:

        set_status(
            supabase,
            "SAFETY STOP: paper-only protection is active.",
            scan=True,
        )

        return

    now = datetime.now(timezone.utc)

    log_db(
        supabase,
        "INFO",
        "Starting research scan.",
    )

    # Retrieve recent intraday bars.
    intraday_data = fetch_bars(
        data_client,
        WATCHLIST,
        TimeFrame.Minute,
        now - timedelta(days=1),
        now,
    )

    # Daily history provides warm-up data for swing indicators.
    daily_symbols = list(
        dict.fromkeys(WATCHLIST + ["SPY"])
    )

    daily_data = fetch_bars(
        data_client,
        daily_symbols,
        TimeFrame.Day,
        now - timedelta(days=330),
        now,
    )

    benchmark = completed_daily_bars(
        daily_data.get(
            "SPY",
            pd.DataFrame(),
        )
    )

    equity, cash, buying_power, held = account_snapshot(
        trading
    )

    if equity <= 0:
        raise RuntimeError(
            "Account equity is unavailable or invalid."
        )

    if cash < 0 or buying_power < 0:
        raise RuntimeError(
            "Account cash or buying power is invalid."
        )

    # Count every existing position, not just positions from this scanner.
    position_count = len(held)

    available_slots = max(
        0,
        MAX_POSITIONS - position_count,
    )

    candidates = []
    buy_signal_count = 0

    # Evaluate every strategy independently for each symbol.
    for symbol in WATCHLIST:

        minute_frame = regular_session_bars(
            intraday_data.get(
                symbol,
                pd.DataFrame(),
            )
        )

        daily_frame = completed_daily_bars(
            daily_data.get(
                symbol,
                pd.DataFrame(),
            )
        )

        signals = evaluate_all(
            minute_frame,
            daily_frame,
            benchmark,
        )

        for signal in signals:

            if signal.action != "BUY":
                continue

            buy_signal_count += 1

            # Never propose a second position in a symbol already held.
            if symbol in held:

                log_db(
                    supabase,
                    "INFO",
                    f"SUPPRESSED {symbol} "
                    f"{signal.strategy_id}: already held.",
                )

                continue

            candidates.append(
                (symbol, signal)
            )

    # Higher signal scores appear first.
    candidates.sort(
        key=lambda item: item[1].score,
        reverse=True,
    )

    # Only one candidate per symbol.
    unique_candidates = []
    seen_symbols = set()

    for symbol, signal in candidates:

        if symbol in seen_symbols:
            continue

        seen_symbols.add(symbol)

        unique_candidates.append(
            (symbol, signal)
        )

    # Never exceed the available position slots.
    selected = unique_candidates[:available_slots]

    remaining_cash = cash

    # Use the more restrictive of cash and reported buying power.
    remaining_funds = min(
        cash,
        buying_power,
    )

    approved_count = 0

    for symbol, signal in selected:

        price = float(signal.price or 0)

        stop_price = float(
            signal.stop_price or 0
        )

        # The existing risk manager calculates a hypothetical position.
        # It does not submit orders.
        decision = calculate_position_size(
            account_equity=equity,
            available_cash=min(
                remaining_cash,
                remaining_funds,
            ),
            entry_price=price,
            stop_price=stop_price,
            open_positions=position_count + approved_count,
            config=RISK_CONFIG,
        )

        if decision.approved:

            approved_count += 1

            remaining_cash = max(
                0.0,
                remaining_cash - decision.estimated_cost,
            )

            remaining_funds = max(
                0.0,
                remaining_funds - decision.estimated_cost,
            )

            sizing_message = (
                f"estimated quantity={decision.quantity:.8f}; "
                f"estimated cost=${decision.estimated_cost:.2f}; "
                f"estimated stop risk=${decision.risk_amount:.2f}"
            )

        else:

            sizing_message = (
                f"POSITION SIZE REJECTED: {decision.reason}"
            )

        log_db(
            supabase,
            "INFO",
            (
                f"RESEARCH CANDIDATE | "
                f"symbol={symbol} | "
                f"strategy={signal.strategy_id} | "
                f"score={signal.score:.3f} | "
                f"price=${price:.4f} | "
                f"stop=${stop_price:.4f} | "
                f"target=${float(signal.target_price or 0):.4f} | "
                f"{sizing_message} | "
                f"reason={signal.reason} | "
                f"orders_submitted=0"
            ),
        )

    summary = (
        f"Research scan complete. "
        f"Equity=${equity:.2f}; "
        f"existing_positions={position_count}/{MAX_POSITIONS}; "
        f"BUY_signals={buy_signal_count}; "
        f"candidates_examined={len(selected)}; "
        f"hypothetical_positions_approved={approved_count}; "
        f"orders_submitted=0. "
        f"Historical validation and paper-trading requirements "
        f"remain unmet."
    )

    log_db(
        supabase,
        "INFO",
        summary,
    )

    set_status(
        supabase,
        summary,
        scan=True,
    )


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    supabase, trading, data_client = init_clients()

    log_db(
        supabase,
        "INFO",
        "Multi-theory research scanner started. "
        "Paper account only; order submission is disabled.",
    )

    while True:

        try:

            scan_once(
                supabase,
                trading,
                data_client,
            )

        except Exception as exc:

            error_message = (
                f"Scanner cycle failed: "
                f"{type(exc).__name__}: {exc}"
            )

            log_db(
                supabase,
                "ERROR",
                error_message,
            )

            set_status(
                supabase,
                f"Scanner safety pause: {error_message}",
                scan=True,
            )

        time.sleep(
            SCAN_INTERVAL_SECONDS
        )


if __name__ == "__main__":
    main()"""
Multi-Theory Trading Research Worker

Evaluates independent day-trading and swing-trading strategies.
Calculates hypothetical position sizes and records candidates.

SAFETY:
- No order submission.
- Paper account only.
- Maximum of three existing or proposed positions.
- Fractional shares supported by the risk calculator.
- Strategies are not approved for live trading.
"""

import logging
import os
import time

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from supabase import create_client
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from strategies import evaluate_all
from risk_manager import calculate_position_size, RiskConfig


# ============================================================
# SAFETY AND CONFIGURATION
# ============================================================

PAPER_MODE = True

MAX_POSITIONS = 3

SCAN_INTERVAL_SECONDS = 300

EASTERN = ZoneInfo("America/New_York")

WATCHLIST = [
    "EDAC",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AMD",
    "AVGO",
    "NFLX",
    "JPM",
    "BAC",
    "WMT",
    "COST",
    "DIS",
    "KO",
    "PEP",
    "XOM",
    "F",
    "INTC",
]

RISK_CONFIG = RiskConfig(
    max_positions=MAX_POSITIONS,
    risk_per_trade=0.01,
    max_position_fraction=0.50,
    cash_reserve_fraction=0.05,
    min_order_value=1.00,
    fractional_shares=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

def required_env(name):
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}"
        )

    return value


def get_alpaca_credentials():
    """
    Accept either the existing APCA_* names or the ALPACA_* names.
    This avoids requiring duplicate secrets.
    """

    api_key = (
        os.getenv("APCA_API_KEY_ID")
        or os.getenv("ALPACA_API_KEY")
    )

    api_secret = (
        os.getenv("APCA_API_SECRET_KEY")
        or os.getenv("ALPACA_SECRET_KEY")
    )

    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Set APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY, or ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY."
        )

    return api_key, api_secret


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# CLIENT INITIALIZATION
# ============================================================

def init_clients():

    supabase_url = required_env("SUPABASE_URL")
    supabase_key = required_env("SUPABASE_SERVICE_ROLE_KEY")

    api_key, api_secret = get_alpaca_credentials()

    supabase = create_client(
        supabase_url,
        supabase_key,
    )

    # Hard-coded paper mode.
    # No order-submission methods are used in this worker.
    trading = TradingClient(
        api_key,
        api_secret,
        paper=True,
    )

    data_client = StockHistoricalDataClient(
        api_key,
        api_secret,
    )

    return supabase, trading, data_client


# ============================================================
# DATABASE LOGGING
# ============================================================

def log_db(supabase, level, message):

    message = str(message)[:1000]

    logging.log(
        getattr(logging, level.upper(), logging.INFO),
        message,
    )

    try:
        supabase.table("bot_logs").insert(
            {
                "level": level.upper(),
                "message": message,
                "created_at": utc_now(),
            }
        ).execute()

    except Exception as exc:
        logging.warning(
            "Could not write bot_logs row: %s",
            exc,
        )


def set_status(supabase, message, scan=False):

    payload = {
        "id": 1,
        "status_message": str(message)[:1000],
        "last_heartbeat": utc_now(),
    }

    if scan:
        payload["last_scan"] = utc_now()

    try:
        supabase.table("bot_status").upsert(
            payload,
            on_conflict="id",
        ).execute()

    except Exception as exc:
        logging.error(
            "Could not update bot_status: %s",
            exc,
        )


def get_config(supabase):

    result = (
        supabase.table("bot_status")
        .select("*")
        .eq("id", 1)
        .limit(1)
        .execute()
    )

    rows = result.data or []

    if not rows:
        raise RuntimeError(
            "The bot_status table has no row with id=1. "
            "The scanner cannot determine whether it is enabled."
        )

    return rows[0]


# ============================================================
# MARKET DATA
# ============================================================

def extract_symbol_bars(raw, symbol):

    empty = pd.DataFrame(
        columns=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    )

    if raw is None or raw.empty:
        return empty

    frame = raw.copy()

    if isinstance(frame.index, pd.MultiIndex):

        try:
            frame = frame.xs(
                symbol,
                level="symbol",
            )

        except (KeyError, TypeError):

            try:
                frame = frame.xs(
                    symbol,
                    level=0,
                )

            except (KeyError, TypeError):
                return empty

    if not isinstance(frame.index, pd.DatetimeIndex):
        return empty

    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")

    frame.index = frame.index.tz_convert(EASTERN)

    frame.columns = [
        str(column).title()
        for column in frame.columns
    ]

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if not all(
        column in frame.columns
        for column in required_columns
    ):
        return empty

    frame = frame[required_columns].copy()

    frame = frame.sort_index()

    frame = frame[
        ~frame.index.duplicated(keep="last")
    ]

    return frame


def fetch_bars(
    data_client,
    symbols,
    timeframe,
    start,
    end,
):

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end,
        limit=10000,
        feed=DataFeed.IEX,
    )

    raw = data_client.get_stock_bars(request).df

    return {
        symbol: extract_symbol_bars(raw, symbol)
        for symbol in symbols
    }


def completed_daily_bars(frame):

    if frame.empty:
        return frame

    today = datetime.now(EASTERN).date()

    # Do not evaluate today's unfinished daily candle.
    return frame[
        frame.index.date < today
    ]


def regular_session_bars(frame):

    if frame.empty:
        return frame

    regular = frame.between_time(
        "09:30",
        "16:00",
        inclusive="both",
    )

    # Exclude the newest bar because it may still be forming.
    if len(regular) > 0:
        regular = regular.iloc[:-1]

    return regular


# ============================================================
# ACCOUNT SNAPSHOT
# ============================================================

def account_snapshot(trading):

    account = trading.get_account()

    equity = float(
        getattr(account, "equity", 0) or 0
    )

    cash = float(
        getattr(account, "cash", 0) or 0
    )

    buying_power = float(
        getattr(account, "buying_power", 0) or 0
    )

    positions = trading.get_all_positions()

    held = {
        str(position.symbol): position
        for position in positions
    }

    return equity, cash, buying_power, held


# ============================================================
# STRATEGY SCANNING
# ============================================================

def scan_once(
    supabase,
    trading,
    data_client,
):

    config = get_config(supabase)

    if config.get("enabled") is not True:

        set_status(
            supabase,
            "Scanner paused: bot_status.enabled is not true.",
            scan=True,
        )

        return

    if str(config.get("mode", "paper")).lower() != "paper":

        set_status(
            supabase,
            "SAFETY STOP: mode must be paper.",
            scan=True,
        )

        log_db(
            supabase,
            "ERROR",
            "Safety stop: scanner requires paper mode.",
        )

        return

    if not PAPER_MODE:

        set_status(
            supabase,
            "SAFETY STOP: paper-only protection is active.",
            scan=True,
        )

        return

    now = datetime.now(timezone.utc)

    log_db(
        supabase,
        "INFO",
        "Starting research scan.",
    )

    # Retrieve recent intraday bars.
    intraday_data = fetch_bars(
        data_client,
        WATCHLIST,
        TimeFrame.Minute,
        now - timedelta(days=1),
        now,
    )

    # Daily history provides warm-up data for swing indicators.
    daily_symbols = list(
        dict.fromkeys(WATCHLIST + ["SPY"])
    )

    daily_data = fetch_bars(
        data_client,
        daily_symbols,
        TimeFrame.Day,
        now - timedelta(days=330),
        now,
    )

    benchmark = completed_daily_bars(
        daily_data.get(
            "SPY",
            pd.DataFrame(),
        )
    )

    equity, cash, buying_power, held = account_snapshot(
        trading
    )

    if equity <= 0:
        raise RuntimeError(
            "Account equity is unavailable or invalid."
        )

    if cash < 0 or buying_power < 0:
        raise RuntimeError(
            "Account cash or buying power is invalid."
        )

    # Count every existing position, not just positions from this scanner.
    position_count = len(held)

    available_slots = max(
        0,
        MAX_POSITIONS - position_count,
    )

    candidates = []
    buy_signal_count = 0

    # Evaluate every strategy independently for each symbol.
    for symbol in WATCHLIST:

        minute_frame = regular_session_bars(
            intraday_data.get(
                symbol,
                pd.DataFrame(),
            )
        )

        daily_frame = completed_daily_bars(
            daily_data.get(
                symbol,
                pd.DataFrame(),
            )
        )

        signals = evaluate_all(
            minute_frame,
            daily_frame,
            benchmark,
        )

        for signal in signals:

            if signal.action != "BUY":
                continue

            buy_signal_count += 1

            # Never propose a second position in a symbol already held.
            if symbol in held:

                log_db(
                    supabase,
                    "INFO",
                    f"SUPPRESSED {symbol} "
                    f"{signal.strategy_id}: already held.",
                )

                continue

            candidates.append(
                (symbol, signal)
            )

    # Higher signal scores appear first.
    candidates.sort(
        key=lambda item: item[1].score,
        reverse=True,
    )

    # Only one candidate per symbol.
    unique_candidates = []
    seen_symbols = set()

    for symbol, signal in candidates:

        if symbol in seen_symbols:
            continue

        seen_symbols.add(symbol)

        unique_candidates.append(
            (symbol, signal)
        )

    # Never exceed the available position slots.
    selected = unique_candidates[:available_slots]

    remaining_cash = cash

    # Use the more restrictive of cash and reported buying power.
    remaining_funds = min(
        cash,
        buying_power,
    )

    approved_count = 0

    for symbol, signal in selected:

        price = float(signal.price or 0)

        stop_price = float(
            signal.stop_price or 0
        )

        # The existing risk manager calculates a hypothetical position.
        # It does not submit orders.
        decision = calculate_position_size(
            account_equity=equity,
            available_cash=min(
                remaining_cash,
                remaining_funds,
            ),
            entry_price=price,
            stop_price=stop_price,
            open_positions=position_count + approved_count,
            config=RISK_CONFIG,
        )

        if decision.approved:

            approved_count += 1

            remaining_cash = max(
                0.0,
                remaining_cash - decision.estimated_cost,
            )

            remaining_funds = max(
                0.0,
                remaining_funds - decision.estimated_cost,
            )

            sizing_message = (
                f"estimated quantity={decision.quantity:.8f}; "
                f"estimated cost=${decision.estimated_cost:.2f}; "
                f"estimated stop risk=${decision.risk_amount:.2f}"
            )

        else:

            sizing_message = (
                f"POSITION SIZE REJECTED: {decision.reason}"
            )

        log_db(
            supabase,
            "INFO",
            (
                f"RESEARCH CANDIDATE | "
                f"symbol={symbol} | "
                f"strategy={signal.strategy_id} | "
                f"score={signal.score:.3f} | "
                f"price=${price:.4f} | "
                f"stop=${stop_price:.4f} | "
                f"target=${float(signal.target_price or 0):.4f} | "
                f"{sizing_message} | "
                f"reason={signal.reason} | "
                f"orders_submitted=0"
            ),
        )

    summary = (
        f"Research scan complete. "
        f"Equity=${equity:.2f}; "
        f"existing_positions={position_count}/{MAX_POSITIONS}; "
        f"BUY_signals={buy_signal_count}; "
        f"candidates_examined={len(selected)}; "
        f"hypothetical_positions_approved={approved_count}; "
        f"orders_submitted=0. "
        f"Historical validation and paper-trading requirements "
        f"remain unmet."
    )

    log_db(
        supabase,
        "INFO",
        summary,
    )

    set_status(
        supabase,
        summary,
        scan=True,
    )


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    supabase, trading, data_client = init_clients()

    log_db(
        supabase,
        "INFO",
        "Multi-theory research scanner started. "
        "Paper account only; order submission is disabled.",
    )

    while True:

        try:

            scan_once(
                supabase,
                trading,
                data_client,
            )

        except Exception as exc:

            error_message = (
                f"Scanner cycle failed: "
                f"{type(exc).__name__}: {exc}"
            )

            log_db(
                supabase,
                "ERROR",
                error_message,
            )

            set_status(
                supabase,
                f"Scanner safety pause: {error_message}",
                scan=True,
            )

        time.sleep(
            SCAN_INTERVAL_SECONDS
        )


if __name__ == "__main__":
    main()
