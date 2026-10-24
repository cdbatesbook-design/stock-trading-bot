"""
Automated Stock Trading Bot
Paper trading only.
Trading remains disabled unless Supabase bot_status.enabled is True.
"""

import logging
import math
import os
import time
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from supabase import create_client
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


# ============================================================
# CONFIGURATION
# ============================================================

PAPER_MODE = True

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

VIRTUAL_BUDGET = 10000.00
MAX_POSITION_VALUE = 900.00
MAX_OPEN_POSITIONS = 5

STOP_LOSS_PERCENT = 0.03
TAKE_PROFIT_PERCENT = 0.06

SCAN_INTERVAL_SECONDS = 300
RSI_PERIOD = 14

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

log = logging.getLogger("stock_trading_bot")


# ============================================================
# CONNECTIONS
# ============================================================

def required_env(name):
    value = os.getenv(name)

    if not value or not value.strip():
        raise RuntimeError(
            f"Missing environment variable: {name}"
        )

    return value.strip()


ALPACA_KEY = required_env("ALPACA_API_KEY")
ALPACA_SECRET = required_env("ALPACA_SECRET_KEY")

SUPABASE_URL = required_env("SUPABASE_URL")
SUPABASE_KEY = required_env("SUPABASE_SERVICE_ROLE_KEY")

# PAPER TRADING IS HARD-CODED.
# Do not change paper=True to paper=False.

trading = TradingClient(
    ALPACA_KEY,
    ALPACA_SECRET,
    paper=True,
)

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def valid_number(value):
    try:
        number = float(value)
        return math.isfinite(number)
    except (TypeError, ValueError):
        return False


def log_db(level, message):
    try:
        supabase.table("bot_logs").insert({
            "level": str(level).upper(),
            "message": str(message)[:8000],
            "created_at": utc_now(),
        }).execute()
    except Exception:
        log.exception("Could not write to bot_logs")


def set_status(message):
    supabase.table("bot_status").upsert(
        {
            "id": 1,
            "mode": "paper",
            "status_message": str(message)[:1000],
            "last_heartbeat": utc_now(),
            "updated_at": utc_now(),
        },
        on_conflict="id",
    ).execute()


def get_bot_config():
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
            "Supabase bot_status row id=1 was not found"
        )

    return rows[0]


def get_open_positions():
    result = (
        supabase.table("positions")
        .select("*")
        .eq("status", "open")
        .execute()
    )

    return result.data or []


def get_open_position(symbol):
    result = (
        supabase.table("positions")
        .select("*")
        .eq("symbol", symbol)
        .eq("status", "open")
        .limit(1)
        .execute()
    )

    rows = result.data or []

    return rows[0] if rows else None


# ============================================================
# MARKET DATA
# ============================================================

def get_price(symbol):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="5d", interval="1d")

    if data is None or data.empty:
        raise RuntimeError(
            f"No market data available for {symbol}"
        )

    price = float(data["Close"].dropna().iloc[-1])

    if not math.isfinite(price) or price <= 0:
        raise RuntimeError(
            f"Invalid price for {symbol}"
        )

    return price


def calculate_rsi(close, period=RSI_PERIOD):
    changes = close.diff()

    gains = changes.clip(lower=0)
    losses = -changes.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    relative_strength = average_gain / average_loss

    rsi = 100 - (100 / (1 + relative_strength))

    rsi = rsi.where(average_loss != 0, 100.0)

    no_movement = (
        (average_gain == 0) &
        (average_loss == 0)
    )

    return rsi.where(~no_movement, 50.0)


def get_signal(symbol):
    try:
        data = yf.download(
            symbol,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if data is None or data.empty:
            return "HOLD"

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if "Close" not in data.columns:
            return "HOLD"

        close = pd.to_numeric(
            data["Close"],
            errors="coerce",
        ).dropna()

        # Ignore the latest candle because it may be incomplete.
        if len(close) > 1:
            close = close.iloc[:-1]

        if len(close) < 60:
            return "HOLD"

        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        rsi = calculate_rsi(close)

        last_close = float(close.iloc[-1])
        last_sma20 = float(sma20.iloc[-1])
        last_sma50 = float(sma50.iloc[-1])
        last_rsi = float(rsi.iloc[-1])

        values = [
            last_close,
            last_sma20,
            last_sma50,
            last_rsi,
        ]

        if not all(math.isfinite(x) for x in values):
            return "HOLD"

        if (
            last_close > last_sma20
            and last_sma20 > last_sma50
            and 45 <= last_rsi <= 68
        ):
            return "BUY"

        return "HOLD"

    except Exception:
        log.exception(
            "Could not calculate signal for %s",
            symbol,
        )
        return "HOLD"


# ============================================================
# ACCOUNT CHECKS
# ============================================================

def get_account_equity():
    account = trading.get_account()
    equity = float(account.equity)

    if not math.isfinite(equity) or equity <= 0:
        raise RuntimeError("Invalid account equity")

    return equity


def get_broker_positions():
    positions = {}

    for position in trading.get_all_positions():
        positions[str(position.symbol)] = position

    return positions


def check_position_consistency():
    """
    Stop trading if broker positions and the database
    disagree. This prevents the bot from guessing ownership.
    """

    broker_positions = get_broker_positions()
    database_positions = get_open_positions()

    database_symbols = {
        row["symbol"]
        for row in database_positions
        if float(row.get("quantity") or 0) > 0
    }

    broker_symbols = set(broker_positions.keys())

    if database_symbols != broker_symbols:
        raise RuntimeError(
            "Broker/database positions do not match. "
            f"Database={sorted(database_symbols)}; "
            f"Alpaca={sorted(broker_symbols)}. "
            "Trading paused for safety."
        )


# ============================================================
# ORDER RECORDING
# ============================================================

def record_trade(
    order_id,
    symbol,
    side,
    quantity,
    price,
    status,
    reason,
):
    payload = {
        "order_id": str(order_id),
        "symbol": str(symbol),
        "side": str(side).lower(),
        "quantity": float(quantity),
        "price": float(price),
        "total_value": float(quantity) * float(price),
        "status": str(status).lower(),
        "strategy": "SMA20_SMA50_RSI",
        "reason": str(reason),
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }

    supabase.table("trades").upsert(
        payload,
        on_conflict="order_id",
    ).execute()


def submit_order(symbol, side, quantity, price, reason):
    """
    Submit a paper market order and record its initial state.

    If the database cannot record the order, trading stops.
    """

    client_order_id = str(uuid.uuid4())

    record_trade(
        order_id=client_order_id,
        symbol=symbol,
        side=side.value,
        quantity=quantity,
        price=price,
        status="submitting",
        reason=reason,
    )

    request = MarketOrderRequest(
        symbol=symbol,
        qty=quantity,
        side=side,
        time_in_force=TimeInForce.DAY,
        client_order_id=client_order_id,
    )

    order = trading.submit_order(order_data=request)

    record_trade(
        order_id=client_order_id,
        symbol=symbol,
        side=side.value,
        quantity=quantity,
        price=price,
        status=str(order.status).lower().split(".")[-1],
        reason=reason,
    )

    log_db(
        "INFO",
        f"Paper order submitted: {side.value} "
        f"{quantity} {symbol}; reason={reason}",
    )

    return order


# ============================================================
# POSITION MANAGEMENT
# ============================================================

def save_position(symbol, quantity, entry_price, current_price):
    existing = get_open_position(symbol)

    payload = {
        "symbol": symbol,
        "quantity": float(quantity),
        "entry_price": float(entry_price),
        "current_price": float(current_price),
        "status": "open",
        "updated_at": utc_now(),
        "closed_at": None,
    }

    if existing:
        (
            supabase.table("positions")
            .update(payload)
            .eq("id", existing["id"])
            .execute()
        )
    else:
        payload["opened_at"] = utc_now()

        supabase.table("positions").insert(
            payload
        ).execute()


def close_position_record(symbol):
    existing = get_open_position(symbol)

    if existing:
        (
            supabase.table("positions")
            .update({
                "quantity": 0,
                "status": "closed",
                "updated_at": utc_now(),
                "closed_at": utc_now(),
            })
            .eq("id", existing["id"])
            .execute()
        )


def manage_positions():
    """
    Check recorded positions for stop-loss and take-profit.

    These are polling-based exits, not broker-held protective
    orders. Price movement between scans can cause larger losses.
    """

    positions = get_open_positions()
    broker_positions = get_broker_positions()

    for position in positions:
        symbol = position["symbol"]

        if symbol not in broker_positions:
            raise RuntimeError(
                f"{symbol} is recorded open in Supabase "
                "but is missing from Alpaca"
            )

        quantity = float(position.get("quantity") or 0)
        entry_price = float(position.get("entry_price") or 0)

        if quantity <= 0 or entry_price <= 0:
            raise RuntimeError(
                f"Invalid recorded position for {symbol}"
            )

        price = get_price(symbol)

        (
            supabase.table("positions")
            .update({
                "current_price": price,
                "updated_at": utc_now(),
            })
            .eq("id", position["id"])
            .execute()
        )

        change = (price - entry_price) / entry_price

        if (
            change <= -STOP_LOSS_PERCENT
            or change >= TAKE_PROFIT_PERCENT
        ):
            broker_quantity = float(
                broker_positions[symbol].qty
            )

            sell_quantity = int(abs(broker_quantity))

            if sell_quantity <= 0:
                raise RuntimeError(
                    f"Invalid broker quantity for {symbol}"
                )

            reason = (
                "stop_loss"
                if change <= -STOP_LOSS_PERCENT
                else "take_profit"
            )

            submit_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=sell_quantity,
                price=price,
                reason=reason,
            )

            # Do not mark the position closed here.
            # The next run must verify the broker position
            # has actually disappeared before any new entry.


# ============================================================
# BUY SCANNER
# ============================================================

def scan_for_buys():
    account_equity = get_account_equity()

    if account_equity < VIRTUAL_BUDGET:
        log_db(
            "WARNING",
            "Account equity is below the configured "
            "minimum budget. New buys are paused.",
        )
        return

    broker_positions = get_broker_positions()
    database_positions = get_open_positions()

    # Never buy if the account has positions the database
    # does not know about.
    check_position_consistency()

    open_symbols = set(broker_positions.keys())

    if len(open_symbols) >= MAX_OPEN_POSITIONS:
        return

    for symbol in WATCHLIST:
        if symbol in open_symbols:
            continue

        if len(open_symbols) >= MAX_OPEN_POSITIONS:
            break

        # Use a fresh account check before considering a buy.
        account = trading.get_account()

        if str(account.trading_blocked).lower() == "true":
            raise RuntimeError(
                "Alpaca reports that trading is blocked"
            )
import json
import logging
import math
import os
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from supabase import create_client

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest

#============================================================

CONFIGURATION

#============================================================

SAFETY: Alpaca paper trading is enforced below.

The bot also requires enabled=True in Supabase before

it can evaluate or submit orders.

PAPER_MODE = True

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

VIRTUAL_BUDGET = 10000.00
MAX_POSITION_VALUE = 900.00
MAX_OPEN_POSITIONS = 5

STOP_LOSS_PERCENT = 0.03
TAKE_PROFIT_PERCENT = 0.06

SCAN_INTERVAL_SECONDS = 300
MAX_PRICE_DRIFT_PERCENT = 0.01
RSI_PERIOD = 14

Broker order statuses that mean the broker considers

an order terminal. Partial fills must still be accounted for.

BROKER_TERMINAL_STATUSES = {
"filled",
"canceled",
"cancelled",
"expired",
"rejected",
}

Application statuses that no longer require reconciliation.

FINAL_ORDER_STATUSES = {
"synced",
"canceled",
"cancelled",
"expiredna 
"rejected",
}

Trade records included in the managed-position ledger.

FILL_ORDER_STATUSES = {
"filled",
"partial_filled",
"partially_filled",
"synced",
}

#============================================================

LOGGING

#============================================================

            try:
                set_status(
                    f"ERROR: {str(exc)[:800]}"
                )
                log_db(
                    "ERROR",
                    f"Worker cycle failed: {str(exc)[:1000]}",
                )
            except Exception:
                log.exception(
                    "Could not update error status"
                )

        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
