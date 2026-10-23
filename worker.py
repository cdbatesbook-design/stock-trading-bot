import os
import time
import logging
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from supabase import create_client
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


# ==========================================================
# AUTOTRADER AI — PAPER TRADING WORKER
# Real-money trading is deliberately disabled.
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

log = logging.getLogger("autotrader")

# Trading rules
STARTING_BUDGET = 10000.00
MAX_POSITION_VALUE = 1000.00
MAX_OPEN_POSITIONS = 5
STOP_LOSS_PERCENT = 0.03
TAKE_PROFIT_PERCENT = 0.06

# Wait five minutes between scans
SCAN_INTERVAL_SECONDS = 300

# Initial watchlist: liquid, widely followed U.S. stocks.
# This is not a scan of every publicly traded stock.
WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AMD", "AVGO", "NFLX",
    "JPM", "BAC", "WMT", "COST", "DIS",
    "KO", "PEP", "XOM", "F", "INTC",
]

EASTERN = ZoneInfo("America/New_York")


def required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}"
        )
    return value


# Load private credentials from the worker host's environment.
ALPACA_KEY = required_env("ALPACA_API_KEY")
ALPACA_SECRET = required_env("ALPACA_SECRET_KEY")
ALPACA_URL = os.getenv(
    "ALPACA_BASE_URL",
    "https://paper-api.alpaca.markets",
).rstrip("/")

SUPABASE_URL = required_env("SUPABASE_URL")
SUPABASE_KEY = required_env("SUPABASE_SERVICE_ROLE_KEY")

# Refuse to start unless the configured endpoint is paper trading.
if "paper-api.alpaca.markets" not in ALPACA_URL.lower():
    raise RuntimeError(
        "Safety stop: this worker only permits the Alpaca paper endpoint."
    )

db = create_client(SUPABASE_URL, SUPABASE_KEY)

# paper=True explicitly requests an Alpaca paper-trading client.
trading = TradingClient(
    ALPACA_KEY,
    ALPACA_SECRET,
    paper=True,
)


def log_event(message, level="INFO", symbol=None):
    """Write an event to the console and Supabase."""
    getattr(log, level.lower(), log.info)(message)

    try:
        db.table("bot_logs").insert({
            "level": level,
            "message": message[:1000],
            "symbol": symbol,
        }).execute()
    except Exception as exc:
        log.error("Could not save log to Supabase: %s", exc)


def update_bot_status(message, last_scan=False):
    """Update the bot heartbeat and status."""
    values = {
        "last_heartbeat": datetime.now().astimezone().isoformat(),
        "status_message": message[:500],
        "updated_at": datetime.now().astimezone().isoformat(),
    }

    if last_scan:
        values["last_scan"] = datetime.now().astimezone().isoformat()

    try:
        db.table("bot_status").update(values).eq("id", 1).execute()
    except Exception as exc:
        log.error("Could not update bot status: %s", exc)


def bot_is_enabled():
    result = (
        db.table("bot_status")
        .select("enabled, mode")
        .eq("id", 1)
        .single()
        .execute()
    )

    row = result.data or {}

    # Fail closed: only an explicit paper-mode enable is accepted.
    return row.get("enabled") is True and row.get("mode") == "paper"


def get_account_state():
    result = (
        db.table("account_state")
        .select("*")
        .eq("id", 1)
        .single()
        .execute()
    )

    return result.data or {}


def get_open_positions():
    result = (
        db.table("positions")
        .select("*")
        .eq("status", "open")
        .execute()
    )

    return result.data or []


def get_price_history(symbol):
    """Download daily adjusted prices for indicator calculations."""
    data = yf.download(
        symbol,
        period="6mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        return None

    # Handle both flat and multi-level yfinance column formats.
    if isinstance(data.columns, pd.MultiIndex):
        close_data = data["Close"]

        if isinstance(close_data, pd.DataFrame):
            if symbol in close_data.columns:
                close = close_data[symbol]
            else:
                close = close_data.iloc[:, 0]
        else:
            close = close_data
    else:
        if "Close" not in data.columns:
            return None
        close = data["Close"]

    close = pd.to_numeric(close, errors="coerce").dropna()

    if len(close) < 55:
        return None

    return close


def calculate_signal(symbol):
    """
    Simple educational strategy:
    Buy candidate: price above 20-day average,
    20-day average above 50-day average, RSI 45–68.
    """
    close = get_price_history(symbol)

    if close is None:
        return None

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    changes = close.diff()
    gains = changes.clip(lower=0).rolling(14).mean()
    losses = (-changes.clip(upper=0)).rolling(14).mean()

    # Avoid division by zero.
    relative_strength = gains / losses.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + relative_strength))

    price = float(close.iloc[-1])
    fast = float(sma20.iloc[-1])
    slow = float(sma50.iloc[-1])
    current_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0

    if not all(pd.notna(x) for x in [price, fast, slow]):
        return None

    buy_signal = (
        price > fast
        and fast > slow
        and 45 <= current_rsi <= 68
    )

    return {
        "symbol": symbol,
        "price": price,
        "rsi": current_rsi,
        "buy": buy_signal,
    }


def submit_order(symbol, quantity, side, price, reason):
    """Submit a paper order and record it for audit purposes."""
    if quantity <= 0:
        return False

    # A stable client ID helps prevent accidental duplicate orders
    # for the same symbol and side on the same calendar day.
    day = datetime.now(EASTERN).strftime("%Y%m%d")
    client_id = f"ata-{side}-{symbol.lower()}-{day}"

    try:
        order = trading.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol,
                qty=round(float(quantity), 6),
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_id,
            )
        )

        db.table("trades").insert({
            "order_id": str(order.id),
            "symbol": symbol,
            "side": side,
            "quantity": round(float
