"""
Stock Trading Bot - Repair and Testing Version

Alpaca paper account only.
Automatic order execution is disabled.
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


# ============================================================
# CONFIGURATION
# ============================================================

PAPER_MODE = True
ORDER_EXECUTION_ENABLED = False

SCAN_INTERVAL_SECONDS = 300
RSI_PERIOD = 14

WATCHLIST = [
    "EDAC", "AAPL", "MSFT", "NVDA", "AMZN",
    "GOOGL", "META", "TSLA", "AMD", "AVGO",
    "NFLX", "JPM", "BAC", "WMT", "COST",
    "DIS", "KO", "PEP", "XOM", "F", "INTC",
]


# ============================================================
# RENDER LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)

log = logging.getLogger("stock_trading_bot")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def required_env(name):
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"Missing Render environment variable: {name}"
        )

    return value


# ============================================================
# CONNECTIONS
# ============================================================

supabase = create_client(
    required_env("SUPABASE_URL"),
    required_env("SUPABASE_SERVICE_ROLE_KEY"),
)

# Always connect to Alpaca PAPER trading.
trading = TradingClient(
    required_env("ALPACA_API_KEY"),
    required_env("ALPACA_SECRET_KEY"),
    paper=True,
)


# ============================================================
# LOGGING TO SUPABASE AND RENDER
# ============================================================

def save_log(level, message):
    level = str(level).upper()
    message = str(message)[:1000]

    # Render captures these console messages.
    log.log(
        getattr(logging, level, logging.INFO),
        message,
    )

    # Save the same event in Supabase.
    try:
        supabase.table("bot_logs").insert({
            "level": level,
            "message": message,
            "created_at": utc_now(),
        }).execute()

    except Exception:
        log.exception(
            "Failed to save message to Supabase bot_logs"
        )


# ============================================================
# BOT STATUS
# ============================================================

def get_bot_config():
    response = (
        supabase.table("bot_status")
        .select("*")
        .eq("id", 1)
        .limit(1)
        .execute()
    )

    rows = response.data or []

    if not rows:
        raise RuntimeError(
            "Supabase bot_status row id=1 was not found"
        )

    return rows[0]


def update_bot_status(message, error=None):
    # Read the existing row to avoid guessing which optional
    # columns exist in the database.
    current = get_bot_config()
    existing_columns = set(current.keys())

    updates = {}

    if "mode" in existing_columns:
        updates["mode"] = "paper"

    if "status_message" in existing_columns:
        updates["status_message"] = str(message)[:1000]

    if "last_heartbeat" in existing_columns:
        updates["last_heartbeat"] = utc_now()

    if "updated_at" in existing_columns:
        updates["updated_at"] = utc_now()

    if "last_scan" in existing_columns:
        updates["last_scan"] = utc_now()

    if error is not None and "last_error" in existing_columns:
        updates["last_error"] = str(error)[:1000]

    if updates:
        (
            supabase.table("bot_status")
            .update(updates)
            .eq("id", 1)
            .execute()
        )

    save_log(
        "INFO",
        f"Bot status updated: {message}",
    )


# ============================================================
# RSI INDICATOR
# ============================================================

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

    safe_loss = average_loss.replace(
        0,
        float("nan"),
    )

    relative_strength = average_gain / safe_loss

    rsi = 100 - (
        100 / (1 + relative_strength)
    )

    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100.0,
    )

    rsi = rsi.mask(
        (average_loss == 0) & (average_gain == 0),
        50.0,
    )

    return rsi


# ============================================================
# STOCK SIGNALS
# ============================================================

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
            return "HOLD", "No market data available"

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if "Close" not in data.columns:
            return "HOLD", "Closing price data unavailable"

        close = pd.to_numeric(
            data["Close"],
            errors="coerce",
        ).dropna()

        if len(close) < 60:
            return "HOLD", "Insufficient price history"

        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        rsi = calculate_rsi(close)

        price = float(close.iloc[-1])
        average20 = float(sma20.iloc[-1])
        average50 = float(sma50.iloc[-1])
        rsi_value = float(rsi.iloc[-1])

        values = [
            price,
            average20,
            average50,
            rsi_value,
        ]

        if not all(math.isfinite(v) for v in values):
            return "HOLD", "Invalid indicator values"

        details = (
            f"Price={price:.2f}; "
            f"SMA20={average20:.2f}; "
            f"SMA50={average50:.2f}; "
            f"RSI={rsi_value:.2f}"
        )

        if (
            price > average20
            and average20 > average50
            and 45 <= rsi_value <= 68
        ):
            return "BUY_SIGNAL", details

        return "HOLD", details

    except Exception as exc:
        save_log(
            "ERROR",
            f"{symbol} signal calculation failed: "
            f"{str(exc)[:500]}",
        )

        return "HOLD", "Market-data calculation failed"


# ============================================================
# ALPACA PAPER ACCOUNT
# ============================================================

def check_paper_account():
    account = trading.get_account()
    equity = float(account.equity)

    if not math.isfinite(equity) or equity <= 0:
        raise RuntimeError(
            "Alpaca returned invalid account equity"
        )

    save_log(
        "INFO",
        f"Alpaca paper account connected; "
        f"equity=${equity:.2f}",
    )


# ============================================================
# WATCHLIST SCAN
# ============================================================

def scan_watchlist():
    update_bot_status(
        "Scanning watchlist; paper mode; orders disabled"
    )

    for symbol in WATCHLIST:
        signal, details = get_signal(symbol)

        save_log(
            "INFO",
            f"{symbol}: {signal} | {details}",
        )

    update_bot_status(
        "Watchlist scan completed; orders disabled"
    )


# ============================================================
# MAIN WORKER
# ============================================================

def main():
    save_log(
        "INFO",
        "Worker started; Alpaca paper mode; "
        "automatic order execution disabled",
    )

    while True:
        try:
            config = get_bot_config()

            if config.get("enabled") is not True:
                update_bot_status(
                    "Paused: bot_status.enabled is not true"
                )

            elif str(config.get("mode", "paper")).lower() != "paper":
                update_bot_status(
                    "Paused: database mode must be paper"
                )

            else:
                check_paper_account()
                scan_watchlist()

        except Exception as exc:
            message = str(exc)[:800]

            log.exception("Worker cycle failed")

            save_log(
                "ERROR",
                f"Worker cycle failed: {message}",
            )

            try:
                update_bot_status(
                    f"Worker error: {message}",
                    error=message,
                )

            except Exception:
                log.exception(
                    "Could not update bot_status after failure"
                )

        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
