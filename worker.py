import os
import time
import logging
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from supabase import create_client
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus


# ============================================================
# CONFIGURATION
# ============================================================

SCAN_INTERVAL_SECONDS = 300
ASSET_REFRESH_SECONDS = 86400
BATCH_SIZE = 50

# Safety: this worker NEVER submits orders.
AUTOMATIC_ORDER_EXECUTION = False

# Alpaca paper trading is required.
PAPER_TRADING = True

MIN_PRICE = 2.00
MIN_AVERAGE_VOLUME = 100000

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger("stock_bot")


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get(
    "SUPABASE_SERVICE_ROLE_KEY"
)

required_variables = {
    "ALPACA_API_KEY": ALPACA_API_KEY,
    "ALPACA_SECRET_KEY": ALPACA_SECRET_KEY,
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_SERVICE_ROLE_KEY,
}

missing_variables = [
    name
    for name, value in required_variables.items()
    if not value
]

if missing_variables:
    raise RuntimeError(
        "Missing environment variables: "
        + ", ".join(missing_variables)
    )


# ============================================================
# CLIENTS
# ============================================================

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
)

alpaca = TradingClient(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    paper=True,
)


# ============================================================
# DATABASE LOGGING
# ============================================================

def log_event(message, level="INFO"):
    """Write messages to Render logs and Supabase."""

    log_level = level.upper()

    if log_level == "ERROR":
        logger.error(message)
    elif log_level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)

    try:
        supabase.table("bot_logs").insert({
            "level": log_level,
            "message": str(message)[:1000],
        }).execute()

    except Exception as exc:
        logger.error(
            "Could not write to Supabase bot_logs: %s",
            exc,
        )


# ============================================================
# BOT STATUS
# ============================================================

def get_bot_config():
    """Read bot configuration from Supabase."""

    try:
        response = (
            supabase.table("bot_status")
            .select("*")
            .eq("id", 1)
            .limit(1)
            .execute()
        )

        if response.data:
            return response.data[0]

        log_event(
            "No bot_status row found with id=1.",
            "WARNING",
        )

    except Exception as exc:
        log_event(
            f"Unable to read bot_status: {exc}",
            "ERROR",
        )

    return {}


def update_bot_status(message, error=None):
    """Update supported status columns without assuming the schema."""

    try:
        response = (
            supabase.table("bot_status")
            .select("*")
            .eq("id", 1)
            .limit(1)
            .execute()
        )

        if not response.data:
            log_event(
                "Cannot update status: bot_status id=1 is missing.",
                "WARNING",
            )
            return

        existing_row = response.data[0]
        now = datetime.now(timezone.utc).isoformat()

        possible_values = {
            "status_message": message,
            "last_heartbeat": now,
            "updated_at": now,
            "last_scan": now,
        }

        if error:
            possible_values["last_error"] = str(error)[:1000]

        else:
            possible_values["last_error"] = None

        updates = {
            key: value
            for key, value in possible_values.items()
            if key in existing_row
        }

        if updates:
            (
                supabase.table("bot_status")
                .update(updates)
                .eq("id", 1)
                .execute()
            )

    except Exception as exc:
        logger.error(
            "Unable to update bot status: %s",
            exc,
        )


# ============================================================
# DISCOVER ALL ELIGIBLE STOCKS
# ============================================================

def get_all_eligible_stocks():
    """
    Retrieve active, tradable U.S. equity assets from Alpaca.

    This removes the old fixed 21-symbol watchlist.
    Actual coverage depends on Alpaca's available assets
    and the market-data subscription.
    """

    log_event("Refreshing eligible stock universe.")

    request = GetAssetsRequest(
        status=AssetStatus.ACTIVE,
        asset_class=AssetClass.US_EQUITY,
    )

    assets = alpaca.get_all_assets(request)

    symbols = []

    for asset in assets:
        try:
            if not asset.tradable:
                continue

            symbol = asset.symbol

            if not symbol:
                continue

            # Skip symbols that are unsuitable for ordinary
            # ticker-based scanning, such as dotted share classes.
            if not symbol.isalnum():
                continue

            symbols.append(symbol)

        except Exception:
            continue

    symbols = sorted(set(symbols))

    log_event(
        f"Eligible stock universe refreshed: "
        f"{len(symbols)} symbols."
    )

    return symbols


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(series, period=14):
    """Calculate the Relative Strength Index."""

    delta = series.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

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

    relative_strength = (
        average_gain / average_loss.replace(0, float("nan"))
    )

    rsi = 100 - (100 / (1 + relative_strength))

    return rsi


# ============================================================
# STOCK ANALYSIS
# ============================================================

def analyze_stock(symbol):
    """
    Analyze a single symbol using recent daily price history.

    Produces signals only. Does not place orders.
    """

    try:
        data = yf.download(
            symbol,
            period="6mo",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
            timeout=20,
        )

        if data is None or data.empty:
            return {
                "symbol": symbol,
                "signal": "NO_DATA",
            }

        # yfinance can return MultiIndex columns.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required_columns = {"Close", "Volume"}

        if not required_columns.issubset(set(data.columns)):
            return {
                "symbol": symbol,
                "signal": "INVALID_DATA",
            }

        close = data["Close"].dropna()
        volume = data["Volume"].dropna()

        if len(close) < 50 or len(volume) < 20:
            return {
                "symbol": symbol,
                "signal": "INSUFFICIENT_HISTORY",
            }

        last_price = float(close.iloc[-1])
        average_volume = float(volume.tail(20).mean())

        sma20 = float(close.tail(20).mean())
        sma50 = float(close.tail(50).mean())

        rsi_series = calculate_rsi(close)
        rsi_value = float(rsi_series.iloc[-1])

        if not all(
            pd.notna(value)
            for value in [
                last_price,
                average_volume,
                sma20,
                sma50,
                rsi_value,
            ]
        ):
            return {
                "symbol": symbol,
                "signal": "INVALID_INDICATORS",
            }

        if last_price < MIN_PRICE:
            signal = "FILTERED_LOW_PRICE"

        elif average_volume < MIN_AVERAGE_VOLUME:
            signal = "FILTERED_LOW_VOLUME"

        elif last_price > sma20 > sma50 and 40 <= rsi_value <= 65:
            signal = "BUY_SIGNAL"

        elif last_price < sma20 and sma20 < sma50:
            signal = "SELL_SIGNAL"

        else:
            signal = "HOLD"

        return {
            "symbol": symbol,
            "signal": signal,
            "price": round(last_price, 4),
            "sma20": round(sma20, 4),
            "sma50": round(sma50, 4),
            "rsi": round(rsi_value, 2),
            "average_volume": int(average_volume),
        }

    except Exception as exc:
        return {
            "symbol": symbol,
            "signal": "ERROR",
            "error": str(exc)[:500],
        }


# ============================================================
# RECORD SCAN RESULTS
# ============================================================

def record_result(result):
    """Record scan outcomes in the bot logs."""

    symbol = result.get("symbol", "UNKNOWN")
    signal = result.get("signal", "UNKNOWN")

    if signal == "ERROR":
        log_event(
            f"Analysis failed for {symbol}: "
            f"{result.get('error', 'Unknown error')}",
            "WARNING",
        )
        return

    if signal in ("BUY_SIGNAL", "SELL_SIGNAL"):
        log_event(
            f"{signal} {symbol} | "
            f"Price={result.get('price')} | "
            f"SMA20={result.get('sma20')} | "
            f"SMA50={result.get('sma50')} | "
            f"RSI={result.get('rsi')}"
        )


# ============================================================
# MARKET SCANNER
# ============================================================

def run_market_scan(symbols):
    """
    Scan the discovered universe in batches.

    This is a broad-market scanner, not a guarantee that every
    symbol can be analyzed on every cycle. Data providers can
    throttle requests or fail to return some symbols.
    """

    total = len(symbols)

    if total == 0:
        log_event(
            "No eligible stocks available to scan.",
            "WARNING",
        )
        return

    log_event(
        f"Starting market scan for {total} eligible symbols."
    )

    scanned = 0
    signals = 0
    errors = 0

    for start in range(0, total, BATCH_SIZE):
        batch = symbols[start:start + BATCH_SIZE]

        for symbol in batch:
            result = analyze_stock(symbol)
            scanned += 1

            if result.get("signal") in (
                "BUY_SIGNAL",
                "SELL_SIGNAL",
            ):
                signals += 1

            if result.get("signal") == "ERROR":
                errors += 1

            record_result(result)

            # Pace requests to reduce the risk of throttling.
            time.sleep(0.25)

        log_event(
            f"Scan progress: {scanned}/{total} symbols; "
            f"signals={signals}; errors={errors}."
        )

    log_event(
        f"Market scan complete. "
        f"Scanned={scanned}; "
        f"signals={signals}; "
        f"errors={errors}."
    )


# ============================================================
# MAIN WORKER
# ============================================================

def main():
    log_event(
        "Stock scanner starting. "
        "Alpaca paper mode enabled. "
        "Automatic order execution disabled."
    )

    update_bot_status(
        "Starting broad-market stock scanner."
    )

    symbols = []
    last_asset_refresh = 0

    while True:
        try:
            config = get_bot_config()

            enabled = config.get("enabled") is True
            mode = str(config.get("mode", "paper")).lower()

            if not enabled:
                update_bot_status(
                    "Paused: bot_status.enabled is not true."
                )

                time.sleep(30)
                continue

            if mode != "paper":
                update_bot_status(
                    "Stopped: only paper mode is allowed."
                )

                log_event(
                    "Safety stop: bot mode is not paper.",
                    "ERROR",
                )

                time.sleep(60)
                continue

            now = time.time()

            if (
                not symbols
                or now - last_asset_refresh >= ASSET_REFRESH_SECONDS
            ):
                symbols = get_all_eligible_stocks()
                last_asset_refresh = time.time()

            if not symbols:
                update_bot_status(
                    "No eligible stocks found."
                )

                time.sleep(60)
                continue

            account = alpaca.get_account()

            log_event(
                f"Paper account connected. "
                f"Equity={account.equity}; "
                f"Buying power={account.buying_power}."
            )

            update_bot_status(
                f"Scanning {len(symbols)} eligible stocks. "
                "Paper mode; orders disabled."
            )

            run_market_scan(symbols)

            update_bot_status(
                f"Last scan completed. "
                f"Universe size={len(symbols)}. "
                "Orders disabled."
            )

            time.sleep(SCAN_INTERVAL_SECONDS)

        except Exception as exc:
            log_event(
                f"Worker cycle failed: {exc}",
                "ERROR",
            )

            update_bot_status(
                "Worker encountered an error.",
                error=str(exc),
            )

            time.sleep(60)


if __name__ == "__main__":
    main()
