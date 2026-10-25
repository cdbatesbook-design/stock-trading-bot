"""
Combined stock and cryptocurrency momentum scanner.

RESEARCH / SIGNAL-ONLY MODE:
- Never submits orders.
- Preserves the existing stock-scanning strategy.
- Adds crypto-market scanning through Alpaca.
- Uses the existing Supabase configuration and logging.
- Does not modify backtest.py or submit live/paper orders.

Crypto signals are research candidates, NOT buy instructions.
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone

import requests
from supabase import create_client


# ============================================================
# CONFIGURATION
# ============================================================

SCAN_INTERVAL_SECONDS = 300
ASSET_REFRESH_SECONDS = 86400

# Existing stock scanner settings
MIN_PRICE = 1.00
MAX_PRICE = 10.00

MIN_AVERAGE_DOLLAR_VOLUME = 5_000_000
MIN_DAILY_MOVE_PERCENT = 3.0
MIN_RELATIVE_VOLUME = 1.5

BREAKOUT_LOOKBACK_BARS = 20
MAX_CANDIDATES = 50

DATA_FEED = "iex"

# Safety: this worker never submits orders.
PAPER_MODE_ONLY = True
SIGNAL_ONLY = True

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"

# Alpaca crypto market-data location.
CRYPTO_DATA_URL = (
    "https://data.alpaca.markets/v1beta3/crypto/us"
)

REQUEST_TIMEOUT = 30
REQUEST_DELAY_SECONDS = 0.25
BATCH_SIZE = 100

# Crypto scanner settings
CRYPTO_MIN_DAILY_MOVE_PERCENT = 3.0
CRYPTO_MIN_RELATIVE_VOLUME = 1.5
CRYPTO_MIN_AVERAGE_DOLLAR_VOLUME = 1_000_000
CRYPTO_BREAKOUT_LOOKBACK_BARS = 20
CRYPTO_MAX_CANDIDATES = 100

# Keep the initial crypto research universe limited to USD,
# USDT, and USDC quote pairs for more comparable volume screening.
CRYPTO_ALLOWED_QUOTES = {"USD", "USDT", "USDC"}

# These are research risk parameters, not order-execution settings.
CRYPTO_MAX_RESEARCH_POSITIONS = 3
CRYPTO_MAX_POSITION_ALLOCATION = 0.05

CRYPTO_PROFIT_TARGETS = [0.005, 0.0075, 0.01, 0.015]
CRYPTO_STOP_METHOD = "ATR_RESEARCH_ONLY"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger("momentum_scanner")


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

ALPACA_API_KEY = (
    os.getenv("APCA_API_KEY_ID")
    or os.getenv("ALPACA_API_KEY")
)

ALPACA_SECRET_KEY = (
    os.getenv("APCA_API_SECRET_KEY")
    or os.getenv("ALPACA_SECRET_KEY")
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY"
)

required_variables = {
    "ALPACA_API_KEY / APCA_API_KEY_ID": ALPACA_API_KEY,
    "ALPACA_SECRET_KEY / APCA_API_SECRET_KEY": ALPACA_SECRET_KEY,
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

session = requests.Session()

session.headers.update({
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
})


# ============================================================
# STATE
# ============================================================

stock_symbols = []
crypto_symbols = []

last_stock_asset_refresh = 0
last_crypto_asset_refresh = 0


# ============================================================
# LOGGING TO SUPABASE
# ============================================================

def log_event(level, message):
    message = str(message)[:4000]

    getattr(logger, level.lower(), logger.info)(message)

    try:
        supabase.table("bot_logs").insert({
            "level": level.upper(),
            "message": message,
        }).execute()

    except Exception as exc:
        logger.error(
            "Supabase logging failed: %s",
            exc,
        )


def update_bot_status(message, error=None):
    try:
        result = (
            supabase.table("bot_status")
            .select("*")
            .eq("id", 1)
            .limit(1)
            .execute()
        )

        rows = result.data or []

        if not rows:
            log_event(
                "WARNING",
                "No bot_status row with id=1 was found.",
            )
            return

        existing = rows[0]
        now = datetime.now(timezone.utc).isoformat()

        possible_values = {
            "status_message": message,
            "last_heartbeat": now,
            "updated_at": now,
            "last_scan": now,
            "last_error": error,
        }

        updates = {
            column: value
            for column, value in possible_values.items()
            if column in existing
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
            "Could not update bot status: %s",
            exc,
        )


def get_bot_config():
    result = (
        supabase.table("bot_status")
        .select("*")
        .eq("id", 1)
        .limit(1)
        .execute()
    )

    rows = result.data or []

    return rows[0] if rows else {}


# ============================================================
# ALPACA HTTP HELPERS
# ============================================================

def alpaca_get(url, params=None, retries=4):
    """
    GET helper with bounded retries for temporary errors
    and rate limiting.
    """

    last_error = None

    for attempt in range(retries):
        try:
            response = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if (
                response.status_code == 429
                or response.status_code >= 500
            ):
                wait = min(30, 2 ** attempt)

                log_event(
                    "WARNING",
                    f"Alpaca HTTP {response.status_code}; "
                    f"retrying in {wait}s.",
                )

                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        except (
            requests.RequestException,
            ValueError,
        ) as exc:
            last_error = exc

            if attempt < retries - 1:
                time.sleep(min(30, 2 ** attempt))

    raise RuntimeError(
        f"Alpaca GET failed after {retries} attempts: "
        f"{last_error}"
    )


# ============================================================
# STOCK SCANNER: EXISTING FUNCTIONALITY
# ============================================================

def get_all_eligible_stocks():
    data = alpaca_get(
        f"{ALPACA_BASE_URL}/v2/assets",
        params={
            "status": "active",
            "asset_class": "us_equity",
        },
    )

    symbols = []

    for asset in data:
        symbol = asset.get("symbol", "")

        if not symbol:
            continue

        if asset.get("status") != "active":
            continue

        if asset.get("tradable") is not True:
            continue

        if not symbol.replace(".", "").isalnum():
            continue

        symbols.append(symbol)

    symbols = sorted(set(symbols))

    log_event(
        "INFO",
        f"Loaded {len(symbols)} active tradable stocks.",
    )

    return symbols


def get_stock_bars(
    symbols,
    timeframe,
    start,
    end,
    limit=10000,
):
    """
    Download stock bars in batches, following pagination tokens.
    """

    all_bars = {}

    for offset in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[offset:offset + BATCH_SIZE]
        page_token = None

        while True:
            params = {
                "symbols": ",".join(batch),
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "limit": limit,
                "feed": DATA_FEED,
                "adjustment": "raw",
                "sort": "asc",
            }

            if page_token:
                params["page_token"] = page_token

            data = alpaca_get(
                f"{ALPACA_DATA_URL}/v2/stocks/bars",
                params=params,
            )

            for symbol, bars in data.get("bars", {}).items():
                all_bars.setdefault(symbol, []).extend(bars)

            page_token = data.get("next_page_token")

            if not page_token:
                break

            time.sleep(REQUEST_DELAY_SECONDS)

        time.sleep(REQUEST_DELAY_SECONDS)

    for bars in all_bars.values():
        bars.sort(key=lambda bar: bar["t"])

    return all_bars


def find_daily_candidates(symbols):
    now = datetime.now(timezone.utc)

    start = (
        now - timedelta(days=45)
    ).isoformat()

    end = now.isoformat()

    log_event(
        "INFO",
        f"Downloading daily stock bars for {len(symbols)} symbols.",
    )

    bars_by_symbol = get_stock_bars(
        symbols=symbols,
        timeframe="1Day",
        start=start,
        end=end,
    )

    candidates = []

    for symbol, bars in bars_by_symbol.items():
        try:
            bars = sorted(bars, key=lambda bar: bar["t"])

            if len(bars) < 22:
                continue

            # Use the last completed daily bar.
            completed = bars[:-1]

            if len(completed) < 21:
                continue

            previous_bar = completed[-2]
            latest_bar = completed[-1]

            price = float(latest_bar["c"])
            previous_close = float(previous_bar["c"])

            if price < MIN_PRICE or price > MAX_PRICE:
                continue

            if previous_close <= 0:
                continue

            daily_move = (
                (price - previous_close)
                / previous_close
            ) * 100

            recent_bars = completed[-21:-1]

            dollar_volumes = [
                float(bar["c"]) * float(bar["v"])
                for bar in recent_bars
            ]

            average_dollar_volume = (
                sum(dollar_volumes) / len(dollar_volumes)
                if dollar_volumes
                else 0
            )

            if (
                average_dollar_volume
                < MIN_AVERAGE_DOLLAR_VOLUME
            ):
                continue

            if daily_move < MIN_DAILY_MOVE_PERCENT:
                continue

            candidates.append({
                "symbol": symbol,
                "price": price,
                "daily_move_percent": daily_move,
                "average_dollar_volume": average_dollar_volume,
            })

        except (
            KeyError,
            TypeError,
            ValueError,
            ZeroDivisionError,
        ):
            continue

    candidates.sort(
        key=lambda item: (
            item["daily_move_percent"],
            item["average_dollar_volume"],
        ),
        reverse=True,
    )

    candidates = candidates[:MAX_CANDIDATES]

    log_event(
        "INFO",
        f"Stock daily filter selected {len(candidates)} candidates.",
    )

    return candidates


def check_intraday_breakouts(candidates):
    if not candidates:
        return []

    symbols = [
        item["symbol"]
        for item in candidates
    ]

    now = datetime.now(timezone.utc)

    start = (
        now - timedelta(hours=6)
    ).isoformat()

    end = now.isoformat()

    bars_by_symbol = get_stock_bars(
        symbols=symbols,
        timeframe="1Min",
        start=start,
        end=end,
    )

    signals = []

    for candidate in candidates:
        symbol = candidate["symbol"]

        bars = sorted(
            bars_by_symbol.get(symbol, []),
            key=lambda bar: bar["t"],
        )

        # Exclude the latest bar because it may still be forming.
        bars = bars[:-1]

        if len(bars) < BREAKOUT_LOOKBACK_BARS + 1:
            continue

        previous_bars = bars[
            -(BREAKOUT_LOOKBACK_BARS + 1):-1
        ]

        latest = bars[-1]

        previous_high = max(
            float(bar["h"])
            for bar in previous_bars
        )

        previous_volumes = [
            float(bar["v"])
            for bar in previous_bars
        ]

        average_volume = (
            sum(previous_volumes)
            / len(previous_volumes)
        )

        latest_close = float(latest["c"])
        latest_volume = float(latest["v"])

        if average_volume <= 0:
            continue

        relative_volume = latest_volume / average_volume

        if latest_close <= previous_high:
            continue

        if relative_volume < MIN_RELATIVE_VOLUME:
            continue

        signals.append({
            "symbol": symbol,
            "signal": "BREAKOUT_CANDIDATE",
            "price": round(latest_close, 8),
            "breakout_level": round(previous_high, 8),
            "relative_volume": round(relative_volume, 2),
            "daily_move_percent": round(
                candidate["daily_move_percent"],
                2,
            ),
            "average_dollar_volume": round(
                candidate["average_dollar_volume"],
                2,
            ),
            "bar_time": latest.get("t"),
        })

    return signals


# ============================================================
# CRYPTO SCANNER: NEW FUNCTIONALITY
# ============================================================

def get_all_eligible_crypto():
    """
    Retrieve active, tradable crypto pairs from Alpaca.

    Only pairs quoted in USD, USDT, or USDC are included so
    dollar-volume estimates are reasonably comparable.

    This is a market scanner, not an order-placement function.
    """

    data = alpaca_get(
        f"{ALPACA_BASE_URL}/v2/assets",
        params={
            "status": "active",
            "asset_class": "crypto",
        },
    )

    symbols = []

    for asset in data:
        symbol = asset.get("symbol", "")

        if not symbol:
            continue

        if asset.get("status") != "active":
            continue

        if asset.get("tradable") is not True:
            continue

        # Modern Alpaca pair format: BTC/USD.
        if "/" not in symbol:
            continue

        parts = symbol.split("/")

        if len(parts) != 2:
            continue

        base, quote = parts

        if not base or quote not in CRYPTO_ALLOWED_QUOTES:
            continue

        symbols.append(symbol)

    symbols = sorted(set(symbols))

    log_event(
        "INFO",
        f"Loaded {len(symbols)} active tradable crypto pairs "
        f"quoted in USD, USDT, or USDC.",
    )

    return symbols


def get_crypto_bars(
    symbols,
    timeframe,
    start,
    end,
    limit=10000,
):
    """
    Download historical crypto bars from Alpaca.

    Crypto symbols use the format BTC/USD.
    The crypto API does not use the stock feed parameter.
    """

    all_bars = {}

    for offset in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[offset:offset + BATCH_SIZE]
        page_token = None

        while True:
            params = {
                "symbols": ",".join(batch),
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "limit": limit,
                "sort": "asc",
            }

            if page_token:
                params["page_token"] = page_token

            data = alpaca_get(
                f"{CRYPTO_DATA_URL}/bars",
                params=params,
            )

            for symbol, bars in data.get("bars", {}).items():
                all_bars.setdefault(symbol, []).extend(bars)

            page_token = data.get("next_page_token")

            if not page_token:
                break

            time.sleep(REQUEST_DELAY_SECONDS)

        time.sleep(REQUEST_DELAY_SECONDS)

    for bars in all_bars.values():
        bars.sort(key=lambda bar: bar["t"])

    return all_bars


def find_crypto_daily_candidates(symbols):
    """
    Find crypto pairs with positive daily momentum and sufficient
    recent dollar volume.

    Crypto trades seven days per week, so no stock-market calendar
    restrictions are applied.
    """

    if not symbols:
        return []

    now = datetime.now(timezone.utc)

    start = (
        now - timedelta(days=35)
    ).isoformat()

    end = now.isoformat()

    log_event(
        "INFO",
        f"Downloading daily crypto bars for {len(symbols)} pairs.",
    )

    bars_by_symbol = get_crypto_bars(
        symbols=symbols,
        timeframe="1Day",
        start=start,
        end=end,
    )

    candidates = []

    for symbol, bars in bars_by_symbol.items():
        try:
            bars = sorted(
                bars,
                key=lambda bar: bar["t"],
            )

            # The most recent daily candle may be incomplete.
            completed = bars[:-1]

            if len(completed) < 21:
                continue

            previous_bar = completed[-2]
            latest_bar = completed[-1]

            price = float(latest_bar["c"])
            previous_close = float(previous_bar["c"])

            if price <= 0 or previous_close <= 0:
                continue

            daily_move = (
                (price - previous_close)
                / previous_close
            ) * 100

            # Use the preceding 20 completed daily bars.
            volume_bars = completed[-21:-1]

            dollar_volumes = [
                float(bar["c"]) * float(bar["v"])
                for bar in volume_bars
            ]

            average_dollar_volume = (
                sum(dollar_volumes) / len(dollar_volumes)
                if dollar_volumes
                else 0
            )

            if (
                average_dollar_volume
                < CRYPTO_MIN_AVERAGE_DOLLAR_VOLUME
            ):
                continue

            if daily_move < CRYPTO_MIN_DAILY_MOVE_PERCENT:
                continue

            candidates.append({
                "symbol": symbol,
                "price": price,
                "daily_move_percent": daily_move,
                "average_dollar_volume": average_dollar_volume,
            })

        except (
            KeyError,
            TypeError,
            ValueError,
            ZeroDivisionError,
        ):
            continue

    candidates.sort(
        key=lambda item: (
            item["daily_move_percent"],
            item["average_dollar_volume"],
        ),
        reverse=True,
    )

    candidates = candidates[:CRYPTO_MAX_CANDIDATES]

    log_event(
        "INFO",
        f"Crypto daily filter selected {len(candidates)} candidates.",
    )

    return candidates


def check_crypto_breakouts(candidates):
    """
    Identify crypto breakout candidates using completed one-minute
    bars. This function does not buy or sell anything.
    """

    if not candidates:
        return []

    symbols = [
        item["symbol"]
        for item in candidates
    ]

    now = datetime.now(timezone.utc)

    start = (
        now - timedelta(hours=6)
    ).isoformat()

    end = now.isoformat()

    bars_by_symbol = get_crypto_bars(
        symbols=symbols,
        timeframe="1Min",
        start=start,
        end=end,
    )

    signals = []

    for candidate in candidates:
        symbol = candidate["symbol"]

        bars = sorted(
            bars_by_symbol.get(symbol, []),
            key=lambda bar: bar["t"],
        )

        # Exclude the latest bar because it may still be forming.
        bars = bars[:-1]

        required_bars = CRYPTO_BREAKOUT_LOOKBACK_BARS + 1

        if len(bars) < required_bars:
            continue

        previous_bars = bars[
            -(CRYPTO_BREAKOUT_LOOKBACK_BARS + 1):-1
        ]

        latest = bars[-1]

        previous_high = max(
            float(bar["h"])
            for bar in previous_bars
        )

        previous_volumes = [
            float(bar["v"])
            for bar in previous_bars
        ]

        average_volume = (
            sum(previous_volumes)
            / len(previous_volumes)
        )

        latest_close = float(latest["c"])
        latest_volume = float(latest["v"])

        if latest_close <= 0 or average_volume <= 0:
            continue

        relative_volume = latest_volume / average_volume

        if latest_close <= previous_high:
            continue

        if relative_volume < CRYPTO_MIN_RELATIVE_VOLUME:
            continue

        signals.append({
            "symbol": symbol,
            "signal": "CRYPTO_BREAKOUT_CANDIDATE",
            "price": round(latest_close, 10),
            "breakout_level": round(previous_high, 10),
            "relative_volume": round(relative_volume, 2),
            "daily_move_percent": round(
                candidate["daily_move_percent"],
                2,
            ),
            "average_dollar_volume": round(
                candidate["average_dollar_volume"],
                2,
            ),
            "bar_time": latest.get("t"),
            "research_only": True,
            "max_research_positions": (
                CRYPTO_MAX_RESEARCH_POSITIONS
            ),
            "max_research_allocation": (
                CRYPTO_MAX_POSITION_ALLOCATION
            ),
            "stop_method": CRYPTO_STOP_METHOD,
            "profit_targets": CRYPTO_PROFIT_TARGETS,
        })

    return signals


# ============================================================
# SAVE RESULTS
# ============================================================

def record_signal(signal):
    message = (
        f"{signal['signal']} {signal['symbol']} | "
        f"Price=${signal['price']:.8f} | "
        f"Breakout=${signal['breakout_level']:.8f} | "
        f"Relative volume={signal['relative_volume']:.2f}x | "
        f"Daily move={signal['daily_move_percent']:.2f}%"
    )

    log_event("INFO", message)


# ============================================================
# ONE COMPLETE STOCK SCAN
# ============================================================

def run_market_scan(symbols):
    candidates = find_daily_candidates(symbols)

    signals = check_intraday_breakouts(candidates)

    if not signals:
        log_event(
            "INFO",
            "Stock scan complete. No qualifying breakout candidates.",
        )
    else:
        for signal in signals:
            record_signal(signal)

        log_event(
            "INFO",
            f"Stock scan complete. "
            f"Found {len(signals)} breakout candidates.",
        )

    return len(signals)


# ============================================================
# ONE COMPLETE CRYPTO SCAN
# ============================================================

def run_crypto_scan(symbols):
    candidates = find_crypto_daily_candidates(symbols)

    signals = check_crypto_breakouts(candidates)

    if not signals:
        log_event(
            "INFO",
            "Crypto scan complete. "
            "No qualifying breakout candidates.",
        )
    else:
        for signal in signals:
            record_signal(signal)

        log_event(
            "INFO",
            f"Crypto scan complete. "
            f"Found {len(signals)} breakout candidates.",
        )

    return len(signals)


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    global stock_symbols
    global crypto_symbols
    global last_stock_asset_refresh
    global last_crypto_asset_refresh

    log_event(
        "INFO",
        "Combined stock and crypto momentum scanner starting. "
        "Signal-only mode. No orders will be submitted.",
    )

    while True:
        try:
            config = get_bot_config()

            if config.get("enabled") is not True:
                update_bot_status(
                    "Scanner disabled in bot_status."
                )

                time.sleep(60)
                continue

            mode = str(
                config.get("mode", "paper")
            ).lower()

            if mode != "paper":
                update_bot_status(
                    "Scanner stopped: mode must be paper.",
                    error="Unsafe mode configuration",
                )

                log_event(
                    "ERROR",
                    "Mode is not paper. No scan performed.",
                )

                time.sleep(60)
                continue

            now_seconds = time.time()

            # Refresh stock symbols once per day.
            if (
                not stock_symbols
                or now_seconds - last_stock_asset_refresh
                >= ASSET_REFRESH_SECONDS
            ):
                stock_symbols = get_all_eligible_stocks()
                last_stock_asset_refresh = now_seconds

            # Refresh crypto symbols once per day.
            if (
                not crypto_symbols
                or now_seconds - last_crypto_asset_refresh
                >= ASSET_REFRESH_SECONDS
            ):
                crypto_symbols = get_all_eligible_crypto()
                last_crypto_asset_refresh = now_seconds

            if not stock_symbols:
                raise RuntimeError(
                    "No eligible stock symbols returned by Alpaca."
                )

            if not crypto_symbols:
                log_event(
                    "WARNING",
                    "No eligible crypto pairs returned by Alpaca. "
                    "The next cycle will retry.",
                )

            update_bot_status(
                "Scanning stocks and eligible crypto pairs. "
                "Signal-only mode."
            )

            stock_signal_count = run_market_scan(
                stock_symbols
            )

            crypto_signal_count = 0

            if crypto_symbols:
                crypto_signal_count = run_crypto_scan(
                    crypto_symbols
                )

            status_message = (
                f"Scan complete. "
                f"Stock candidates: {stock_signal_count}; "
                f"crypto candidates: {crypto_signal_count}. "
                f"No orders submitted."
            )

            update_bot_status(status_message)

        except Exception as exc:
            logger.exception(
                "Scanner cycle failed."
            )

            update_bot_status(
                "Scanner cycle failed.",
                error=str(exc)[:1000],
            )

            log_event(
                "ERROR",
                f"Scanner cycle failed: {exc}",
            )

        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
