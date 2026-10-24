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

MIN_PRICE = 1.00
MAX_PRICE = 10.00

MIN_AVERAGE_DOLLAR_VOLUME = 5_000_000
MIN_DAILY_MOVE_PERCENT = 3.0
MIN_RELATIVE_VOLUME = 1.5

BREAKOUT_LOOKBACK_BARS = 20
MAX_CANDIDATES = 50

DATA_FEED = "iex"
PAPER_MODE_ONLY = True

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"

REQUEST_TIMEOUT = 30
REQUEST_DELAY_SECONDS = 0.25
BATCH_SIZE = 100


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

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

required_variables = {
    "ALPACA_API_KEY": ALPACA_API_KEY,
    "ALPACA_SECRET_KEY": ALPACA_SECRET_KEY,
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_SERVICE_ROLE_KEY,
}

missing_variables = [
    name for name, value in required_variables.items()
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
last_asset_refresh = 0


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
        logger.error("Supabase logging failed: %s", exc)


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

        updates = {}

        possible_values = {
            "status_message": message,
            "last_heartbeat": now,
            "updated_at": now,
            "last_scan": now,
            "last_error": error,
        }

        for column, value in possible_values.items():
            if column in existing:
                updates[column] = value

        if updates:
            (
                supabase.table("bot_status")
                .update(updates)
                .eq("id", 1)
                .execute()
            )

    except Exception as exc:
        logger.error("Could not update bot status: %s", exc)


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

def alpaca_get(url, params=None):
    response = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code == 429:
        raise RuntimeError(
            "Alpaca rate limit reached. The next scan will retry."
        )

    response.raise_for_status()

    return response.json()


# ============================================================
# LOAD STOCK UNIVERSE
# ============================================================

def get_all_eligible_stocks():
    url = f"{ALPACA_BASE_URL}/v2/assets"

    data = alpaca_get(
        url,
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

        # Avoid unusual symbol formats that may be incompatible
        # with data providers or common stock symbols.
        if not symbol.replace(".", "").isalnum():
            continue

        symbols.append(symbol)

    symbols = sorted(set(symbols))

    log_event(
        "INFO",
        f"Loaded {len(symbols)} active tradable stock symbols.",
    )

    return symbols


# ============================================================
# BATCH STOCK BARS
# ============================================================

def get_stock_bars(symbols, timeframe, start, end, limit=10000):
    """
    Download bars in batches from Alpaca's market-data API.
    Supports pagination through next_page_token.
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

    return all_bars


# ============================================================
# DAILY CANDIDATE FILTER
# ============================================================

def find_daily_candidates(symbols):
    now = datetime.now(timezone.utc)

    start = (now - timedelta(days=45)).isoformat()
    end = now.isoformat()

    log_event(
        "INFO",
        f"Downloading daily bars for {len(symbols)} symbols.",
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
            if len(bars) < 22:
                continue

            bars = sorted(bars, key=lambda bar: bar["t"])

            previous_bar = bars[-2]
            latest_bar = bars[-1]

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

            recent_bars = bars[-21:-1]

            dollar_volumes = [
                float(bar["c"]) * float(bar["v"])
                for bar in recent_bars
            ]

            average_dollar_volume = (
                sum(dollar_volumes) / len(dollar_volumes)
                if dollar_volumes else 0
            )

            if average_dollar_volume < MIN_AVERAGE_DOLLAR_VOLUME:
                continue

            if daily_move < MIN_DAILY_MOVE_PERCENT:
                continue

            candidates.append({
                "symbol": symbol,
                "price": price,
                "daily_move_percent": daily_move,
                "average_dollar_volume": average_dollar_volume,
            })

        except (KeyError, TypeError, ValueError, ZeroDivisionError):
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
        f"Daily filter selected {len(candidates)} candidates.",
    )

    return candidates


# ============================================================
# INTRADAY BREAKOUT CHECK
# ============================================================

def check_intraday_breakouts(candidates):
    if not candidates:
        return []

    symbols = [item["symbol"] for item in candidates]

    now = datetime.now(timezone.utc)
    start = (now - timedelta(hours=6)).isoformat()
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

        bars = bars_by_symbol.get(symbol, [])
        bars = sorted(bars, key=lambda bar: bar["t"])

        # Need 20 completed comparison bars plus the latest bar.
        if len(bars) < BREAKOUT_LOOKBACK_BARS + 1:
            continue

        previous_bars = bars[-(BREAKOUT_LOOKBACK_BARS + 1):-1]
        latest = bars[-1]

        previous_high = max(
            float(bar["h"]) for bar in previous_bars
        )

        previous_volumes = [
            float(bar["v"]) for bar in previous_bars
        ]

        average_volume = (
            sum(previous_volumes) / len(previous_volumes)
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

        signal = {
            "symbol": symbol,
            "signal": "BREAKOUT_CANDIDATE",
            "price": round(latest_close, 4),
            "breakout_level": round(previous_high, 4),
            "relative_volume": round(relative_volume, 2),
            "daily_move_percent": round(
                candidate["daily_move_percent"], 2
            ),
            "average_dollar_volume": round(
                candidate["average_dollar_volume"], 2
            ),
            "bar_time": latest.get("t"),
        }

        signals.append(signal)

    return signals


# ============================================================
# SAVE RESULTS
# ============================================================

def record_signal(signal):
    message = (
        f"{signal['signal']} {signal['symbol']} | "
        f"Price=${signal['price']:.4f} | "
        f"Breakout=${signal['breakout_level']:.4f} | "
        f"Relative volume={signal['relative_volume']:.2f}x | "
        f"Daily move={signal['daily_move_percent']:.2f}%"
    )

    log_event("INFO", message)


# ============================================================
# ONE COMPLETE SCAN
# ============================================================

def run_market_scan(symbols):
    candidates = find_daily_candidates(symbols)

    signals = check_intraday_breakouts(candidates)

    if not signals:
        log_event(
            "INFO",
            "Scan complete. No qualifying breakout candidates.",
        )
    else:
        for signal in signals:
            record_signal(signal)

        log_event(
            "INFO",
            f"Scan complete. Found {len(signals)} breakout candidates.",
        )

    return len(signals)


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    global stock_symbols, last_asset_refresh

    log_event(
        "INFO",
        "Momentum breakout scanner starting. "
        "Signal-only mode; no orders will be submitted.",
    )

    while True:
        try:
            config = get_bot_config()

            if config.get("enabled") is not True:
                update_bot_status("Scanner disabled in bot_status.")
                time.sleep(60)
                continue

            mode = str(config.get("mode", "paper")).lower()

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

            if (
                not stock_symbols
                or now_seconds - last_asset_refresh
                >= ASSET_REFRESH_SECONDS
            ):
                stock_symbols = get_all_eligible_stocks()
                last_asset_refresh = now_seconds

            if not stock_symbols:
                raise RuntimeError(
                    "No eligible stock symbols were returned by Alpaca."
                )

            update_bot_status("Scanning for momentum breakouts.")

            signal_count = run_market_scan(stock_symbols)

            update_bot_status(
                f"Scan complete. {signal_count} breakout candidates found."
            )

        except Exception as exc:
            logger.exception("Scanner cycle failed.")

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
