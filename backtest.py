import os
import time
import logging
import uuid
from datetime import datetime, timedelta, timezone, time as dt_time
from zoneinfo import ZoneInfo

import requests


# ============================================================
# CONFIGURATION
# ============================================================

ALPACA_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY")

ALPACA_DATA_URL = "https://data.alpaca.markets/v2"
ALPACA_TRADING_URL = "https://paper-api.alpaca.markets"

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

EASTERN = ZoneInfo("America/New_York")

BACKTEST_DAYS = 180
DAILY_WARMUP_DAYS = 150

MAX_SYMBOLS = 100
SYMBOL_BATCH_SIZE = 10
DAILY_BATCH_SIZE = 100

STARTING_CAPITALS = [500, 1000, 2500, 5000]

ORB_STRATEGIES = [5, 15, 30]

MAX_POSITIONS = 3

RISK_PER_TRADE = 0.005
MAX_POSITION_EQUITY_PCT = 0.30

SLIPPAGE_BPS = 5
REWARD_RISK_RATIO = 2.0

MIN_STOCK_PRICE = 5.00
MIN_AVG_DOLLAR_VOLUME = 20_000_000

MAX_ENTRY_GAP_PCT = 0.03

DATA_FEED = "iex"

# Four momentum configurations:
# (name, breakout lookback, volume multiplier, ATR stop multiplier)
MOMENTUM_CONFIGS = [
    ("MOM_20D_VOL15_ATR2", 20, 1.5, 2.0),
    ("MOM_20D_VOL10_ATR2", 20, 1.0, 2.0),
    ("MOM_55D_VOL15_ATR2", 55, 1.5, 2.0),
    ("MOM_20D_VOL15_ATR3", 20, 1.5, 3.0),
]

LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s: %(message)s",
)

log = logging.getLogger("stock_backtest")

SESSION = requests.Session()

if not ALPACA_KEY or not ALPACA_SECRET:
    raise RuntimeError(
        "Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY."
    )

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def parse_timestamp(value):
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )


def iso_utc(value):
    return value.astimezone(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def request_json(url, params=None, headers=None, retries=6):
    last_error = None

    for attempt in range(retries):
        try:
            response = SESSION.get(
                url,
                params=params,
                headers=headers or HEADERS,
                timeout=90,
            )

            if response.status_code == 429:
                wait = min(60, 2 ** attempt)
                log.warning(
                    "Rate limited. Waiting %s seconds.",
                    wait,
                )
                time.sleep(wait)
                continue

            if response.status_code >= 500:
                wait = min(60, 2 ** attempt)
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(60, 2 ** attempt))

    raise RuntimeError(
        f"API request failed after retries: {last_error}"
    )


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# ALPACA ASSET UNIVERSE
# ============================================================

def discover_symbols():
    log.info("Discovering active US equity assets.")

    assets = request_json(
        f"{ALPACA_TRADING_URL}/v2/assets",
        params={
            "status": "active",
            "asset_class": "us_equity",
        },
    )

    symbols = []

    for asset in assets:
        if asset.get("status") != "active":
            continue

        if asset.get("tradable") is not True:
            continue

        symbol = asset.get("symbol", "")

        if not symbol or len(symbol) > 6:
            continue

        if not symbol.replace(".", "").isalnum():
            continue

        symbols.append(symbol)

    symbols = sorted(set(symbols))

    log.info(
        "Discovered %s active, tradable equity symbols.",
        len(symbols),
    )

    return symbols


# ============================================================
# HISTORICAL BAR DOWNLOAD
# ============================================================

def download_bars(symbols, timeframe, start, end):
    if not symbols:
        return {}

    collected = {symbol: [] for symbol in symbols}

    params = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": iso_utc(start),
        "end": iso_utc(end),
        "limit": 10000,
        "adjustment": "split",
        "feed": DATA_FEED,
        "sort": "asc",
    }

    page_token = None
    page_count = 0

    while True:
        query = dict(params)

        if page_token:
            query["page_token"] = page_token

        payload = request_json(
            f"{ALPACA_DATA_URL}/stocks/bars",
            params=query,
        )

        page_count += 1

        bars_by_symbol = payload.get("bars") or {}

        for symbol, rows in bars_by_symbol.items():
            if symbol in collected:
                collected[symbol].extend(rows)

        page_token = payload.get("next_page_token")

        if not page_token:
            break

        if page_count % 20 == 0:
            log.info(
                "%s data: downloaded %s pages.",
                timeframe,
                page_count,
            )

    for symbol in collected:
        collected[symbol].sort(
            key=lambda row: row["t"]
        )

    return collected


# ============================================================
# LIQUIDITY SCREEN
# ============================================================

def screen_universe(symbols, start, end):
    log.info("Screening stocks for price and liquidity.")

    candidates = []
    total = len(symbols)

    for batch_number, batch in enumerate(
        chunks(symbols, DAILY_BATCH_SIZE),
        start=1,
    ):
        daily = download_bars(
            batch,
            "1Day",
            start,
            end,
        )

        for symbol, rows in daily.items():
            if len(rows) < 20:
                continue

            recent = rows[-20:]
            last_close = safe_float(recent[-1].get("c"))

            if last_close < MIN_STOCK_PRICE:
                continue

            dollar_volumes = [
                safe_float(row.get("c"))
                * safe_float(row.get("v"))
                for row in recent
            ]

            avg_dollar_volume = (
                sum(dollar_volumes) / len(dollar_volumes)
            )

            if avg_dollar_volume < MIN_AVG_DOLLAR_VOLUME:
                continue

            candidates.append({
                "symbol": symbol,
                "price": last_close,
                "avg_dollar_volume": avg_dollar_volume,
            })

        log.info(
            "Screening batch %s complete. "
            "%s candidates qualify so far.",
            batch_number,
            len(candidates),
        )

    candidates.sort(
        key=lambda item: item["avg_dollar_volume"],
        reverse=True,
    )

    selected = candidates[:MAX_SYMBOLS]

    log.info(
        "%s stocks passed the liquidity screen; "
        "selected %s.",
        len(candidates),
        len(selected),
    )

    for item in selected[:20]:
        log.info(
            "Selected %s: price=$%.2f, average dollar volume=$%.0f",
            item["symbol"],
            item["price"],
            item["avg_dollar_volume"],
        )

    return [item["symbol"] for item in selected]


# ============================================================
# MINUTE BARS: REGULAR SESSION
# ============================================================

def regular_session_by_day(rows):
    by_day = {}

    for row in rows:
        timestamp = parse_timestamp(row["t"])
        local = timestamp.astimezone(EASTERN)

        if local.weekday() >= 5:
            continue

        if not (
            dt_time(9, 30)
            <= local.time()
            < dt_time(16, 0)
        ):
            continue

        day = local.date()

        by_day.setdefault(day, []).append({
            "timestamp": timestamp,
            "local": local,
            "open": safe_float(row.get("o")),
            "high": safe_float(row.get("h")),
            "low": safe_float(row.get("l")),
            "close": safe_float(row.get("c")),
            "volume": safe_float(row.get("v")),
        })

    for day in by_day:
        by_day[day].sort(
            key=lambda bar: bar["timestamp"]
        )

    return by_day


# ============================================================
# OPENING-RANGE BREAKOUT
# ============================================================

def generate_orb_candidates(symbol, rows, opening_minutes):
    candidates = []
    days = regular_session_by_day(rows)

    for trading_day, bars in days.items():
        if len(bars) <= opening_minutes:
            continue

        opening_bars = bars[:opening_minutes]

        range_high = max(
            bar["high"] for bar in opening_bars
        )

        range_low = min(
            bar["low"] for bar in opening_bars
        )

        if range_high <= range_low:
            continue

        signal_index = None

        for index in range(
            opening_minutes,
            len(bars) - 1,
        ):
            if bars[index]["close"] > range_high:
                signal_index = index
                break

        if signal_index is None:
            continue

        entry_bar = bars[signal_index + 1]
        raw_entry = entry_bar["open"]
        raw_stop = range_low

        if raw_entry <= raw_stop or raw_entry <= 0:
            continue

        risk_per_share = raw_entry - raw_stop

        target = (
            raw_entry
            + REWARD_RISK_RATIO * risk_per_share
        )

        exit_price = None
        exit_timestamp = None
        exit_reason = None

        for index in range(signal_index + 1, len(bars)):
            bar = bars[index]

            if bar["open"] <= raw_stop:
                exit_price = bar["open"]
                exit_timestamp = bar["timestamp"]
                exit_reason = "STOP_GAP"
                break

            if bar["open"] >= target:
                exit_price = target
                exit_timestamp = bar["timestamp"]
                exit_reason = "TARGET"
                break

            hit_stop = bar["low"] <= raw_stop
            hit_target = bar["high"] >= target

            if hit_stop and hit_target:
                exit_price = raw_stop
                exit_timestamp = bar["timestamp"]
                exit_reason = "STOP_AMBIGUOUS"
                break

            if hit_stop:
                exit_price = raw_stop
                exit_timestamp = bar["timestamp"]
                exit_reason = "STOP"
                break

            if hit_target:
                exit_price = target
                exit_timestamp = bar["timestamp"]
                exit_reason = "TARGET"
                break

        if exit_price is None:
            last_bar = bars[-1]
            exit_price = last_bar["close"]
            exit_timestamp = last_bar["timestamp"]
            exit_reason = "SESSION_CLOSE"

        candidates.append({
            "symbol": symbol,
            "trading_day": str(trading_day),
            "entry_time": entry_bar["timestamp"],
            "exit_time": exit_timestamp,
            "entry_price": raw_entry,
            "exit_price": exit_price,
            "stop_price": raw_stop,
            "target_price": target,
            "exit_reason": exit_reason,
            "strategy_type": f"ORB_{opening_minutes}M",
        })

    return candidates


# ============================================================
# DAILY BARS: PREPARE INDICATORS
# ============================================================

def prepare_daily_bars(rows):
    """
    Adds moving averages, volume average, ATR, and returns.

    All indicators at index i use data available by the
    close of bar i. Entries are scheduled for the next
    available session's open.
    """

    prepared = []

    for index, row in enumerate(rows):
        bar = {
            "timestamp": parse_timestamp(row["t"]),
            "date": parse_timestamp(row["t"]).astimezone(
                EASTERN
            ).date(),
            "open": safe_float(row.get("o")),
            "high": safe_float(row.get("h")),
            "low": safe_float(row.get("l")),
            "close": safe_float(row.get("c")),
            "volume": safe_float(row.get("v")),
        }

        if index >= 19:
            last20 = rows[index - 19:index + 1]

            bar["ma20"] = sum(
                safe_float(item.get("c"))
                for item in last20
            ) / 20

        if index >= 49:
            last50 = rows[index - 49:index + 1]

            bar["ma50"] = sum(
                safe_float(item.get("c"))
                for item in last50
            ) / 50

        if index >= 20:
            prior20 = rows[index - 20:index]

            bar["prior20_high"] = max(
                safe_float(item.get("h"))
                for item in prior20
            )

            bar["prior20_volume_avg"] = sum(
                safe_float(item.get("v"))
                for item in prior20
            ) / 20

            prior_close = safe_float(
                rows[index - 20].get("c")
            )

            if prior_close > 0:
                bar["return20"] = (
                    bar["close"] / prior_close
                ) - 1

        if index >= 55:
            prior55 = rows[index - 55:index]

            bar["prior55_high"] = max(
                safe_float(item.get("h"))
                for item in prior55
            )

        if index >= 14:
            true_ranges = []

            for j in range(index - 13, index + 1):
                current = rows[j]

                high = safe_float(current.get("h"))
                low = safe_float(current.get("l"))

                previous_close = safe_float(
                    rows[j - 1].get("c")
                ) if j > 0 else safe_float(
                    current.get("c")
                )

                true_range = max(
                    high - low,
                    abs(high - previous_close),
                    abs(low - previous_close),
                )

                true_ranges.append(true_range)

            bar["atr14"] = (
                sum(true_ranges) / len(true_ranges)
            )

        prepared.append(bar)

    return prepared


# ============================================================
# MOMENTUM BREAKOUT SIGNALS
# ============================================================

def generate_momentum_candidates(
    symbol,
    stock_rows,
    spy_rows,
    lookback,
    volume_multiplier,
    atr_multiplier,
):
    """
    Long-only daily momentum breakout.

    Signal is evaluated after a completed daily close.
    Entry is at the next available session's open.

    The breakout level excludes the signal day's high.
    The volume average excludes the signal day's volume.
    """

    candidates = []

    stock = prepare_daily_bars(stock_rows)
    spy = prepare_daily_bars(spy_rows)

    spy_by_date = {
        bar["date"]: bar
        for bar in spy
    }

    strategy_name = (
        f"MOM_{lookback}D_VOL"
        f"{int(volume_multiplier * 10)}_ATR"
        f"{int(atr_multiplier)}"
    )

    for index in range(1, len(stock) - 1):
        signal = stock[index]
        entry_bar = stock[index + 1]

        if signal["date"] not in spy_by_date:
            continue

        market = spy_by_date[signal["date"]]

        if (
            signal["close"] < MIN_STOCK_PRICE
            or signal["close"] <= 0
        ):
            continue

        if "ma20" not in signal or "ma50" not in signal:
            continue

        if "atr14" not in signal:
            continue

        if "return20" not in signal:
            continue

        if signal["close"] <= signal["ma20"]:
            continue

        if signal["close"] <= signal["ma50"]:
            continue

        if market["close"] <= 0:
            continue

        if "ma50" not in market:
            continue

        if "return20" not in market:
            continue

        # Market regime filter.
        if market["close"] <= market["ma50"]:
            continue

        # Stock must outperform SPY over 20 trading days.
        if signal["return20"] <= market["return20"]:
            continue

        # Previous completed highs exclude the signal bar.
        if lookback == 20:
            breakout_level = signal.get("prior20_high")
        else:
            breakout_level = signal.get("prior55_high")

        if breakout_level is None or breakout_level <= 0:
            continue

        # Close must break the previous high.
        if signal["close"] <= breakout_level:
            continue

        volume_avg = signal.get("prior20_volume_avg", 0)

        if volume_avg <= 0:
            continue

        if signal["volume"] < volume_multiplier * volume_avg:
            continue

        # Avoid chasing an excessive overnight gap.
        entry_price = entry_bar["open"]

        if entry_price <= 0:
            continue

        if (
            entry_price
            > breakout_level * (1 + MAX_ENTRY_GAP_PCT)
        ):
            continue

        atr = signal["atr14"]

        if atr <= 0:
            continue

        stop_price = (
            entry_price - atr_multiplier * atr
        )

        if stop_price <= 0 or stop_price >= entry_price:
            continue

        risk_per_share = entry_price - stop_price

        target_price = (
            entry_price
            + REWARD_RISK_RATIO * risk_per_share
        )

        exit_price = None
        exit_timestamp = None
        exit_reason = None

        # Start examining prices on the entry session.
        for future in stock[index + 1:]:
            if future["timestamp"] < entry_bar["timestamp"]:
                continue

            # Gap-aware exits.
            if future["open"] <= stop_price:
                exit_price = future["open"]
                exit_timestamp = future["timestamp"]
                exit_reason = "STOP_GAP"
                break

            if future["open"] >= target_price:
                exit_price = target_price
                exit_timestamp = future["timestamp"]
                exit_reason = "TARGET"
                break

            hit_stop = future["low"] <= stop_price
            hit_target = future["high"] >= target_price

            # Daily OHLC cannot reveal which was hit first.
            # Assume the stop was hit first.
            if hit_stop and hit_target:
                exit_price = stop_price
                exit_timestamp = future["timestamp"]
                exit_reason = "STOP_AMBIGUOUS"
                break

            if hit_stop:
                exit_price = stop_price
                exit_timestamp = future["timestamp"]
                exit_reason = "STOP"
                break

            if hit_target:
                exit_price = target_price
                exit_timestamp = future["timestamp"]
                exit_reason = "TARGET"
                break

        if exit_price is None:
            last_bar = stock[-1]
            exit_price = last_bar["close"]
            exit_timestamp = last_bar["timestamp"]
            exit_reason = "END_OF_TEST"

        candidates.append({
            "symbol": symbol,
            "trading_day": str(signal["date"]),
            "entry_time": entry_bar["timestamp"],
            "exit_time": exit_t
