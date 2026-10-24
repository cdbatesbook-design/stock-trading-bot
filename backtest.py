import os
import time
import math
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
DAILY_WARMUP_DAYS = 180

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

MOMENTUM_CONFIGS = [
    ("MOM_20D_VOL15_ATR2", 20, 1.5, 2.0),
    ("MOM_20D_VOL10_ATR2", 20, 1.0, 2.0),
    ("MOM_55D_VOL15_ATR2", 55, 1.5, 2.0),
    ("MOM_20D_VOL15_ATR3", 20, 1.5, 3.0),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

log = logging.getLogger("stock_backtest")
SESSION = requests.Session()

if not ALPACA_KEY or not ALPACA_SECRET:
    raise RuntimeError(
        "Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY environment variables."
    )

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def parse_timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def iso_utc(value):
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


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

            if response.status_code == 429 or response.status_code >= 500:
                wait = min(60, 2 ** attempt)

                log.warning(
                    "API returned HTTP %s; retrying in %s seconds.",
                    response.status_code,
                    wait,
                )

                time.sleep(wait)
                continue

            response.raise_for_status()

            try:
                return response.json()
            except ValueError as exc:
                raise RuntimeError(
                    f"API returned invalid JSON from {url}: "
                    f"{response.text[:300]}"
                ) from exc

        except requests.RequestException as exc:
            last_error = exc
            wait = min(60, 2 ** attempt)
            time.sleep(wait)

    raise RuntimeError(
        f"API request failed after {retries} attempts: {last_error}"
    )


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


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

    while True:
        query = dict(params)

        if page_token:
            query["page_token"] = page_token

        payload = request_json(
            f"{ALPACA_DATA_URL}/stocks/bars",
            params=query,
        )

        for symbol, rows in (payload.get("bars") or {}).items():
            if symbol in collected:
                collected[symbol].extend(rows)

        page_token = payload.get("next_page_token")

        if not page_token:
            break

    for symbol in collected:
        collected[symbol].sort(
            key=lambda row: row.get("t", "")
        )

    return collected


# ============================================================
# STOCK UNIVERSE
# ============================================================

def discover_symbols():
    log.info("Discovering active, tradable US equity assets.")

    assets = request_json(
        f"{ALPACA_TRADING_URL}/v2/assets",
        params={
            "status": "active",
            "asset_class": "us_equity",
        },
    )

    symbols = []

    for asset in assets:
        symbol = asset.get("symbol", "")

        if asset.get("status") != "active":
            continue

        if asset.get("tradable") is not True:
            continue

        if not symbol or len(symbol) > 6:
            continue

        if not symbol.replace(".", "").isalnum():
            continue

        symbols.append(symbol)

    symbols = sorted(set(symbols))

    log.info("Discovered %s symbols.", len(symbols))

    return symbols


def screen_universe(symbols, start, end):
    log.info("Screening universe for recent price and liquidity.")

    candidates = []

    for batch_number, batch in enumerate(
        chunks(symbols, DAILY_BATCH_SIZE),
        start=1,
    ):
        daily = download_bars(batch, "1Day", start, end)

        for symbol, rows in daily.items():
            if len(rows) < 20:
                continue

            recent = rows[-20:]
            last_close = safe_float(recent[-1].get("c"))

            if last_close < MIN_STOCK_PRICE:
                continue

            avg_dollar_volume = sum(
                safe_float(row.get("c")) * safe_float(row.get("v"))
                for row in recent
            ) / len(recent)

            if avg_dollar_volume >= MIN_AVG_DOLLAR_VOLUME:
                candidates.append({
                    "symbol": symbol,
                    "price": last_close,
                    "avg_dollar_volume": avg_dollar_volume,
                })

        log.info(
            "Screening batch %s complete; %s candidates qualify.",
            batch_number,
            len(candidates),
        )

    candidates.sort(
        key=lambda item: item["avg_dollar_volume"],
        reverse=True,
    )

    selected = candidates[:MAX_SYMBOLS]

    log.info(
        "Selected %s stocks from %s qualifying stocks.",
        len(selected),
        len(candidates),
    )

    for item in selected[:20]:
        log.info(
            "Selected %s: $%.2f, avg dollar volume $%.0f",
            item["symbol"],
            item["price"],
            item["avg_dollar_volume"],
        )

    return [item["symbol"] for item in selected]


# ============================================================
# OPENING-RANGE BREAKOUT
# ============================================================

def regular_session_by_day(rows):
    by_day = {}

    for row in rows:
        timestamp = parse_timestamp(row["t"])
        local = timestamp.astimezone(EASTERN)

        if local.weekday() >= 5:
            continue

        if not (
            dt_time(9, 30) <= local.time() < dt_time(16, 0)
        ):
            continue

        by_day.setdefault(local.date(), []).append({
            "timestamp": timestamp,
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


def generate_orb_candidates(symbol, rows, opening_minutes):
    candidates = []

    for trading_day, bars in regular_session_by_day(rows).items():
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
        entry_price = entry_bar["open"]
        stop_price = range_low

        if entry_price <= stop_price or entry_price <= 0:
            continue

        target_price = (
            entry_price
            + REWARD_RISK_RATIO * (entry_price - stop_price)
        )

        exit_price = None
        exit_time = None
        exit_reason = None

        for bar in bars[signal_index + 1:]:
            if bar["open"] <= stop_price:
                exit_price = bar["open"]
                exit_time = bar["timestamp"]
                exit_reason = "STOP_GAP"
                break

            if bar["open"] >= target_price:
                exit_price = target_price
                exit_time = bar["timestamp"]
                exit_reason = "TARGET"
                break

            hit_stop = bar["low"] <= stop_price
            hit_target = bar["high"] >= target_price

            if hit_stop:
                exit_price = stop_price
                exit_time = bar["timestamp"]
                exit_reason = (
                    "STOP_AMBIGUOUS" if hit_target else "STOP"
                )
                break

            if hit_target:
                exit_price = target_price
                exit_time = bar["timestamp"]
                exit_reason = "TARGET"
                break

        if exit_price is None:
            last_bar = bars[-1]
            exit_price = last_bar["close"]
            exit_time = last_bar["timestamp"]
            exit_reason = "SESSION_CLOSE"

        candidates.append({
            "symbol": symbol,
            "trading_day": str(trading_day),
            "entry_time": entry_bar["timestamp"],
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "exit_reason": exit_reason,
            "strategy_type": f"ORB_{opening_minutes}M",
        })

    return candidates


# ============================================================
# DAILY INDICATORS / MOMENTUM BREAKOUT
# ============================================================

def prepare_daily_bars(rows):
    prepared = []

    for index, row in enumerate(rows):
        timestamp = parse_timestamp(row["t"])

        bar = {
            "timestamp": timestamp,
            "date": timestamp.astimezone(EASTERN).date(),
            "open": safe_float(row.get("o")),
            "high": safe_float(row.get("h")),
            "low": safe_float(row.get("l")),
            "close": safe_float(row.get("c")),
            "volume": safe_float(row.get("v")),
        }

        if index >= 19:
            bar["ma20"] = sum(
                safe_float(x.get("c"))
                for x in rows[index - 19:index + 1]
            ) / 20

        if index >= 49:
            bar["ma50"] = sum(
                safe_float(x.get("c"))
                for x in rows[index - 49:index + 1]
            ) / 50

        if index >= 20:
            prior20 = rows[index - 20:index]

            bar["prior20_high"] = max(
                safe_float(x.get("h"))
                for x in prior20
            )

            bar["prior20_volume_avg"] = sum(
                safe_float(x.get("v"))
                for x in prior20
            ) / 20

            previous_close = safe_float(
                rows[index - 20].get("c")
            )

            if previous_close > 0:
                bar["return20"] = (
                    bar["close"] / previous_close
                ) - 1

        if index >= 55:
            prior55 = rows[index - 55:index]

            bar["prior55_high"] = max(
                safe_float(x.get("h"))
                for x in prior55
            )

        if index >= 14:
            true_ranges = []

            for j in range(index - 13, index + 1):
                high = safe_float(rows[j].get("h"))
                low = safe_float(rows[j].get("l"))

                previous_close = (
                    safe_float(rows[j - 1].get("c"))
                    if j > 0
                    else safe_float(rows[j].get("c"))
                )

                true_ranges.append(
                    max(
                        high - low,
                        abs(high - previous_close),
                        abs(low - previous_close),
                    )
                )

            bar["atr14"] = (
                sum(true_ranges) / len(true_ranges)
            )

        prepared.append(bar)

    return prepared


def generate_momentum_candidates(
    symbol,
    stock_rows,
    spy_rows,
    lookback,
    volume_multiplier,
    atr_multiplier,
    test_start,
    test_end,
):
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

        if signal["timestamp"] < test_start:
            continue

        if signal["timestamp"] >= test_end:
            continue

        market = spy_by_date.get(signal["date"])

        if market is None:
            continue

        if signal["close"] < MIN_STOCK_PRICE:
            continue

        if signal["close"] <= 0:
            continue

        if not all(
            key in signal
            for key in ("ma20", "ma50", "atr14", "return20")
        ):
            continue

        if not all(
            key in market
            for key in ("ma50", "return20")
        ):
            continue

        if signal["close"] <= signal["ma20"]:
            continue

        if signal["close"] <= signal["ma50"]:
            continue

        if market["close"] <= 0:
            continue

        if market["close"] <= market["ma50"]:
            continue

        if signal["return20"] <= market["return20"]:
            continue

        if lookback == 20:
            breakout_level = signal.get("prior20_high")
        else:
            breakout_level = signal.get("prior55_high")

        if breakout_level is None or breakout_level <= 0:
            continue

        if signal["close"] <= breakout_level:
            continue

        volume_avg = signal.get("prior20_volume_avg", 0)

        if volume_avg <= 0:
            continue

        if signal["volume"] < volume_multiplier * volume_avg:
            continue

        entry_price = entry_bar["open"]

        if entry_price <= 0:
            continue

        if entry_price > breakout_level * (1 + MAX_ENTRY_GAP_PCT):
            continue

        atr = signal["atr14"]

        if atr <= 0:
            continue

        stop_price = entry_price - atr_multiplier * atr

        if stop_price <= 0 or stop_price >= entry_price:
            continue

        target_price = (
            entry_price
            + REWARD_RISK_RATIO * (entry_price - stop_price)
        )

        exit_price = None
        exit_time = None
        exit_reason = None

        for future in stock[index + 1:]:
            if future["timestamp"] >= test_end:
                break

            if future["open"] <= stop_price:
                exit_price = future["open"]
                exit_time = future["timestamp"]
                exit_reason = "STOP_GAP"
                break

            if future["open"] >= target_price:
                exit_price = target_price
                exit_time = future["timestamp"]
                exit_reason = "TARGET"
                break

            hit_stop = future["low"] <= stop_price
            hit_target = future["high"] >= target_price

            # Conservative assumption when daily OHLC cannot
            # establish whether stop or target occurred first.
            if hit_stop:
                exit_price = stop_price
                exit_time = future["timestamp"]
                exit_reason = (
                    "STOP_AMBIGUOUS" if hit_target else "STOP"
                )
                break

            if hit_target:
                exit_price = target_price
                exit_time = future["timestamp"]
                exit_reason = "TARGET"
                break

        if exit_price is None:
            eligible = [
                bar for bar in stock
                if entry_bar["timestamp"] <= bar["timestamp"] < test_end
            ]

            if not eligible:
                continue

            last_bar = eligible[-1]
            exit_price = last_bar["close"]
            exit_time = last_bar["timestamp"]
            exit_reason = "END_OF_TEST"

        if exit_time < entry_bar["timestamp"]:
            continue

        candidates.append({
            "symbol": symbol,
            "trading_day": str(signal["date"]),
            "entry_time": entry_bar["timestamp"],
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "exit_reason": exit_reason,
            "strategy_type": strategy_name,
        })

    return candidates


# ============================================================
# SIMULATION / PERFORMANCE METRICS
# ============================================================

def simulate_strategy(candidates, starting_capital):
    candidates = sorted(
        candidates,
        key=lambda item: (item["entry_time"], item["symbol"]),
    )

    cash = float(starting_capital)
    open_positions = []
    trades = []
    skipped = 0

    for candidate in candidates:
        entry_time = candidate["entry_time"]
        still_open = []

        # Settle positions whose exits occur before this entry.
        for position in open_positions:
            if position["exit_time"] <= entry_time:
                proceeds = (
                    position["shares"]
                    * position["exit_price"]
                    * (1 - SLIPPAGE_BPS / 10000.0)
                )

                cash += proceeds
                trades.append(position["trade"])
            else:
                still_open.append(position)

        open_positions = still_open

        if len(open_positions) >= MAX_POSITIONS:
            skipped += 1
            continue

        entry_price = safe_float(candidate["entry_price"])
        exit_price = safe_float(candidate["exit_price"])
        stop_price = safe_float(candidate["stop_price"])

        if (
            entry_price <= 0
            or exit_price <= 0
            or stop_price <= 0
            or stop_price >= entry_price
        ):
            skipped += 1
            continue

        # Cash is reduced when a position is purchased.
        # Fractional shares are allowed.
        committed = sum(
            position["shares"] * position["entry_price"]
            for position in open_positions
        )

        current_equity = max(0.0, cash + committed)

        risk_budget = current_equity * RISK_PER_TRADE
        risk_per_share = entry_price - stop_price

        if risk_per_share <= 0:
            skipped += 1
            continue

        shares_by_risk = risk_budget / risk_per_share

        shares_by_position_cap = (
            current_equity * MAX_POSITION_EQUITY_PCT
        ) / entry_price

        entry_fill = (
            entry_price * (1 + SLIPPAGE_BPS / 10000.0)
        )

        available_cash = max(0.0, cash)

        shares_by_cash = available_cash / entry_fill

        shares = min(
            shares_by_risk,
            shares_by_position_cap,
            shares_by_cash,
        )

        if not math.isfinite(shares) or shares <= 0:
            skipped += 1
            continue

        exit_fill = (
            exit_price * (1 - SLIPPAGE_BPS / 10000.0)
        )

        entry_cost = shares * entry_fill
        cash -= entry_cost

        pnl = shares * (exit_fill - entry_fill)

        trade = {
            "symbol": candidate["symbol"],
            "strategy_type": candidate["strategy_type"],
            "entry_time": entry_time,
            "exit_time": candidate["exit_time"],
            "entry_price": entry_fill,
            "exit_price": exit_fill,
            "shares": shares,
            "pnl": pnl,
            "return_pct": (
                (exit_fill / entry_fill - 1) * 100
            ),
            "exit_reason": candidate["exit_reason"],
        }

        open_positions.append({
            "exit_time": candidate["exit_time"],
            "shares": shares,
            "entry_price": entry_fill,
            "exit_price": exit_price,
            "trade": trade,
        })

    # Settle all remaining positions in exit-time order.
    for position in sorted(
        open_positions,
        key=lambda item: item["exit_time"],
    ):
        proceeds = (
            position["shares"]
            * position["exit_price"]
            * (1 - SLIPPAGE_BPS / 10000.0)
        )

        cash += proceeds
        trades.append(position["trade"])

    trades.sort(key=lambda trade: trade["exit_time"])

    ending_equity = cash

    pnl_values = [
        trade["pnl"]
        for trade in trades
    ]

    wins = sum(1 for pnl in pnl_values if pnl > 0)
    losses = sum(1 for pnl in pnl_values if pnl < 0)

    gross_profit = sum(
        pnl for pnl in pnl_values if pnl > 0
    )

    gross_loss = abs(sum(
        pnl for pnl in pnl_values if pnl < 0
    ))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    # Approximate drawdown from realized trade-by-trade equity.
    equity = float(starting_capital)
    peak = equity
    max_drawdown = 0.0

    for trade in trades:
        equity += trade["pnl"]
        peak = max(peak, equity)

        if peak > 0:
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

    return {
        "starting_capital": float(starting_capital),
        "ending_equity": ending_equity,
        "net_profit": ending_equity - starting_capital,
        "return_pct": (
            (ending_equity / starting_capital - 1) * 100
            if starting_capital
            else 0
        ),
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": (
            wins / len(trades) * 100
            if trades
            else 0.0
        ),
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown,
        "skipped_trades": skipped,
    }


# ============================================================
# SPY BENCHMARK
# ============================================================

def calculate_spy_benchmark(spy_rows, start, end, capital):
    rows = prepare_daily_bars(spy_rows)

    eligible = [
        bar for bar in rows
        if start <= bar["timestamp"] < end and bar["close"] > 0
    ]

    if len(eligible) < 2:
        return None

    first_close = eligible[0]["close"]
    last_close = eligible[-1]["close"]

    ending = capital * last_close / first_close

    return {
        "strategy": "SPY_BUY_HOLD",
        "starting_capital": capital,
        "ending_equity": ending,
        "net_profit": ending - capital,
        "return_pct": (ending / capital - 1) * 100,
        "trades": 1,
        "wins": int(ending > capital),
        "losses": int(ending < capital),
        "win_rate": 100.0 if ending > capital else 0.0,
        "profit_factor": 0.0,
        "max_drawdown_pct": 0.0,
        "skipped_trades": 0,
    }


# ============================================================
# SUPABASE RESULTS
# ============================================================

def save_result(result, start_date, end_date):
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.warning(
            "Supabase credentials missing; results will only appear in logs."
        )
        return False

    payload = {
        "run_id": str(uuid.uuid4()),
        "strategy": result["strategy"],
        "starting_capital": result["starting_capital"],
        "ending_equity": result["ending_equity"],
        "net_profit": result["net_profit"],
        "return_pct": result["return_pct"],
        "trades": result["trades"],
        "wins": result["wins"],
        "losses": result["losses"],
        "win_rate": result["win_rate"],
        "profit_factor": result["profit_factor"],
        "max_drawdown_pct": result["max_drawdown_pct"],
        "skipped_trades": result["skipped_trades"],
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
    }

    if not math.isfinite(payload["profit_factor"]):
        payload["profit_factor"] = None

    try:
        response = SESSION.post(
            f"{SUPABASE_URL}/rest/v1/backtest_results",
            json=payload,
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            timeout=45,
        )

        response.raise_for_status()
        return True

    except requests.RequestException as exc:
        log.error(
            "Could not save %s to Supabase: %s; response=%s",
            result["strategy"],
            exc,
            getattr(exc.response, "text", "")[:500],
        )
        return False


def print_result(result):
    pf = result["profit_factor"]

    if not math.isfinite(pf):
        pf_text = "infinity"
    else:
        pf_text = f"{pf:.2f}"

    log.info(
        "RESULT %(strategy)s | capital=$%(starting_capital).2f | "
        "ending=$%(ending_equity).2f | "
        "P/L=$%(net_profit).2f (%(return_pct).2f%%) | "
        "trades=%(trades)d | win rate=%(win_rate).1f%% | "
        "PF=%(pf)s | max DD=%(max_drawdown_pct).2f%% | "
        "skipped=%(skipped_trades)d",
        {**result, "pf": pf_text},
    )


# ============================================================
# MAIN
# ============================================================

def main():
    now = datetime.now(timezone.utc).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    test_start = now - timedelta(days=BACKTEST_DAYS)
    data_start = test_start - timedelta(days=DAILY_WARMUP_DAYS)
    data_end = now

    log.info(
        "Backtest period: %s through %s UTC.",
        test_start.date(),
        data_end.date(),
    )

    all_symbols = discover_symbols()

    symbols = screen_universe(
        all_symbols,
        data_start,
        data_end,
    )

    if not symbols:
        raise RuntimeError(
            "No stocks passed the liquidity screen. "
            "Check Alpaca data access, feed, and date range."
        )

    log.info(
        "Downloading daily bars for %s selected symbols plus SPY.",
        len(symbols),
    )

    daily_data = {}

    symbols_with_spy = list(dict.fromkeys(symbols + ["SPY"]))

    for batch_number, batch in enumerate(
        chunks(symbols_with_spy, DAILY_BATCH_SIZE),
        start=1,
    ):
        daily_data.update(
            download_bars(
                batch,
                "1Day",
                data_start,
                data_end,
            )
        )

        log.info(
            "Daily data batch %s downloaded.",
            batch_number,
        )

    spy_rows = daily_data.get("SPY", [])

    if len(spy_rows) < 60:
        raise RuntimeError(
            f"Not enough SPY daily bars for indicators: {len(spy_rows)}"
        )

    log.info(
        "Downloading minute bars for ORB strategies; this can take a while."
    )

    minute_data = {}

    for batch_number, batch in enumerate(
        chunks(symbols, SYMBOL_BATCH_SIZE),
        start=1,
    ):
        minute_data.update(
            download_bars(
                batch,
                "1Min",
                test_start,
                data_end,
            )
        )

        log.info(
            "Minute data batch %s/%s downloaded.",
            batch_number,
            math.ceil(len(symbols) / SYMBOL_BATCH_SIZE),
        )

    candidates_by_strategy = {}

    # Opening-range breakout strategies.
    for opening_minutes in ORB_STRATEGIES:
        name = f"ORB_{opening_minutes}M"
        candidates_by_strategy[name] = []

        for symbol in symbols:
            candidates_by_strategy[name].extend(
                generate_orb_candidates(
                    symbol,
                    minute_data.get(symbol, []),
                    opening_minutes,
                )
            )

        log.info(
            "%s generated %s candidate trades.",
            name,
            len(candidates_by_strategy[name]),
        )

    # Momentum strategies.
    for (
        name,
        lookback,
        volume_multiplier,
        atr_multiplier,
    ) in MOMENTUM_CONFIGS:
        candidates_by_strategy[name] = []

        for symbol in symbols:
            candidates_by_strategy[name].extend(
                generate_momentum_candidates(
                    symbol,
                    daily_data.get(symbol, []),
                    spy_rows,
                    lookback,
                    volume_multiplier,
                    atr_multiplier,
                    test_start,
                    data_end,
                )
            )

        log.info(
            "%s generated %s candidate trades.",
            name,
            len(candidates_by_strategy[name]),
        )

    results = []

    for strategy_name, candidates in candidates_by_strategy.items():
        for capital in STARTING_CAPITALS:
            result = simulate_strategy(
                candidates,
                capital,
            )

            result["strategy"] = strategy_name

            print_result(result)
            save_result(result, test_start, data_end)

            results.append(result)

    # Compare each starting capital against SPY buy-and-hold.
    for capital in STARTING_CAPITALS:
        benchmark = calculate_spy_benchmark(
            spy_rows,
            test_start,
            data_end,
            capital,
        )

        if benchmark:
            print_result(benchmark)
            save_result(benchmark, test_start, data_end)
            results.append(benchmark)

    log.info(
        "Backtest completed. Saved or printed %s result rows.",
        len(results),
    )

    log.warning(
        "This is a historical simulation, not a prediction of future performance. "
        "The current stock universe may introduce survivorship bias."
    )


if __name__ == "__main__":
    main()
