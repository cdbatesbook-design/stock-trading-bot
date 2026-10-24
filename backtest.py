import os
import time
import logging
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
MAX_SYMBOLS = 100
SYMBOL_BATCH_SIZE = 10
DAILY_BATCH_SIZE = 200

STARTING_CAPITALS = [500, 1000, 2500, 5000]
STRATEGIES = [5, 15, 30]

MAX_POSITIONS = 3

# Risk no more than 1% of current equity per trade.
RISK_PER_TRADE = 0.01

# Limit each position to 33% of current equity.
MAX_POSITION_EQUITY_PCT = 0.33

# Simulated slippage, applied on both entry and exit.
SLIPPAGE_BPS = 5

# Target is 2 times the entry-to-stop risk.
REWARD_RISK_RATIO = 2.0

# Stock must be at least $5 and have sufficient liquidity.
MIN_STOCK_PRICE = 5.00
MIN_AVG_DOLLAR_VOLUME = 20_000_000

# Alpaca IEX historical data feed.
DATA_FEED = "iex"

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

        # Exclude unusual symbol formats and OTC-style symbols.
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
    """
    Downloads all available pages for the requested symbols.

    Returns:
        {symbol: [bar, bar, ...]}

    Each bar includes its actual timestamp and OHLCV values.
    """

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
            if symbol not in collected:
                continue

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
    log.info("Screening symbols using daily historical bars.")

    candidates = []

    for batch in chunks(symbols, DAILY_BATCH_SIZE):
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

            last_close = float(recent[-1]["c"])

            if last_close < MIN_STOCK_PRICE:
                continue

            dollar_volumes = [
                float(row["c"]) * float(row["v"])
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
            "Daily screening progress: %s/%s symbols; "
            "%s currently qualify.",
            min(
                symbols.index(batch[-1]) + 1,
                len(symbols),
            ) if batch else 0,
            len(symbols),
            len(candidates),
        )

    candidates.sort(
        key=lambda item: item["avg_dollar_volume"],
        reverse=True,
    )

    selected = candidates[:MAX_SYMBOLS]

    log.info(
        "%s symbols passed the liquidity screen; "
        "selecting the top %s.",
        len(candidates),
        len(selected),
    )

    for item in selected[:20]:
        log.info(
            "Candidate: %s price=$%.2f average daily "
            "dollar volume=$%.0f",
            item["symbol"],
            item["price"],
            item["avg_dollar_volume"],
        )

    return [item["symbol"] for item in selected]


# ============================================================
# REGULAR-SESSION MINUTE BARS
# ============================================================

def regular_session_by_day(rows):
    """
    Groups minute bars by Eastern trading date.

    Only regular-session bars from 9:30 AM through
    3:59 PM Eastern are included.
    """

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
            "open": float(row["o"]),
            "high": float(row["h"]),
            "low": float(row["l"]),
            "close": float(row["c"]),
            "volume": float(row["v"]),
        })

    for day in by_day:
        by_day[day].sort(
            key=lambda bar: bar["timestamp"]
        )

    return by_day


# ============================================================
# OPENING-RANGE BREAKOUT CANDIDATES
# ============================================================

def generate_candidates(symbol, rows, opening_minutes):
    """
    Long-only opening-range breakout.

    Rules:
      1. Calculate the opening range using the first N bars.
      2. Signal when a later bar closes above the range high.
      3. Enter at the NEXT bar's open.
      4. Stop at the opening-range low.
      5. Target at 2R.
      6. If stop and target are touched in the same bar,
         assume the stop was hit first.
      7. If neither is hit, exit at the final regular-session
         bar's close.

    Entry and exit timestamps are actual bar timestamps.
    """

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

        # The opening range must finish before a breakout
        # can be signaled.
        signal_index = None

        for index in range(opening_minutes, len(bars) - 1):
            if bars[index]["close"] > range_high:
                signal_index = index
                break

        if signal_index is None:
            continue

        entry_index = signal_index + 1
        entry_bar = bars[entry_index]

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

        for index in range(entry_index, len(bars)):
            bar = bars[index]

            # Handle gaps through stop/target levels.
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

            # Conservative assumption when intrabar order
            # cannot be determined from one-minute OHLC data.
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

        if exit_timestamp < entry_bar["timestamp"]:
            continue

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
            "opening_minutes": opening_minutes,
        })

    return candidates


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

def simulate_portfolio(candidates, starting_capital):
    """
    Simulates one strategy/account combination.

    Constraints:
      - At most three simultaneous positions.
      - Risk-based position sizing.
      - No borrowing or negative cash.
      - Fractional shares are allowed in the simulation.
      - Slippage is applied on entry and exit.
      - Positions are closed using their actual exit times.
    """

    slippage = SLIPPAGE_BPS / 10000.0

    candidates = sorted(
        candidates,
        key=lambda trade: (
            trade["entry_time"],
            trade["symbol"],
        ),
    )

    cash = float(starting_capital)
    active = []
    closed_trades = []
    skipped = 0

    equity_curve = [float(starting_capital)]

    def close_position(position):
        nonlocal cash

        cash += (
            position["shares"]
            * position["effective_exit"]
        )

        closed_trades.append({
            "symbol": position["symbol"],
            "entry_time": position["entry_time"],
            "exit_time": position["exit_time"],
            "shares": position["shares"],
            "entry_price": position["effective_entry"],
            "exit_price": position["effective_exit"],
            "pnl": position["pnl"],
            "exit_reason": position["exit_reason"],
        })

        equity_curve.append(cash)

    for trade in candidates:
        entry_time = trade["entry_time"]

        # Close every position whose actual exit time has
        # arrived before processing this entry.
        still_active = []

        for position in active:
            if position["exit_time"] <= entry_time:
                close_position(position)
            else:
                still_active.append(position)

        active = still_active

        # Reject a new trade if three positions are open.
        if len(active) >= MAX_POSITIONS:
            skipped += 1
            continue

        effective_entry = (
            trade["entry_price"] * (1 + slippage)
        )

        effective_exit = (
            trade["exit_price"] * (1 - slippage)
        )

        effective_stop = (
            trade["stop_price"] * (1 - slippage)
        )

        risk_per_share = (
            effective_entry - effective_stop
        )

        if (
            effective_entry <= 0
            or risk_per_share <= 0
            or cash <= 0
        ):
            skipped += 1
            continue

        # Equity is cash plus the entry cost of currently
        # open positions. Unrealized gains are not assumed.
        equity = cash + sum(
            position["shares"]
            * position["effective_entry"]
            for position in active
        )

        risk_budget = equity * RISK_PER_TRADE

        max_position_value = (
            equity * MAX_POSITION_EQUITY_PCT
        )

        shares_by_risk = (
            risk_budget / risk_per_share
        )

        shares_by_position_limit = (
            max_position_value / effective_entry
        )

        shares_by_cash = cash / effective_entry

        shares = min(
            shares_by_risk,
            shares_by_position_limit,
            shares_by_cash,
        )

        if shares <= 0.000001:
            skipped += 1
            continue

        cost = shares * effective_entry

        if cost > cash + 0.000001:
            skipped += 1
            continue

        cash -= cost

        estimated_pnl = shares * (
            effective_exit - effective_entry
        )

        active.append({
            "symbol": trade["symbol"],
            "entry_time": entry_time,
            "exit_time": trade["exit_time"],
            "shares": shares,
            "effective_entry": effective_entry,
            "effective_exit": effective_exit,
            "pnl": estimated_pnl,
            "exit_reason": trade["exit_reason"],
        })

    # Close remaining positions in actual exit-time order.
    active.sort(key=lambda position: position["exit_time"])

    for position in active:
        close_position(position)

    ending_equity = cash

    profits = [
        trade["pnl"]
        for trade in closed_trades
        if trade["pnl"] > 0
    ]

    losses = [
        trade["pnl"]
        for trade in closed_trades
        if trade["pnl"] < 0
    ]

    wins = len(profits)
    losing_trades = len(losses)
    trade_count = len(closed_trades)

    win_rate = (
        wins / trade_count * 100
        if trade_count
        else 0.0
    )

    gross_profit = sum(profits)
    gross_loss = abs(sum(losses))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = None
    else:
        profit_factor = 0.0

    net_profit = ending_equity - starting_capital

    return_pct = (
        net_profit / starting_capital * 100
        if starting_capital
        else 0.0
    )

    # Drawdown calculated from the closed-trade equity curve.
    # It does not capture every unrealized intratrade decline.
    peak = equity_curve[0]
    max_drawdown = 0.0

    for equity in equity_curve:
        peak = max(peak, equity)

        if peak > 0:
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(
                max_drawdown,
                drawdown,
            )

    return {
        "starting_capital": round(starting_capital, 2),
        "ending_equity": round(ending_equity, 2),
        "net_profit": round(net_profit, 2),
        "return_pct": round(return_pct, 2),
        "trades": trade_count,
        "wins": wins,
        "losses": losing_trades,
        "win_rate": round(win_rate, 2),
        "profit_factor": (
            round(profit_factor, 4)
            if profit_factor is not None
            else None
        ),
        "max_drawdown_pct": round(max_drawdown, 2),
        "skipped_trades": skipped,
        "closed_trades": closed_trades,
    }


# ============================================================
# SUPABASE RESULTS
# ============================================================

def save_result(strategy, result, start_date, end_date):
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.warning(
            "Supabase credentials missing; result not saved."
        )
        return

    payload = {
        "strategy": strategy,
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
        "start_date": start_date,
        "end_date": end_date,
    }

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    try:
        response = SESSION.post(
            f"{SUPABASE_URL}/rest/v1/backtest_results",
            json=payload,
            headers=headers,
            timeout=30,
        )

        if response.status_code >= 300:
            log.error(
                "Supabase save failed: %s %s",
                response.status_code,
                response.text[:1000],
            )
        else:
            log.info(
                "Saved %s result for $%.0f.",
                strategy,
                result["starting_capital"],
            )

    except requests.RequestException as exc:
        log.error("Supabase request failed: %s", exc)


# ============================================================
# REPORTING
# ============================================================

def report_result(strategy, result):
    log.info("=" * 65)
    log.info("STRATEGY: %s", strategy)
    log.info(
        "STARTING CAPITAL: $%.2f",
        result["starting_capital"],
    )
    log.info(
        "ENDING EQUITY: $%.2f",
        result["ending_equity"],
    )
    log.info(
        "NET PROFIT: $%.2f",
        result["net_profit"],
    )
    log.info(
        "RETURN: %.2f%%",
        result["return_pct"],
    )
    log.info("TRADES: %s", result["trades"])
    log.info("WINS: %s", result["wins"])
    log.info("LOSSES: %s", result["losses"])
    log.info("WIN RATE: %.2f%%", result["win_rate"])

    pf = result["profit_factor"]

    if pf is None:
        log.info("PROFIT FACTOR: Undefined (no losing trades)")
    else:
        log.info("PROFIT FACTOR: %.4f", pf)

    log.info(
        "CLOSED-TRADE MAX DRAWDOWN: %.2f%%",
        result["max_drawdown_pct"],
    )
    log.info(
        "SKIPPED TRADES: %s",
        result["skipped_trades"],
    )
    log.info("=" * 65)


# ============================================================
# MAIN
# ============================================================

def main():
    now = datetime.now(timezone.utc)

    # End at today's UTC midnight to avoid using a partial
    # current trading session.
    end = now.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    start = end - timedelta(days=BACKTEST_DAYS)

    log.info("Historical backtest starting.")
    log.info("Start date: %s", start.date())
    log.info("End date: %s", end.date())
    log.info("Maximum simultaneous positions: %s", MAX_POSITIONS)
    log.info("Capital buckets: %s", STARTING_CAPITALS)
    log.info("Strategies: %s", STRATEGIES)
    log.info("Slippage per side: %s basis points", SLIPPAGE_BPS)
    log.info("Risk per trade: %.2f%%", RISK_PER_TRADE * 100)

    symbols = discover_symbols()

    if not symbols:
        raise RuntimeError("No eligible Alpaca symbols discovered.")

    selected_symbols = screen_universe(
        symbols,
        start,
        end,
    )

    if not selected_symbols:
        raise RuntimeError(
            "No stocks passed the liquidity screen."
        )

    all_candidates = {
        minutes: []
        for minutes in STRATEGIES
    }

    total_bars = 0

    log.info(
        "Downloading minute bars for %s selected stocks.",
        len(selected_symbols),
    )

    # Process small batches to limit memory consumption.
    for batch_number, batch in enumerate(
        chunks(selected_symbols, SYMBOL_BATCH_SIZE),
        start=1,
    ):
        log.info(
            "Minute-data batch %s: %s symbols.",
            batch_number,
            len(batch),
        )

        minute_data = download_bars(
            batch,
            "1Min",
            start,
            end,
        )

        for symbol, rows in minute_data.items():
            total_bars += len(rows)

            if not rows:
                continue

            for minutes in STRATEGIES:
                trades = generate_candidates(
                    symbol,
                    rows,
                    minutes,
                )

                all_candidates[minutes].extend(trades)

        log.info(
            "Minute bars downloaded so far: %s",
            total_bars,
        )

        # Release references before moving to the next batch.
        del minute_data

    log.info(
        "Total minute bars downloaded: %s",
        total_bars,
    )

    start_date = start.date().isoformat()
    end_date = end.date().isoformat()

    for minutes in STRATEGIES:
        strategy = f"ORB_{minutes}M"
        candidates = all_candidates[minutes]

        log.info(
            "%s generated %s trade candidates.",
            strategy,
            len(candidates),
        )

        for capital in STARTING_CAPITALS:
            result = simulate_portfolio(
                candidates,
                capital,
            )

            report_result(strategy, result)

            save_result(
                strategy,
                result,
                start_date,
                end_date,
            )

    log.info("All backtests completed.")
    log.info(
        "WARNING: These are historical simulations, not "
        "evidence of future profitability."
    )


if __name__ == "__main__":
    main()
