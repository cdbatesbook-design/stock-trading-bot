import os
import sys
import time
import logging
from datetime import datetime, timedelta, timezone
from itertools import islice

import requests
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ALPACA_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

DATA_URL = "https://data.alpaca.markets/v2"
TRADING_URL = "https://paper-api.alpaca.markets/v2"

# Historical testing period.
# Longer periods require significantly more data and API time.
BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "180"))

# Dynamically discovered stocks are screened for liquidity.
MIN_PRICE = 5.00
MAX_PRICE = 500.00
MIN_AVG_DOLLAR_VOLUME = 10_000_000

# Maximum number of dynamically selected stocks tested intraday.
# This is NOT a fixed ticker list.
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "100"))

CAPITAL_BUCKETS = [500, 1000, 2500, 5000]

MAX_OPEN_POSITIONS = 3
RISK_PER_TRADE = 0.01

REWARD_RISK_RATIO = 2.0
VOLUME_MULTIPLIER = 1.5

# Estimated total trading friction per side, in basis points.
# 5 basis points = 0.05%.
COST_BPS_PER_SIDE = 5

# Additional assumed slippage per side.
SLIPPAGE_BPS_PER_SIDE = 3

# Minimum entry price above the opening range.
# 0.0005 = 0.05%.
BREAKOUT_BUFFER = 0.0005

STRATEGIES = {
    "ORB_5M": 5,
    "ORB_15M": 15,
    "ORB_30M": 30,
}

# Conservative risk controls.
MIN_STOP_PERCENT = 0.002
MAX_STOP_PERCENT = 0.05

# Fractional shares are simulated mathematically.
# Actual brokerage support and order rules must be checked separately.

REQUEST_TIMEOUT = 60
MAX_RETRIES = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

log = logging.getLogger("breakout-backtest")


# ============================================================
# API HELPERS
# ============================================================

if not ALPACA_KEY or not ALPACA_SECRET:
    raise SystemExit(
        "Missing GitHub secrets APCA_API_KEY_ID "
        "or APCA_API_SECRET_KEY."
    )

SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
    "accept": "application/json",
})


def api_get(url, params=None):
    """GET with retries for temporary errors and rate limiting."""

    for attempt in range(MAX_RETRIES):
        try:
            response = SESSION.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 429:
                wait = min(60, 5 * (attempt + 1))
                log.warning("Rate limited. Waiting %s seconds.", wait)
                time.sleep(wait)
                continue

            if response.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    f"API request failed: {exc}"
                ) from exc

            time.sleep(2 * (attempt + 1))

    raise RuntimeError("API request failed after retries.")


def chunks(items, size):
    iterator = iter(items)

    while True:
        batch = list(islice(iterator, size))

        if not batch:
            break

        yield batch


# ============================================================
# DISCOVER THE STOCK UNIVERSE
# ============================================================

def discover_assets():
    """
    Discover active US equities from Alpaca.
    No hardcoded ticker universe is used.
    """

    url = f"{TRADING_URL}/assets"

    payload = api_get(url, {
        "status": "active",
        "asset_class": "us_equity",
    })

    symbols = []

    for asset in payload:
        symbol = asset.get("symbol", "")
        exchange = asset.get("exchange", "")
        tradable = asset.get("tradable", False)

        if not symbol or not tradable:
            continue

        # Exclude symbols that are unlikely to be ordinary stocks.
        if any(char in symbol for char in "^/$"):
            continue

        if len(symbol) > 5:
            continue

        if exchange not in {
            "NYSE",
            "NASDAQ",
            "NYSEARCA",
            "NYSEAMERICAN",
            "ARCA",
            "BATS",
        }:
            continue

        symbols.append(symbol)

    symbols = sorted(set(symbols))

    log.info("Discovered %s active equity symbols.", len(symbols))

    return symbols


# ============================================================
# HISTORICAL BAR RETRIEVAL
# ============================================================

def fetch_bars(symbols, timeframe, start, end, feed="iex"):
    """
    Fetch historical bars in batches and follow pagination tokens.
    Returns a DataFrame indexed by symbol and timestamp.
    """

    collected = []

    for batch in chunks(symbols, 100):
        token = None

        while True:
            params = {
                "symbols": ",".join(batch),
                "timeframe": timeframe,
                "start": start.isoformat().replace("+00:00", "Z"),
                "end": end.isoformat().replace("+00:00", "Z"),
                "limit": 10000,
                "adjustment": "split",
                "feed": feed,
                "sort": "asc",
            }

            if token:
                params["page_token"] = token

            payload = api_get(
                f"{DATA_URL}/stocks/bars",
                params,
            )

            bars = payload.get("bars", {})

            for symbol, rows in bars.items():
                for row in rows:
                    collected.append({
                        "symbol": symbol,
                        "timestamp": pd.to_datetime(
                            row["t"], utc=True
                        ),
                        "open": float(row["o"]),
                        "high": float(row["h"]),
                        "low": float(row["l"]),
                        "close": float(row["c"]),
                        "volume": float(row["v"]),
                    })

            token = payload.get("next_page_token")

            if not token:
                break

        log.info(
            "Retrieved %s bars so far.",
            len(collected),
        )

    if not collected:
        return pd.DataFrame(
            columns=[
                "symbol", "timestamp", "open",
                "high", "low", "close", "volume",
            ]
        )

    df = pd.DataFrame(collected)

    df = df.drop_duplicates(
        subset=["symbol", "timestamp"]
    )

    return df.sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)


# ============================================================
# LIQUIDITY SCREEN
# ============================================================

def screen_universe(all_symbols, end):
    """
    Screen all discoverable symbols using recent daily bars.
    Eligible symbols are ranked by average dollar volume.
    """

    start = end - timedelta(days=45)

    log.info("Screening discovered stocks using daily bars.")

    daily = fetch_bars(
        all_symbols,
        "1Day",
        start,
        end,
        feed="iex",
    )

    if daily.empty:
        raise RuntimeError(
            "No daily bars returned. Check Alpaca data access."
        )

    daily["dollar_volume"] = (
        daily["close"] * daily["volume"]
    )

    candidates = []

    for symbol, group in daily.groupby("symbol"):
        group = group.sort_values("timestamp")

        # Require sufficient observations to reduce thin-data cases.
        if len(group) < 10:
            continue

        last_price = float(group.iloc[-1]["close"])

        avg_dollar_volume = float(
            group["dollar_volume"].tail(20).mean()
        )

        if not MIN_PRICE <= last_price <= MAX_PRICE:
            continue

        if avg_dollar_volume < MIN_AVG_DOLLAR_VOLUME:
            continue

        candidates.append({
            "symbol": symbol,
            "last_price": last_price,
            "avg_dollar_volume": avg_dollar_volume,
        })

    candidates.sort(
        key=lambda row: row["avg_dollar_volume"],
        reverse=True,
    )

    if MAX_CANDIDATES > 0:
        candidates = candidates[:MAX_CANDIDATES]

    symbols = [row["symbol"] for row in candidates]

    log.info(
        "%s stocks passed the liquidity screen.",
        len(symbols),
    )

    for row in candidates[:20]:
        log.info(
            "Candidate %-8s price=$%.2f avg dollar volume=$%.0f",
            row["symbol"],
            row["last_price"],
            row["avg_dollar_volume"],
        )

    return symbols


# ============================================================
# BAR PREPARATION
# ============================================================

def prepare_intraday_data(df):
    """
    Convert UTC timestamps to New York time and retain
    regular-session bars.
    """

    if df.empty:
        return df

    df = df.copy()

    df["timestamp_ny"] = (
        df["timestamp"].dt.tz_convert("America/New_York")
    )

    df["date"] = df["timestamp_ny"].dt.date

    df["minute"] = (
        df["timestamp_ny"].dt.hour * 60
        + df["timestamp_ny"].dt.minute
    )

    # Regular US stock session: 9:30 AM to 4:00 PM ET.
    df = df[
        (df["minute"] >= 570)
        & (df["minute"] < 960)
    ].copy()

    return df.sort_values(
        ["date", "symbol", "timestamp"]
    ).reset_index(drop=True)


# ============================================================
# CREATE BREAKOUT SIGNALS
# ============================================================

def generate_trades(df, strategy_name, opening_minutes):
    """
    One long trade opportunity per symbol per session.
    Entry is modeled at the next available bar open after
    a confirmed breakout.
    """

    trades = []

    if df.empty:
        return trades

    for (symbol, session_date), day in df.groupby(
        ["symbol", "date"]
    ):
        day = day.sort_values("timestamp").reset_index(drop=True)

        opening_end = 570 + opening_minutes

        opening = day[
            (day["minute"] >= 570)
            & (day["minute"] < opening_end)
        ]

        if len(opening) < opening_minutes:
            continue

        range_high = float(opening["high"].max())
        range_low = float(opening["low"].min())

        if range_low <= 0 or range_high <= range_low:
            continue

        risk_distance = range_high - range_low

        risk_percent = risk_distance / range_high

        if not MIN_STOP_PERCENT <= risk_percent <= MAX_STOP_PERCENT:
            continue

        after_opening = day[
            day["minute"] >= opening_end
        ].reset_index(drop=True)

        if len(after_opening) < 2:
            continue

        average_opening_volume = float(
            opening["volume"].mean()
        )

        if average_opening_volume <= 0:
            continue

        entered = False

        for index in range(len(after_opening) - 1):
            bar = after_opening.iloc[index]
            next_bar = after_opening.iloc[index + 1]

            breakout_level = (
                range_high * (1 + BREAKOUT_BUFFER)
            )

            if (
                float(bar["close"]) > breakout_level
                and float(bar["volume"])
                >= average_opening_volume * VOLUME_MULTIPLIER
            ):
                entry = float(next_bar["open"])

                # Do not enter if the next bar opens beyond
                # the intended target or stop geometry.
                stop = range_low

                if entry <= stop:
                    continue

                risk = entry - stop
                target = entry + risk * REWARD_RISK_RATIO

                # Walk forward through subsequent bars.
                exit_price = float(
                    after_opening.iloc[-1]["close"]
                )

                exit_reason = "EOD"

                for future_index in range(index + 1, len(after_opening)):
                    future = after_opening.iloc[future_index]

                    future_low = float(future["low"])
                    future_high = float(future["high"])

                    # Conservative assumption: if stop and target
                    # both occur within the same bar, assume stop first.
                    if future_low <= stop:
                        exit_price = min(
                            stop,
                            float(future["open"])
                        )
                        exit_reason = "STOP"
                        break

                    if future_high >= target:
                        exit_price = target
                        exit_reason = "TARGET"
                        break

                trades.append({
                    "symbol": symbol,
                    "date": str(session_date),
                    "strategy": strategy_name,
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "exit": exit_price,
                    "exit_reason": exit_reason,
                    "entry_time": str(next_bar["timestamp"]),
                })

                entered = True
                break

        if entered:
            continue

    return trades


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

def simulate_portfolio(trades, starting_capital):
    """
    Simulates one portfolio with:
      - Maximum three simultaneous positions.
      - Risk-based position sizing.
      - Cash constraints.
      - Fractional shares.
      - Estimated costs and slippage.
    """

    ordered = sorted(
        trades,
        key=lambda trade: trade["entry_time"],
    )

    cash = float(starting_capital)
    equity = float(starting_capital)

    open_positions = []
    completed = []

    equity_curve = [starting_capital]
    skipped = 0

    for trade in ordered:
        entry_time = pd.to_datetime(
            trade["entry_time"], utc=True
        )

        # Close positions before the next trade timestamp.
        remaining = []

        for position in open_positions:
            if position["exit_time"] <= entry_time:
                cash += position["proceeds"]
                completed.append(position["result"])
            else:
                remaining.append(position)

        open_positions = remaining

        equity = cash + sum(
            position["market_value"]
            for position in open_positions
        )

        equity_curve.append(equity)

        if len(open_positions) >= MAX_OPEN_POSITIONS:
            skipped += 1
            continue

        entry = float(trade["entry"])
        stop = float(trade["stop"])
        exit_price = float(trade["exit"])

        risk_per_share = entry - stop

        if entry <= 0 or risk_per_share <= 0:
            skipped += 1
            continue

        risk_budget = equity * RISK_PER_TRADE

        shares_by_risk = risk_budget / risk_per_share
        shares_by_cash = cash / entry

        shares = min(
            shares_by_risk,
            shares_by_cash,
        )

        if shares <= 0:
            skipped += 1
            continue

        entry_cost_rate = (
            COST_BPS_PER_SIDE + SLIPPAGE_BPS_PER_SIDE
        ) / 10000

        exit_cost_rate = entry_cost_rate

        entry_cost = entry * shares * entry_cost_rate
        exit_cost = exit_price * shares * exit_cost_rate

        entry_cash = entry * shares + entry_cost

        if entry_cash > cash:
            shares = cash / (
                entry * (1 + entry_cost_rate)
            )

            if shares <= 0:
                skipped += 1
                continue

            entry_cost = entry * shares * entry_cost_rate
            exit_cost = exit_price * shares * exit_cost_rate
            entry_cash = entry * shares + entry_cost

        cash -= entry_cash

        exit_proceeds = (
            exit_price * shares - exit_cost
        )

        pnl = (
            (exit_price - entry) * shares
            - entry_cost
            - exit_cost
        )

        result = {
            **trade,
            "shares": shares,
            "pnl": pnl,
            "return_pct": (
                pnl / entry_cash * 100
                if entry_cash > 0 else 0
            ),
        }

        # Position's historical exit is used to enforce
        # the maximum concurrent-position rule.
        exit_time = pd.to_datetime(
            trade["entry_time"], utc=True
        ) + pd.Timedelta(days=1)

        open_positions.append({
            "exit_time": exit_time,
            "proceeds": exit_proceeds,
            "market_value": exit_proceeds,
            "result": result,
        })

    # Close remaining positions at the end of the simulation.
    for position in open_positions:
        cash += position["proceeds"]
        completed.append(position["result"])

    final_equity = cash

    pnls = [float(row["pnl"]) for row in completed]
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]

    total_return_pct = (
        (final_equity / starting_capital) - 1
    ) * 100

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf") if gross_profit > 0 else 0
    )

    max_drawdown_pct = calculate_drawdown(
        starting_capital,
        pnls,
    )

    return {
        "starting_capital": starting_capital,
        "ending_equity": final_equity,
        "net_profit": final_equity - starting_capital,
        "return_pct": total_return_pct,
        "trades": len(completed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": (
            len(wins) / len(completed) * 100
            if completed else 0
        ),
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct,
        "skipped_trades": skipped,
        "completed_trades": completed,
    }


def calculate_drawdown(starting_capital, pnls):
    equity = float(starting_capital)
    peak = equity
    max_drawdown = 0.0

    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)

        if peak > 0:
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown


# ============================================================
# RESULTS
# ============================================================

def print_results(strategy_name, capital, result):
    log.info("")
    log.info("=" * 65)
    log.info("STRATEGY: %s", strategy_name)
    log.info("STARTING CAPITAL: $%.2f", capital)
    log.info("ENDING EQUITY: $%.2f", result["ending_equity"])
    log.info("NET PROFIT: $%.2f", result["net_profit"])
    log.info("RETURN: %.2f%%", result["return_pct"])
    log.info("TRADES: %s", result["trades"])
    log.info("WINS: %s", result["wins"])
    log.info("LOSSES: %s", result["losses"])
    log.info("WIN RATE: %.2f%%", result["win_rate_pct"])
    log.info("PROFIT FACTOR: %.3f", result["profit_factor"])
    log.info("MAX DRAWDOWN: %.2f%%", result["max_drawdown_pct"])
    log.info("SKIPPED TRADES: %s", result["skipped_trades"])
    log.info("=" * 65)


def save_to_supabase(rows):
    """
    Optional results storage.
    Backtesting still runs when Supabase is not configured.
    """

    if not SUPABASE_URL or not SUPABASE_KEY:
        log.info("Supabase not configured; skipping database save.")
        return

    url = (
        f"{SUPABASE_URL}/rest/v1/backtest_results"
    )

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    for row in rows:
        payload = {
            "strategy": row["strategy"],
            "starting_capital": row["starting_capital"],
            "ending_equity": row["ending_equity"],
            "net_profit": row["net_profit"],
            "return_pct": row["return_pct"],
            "trades": row["trades"],
            "wins": row["wins"],
            "losses": row["losses"],
            "win_rate_pct": row["win_rate_pct"],
            "profit_factor": (
                row["profit_factor"]
                if row["profit_factor"] != float("inf")
                else None
            ),
            "max_drawdown_pct": row["max_drawdown_pct"],
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code not in (200, 201, 204):
            log.error(
                "Supabase save failed: %s %s",
                response.status_code,
                response.text[:500],
            )
        else:
            log.info("Saved %s result to Supabase.", row["strategy"])


# ============================================================
# MAIN
# ============================================================

def main():
    end = datetime.now(timezone.utc)

    # Request extra calendar days to cover weekends and holidays.
    start = end - timedelta(days=BACKTEST_DAYS)

    log.info("Starting historical breakout backtest.")
    log.info("Date range: %s to %s", start.date(), end.date())
    log.info("Maximum simultaneous positions: %s", MAX_OPEN_POSITIONS)
    log.info("Capital buckets: %s", CAPITAL_BUCKETS)

    all_symbols = discover_assets()

    if not all_symbols:
        raise RuntimeError("No eligible assets were discovered.")

    candidates = screen_universe(all_symbols, end)

    if not candidates:
        raise RuntimeError(
            "No symbols passed the liquidity screen. "
            "Check Alpaca data access or relax the screening thresholds."
        )

    log.info(
        "Downloading minute bars for %s selected stocks.",
        len(candidates),
    )

    minute_bars = fetch_bars(
        candidates,
        "1Min",
        start,
        end,
        feed="iex",
    )

    if minute_bars.empty:
        raise RuntimeError(
            "No minute bars returned. Check Alpaca subscription "
            "and historical-data permissions."
        )

    minute_bars = prepare_intraday_data(minute_bars)

    all_results = []

    for strategy_name, opening_minutes in STRATEGIES.items():
        log.info(
            "Evaluating %s opening-range breakout.",
            strategy_name,
        )

        trades = generate_trades(
            minute_bars,
            strategy_name,
            opening_minutes,
        )

        log.info(
            "%s generated %s trade candidates.",
            strategy_name,
            len(trades),
        )

        for capital in CAPITAL_BUCKETS:
            result = simulate_portfolio(
                trades,
                capital,
            )

            result["strategy"] = strategy_name

            print_results(
                strategy_name,
                capital,
                result,
            )

            all_results.append(result)

    save_to_supabase(all_results)

    log.info("")
    log.info("BACKTEST FINISHED.")
    log.info(
        "Results evaluated: %s strategy/capital combinations.",
        len(all_results),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.exception("BACKTEST FAILED: %s", exc)
        sys.exit(1)
