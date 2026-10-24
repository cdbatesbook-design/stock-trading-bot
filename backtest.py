"""One-minute momentum research backtest. PAPER/RESEARCH ONLY: never submits orders.

Tests 0.50%, 0.75%, 1.00%, and 1.50% profit targets with a 1% stop from
entry fill, 5% equity/buying-power allocation per position, fractional shares,
slippage, cash accounting, and mark-to-market portfolio drawdown.

This is a research backtest, not a live trading bot. It intentionally does not
place orders or change the existing backtest.py / worker.py files.
"""
from __future__ import annotations

import logging
import math
import os
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone, time as dt_time
from zoneinfo import ZoneInfo

import requests

# ---------------------------- configuration ----------------------------
ALPACA_KEY = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"
ALPACA_TRADING_URL = "https://paper-api.alpaca.markets"
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
EASTERN = ZoneInfo("America/New_York")

BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "90"))
MAX_SYMBOLS = int(os.getenv("BACKTEST_MAX_SYMBOLS", "50"))
DAILY_BATCH_SIZE = 100
MIN_STOCK_PRICE = 5.0
MIN_AVG_DOLLAR_VOLUME = 20_000_000
DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")
STARTING_CAPITALS = [100, 500, 1000, 2500, 5000]
PROFIT_TARGETS = [0.005, 0.0075, 0.01, 0.015]
STOP_LOSS_PCT = 0.01
POSITION_ALLOCATION_PCT = 0.05
SLIPPAGE_BPS_PER_SIDE = 5
MAX_HOLD_MINUTES = 120
MIN_AVG_VOLUME_MULTIPLIER = 1.2
LOOKBACK_BARS = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger("stock_backtest_v2")
SESSION = requests.Session()

if not ALPACA_KEY or not ALPACA_SECRET:
    raise RuntimeError(
        "Missing Alpaca credentials. Set APCA_API_KEY_ID and "
        "APCA_API_SECRET_KEY in GitHub Actions secrets."
    )

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}


def utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def number(value, default=0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def get_json(url, params=None, retries=6):
    last = None

    for attempt in range(retries):
        try:
            response = SESSION.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=90,
            )

            if response.status_code == 429 or response.status_code >= 500:
                wait = min(60, 2 ** attempt)
                log.warning(
                    "Alpaca HTTP %s; retrying in %ss",
                    response.status_code,
                    wait,
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(min(60, 2 ** attempt))

    raise RuntimeError(
        f"Alpaca request failed after {retries} attempts: {last}"
    )


def download_bars(symbols, timeframe, start, end):
    collected = {symbol: [] for symbol in symbols}

    if not symbols:
        return collected

    params = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": utc_iso(start),
        "end": utc_iso(end),
        "limit": 10000,
        "adjustment": "split",
        "feed": DATA_FEED,
        "sort": "asc",
    }

    token = None

    while True:
        query = dict(params)

        if token:
            query["page_token"] = token

        payload = get_json(
            f"{ALPACA_DATA_URL}/stocks/bars",
            query,
        )

        for symbol, rows in (payload.get("bars") or {}).items():
            if symbol in collected:
                collected[symbol].extend(rows)

        token = payload.get("next_page_token")

        if not token:
            break

    for rows in collected.values():
        rows.sort(key=lambda row: row.get("t", ""))

    return collected


def discover_and_screen(start, end):
    log.info("Discovering active tradable US equities")

    assets = get_json(
        f"{ALPACA_TRADING_URL}/v2/assets",
        {
            "status": "active",
            "asset_class": "us_equity",
        },
    )

    symbols = sorted({
        asset.get("symbol", "")
        for asset in assets
        if asset.get("status") == "active"
        and asset.get("tradable") is True
        and asset.get("symbol")
        and len(asset["symbol"]) <= 6
        and asset["symbol"].replace(".", "").isalnum()
    })

    log.info(
        "Checking daily liquidity for %s symbols; this may take several minutes",
        len(symbols),
    )

    qualified = []

    for batch_no, batch in enumerate(
        chunks(symbols, DAILY_BATCH_SIZE),
        1,
    ):
        daily = download_bars(
            batch,
            "1Day",
            start - timedelta(days=45),
            end,
        )

        for symbol, rows in daily.items():
            if len(rows) < 20:
                continue

            recent = rows[-20:]
            close = number(recent[-1].get("c"))

            avg_dv = sum(
                number(row.get("c")) * number(row.get("v"))
                for row in recent
            ) / len(recent)

            if (
                close >= MIN_STOCK_PRICE
                and avg_dv >= MIN_AVG_DOLLAR_VOLUME
            ):
                qualified.append((symbol, avg_dv))

        if batch_no % 10 == 0:
            log.info(
                "Liquidity screen batches: %s; qualified so far: %s",
                batch_no,
                len(qualified),
            )

    qualified.sort(key=lambda item: item[1], reverse=True)

    selected = [
        symbol
        for symbol, _ in qualified[:MAX_SYMBOLS]
    ]

    if "SPY" not in selected:
        selected.append("SPY")

    log.info(
        "Selected %s symbols including SPY benchmark",
        len(selected),
    )

    return selected


def regular_session(row):
    ts = parse_ts(row["t"]).astimezone(EASTERN)

    return (
        ts.weekday() < 5
        and dt_time(9, 30) <= ts.time() < dt_time(16, 0)
    )


def build_signal_flags(rows):
    """Build completed-bar signals without using future data."""

    parsed = []
    ema9 = None
    ema21 = None

    alpha9 = 2 / 10
    alpha21 = 2 / 22

    session_date = None
    cumulative_pv = 0.0
    cumulative_volume = 0.0

    vol_window = deque(maxlen=LOOKBACK_BARS)
    high_window = deque(maxlen=LOOKBACK_BARS)
    flags = []

    for row in rows:
        ts = parse_ts(row["t"])
        local = ts.astimezone(EASTERN)

        bar = {
            "timestamp": ts,
            "local_date": local.date(),
            "open": number(row.get("o")),
            "high": number(row.get("h")),
            "low": number(row.get("l")),
            "close": number(row.get("c")),
            "volume": number(row.get("v")),
        }

        parsed.append(bar)

        valid_session = (
            local.weekday() < 5
            and dt_time(9, 30) <= local.time() < dt_time(16, 0)
        )

        if (
            not valid_session
            or min(
                bar["open"],
                bar["high"],
                bar["low"],
                bar["close"],
            ) <= 0
        ):
            flags.append(False)
            continue

        if session_date != local.date():
            session_date = local.date()
            cumulative_pv = 0.0
            cumulative_volume = 0.0
            vol_window.clear()
            high_window.clear()
            ema9 = None
            ema21 = None

        typical = (
            bar["high"] + bar["low"] + bar["close"]
        ) / 3

        cumulative_pv += typical * bar["volume"]
        cumulative_volume += bar["volume"]

        vwap = (
            cumulative_pv / cumulative_volume
            if cumulative_volume > 0
            else bar["close"]
        )

        old_vols = list(vol_window)

        prior_high = (
            max(high_window)
            if high_window
            else float("inf")
        )

        ema9 = (
            bar["close"]
            if ema9 is None
            else alpha9 * bar["close"] + (1 - alpha9) * ema9
        )

        ema21 = (
            bar["close"]
            if ema21 is None
            else alpha21 * bar["close"] + (1 - alpha21) * ema21
        )

        avg_volume = (
            sum(old_vols) / len(old_vols)
            if old_vols
            else 0
        )

        flag = (
            len(old_vols) == LOOKBACK_BARS
            and bar["close"] > prior_high
            and ema9 > ema21
            and bar["close"] > vwap
            and avg_volume > 0
            and bar["volume"]
            >= MIN_AVG_VOLUME_MULTIPLIER * avg_volume
        )

        flags.append(flag)

        vol_window.append(bar["volume"])
        high_window.append(bar["high"])

    return parsed, flags


def simulate_one_target(
    bars_by_symbol,
    flags_by_symbol,
    starting_capital,
    target_pct,
    trade_start=None,
    trade_end=None,
):
    """Event-driven portfolio simulation with cash, 5% positions, and drawdown."""

    slippage = SLIPPAGE_BPS_PER_SIDE / 10000.0

    cash = float(starting_capital)
    positions = {}
    trades = []
    current_prices = {}
    pending = {}

    max_dd = 0.0
    peak_equity = float(starting_capital)
    equity_curve = []
    skip_count = 0

    timeline = sorted({
        bar["timestamp"]
        for rows in bars_by_symbol.values()
        for bar in rows
        if trade_end is None or bar["timestamp"] < trade_end
    })

    at_time = defaultdict(list)

    for symbol, rows in bars_by_symbol.items():
        for i, bar in enumerate(rows):
            at_time[bar["timestamp"]].append(
                (symbol, i, bar)
            )

    for ts in timeline:
        todays_bars = sorted(
            at_time[ts],
            key=lambda item: item[0],
        )

        # Execute pending signals at the next available bar's open.
        for symbol, i, bar in todays_bars:
            if symbol == "SPY":
                continue

            if symbol in pending and symbol not in positions:
                local = ts.astimezone(EASTERN)

                if (
                    local.weekday() < 5
                    and dt_time(9, 30) <= local.time() < dt_time(16, 0)
                ):
                    raw_entry = bar["open"]
                    entry_fill = raw_entry * (1 + slippage)

                    equity_now = cash + sum(
                        position["shares"]
                        * current_prices.get(
                            held_symbol,
                            position["entry_fill"],
                        )
                        for held_symbol, position in positions.items()
                    )

                    allocation = min(
                        equity_now * POSITION_ALLOCATION_PCT,
                        cash,
                    )

                    shares = (
                        allocation / entry_fill
                        if entry_fill > 0
                        else 0.0
                    )

                    if shares > 0 and allocation >= 1.0:
                        stop = entry_fill * (1 - STOP_LOSS_PCT)
                        target = entry_fill * (1 + target_pct)

                        cash -= shares * entry_fill

                        positions[symbol] = {
                            "symbol": symbol,
                            "entry_time": ts,
                            "entry_fill": entry_fill,
                            "shares": shares,
                            "stop": stop,
                            "target": target,
                            "bars_held": 0,
                            "last_price": raw_entry,
                        }
                    else:
                        skip_count += 1

                pending.pop(symbol, None)

        # If stop and target are both hit within a candle, assume stop first.
        for symbol, i, bar in todays_bars:
            current_prices[symbol] = bar["close"]

            position = positions.get(symbol)

            if position:
                position["bars_held"] += 1
                position["last_price"] = bar["close"]

                stop = position["stop"]
                target = position["target"]

                exit_raw = None
                reason = None

                if bar["open"] <= stop:
                    exit_raw = bar["open"]
                    reason = "STOP_GAP"

                elif bar["open"] >= target:
                    exit_raw = target
                    reason = "TARGET_GAP"

                else:
                    hit_stop = bar["low"] <= stop
                    hit_target = bar["high"] >= target

                    if hit_stop:
                        exit_raw = stop
                        reason = (
                            "STOP_AMBIGUOUS"
                            if hit_target
                            else "STOP_1PCT"
                        )

                    elif hit_target:
                        exit_raw = target
                        reason = "TARGET"

                local = ts.astimezone(EASTERN)

                session_end = (
                    local.weekday() < 5
                    and local.time() >= dt_time(15, 59)
                )

                timed_out = (
                    position["bars_held"] >= MAX_HOLD_MINUTES
                )

                if exit_raw is None and (session_end or timed_out):
                    exit_raw = bar["close"]
                    reason = (
                        "SESSION_CLOSE"
                        if session_end
                        else "TIME_EXIT"
                    )

                if exit_raw is not None:
                    exit_fill = max(
                        0.0001,
                        exit_raw * (1 - slippage),
                    )

                    proceeds = position["shares"] * exit_fill
                    cash += proceeds

                    pnl = (
                        proceeds
                        - position["shares"] * position["entry_fill"]
                    )

                    trades.append({
                        "symbol": symbol,
                        "entry_time": position["entry_time"],
                        "exit_time": ts,
                        "entry_price": position["entry_fill"],
                        "exit_price": exit_fill,
                        "shares": position["shares"],
                        "pnl": pnl,
                        "exit_reason": reason,
                    })

                    del positions[symbol]

        # Schedule signals only after their candles have completed.
        for symbol, i, bar in todays_bars:
            flags = flags_by_symbol.get(symbol, [])

            in_window = (
                (trade_start is None or ts >= trade_start)
                and (trade_end is None or ts < trade_end)
            )

            if (
                in_window
                and symbol != "SPY"
                and i < len(flags)
                and flags[i]
                and symbol not in positions
                and symbol not in pending
            ):
                pending[symbol] = ts

        # Mark the portfolio to market, including open positions.
        equity = cash + sum(
            position["shares"]
            * current_prices.get(
                symbol,
                position["entry_fill"],
            )
            for symbol, position in positions.items()
        )

        peak_equity = max(peak_equity, equity)

        if peak_equity > 0:
            max_dd = max(
                max_dd,
                (peak_equity - equity) / peak_equity * 100.0,
            )

        equity_curve.append((ts, equity))

    # Close remaining positions at the last available prices.
    if positions:
        final_ts = timeline[-1] if timeline else datetime.now(timezone.utc)

        for symbol, position in list(positions.items()):
            raw_exit = current_prices.get(
                symbol,
                position["last_price"],
            )

            exit_fill = max(
                0.0001,
                raw_exit * (1 - slippage),
            )

            proceeds = position["shares"] * exit_fill
            cash += proceeds

            pnl = (
                proceeds
                - position["shares"] * position["entry_fill"]
            )

            trades.append({
                "symbol": symbol,
                "entry_time": position["entry_time"],
                "exit_time": final_ts,
                "entry_price": position["entry_fill"],
                "exit_price": exit_fill,
                "shares": position["shares"],
                "pnl": pnl,
                "exit_reason": "END_OF_TEST",
            })

            del positions[symbol]

    ending_equity = cash
    pnls = [trade["pnl"] for trade in trades]

    gross_profit = sum(pnl for pnl in pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (float("inf") if gross_profit > 0 else 0.0)
    )

    wins = sum(pnl > 0 for pnl in pnls)
    losses = sum(pnl < 0 for pnl in pnls)

    return {
        "starting_capital": starting_capital,
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
        "profit_factor": pf,
        "max_drawdown_pct": max_dd,
        "skipped_trades": skip_count,
        "trade_details": trades,
    }


def calculate_spy_benchmark(spy_rows, start, end, capital):
    eligible = [
        row
        for row in spy_rows
        if start <= parse_ts(row["t"]) < end
        and number(row.get("c")) > 0
    ]

    if len(eligible) < 2:
        return None

    first = number(eligible[0].get("c"))

    if first <= 0:
        return None

    values = [
        capital * number(row.get("c")) / first
        for row in eligible
    ]

    peak = values[0]
    drawdown = 0.0

    for value in values:
        peak = max(peak, value)

        if peak:
            drawdown = max(
                drawdown,
                (peak - value) / peak * 100,
            )

    ending = values[-1]
    profit = ending - capital

    return {
        "starting_capital": capital,
        "ending_equity": ending,
        "net_profit": profit,
        "return_pct": profit / capital * 100,
        "trades": 1,
        "wins": int(profit > 0),
        "losses": int(profit < 0),
        "win_rate": 100.0 if profit > 0 else 0.0,
        "profit_factor": 0.0,
        "max_drawdown_pct": drawdown,
        "skipped_trades": 0,
    }


def save_result(run_id, strategy, result, start, end):
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.warning(
            "Supabase credentials missing; result will only appear in workflow logs"
        )
        return False

    payload = {
        "run_id": run_id,
        "strategy": strategy,
        "symbol": "PORTFOLIO",
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
        "start_date": str(start.date()),
        "end_date": str(end.date()),
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
            "Supabase save failed for %s: %s; response=%s",
            strategy,
            exc,
            getattr(exc.response, "text", "")[:400],
        )
        return False


def log_result(strategy, result):
    pf = result["profit_factor"]

    pf_text = (
        "infinity"
        if not math.isfinite(pf)
        else f"{pf:.2f}"
    )

    log.info(
        "RESULT %s | start=$%.2f end=$%.2f P/L=$%.2f (%.2f%%) "
        "trades=%s win=%.1f%% PF=%s maxDD=%.2f%% skipped=%s",
        strategy,
        result["starting_capital"],
        result["ending_equity"],
        result["net_profit"],
        result["return_pct"],
        result["trades"],
        result["win_rate"],
        pf_text,
        result["max_drawdown_pct"],
        result["skipped_trades"],
    )


def main():
    end = datetime.now(timezone.utc).replace(
        second=0,
        microsecond=0,
    )

    start = end - timedelta(days=BACKTEST_DAYS)

    log.info(
        "ONE-MINUTE MOMENTUM BACKTEST | %s to %s | targets=%s | "
        "stop=1%% | position=5%% | paper/research only",
        start.date(),
        end.date(),
        [f"{target * 100:.2f}%" for target in PROFIT_TARGETS],
    )

    symbols = discover_and_screen(start, end)

    all_bars = {}
    stock_symbols = [
        symbol for symbol in symbols
        if symbol != "SPY"
    ]

    for batch_no, batch in enumerate(
        chunks(stock_symbols, 10),
        1,
    ):
        all_bars.update(
            download_bars(batch, "1Min", start, end)
        )

        log.info(
            "Downloaded minute-bar batch %s/%s",
            batch_no,
            math.ceil(len(stock_symbols) / 10),
        )

    spy_daily = download_bars(
        ["SPY"],
        "1Day",
        start - timedelta(days=5),
        end,
    ).get("SPY", [])

    bars_by_symbol = {}
    flags_by_symbol = {}

    for symbol, rows in all_bars.items():
        parsed, flags = build_signal_flags(rows)

        keep = [
            (i, bar)
            for i, bar in enumerate(parsed)
            if regular_session(rows[i])
        ]

        if len(keep) < 100:
            continue

        bars_by_symbol[symbol] = [
            bar for _, bar in keep
        ]

        flags_by_symbol[symbol] = [
            flags[i] for i, _ in keep
        ]

    log.info(
        "Ready to simulate %s symbols with usable minute bars",
        len(bars_by_symbol),
    )

    if not bars_by_symbol:
        raise RuntimeError(
            "No usable minute bars downloaded. Check Alpaca data "
            "permissions/feed and date range."
        )

    run_id = str(uuid.uuid4())
    split = start + (end - start) * 0.70

    log.info(
        "Chronological split: in-sample %s through %s; "
        "out-of-sample %s through %s",
        start.date(),
        split.date(),
        split.date(),
        end.date(),
    )

    for target in PROFIT_TARGETS:
        base = (
            f"M1_MOM_T{target * 100:.2f}_SL1.00_ALLOC5"
        )

        for phase, phase_start, phase_end in (
            ("IS", start, split),
            ("OOS", split, end),
        ):
            for capital in STARTING_CAPITALS:
                result = simulate_one_target(
                    bars_by_symbol,
                    flags_by_symbol,
                    capital,
                    target,
                    trade_start=phase_start,
                    trade_end=phase_end,
                )

                strategy = f"{phase}_{base}"

                log_result(
                    strategy + f"_C{capital}",
                    result,
                )

                if not save_result(
                    run_id,
                    strategy,
                    result,
                    phase_start,
                    phase_end,
                ):
                    log.warning(
                        "Could not save %s capital $%s result",
                        strategy,
                        capital,
                    )

    for phase, phase_start, phase_end in (
        ("IS", start, split),
        ("OOS", split, end),
    ):
        for capital in STARTING_CAPITALS:
            benchmark = calculate_spy_benchmark(
                spy_daily,
                phase_start,
                phase_end,
                capital,
            )

            if benchmark:
                strategy = f"{phase}_SPY_BUY_HOLD"

                log_result(
                    strategy + f"_C{capital}",
                    benchmark,
                )

                save_result(
                    run_id,
                    strategy,
                    benchmark,
                    phase_start,
                    phase_end,
                )

    log.info(
        "Completed. Results include separate chronological "
        "in-sample/out-of-sample tests, slippage, 5%% position "
        "allocation, 1%% stop, and mark-to-market drawdown."
    )

    log.warning(
        "Limits: current active universe creates survivorship bias; "
        "IEX data may be incomplete; a short pilot is not evidence "
        "of profitability. No orders were placed."
    )


if __name__ == "__main__":
    main()
