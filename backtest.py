import os
import time
import logging
import uuid
from datetime import datetime, timezone

import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

START_DATE = "2024-10-23T00:00:00Z"
END_DATE = "2026-10-24T00:00:00Z"

# Initial test universe. Expand only after validating data access.
SYMBOLS = [
    "SPY",
    "QQQ",
    "IWM",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMD",
    "META",
    "AMZN",
    "TSLA",
]

STRATEGIES = {
    "ORB_5M": 5,
    "ORB_15M": 15,
    "ORB_30M": 30,
}

# Volume confirmation threshold.
VOLUME_MULTIPLIER = 1.5

# Target is two times the initial risk.
REWARD_RISK_RATIO = 2.0

# Estimated transaction cost per side, in basis points.
# 5 basis points = 0.05% per side.
COST_BPS_PER_SIDE = 5.0

ALPACA_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

RUN_ID = str(uuid.uuid4())

API_URL = "https://data.alpaca.markets/v2/stocks/bars"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

session = requests.Session()

if not ALPACA_KEY or not ALPACA_SECRET:
    raise RuntimeError(
        "Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY."
    )

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError(
        "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY. "
        "Configure these in Render before running the backtest."
    )

session.headers.update({
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
})


# ============================================================
# DOWNLOAD HISTORICAL MINUTE BARS
# ============================================================

def download_bars(symbol):
    logging.info("Downloading historical minute bars for %s", symbol)

    params = {
        "symbols": symbol,
        "timeframe": "1Min",
        "start": START_DATE,
        "end": END_DATE,
        "limit": 10000,
        "adjustment": "all",
        "feed": "iex",
        "sort": "asc",
    }

    all_rows = []
    page_token = None

    while True:
        if page_token:
            params["page_token"] = page_token
        else:
            params.pop("page_token", None)

        response = session.get(
            API_URL,
            params=params,
            timeout=60,
        )

        if response.status_code == 429:
            logging.warning(
                "Rate limit reached for %s. Waiting 30 seconds.",
                symbol,
            )
            time.sleep(30)
            continue

        if response.status_code != 200:
            raise RuntimeError(
                f"Alpaca request failed for {symbol}: "
                f"HTTP {response.status_code}: {response.text[:500]}"
            )

        payload = response.json()
        rows = payload.get("bars", {}).get(symbol, [])

        all_rows.extend(rows)

        page_token = payload.get("next_page_token")

        logging.info(
            "%s: downloaded %s bars so far",
            symbol,
            len(all_rows),
        )

        if not page_token:
            break

        time.sleep(0.25)

    if not all_rows:
        logging.warning(
            "No historical bars returned for %s.", symbol
        )
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    df["timestamp"] = pd.to_datetime(
        df["t"],
        utc=True,
    ).dt.tz_convert("America/New_York")

    df = df.rename(columns={
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    })

    df = df.sort_values("timestamp")
    df = df.drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    # Keep regular US stock-market hours only.
    df = df.between_time(
        "09:30",
        "15:59",
        inclusive="both",
    )

    # Exclude weekends.
    df = df[df.index.dayofweek < 5]

    return df[["open", "high", "low", "close", "volume"]]


# ============================================================
# SIMULATE ONE OPENING-RANGE STRATEGY
# ============================================================

def simulate_strategy(symbol, df, strategy_name, opening_minutes):
    trades = []

    if df.empty:
        return trades

    grouped = df.groupby(df.index.date)

    for trading_date, day in grouped:
        day = day.sort_index()

        # Require enough data for the opening range and confirmation.
        if len(day) < opening_minutes + 10:
            continue

        # Minute timestamps identify the beginning of each bar.
        opening_end_minute = 9 * 60 + 30 + opening_minutes

        minute_of_day = day.index.hour * 60 + day.index.minute

        opening_mask = minute_of_day < opening_end_minute
        opening_bars = day[opening_mask]

        if len(opening_bars) < opening_minutes:
            continue

        range_high = float(opening_bars["high"].max())
        range_low = float(opening_bars["low"].min())

        if range_high <= range_low:
            continue

        after_opening = day[~opening_mask]

        if len(after_opening) < 2:
            continue

        entered = False

        for i in range(5, len(after_opening) - 1):
            bar = after_opening.iloc[i]

            # Volume must exceed the average of the previous
            # five available one-minute bars.
            previous_volumes = after_opening["volume"].iloc[i - 5:i]
            average_volume = float(previous_volumes.mean())

            if average_volume <= 0:
                continue

            volume_confirmed = (
                float(bar["volume"])
                >= VOLUME_MULTIPLIER * average_volume
            )

            breakout_confirmed = (
                float(bar["close"]) > range_high
            )

            if not (volume_confirmed and breakout_confirmed):
                continue

            # Enter at the next bar's open to avoid assuming
            # execution at the same close that created the signal.
            entry_bar = after_opening.iloc[i + 1]
            entry_price = float(entry_bar["open"])

            if entry_price <= range_low:
                continue

            risk = entry_price - range_low

            if risk <= 0:
                continue

            stop_price = range_low
            target_price = (
                entry_price
                + REWARD_RISK_RATIO * risk
            )

            entry_time = after_opening.index[i + 1]

            exit_price = float(day.iloc[-1]["close"])
            exit_time = day.index[-1]
            exit_reason = "END_OF_DAY"

            # Check the bars after entry. If stop and target
            # are both touched in one bar, assume stop first.
            for j in range(i + 1, len(after_opening)):
                future_bar = after_opening.iloc[j]
                future_time = after_opening.index[j]

                low_price = float(future_bar["low"])
                high_price = float(future_bar["high"])

                if low_price <= stop_price:
                    exit_price = stop_price
                    exit_time = future_time
                    exit_reason = "STOP_LOSS"
                    break

                if high_price >= target_price:
                    exit_price = target_price
                    exit_time = future_time
                    exit_reason = "TAKE_PROFIT"
                    break

            gross_return_pct = (
                (exit_price - entry_price)
                / entry_price
            ) * 100.0

            total_cost_pct = (
                2 * COST_BPS_PER_SIDE / 100.0
            )

            net_return_pct = (
                gross_return_pct - total_cost_pct
            )

            # One-share equivalent. This is not a recommended
            # position size or a portfolio-level dollar forecast.
            net_pnl = (
                exit_price - entry_price
                - entry_price * total_cost_pct / 100.0
            )

            trades.append({
                "run_id": RUN_ID,
                "strategy": strategy_name,
                "symbol": symbol,
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "shares": 1,
                "pnl": round(net_pnl, 4),
                "pnl_percent": round(net_return_pct, 4),
                "exit_reason": exit_reason,
            })

            entered = True
            break

        # At most one long trade per symbol per day per strategy.
        if entered:
            continue

    return trades


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(trades):
    if not trades:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "average_return_pct": 0.0,
            "average_win_pct": 0.0,
            "average_loss_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
        }

    results = pd.DataFrame(trades)
    returns = results["pnl_percent"].astype(float)

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    if len(losses):
        profit_factor = (
            float(wins.sum()) / abs(float(losses.sum()))
            if float(losses.sum()) != 0
            else float("inf")
        )
    else:
        profit_factor = (
            float("inf") if len(wins) else 0.0
        )

    # Approximate equal-sized sequential trade portfolio.
    # Trades are sorted by exit time.
    ordered = results.sort_values("exit_time")
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0

    for ret in ordered["pnl_percent"].astype(float):
        equity *= 1.0 + ret / 100.0
        peak = max(peak, equity)

        if peak > 0:
            drawdown = (peak - equity) / peak * 100.0
            max_drawdown = max(max_drawdown, drawdown)

    return {
        "trades": int(len(results)),
        "win_rate_pct": round(
            float((returns > 0).mean() * 100.0), 2
        ),
        "average_return_pct": round(
            float(returns.mean()), 4
        ),
        "average_win_pct": round(
            float(wins.mean()), 4
        ) if len(wins) else 0.0,
        "average_loss_pct": round(
            float(losses.mean()), 4
        ) if len(losses) else 0.0,
        "profit_factor": round(
            profit_factor, 4
        ) if profit_factor != float("inf") else "Infinity",
        "max_drawdown_pct": round(max_drawdown, 2),
    }


# ============================================================
# SAVE RESULTS TO SUPABASE
# ============================================================

def save_trades(trades):
    if not trades:
        logging.warning("No trades to save.")
        return

    url = f"{SUPABASE_URL}/rest/v1/backtest_results"

    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    # Insert in small batches to avoid oversized requests.
    for start in range(0, len(trades), 100):
        batch = trades[start:start + 100]

        response = requests.post(
            url,
            headers=headers,
            json=batch,
            timeout=60,
        )

        if response.status_code not in (200, 201, 204):
            raise RuntimeError(
                "Supabase insert failed: "
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        logging.info(
            "Saved %s of %s trades",
            min(start + 100, len(trades)),
            len(trades),
        )


# ============================================================
# MAIN
# ============================================================

def main():
    logging.info("Starting historical backtest.")
    logging.info("Run ID: %s", RUN_ID)
    logging.info("Period: %s through %s", START_DATE, END_DATE)
    logging.info("Symbols: %s", ", ".join(SYMBOLS))
    logging.info("Strategies: %s", ", ".join(STRATEGIES))

    all_trades = []
    symbols_with_data = 0

    for symbol in SYMBOLS:
        try:
            df = download_bars(symbol)

            if df.empty:
                logging.warning(
                    "Skipping %s because no data was returned.",
                    symbol,
                )
                continue

            symbols_with_data += 1

            logging.info(
                "%s: received %s regular-session bars.",
                symbol,
                len(df),
            )

            for strategy_name, minutes in STRATEGIES.items():
                trades = simulate_strategy(
                    symbol,
                    df,
                    strategy_name,
                    minutes,
                )

                logging.info(
                    "%s / %s: %s simulated trades",
                    symbol,
                    strategy_name,
                    len(trades),
                )

                all_trades.extend(trades)

        except Exception:
            logging.exception(
                "Failed while processing %s.", symbol
            )
            raise

    logging.info(
        "Historical data received for %s of %s symbols.",
        symbols_with_data,
        len(SYMBOLS),
    )

    logging.info(
        "Total simulated trades: %s",
        len(all_trades),
    )

    if all_trades:
        trade_frame = pd.DataFrame(all_trades)

        for strategy_name in STRATEGIES:
            strategy_trades = trade_frame[
                trade_frame["strategy"] == strategy_name
            ].to_dict("records")

            metrics = calculate_metrics(strategy_trades)

            logging.info(
                "RESULT %s: %s",
                strategy_name,
                metrics,
            )

        save_trades(all_trades)

    else:
        logging.warning(
            "No trades were generated. Check historical data access "
            "and strategy conditions before drawing conclusions."
        )

    logging.info("Backtest finished. Run ID: %s", RUN_ID)


if __name__ == "__main__":
    main()
