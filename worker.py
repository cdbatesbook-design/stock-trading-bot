"""Conservative paper-trading worker. Trading stays disabled unless bot_status.enabled is true.

Database note: to preserve the existing schema, trades.order_id stores Alpaca's
client_order_id (not the broker's server-generated order ID). Never reuse it.
"""
import logging
import math
import os
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from supabase import create_client
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import StockHisto
ricalDataClient
from alpaca.data.requests import StockLatestTradeRequest

# ---- Conservative settings ----
PAPER_MODE = True
WATCHLIST = [
    "EDAC", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "AMD", "AVGO", "NFLX", "JPM", "BAC", "WMT", "COST", "DIS", "KO",
    "PEP", "XOM", "F", "INTC",
]
VIRTUAL_BUDGET = 10_000.00
MAX_POSITION_VALUE = 900.00
MAX_OPEN_POSITIONS = 5
STOP_LOSS_PERCENT = 0.03
TAKE_PROFIT_PERCENT = 0.06
SCAN_INTERVAL_SECONDS = 300
MAX_PRICE_DRIFT_PERCENT = 0.01
RSI_PERIOD = 14

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def finite_positive(value):
    try:
        number = float(value)
        return math.isfinite(number) and number > 0
    except (TypeError, ValueError):
        return False


SUPABASE_URL = required_env("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = required_env("SUPABASE_SERVICE_ROLE_KEY")
ALPACA_API_KEY = required_env("ALPACA_API_KEY")
ALPACA_SECRET_KEY = required_env("ALPACA_SECRET_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
trading = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


# ---- Database helpers ----
def log_db(level, message):
    message = str(message)[:1000]
    logging.log(getattr(logging, level.upper(), logging.INFO), message)
    try:
        supabase.table("bot_logs").insert({
            "level": level.upper(),
            "message": message,
            "created_at": utc_now(),
        }).execute()
    except Exception as exc:
        logging.warning("Could not write bot_logs row: %s", exc)


def set_status(message, *, heartbeat=True, scan=False):
    row = {"id": 1, "status_message": str(message)[:1000]}
    if heartbeat:
        row["last_heartbeat"] = utc_now()
    if scan:
        row["last_scan"] = utc_now()
    try:
        supabase.table("bot_status").upsert(
            row, on_conflict="id"
        ).execute()
    except Exception as exc:
        log_db("ERROR", f"Could not update bot_status: {exc}")


def get_bot_config():
    result = (
        supabase.table("bot_status")
        .select("*")
        .eq("id", 1)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(
            "bot_status row id=1 is missing; refusing to trade"
        )
    return rows[0]


def fetch_rows(table, limit=1000):
    result = (
        supabase.table(table)
        .select("*")
        .order("id", desc=False)
        .limit(limit)
        .execute()
    )
    return result.data or []


def get_trade_rows():
    return fetch_rows("trades", 2000)


def get_position_rows():
    return fetch_rows("positions", 1000)


def get_trade_by_client_id(client_order_id):
    result = (
        supabase.table("trades")
        .select("*")
        .eq("order_id", client_order_id)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return rows[0] if rows else None


def write_trade(
    client_order_id,
    symbol,
    side,
    quantity,
    status,
    price=None,
    strategy="SMA_RSI",
    reason="",
):
    # Existing schema has a unique constraint on order_id.
    # This worker consistently stores Alpaca's client_order_id there.
    payload = {
        "order_id": client_order_id,
        "symbol": symbol,
        "side": side,
        "quantity": float(quantity),
        "price": float(price) if finite_positive(price) else None,
        "total_value": (
            float(quantity) * float(price)
            if finite_positive(quantity) and finite_positive(price)
            else None
        ),
        "strategy": strategy,
        "reason": str(reason)[:500],
        "status": str(status).upper(),
        "updated_at": utc_now(),
    }

    existing = get_trade_by_client_id(client_order_id)
    if existing:
        (
            supabase.table("trades")
            .update(payload)
            .eq("order_id", client_order_id)
            .execute()
        )
    else:
        payload["created_at"] = utc_now()
        supabase.table("trades").insert(payload).execute()


def find_position(symbol):
    result = (
        supabase.table("positions")
        .select("*")
        .eq("symbol", symbol)
        .eq("status", "OPEN")
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return rows[0] if rows else None


def save_position(
    symbol,
    quantity,
    entry_price,
    current_price=None,
    status="OPEN",
):
    existing = find_position(symbol)
    payload = {
        "symbol": symbol,
        "quantity": float(quantity),
        "entry_price": float(entry_price),
        "current_price": (
            float(current_price)
            if finite_positive(current_price)
            else None
        ),
        "status": status,
        "updated_at": utc_now(),
    }

    if status == "CLOSED":
        payload["closed_at"] = utc_now()

    if existing:
        # Preserve the original opened_at on updates.
        (
            supabase.table("positions")
            .update(payload)
            .eq("id", existing["id"])
            .execute()
        )
    else:
        payload["opened_at"] = utc_now()
        supabase.table("positions").insert(payload).execute()


def latest_price(symbol):
    try:
        request = StockLatestTradeRequest(symbol_or_symbols=symbol)
        response = data_client.get_stock_latest_trade(request)
        trade = response.get(symbol)
        price = getattr(trade, "price", None) if trade else None
        return float(price) if finite_positive(price) else None
    except Exception as exc:
        log_db(
            "WARNING",
            f"Latest-price lookup failed for {symbol}: {exc}",
        )
        return None


def calculate_rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def get_signal(symbol):
    """Return BUY/HOLD using daily candles; errors fail closed to HOLD."""
    try:
        frame = yf.download(
            symbol,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if frame is None or frame.empty:
            return "HOLD", "No daily market data"

        close = frame["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = close.dropna()
        if len(close) < 55:
            return "HOLD", "Not enough daily candles"

        # Ignore the latest candle because it may still be forming.
        close = close.iloc[:-1]
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        rsi = calculate_rsi(close)

        last = float(close.iloc[-1])
        current_sma20 = float(sma20.iloc[-1])
        current_sma50 = float(sma50.iloc[-1])
        current_rsi = float(rsi.iloc[-1])

        if (
            last > current_sma20
            and current_sma20 > current_sma50
            and 45 <= current_rsi <= 68
        ):
            return (
                "BUY",
                f"Close above SMA20, SMA20 above SMA50, "
                f"RSI={current_rsi:.1f}",
            )

        return (
            "HOLD",
            f"Entry conditions not met; RSI={current_rsi:.1f}",
        )

    except Exception as exc:
        log_db(
            "WARNING",
            f"Signal calculation failed for {symbol}: {exc}",
        )
        return "HOLD", "Signal error; fail closed"


def active_trade_for_symbol(symbol):
    active_statuses = {
        "NEW",
        "ACCEPTED",
        "PENDING_NEW",
        "PENDING_CANCEL",
        "PARTIALLY_FILLED",
        "SUBMITTING",
        "UNKNOWN",
    }

    for row in get_trade_rows():
        if (
            row.get("symbol") == symbol
            and str(row.get("status", "")).upper() in active_statuses
        ):
            return row

    return None


def get_order_by_client_id(client_order_id):
    try:
        return trading.get_order_by_client_id(client_order_id)
    except Exception:
        return None


def reconcile_orders():
    """Refresh known orders; uncertain orders block further trading."""
    active_statuses = {
        "SUBMITTING",
        "UNKNOWN",
        "NEW",
        "ACCEPTED",
        "PENDING_NEW",
        "PENDING_CANCEL",
        "PARTIALLY_FILLED",
    }

    for row in get_trade_rows():
        status = str(row.get("status", "")).upper()
        client_order_id = row.get("order_id")

        if status not in active_statuses or not client_order_id:
            continue

        order = get_order_by_client_id(client_order_id)

        if order is None:
            if status != "UNKNOWN":
                write_trade(
                    client_order_id,
                    row["symbol"],
                    row["side"],
                    row["quantity"],
                    "UNKNOWN",
                    row.get("price"),
                    row.get("strategy") or "SMA_RSI",
                    "Order status cannot be verified; manual review required",
                )
            continue

        broker_status = str(
            getattr(order, "status", "unknown")
        ).split(".")[-1].upper()

        filled_qty = getattr(order, "filled_qty", None)
        avg_price = getattr(order, "filled_avg_price", None)

        qty = (
            float(filled_qty)
            if finite_positive(filled_qty)
            else float(row.get("quantity") or 0)
        )

        write_trade(
            client_order_id,
            row["symbol"],
            row["side"],
            qty,
            broker_status,
            avg_price or row.get("price"),
            row.get("strategy") or "SMA_RSI",
            row.get("reason") or "",
        )

        side = str(row.get("side", "")).upper()

        if broker_status == "FILLED" and side == "BUY":
            if finite_positive(avg_price) and finite_positive(filled_qty):
                save_position(
                    row["symbol"],
                    float(filled_qty),
                    float(avg_price),
                    float(avg_price),
                    "OPEN",
                )

        elif broker_status == "FILLED" and side == "SELL":
            position = find_position(row["symbol"])
            if position:
                (
                    supabase.table("positions")
                    .update({
                        "status": "CLOSED",
                        "closed_at": utc_now(),
                        "updated_at": utc_now(),
                    })
                    .eq("id", position["id"])
                    .execute()
                )


def broker_positions_by_symbol():
    try:
        return {
            position.symbol: position
            for position in trading.get_all_positions()
        }
    except Exception as exc:
        raise RuntimeError(
            f"Could not read broker positions; refusing new orders: {exc}"
        )


def open_position_rows():
    return [
        position
        for position in get_position_rows()
        if str(position.get("status", "")).upper() == "OPEN"
    ]


def reconcile_positions():
    """Fail closed if broker positions and database records disagree."""
    broker = broker_positions_by_symbol()
    ledger = {
        position["symbol"]: position
        for position in open_position_rows()
    }

    symbols = set(broker) | set(ledger)

    for symbol in symbols:
        broker_position = broker.get(symbol)
        ledger_position = ledger.get(symbol)

        broker_qty = (
            float(getattr(broker_position, "qty", 0) or 0)
            if broker_position
            else 0.0
        )
        ledger_qty = (
            float(ledger_position.get("quantity", 0) or 0)
            if ledger_position
            else 0.0
        )

        if abs(broker_qty - ledger_qty) > 0.0001:
            raise RuntimeError(
                f"Position mismatch for {symbol}: "
                f"broker={broker_qty}, ledger={ledger_qty}; "
                "no new trades will be submitted until reconciled"
            )

    return broker, ledger


def has_unresolved_orders():
    active_statuses = {
        "SUBMITTING",
        "UNKNOWN",
        "NEW",
        "ACCEPTED",
        "PENDING_NEW",
        "PENDING_CANCEL",
        "PARTIALLY_FILLED",
    }

    return any(
        str(row.get("status", "")).upper() in active_statuses
        for row in get_trade_rows()
    )


def submit_market_order(symbol, side, quantity, reason):
    """Record intent first; never auto-resubmit an uncertain order."""
    client_order_id = f"worker-{uuid.uuid4().hex}"

    write_trade(
        client_order_id,
        symbol,
        side,
        quantity,
        "SUBMITTING",
        None,
        "SMA_RSI",
        reason,
    )

    try:
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=(
                OrderSide.BUY
                if side == "BUY"
                else OrderSide.SELL
            ),
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )

        order = trading.submit_order(order_request)

        broker_status = str(
            getattr(order, "status", "unknown")
        ).split(".")[-1].upper()

        filled_qty = getattr(order, "filled_qty", None)
        avg_price = getattr(order, "filled_avg_price", None)

        effective_qty = (
            float(filled_qty)
            if finite_positive(filled_qty)
            else quantity
        )

        write_trade(
            client_order_id,
            symbol,
            side,
            effective_qty,
            broker_status,
            avg_price,
            "SMA_RSI",
            reason,
        )

        if broker_status == "FILLED":
            if (
                side == "BUY"
                and finite_positive(avg_price)
                and finite_positive(filled_qty)
            ):
                save_position(
                    symbol,
                    float(filled_qty),
                    float(avg_price),
                    float(avg_price),
                    "OPEN",
                )

            elif side == "SELL":
                position = find_position(symbol)
                if position:
                    (
                        supabase.table("positions")
                        .update({
                            "status": "CLOSED",
                            "closed_at": utc_now(),
                            "updated_at": utc_now(),
                        })
                        .eq("id", position["id"])
                        .execute()
                    )

        return order

    except Exception as exc:
        # Do not retry: the broker may have accepted the order before
        # a network error occurred.
        try:
            write_trade(
                client_order_id,
                symbol,
                side,
                quantity,
                "UNKNOWN",
                None,
                "SMA_RSI",
                f"Submission outcome uncertain: {exc}",
            )
        finally:
            log_db(
                "ERROR",
                f"Order submission uncertain for {symbol}; "
                f"will not retry automatically: {exc}",
            )

        return None


def manage_exits(broker_positions, ledger_positions):
    for symbol, ledger in ledger_positions.items():
        broker_position = broker_positions.get(symbol)

        if broker_position is None:
            raise RuntimeError(
                f"Ledger says {symbol} is open but broker has no position"
            )

        if active_trade_for_symbol(symbol):
            continue

        quantity = float(getattr(broker_position, "qty", 0) or 0)
        entry = float(ledger.get("entry_price", 0) or 0)
        price = latest_price(symbol)

        if (
            quantity <= 0
            or not finite_positive(entry)
            or not finite_positive(price)
        ):
            continue

        try:
            (
                supabase.table("positions")
                .update({
                    "current_price": price,
                    "updated_at": utc_now(),
                })
                .eq("id", ledger["id"])
                .execute()
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not update position ledger for {symbol}: {exc}"
            )

        change = (price - entry) / entry
        reason = None

        if change <= -STOP_LOSS_PERCENT:
            reason = f"Stop-loss threshold reached ({change:.2%})"
        elif change >= TAKE_PROFIT_PERCENT:
            reason = f"Take-profit threshold reached ({change:.2%})"

        if reason:
            submit_market_order(
                symbol,
                "SELL",
                abs(quantity),
                reason,
            )


def run_scan():
    config = get_bot_config()
    set_status("Worker heartbeat; checking control switch")

    if config.get("enabled") is not True:
        set_status(
            "Paused: bot_status.enabled is not true",
            scan=True,
        )
        return

    mode = str(config.get("mode", "paper")).lower()
    if mode != "paper" or not PAPER_MODE:
        set_status(
            "Safety stop: only paper mode is permitted",
            scan=True,
        )
        log_db("ERROR", "Safety stop: refusing to run outside paper mode")
        return

    reconcile_orders()

    if has_unresolved_orders():
        set_status(
            "Paused: order status unresolved; manual review required",
            scan=True,
        )
        return

    broker_positions, ledger_positions = reconcile_positions()

    manage_exits(broker_positions, ledger_positions)

    if has_unresolved_orders():
        set_status(
            "Exit order submitted or unresolved; awaiting reconciliation",
            scan=True,
        )
        return

    if len(ledger_positions) >= MAX_OPEN_POSITIONS:
        set_status(
            "Scan complete: maximum open positions reached",
            scan=True,
        )
        return

    try:
        account = trading.get_account()
        buying_power = float(
            getattr(account, "buying_power", 0) or 0
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not read paper account; refusing new orders: {exc}"
        )

    committed = sum(
        float(position.get("quantity", 0) or 0)
        * float(position.get("entry_price", 0) or 0)
        for position in ledger_positions.values()
    )

    available_virtual = max(0.0, VIRTUAL_BUDGET - committed)

    for symbol in WATCHLIST:
        if len(ledger_positions) >= MAX_OPEN_POSITIONS:
            break

        if symbol in ledger_positions or active_trade_for_symbol(symbol):
            continue

        signal, reason = get_signal(symbol)
        if signal != "BUY":
            continue

        price = latest_price(symbol)
        if not finite_positive(price):
            log_db(
                "WARNING",
                f"Skipping {symbol}: no valid latest trade price",
            )
            continue

        target_value = min(
            MAX_POSITION_VALUE,
            available_virtual,
            buying_power,
        )

        if target_value <= 0:
            set_status(
                "Scan stopped: budget or paper buying power exhausted",
                scan=True,
            )
            return

        # Use whole shares only.
        quantity = int(target_value // price)
        if quantity < 1:
            continue

        # Recheck the quote immediately before submitting.
        confirm_price = latest_price(symbol)
        if not finite_positive(confirm_price):
            continue

        drift = abs(confirm_price - price) / price
        if drift > MAX_PRICE_DRIFT_PERCENT:
            log_db(
                "INFO",
                f"Skipping {symbol}: quote drift {drift:.2%} exceeds limit",
            )
            continue

        if (
            confirm_price * quantity > target_value
            or confirm_price * quantity > buying_power
        ):
            continue

        submit_market_order(symbol, "BUY", quantity, reason)

        # Stop after one attempt and reconcile before another order.
        set_status(
            f"Submitted/attempted paper BUY for {symbol}; "
            "waiting for reconciliation",
            scan=True,
        )
        return

    set_status("Scan complete: no eligible entry found", scan=True)


def main():
    log_db(
        "INFO",
        "Paper worker started; trading remains controlled by "
        "bot_status.enabled",
    )

    while True:
        try:
            run_scan()
        except Exception as exc:
            log_db(
                "ERROR",
                f"Worker safety pause: {type(exc).__name__}: {exc}",
            )
            set_status(
                f"Safety pause: {type(exc).__name__}: {exc}",
                scan=True,
            )

        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
