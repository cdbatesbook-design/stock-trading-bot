"""
Automated Stock Trading Worker
PAPER TRADING ONLY

Required environment variables:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY

Run only ONE instance of this worker.
Keep bot_status.enabled = false until setup is verified.
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

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest


# ============================================================
# CONFIGURATION
# ============================================================

PAPER_MODE = True

WATCHLIST = [
    "EDAC",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AMD",
    "AVGO",
    "NFLX",
    "JPM",
    "BAC",
    "WMT",
    "COST",
    "DIS",
    "KO",
    "PEP",
    "XOM",
    "F",
    "INTC",
]

VIRTUAL_BUDGET = 10_000.00
MAX_POSITION_VALUE = 900.00
MAX_OPEN_POSITIONS = 5

STOP_LOSS_PERCENT = 0.03
TAKE_PROFIT_PERCENT = 0.06

SCAN_INTERVAL_SECONDS = 300
MAX_PRICE_DRIFT_PERCENT = 0.01
RSI_PERIOD = 14

FINAL_ORDER_STATUSES = {
    "synced",
    "canceled",
    "cancelled",
    "expired",
    "rejected",
}

FILL_ORDER_STATUSES = {
    "filled",
    "partial_filled",
    "partially_filled",
    "synced",
}

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)

log = logging.getLogger("paper-trading-worker")


# ============================================================
# ENVIRONMENT VARIABLES AND CLIENTS
# ============================================================

def required_env(name):
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}"
        )

    return value


ALPACA_KEY = required_env("ALPACA_API_KEY")
ALPACA_SECRET = required_env("ALPACA_SECRET_KEY")

SUPABASE_URL = required_env("SUPABASE_URL")
SUPABASE_KEY = required_env("SUPABASE_SERVICE_ROLE_KEY")

# Paper mode is deliberately hard-coded.
trading = TradingClient(
    ALPACA_KEY,
    ALPACA_SECRET,
    paper=True,
)

market_data = StockHistoricalDataClient(
    ALPACA_KEY,
    ALPACA_SECRET,
)

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def log_db(level, message, details=None):
    try:
        supabase.table("bot_logs").insert({
            "level": str(level).upper(),
            "message": str(message)[:1000],
            "details": details or {},
            "created_at": utc_now(),
        }).execute()

    except Exception:
        log.exception("Could not write bot_logs record")


def set_status(message, **extra):
    row = {
        "id": 1,
        "last_heartbeat": utc_now(),
        "status_message": str(message)[:1000],
        "updated_at": utc_now(),
        **extra,
    }

    try:
        supabase.table("bot_status").upsert(
            row,
            on_conflict="id",
        ).execute()

    except Exception:
        log.exception("Could not update bot_status")


def get_bot_config():
    response = (
        supabase.table("bot_status")
        .select("*")
        .eq("id", 1)
        .limit(1)
        .execute()
    )

    rows = response.data or []

    if not rows:
        raise RuntimeError(
            "bot_status row with id=1 is missing"
        )

    return rows[0]


def fetch_rows(table):
    response = (
        supabase.table(table)
        .select("*")
        .execute()
    )

    return response.data or []


def get_trade_rows():
    return fetch_rows("trades")


def get_position_rows():
    return fetch_rows("positions")


def latest_price(symbol):
    response = market_data.get_stock_latest_trade(
        StockLatestTradeRequest(
            symbol_or_symbols=[symbol]
        )
    )

    trade = response.get(symbol)

    if trade is None or trade.price is None:
        raise RuntimeError(
            f"No latest trade price for {symbol}"
        )

    price = float(trade.price)

    if not math.isfinite(price) or price <= 0:
        raise RuntimeError(
            f"Invalid latest price for {symbol}: {price}"
        )

    return price


def order_status(order):
    status = getattr(order, "status", "")

    return str(
        getattr(status, "value", status)
    ).lower()


# ============================================================
# TRADING SIGNALS
# ============================================================

def get_signal(symbol):
    """
    Strategy:
      - Price above SMA20
      - SMA20 above SMA50
      - Wilder RSI between 45 and 68

    The newest daily candle is excluded because it might
    still be incomplete.
    """

    frame = yf.download(
        symbol,
        period="6mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if frame is None or frame.empty:
        return "HOLD", {
            "reason": "Daily history unavailable"
        }

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    if "Close" not in frame.columns:
        return "HOLD", {
            "reason": "Close prices unavailable"
        }

    close = pd.to_numeric(
        frame["Close"],
        errors="coerce",
    ).dropna()

    if len(close) < 61:
        return "HOLD", {
            "reason": "Insufficient daily history"
        }

    close = close.iloc[:-1]

    if len(close) < 50:
        return "HOLD", {
            "reason": "Insufficient completed daily bars"
        }

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / RSI_PERIOD,
        min_periods=RSI_PERIOD,
        adjust=False,
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / RSI_PERIOD,
        min_periods=RSI_PERIOD,
        adjust=False,
    ).mean()

    rsi = pd.Series(
        index=close.index,
        dtype="float64",
    )

    both_zero = (
        (avg_gain == 0)
        & (avg_loss == 0)
    )

    no_losses = (
        (avg_loss == 0)
        & (avg_gain > 0)
    )

    no_gains = (
        (avg_gain == 0)
        & (avg_loss > 0)
    )

    normal = (
        (avg_gain > 0)
        & (avg_loss > 0)
    )

    rsi.loc[both_zero] = 50.0
    rsi.loc[no_losses] = 100.0
    rsi.loc[no_gains] = 0.0

    rs = (
        avg_gain.loc[normal]
        / avg_loss.loc[normal]
    )

    rsi.loc[normal] = (
        100 - (100 / (1 + rs))
    )

    values = [
        close.iloc[-1],
        sma20.iloc[-1],
        sma50.iloc[-1],
        rsi.iloc[-1],
    ]

    if any(pd.isna(value) for value in values):
        return "HOLD", {
            "reason": "Indicator values unavailable"
        }

    last_close = float(values[0])
    last_sma20 = float(values[1])
    last_sma50 = float(values[2])
    last_rsi = float(values[3])

    info = {
        "daily_close": last_close,
        "sma20": last_sma20,
        "sma50": last_sma50,
        "rsi": last_rsi,
    }

    if (
        last_close > last_sma20 > last_sma50
        and 45 <= last_rsi <= 68
    ):
        return "BUY", info

    return "HOLD", info


# ============================================================
# DATABASE RECORDS
# ============================================================

def write_trade(
    order_id,
    symbol,
    side,
    quantity,
    price,
    status,
    strategy,
    reason,
    total_value=None,
):
    row = {
        "order_id": order_id,
        "symbol": symbol,
        "side": side.lower(),
        "quantity": float(quantity),
        "price": (
            float(price)
            if price is not None
            else None
        ),
        "total_value": (
            float(total_value)
            if total_value is not None
            else None
        ),
        "strategy": strategy,
        "reason": str(reason)[:1000],
        "status": status.lower(),
    }

    # trades.order_id must have a UNIQUE constraint.
    supabase.table("trades").upsert(
        row,
        on_conflict="order_id",
    ).execute()


def upsert_position(
    symbol,
    quantity,
    entry_price,
    current_price,
    status,
):
    row = {
        "symbol": symbol,
        "quantity": float(quantity),
        "entry_price": float(entry_price),
        "current_price": float(current_price),
        "status": status,
        "updated_at": utc_now(),
    }

    if status == "closed":
        row["closed_at"] = utc_now()

    # positions.symbol must have a UNIQUE constraint.
    supabase.table("positions").upsert(
        row,
        on_conflict="symbol",
    ).execute()


# ============================================================
# POSITION ACCOUNTING
# ============================================================

def calculate_managed_position(trades, symbol):
    """
    Calculate net bot-managed shares and average cost.

    Trades must have reliable created_at timestamps for accurate
    chronological accounting. A missing timestamp is treated as
    a reason to stop rather than guess the trade order.
    """

    relevant = [
        trade for trade in trades
        if str(trade.get("symbol") or "") == symbol
        and str(trade.get("status") or "").lower()
        in FILL_ORDER_STATUSES
        and str(trade.get("side") or "").lower()
        in {"buy", "sell"}
    ]

    for trade in relevant:
        if not trade.get("created_at"):
            raise RuntimeError(
                f"Trade {trade.get('order_id')} for {symbol} "
                "has no created_at timestamp. Manual review required."
            )

    relevant.sort(
        key=lambda trade: str(trade["created_at"])
    )

    quantity = 0.0
    average_cost = 0.0

    for trade in relevant:
        side = str(trade["side"]).lower()
        qty = float(trade.get("quantity") or 0)
        price = float(trade.get("price") or 0)

        if qty <= 0 or price <= 0:
            raise RuntimeError(
                f"Invalid fill data for {symbol}; manual review required."
            )

        if side == "buy":
            new_quantity = quantity + qty

            average_cost = (
                quantity * average_cost + qty * price
            ) / new_quantity

            quantity = new_quantity

        else:
            if qty > quantity + 1e-8:
                raise RuntimeError(
                    f"Sell quantity exceeds managed holdings for {symbol}."
                )

            quantity -= qty

            if quantity <= 1e-8:
                quantity = 0.0
                average_cost = 0.0

    return quantity, average_cost


def sync_position_from_broker(symbol):
    trades = (
        supabase.table("trades")
        .select("*")
        .eq("symbol", symbol)
        .execute()
        .data
        or []
    )

    managed_quantity, average_cost = (
        calculate_managed_position(trades, symbol)
    )

    existing_rows = (
        supabase.table("positions")
        .select("*")
        .eq("symbol", symbol)
        .limit(1)
        .execute()
        .data
        or []
    )

    existing = (
        existing_rows[0]
        if existing_rows
        else None
    )

    if managed_quantity <= 1e-8:
        if existing:
            upsert_position(
                symbol,
                0,
                0,
                0,
                "closed",
            )

        return

    broker_positions = trading.get_all_positions()

    broker_position = next(
        (
            position
            for position in broker_positions
            if position.symbol == symbol
        ),
        None,
    )

    if broker_position is None:
        raise RuntimeError(
            f"Ledger shows {managed_quantity} shares of {symbol}, "
            "but Alpaca reports no position."
        )

    broker_quantity = float(broker_position.qty)

    if broker_quantity + 1e-8 < managed_quantity:
        raise RuntimeError(
            f"Ledger shows {managed_quantity} shares of {symbol}, "
            f"but Alpaca reports {broker_quantity}."
        )

    upsert_position(
        symbol,
        managed_quantity,
        average_cost,
        float(broker_position.current_price),
        "open",
    )


def active_bot_positions():
    return [
        position
        for position in get_position_rows()
        if str(position.get("status") or "").lower() == "open"
        and float(position.get("quantity") or 0) > 0
    ]


def unresolved_status(status):
    return status not in FINAL_ORDER_STATUSES


def has_unresolved_order(symbol=None):
    for trade in get_trade_rows():
        if not trade.get("order_id"):
            continue

        if (
            symbol is not None
            and trade.get("symbol") != symbol
        ):
            continue

        status = str(
            trade.get("status") or ""
        ).lower()

        if unresolved_status(status):
            return True

    return False


def account_equity():
    account = trading.get_account()
    equity = float(account.equity)

    if not math.isfinite(equity) or equity <= 0:
        raise RuntimeError(
            f"Invalid account equity: {equity}"
        )

    return equity


def broker_positions_by_symbol():
    return {
        position.symbol: position
        for position in trading.get_all_positions()
    }


def committed_value(positions):
    total = 0.0

    for position in positions:
        quantity = float(
            position.get("quantity") or 0
        )

        price = float(
            position.get("current_price")
            or position.get("entry_price")
            or 0
        )

        total += quantity * price

    return total


# ============================================================
# ORDER RECONCILIATION
# ============================================================

def reconcile_orders():
    """
    Resolve orders using Alpaca's client order ID.

    Do not resubmit an order when its outcome is uncertain.
    Filled orders must be synchronized before being marked synced.
    """

    for trade in get_trade_rows():
        order_id = str(trade.get("order_id") or "")
        db_status = str(
            trade.get("status") or ""
        ).lower()

        if not order_id:
            continue

        # Revisit filled orders until synchronization is confirmed.
        if db_status in FINAL_ORDER_STATUSES:
            continue

        try:
            order = trading.get_order_by_client_id(
                order_id
            )

        except Exception as exc:
            log.warning(
                "Order %s unresolved: %s",
                order_id,
                exc,
            )
            continue

        broker_status = order_status(order)

        filled_quantity = float(
            getattr(order, "filled_qty", 0) or 0
        )

        average_price = getattr(
            order,
            "filled_avg_price",
            None,
        )

        fill_price = (
            float(average_price)
            if average_price not in (None, "")
            else None
        )

        if filled_quantity > 0:
            recorded_status = (
                "filled"
                if broker_status == "filled"
                else "partial_filled"
            )

            recorded_quantity = filled_quantity

            recorded_price = (
                fill_price
                if fill_price is not None
                else trade.get("price")
            )

        else:
            recorded_status = broker_status
            recorded_quantity = float(
                trade.get("quantity") or 0
            )

            recorded_price = trade.get("price")

        try:
            write_trade(
                order_id=order_id,
                symbol=str(trade["symbol"]),
                side=str(trade["side"]),
                quantity=recorded_quantity,
                price=recorded_price,
                status=recorded_status,
                strategy=str(
                    trade.get("strategy") or "daily-trend"
                ),
                reason=str(trade.get("reason") or ""),
                total_value=(
                    filled_quantity * fill_price
                    if filled_quantity > 0
                    and fill_price is not None
                    else trade.get("total_value")
                ),
            )

        except Exception:
            log.exception(
                "Failed to update order %s",
                order_id,
            )
            continue

        if filled_quantity > 0:
            try:
                sync_position_from_broker(
                    str(trade["symbol"])
                )

                if broker_status in {
                    "filled",
                    "canceled",
                    "cancelled",
                    "expired",
                    "rejected",
                }:
                    supabase.table("trades").update({
                        "status": "synced",
                    }).eq(
                        "order_id",
                        order_id,
                    ).execute()

            except Exception:
                log.exception(
                    "Position synchronization failed for %s",
                    order_id,
                )


# ============================================================
# ORDER SUBMISSION
# ============================================================

def place_market_order(
    symbol,
    side,
    quantity,
    reason,
):
    if quantity < 1:
        raise ValueError(
            "Quantity must be at least one whole share."
        )

    if side not in {
        OrderSide.BUY,
        OrderSide.SELL,
    }:
        raise ValueError("Invalid order side.")

    if has_unresolved_order(symbol):
        raise RuntimeError(
            f"An unresolved order exists for {symbol}."
        )

    client_order_id = (
        f"bot-{symbol.lower()}-{side.value}-"
        f"{uuid.uuid4().hex[:20]}"
    )

    # Save intent before sending the order.
    write_trade(
        client_order_id,
        symbol,
        side.value,
        quantity,
        None,
        "intent",
        "daily-trend",
        reason,
    )

    request = MarketOrderRequest(
        symbol=symbol,
        qty=quantity,
        side=side,
        time_in_force=TimeInForce.DAY,
        client_order_id=client_order_id,
    )

    try:
        order = trading.submit_order(
            order_data=request
        )

    except Exception as exc:
        # Do not resubmit automatically: the broker may have
        # received the order despite a timeout.
        log.exception(
            "Submission result uncertain for %s",
            symbol,
        )

        log_db(
            "ERROR",
            "Order submission outcome uncertain",
            {
                "symbol": symbol,
                "order_id": client_order_id,
                "error": str(exc),
            },
        )

        raise

    try:
        write_trade(
            client_order_id,
            symbol,
            side.value,
            quantity,
            None,
            order_status(order),
            "daily-trend",
            reason,
        )

    except Exception:
        log.exception(
            "Order submitted but database update failed: %s",
            client_order_id,
        )

    return client_order_id


# ============================================================
# EXIT MANAGEMENT
# ============================================================

def manage_exits():
    for position in active_bot_positions():
        symbol = str(position["symbol"])

        quantity = int(
            math.floor(
                float(position.get("quantity") or 0) + 1e-8
            )
        )

        entry_price = float(
            position.get("entry_price") or 0
        )

        if (
            quantity < 1
            or entry_price <= 0
            or has_unresolved_order(symbol)
        ):
            continue

        try:
            price = latest_price(symbol)

            pnl = (
                (price - entry_price) / entry_price
            )

            if pnl <= -STOP_LOSS_PERCENT:
                place_market_order(
                    symbol,
                    OrderSide.SELL,
                    quantity,
                    f"Stop-loss threshold reached: {pnl:.2%}",
                )

            elif pnl >= TAKE_PROFIT_PERCENT:
                place_market_order(
                    symbol,
                    OrderSide.SELL,
                    quantity,
                    f"Take-profit threshold reached: {pnl:.2%}",
                )

            else:
                supabase.table("positions").update({
                    "current_price": price,
                    "updated_at": utc_now(),
                }).eq(
                    "symbol",
                    symbol,
                ).execute()

        except Exception as exc:
            log.exception(
                "Exit management failed for %s",
                symbol,
            )

            log_db(
                "ERROR",
                "Exit management failed",
                {
                    "symbol": symbol,
                    "error": str(exc),
                },
            )


# ============================================================
# ENTRY VALIDATION
# ============================================================

def can_buy(
    symbol,
    price,
    positions,
    broker_positions,
):
    if not math.isfinite(price) or price <= 0:
        return False, "Invalid price"

    if has_unresolved_order(symbol):
        return False, "Unresolved order exists"

    if any(
        str(position.get("symbol")) == symbol
        for position in positions
    ):
        return False, "Already have a managed position"

    if symbol in broker_positions:
        return False, (
            "Symbol already held at broker; "
            "manual reconciliation required"
        )

    if len(positions) >= MAX_OPEN_POSITIONS:
        return False, "Maximum open positions reached"

    remaining = max(
        0.0,
        VIRTUAL_BUDGET - committed_value(positions),
    )

    allowed_value = min(
        MAX_POSITION_VALUE,
        remaining,
    )

    if allowed_value < price:
        return False, "Insufficient virtual budget"

    return True, f"Estimated position limit: {allowed_value:.2f}"


# ============================================================
# MAIN SCAN
# ============================================================

def scan_once():
    config = get_bot_config()

    if config.get("enabled") is not True:
        set_status(
            "Paused: bot is disabled",
            mode="paper",
        )

        log.info("Bot disabled in Supabase.")
        return

    if (
        str(config.get("mode") or "").lower() != "paper"
        or not PAPER_MODE
    ):
        set_status(
            "Blocked: paper mode is required",
            mode="paper",
        )

        raise RuntimeError(
            "Refusing to trade unless paper mode is enabled."
        )

    reconcile_orders()

    try:
        clock = trading.get_clock()
        equity = account_equity()
        broker_positions = broker_positions_by_symbol()

    except Exception:
        set_status(
            "Safety stop: broker account unavailable",
            mode="paper",
        )

        log.exception(
            "Broker verification failed; no new buys."
        )

        return

    if not clock.is_open:
        set_status(
            "Waiting: market is closed",
            mode="paper",
        )
        return

    manage_exits()

    if has_unresolved_order():
        set_status(
            "Waiting: unresolved order requires reconciliation",
            mode="paper",
        )
        return

    for symbol in WATCHLIST:
        positions = active_bot_positions()

        if len(positions) >= MAX_OPEN_POSITIONS:
            break

        if any(
            str(position.get("symbol")) == symbol
            for position in positions
        ):
            continue

        if symbol in broker_positions:
            log.warning(
                "Skipping %s: already held at broker.",
                symbol,
            )
            continue

        try:
            signal, info = get_signal(symbol)

            if signal != "BUY":
                continue

            price_before = latest_price(symbol)

            allowed, reason = can_buy(
                symbol,
                price_before,
                positions,
                broker_positions,
            )

            if not allowed:
                log.info(
                    "Skipping %s: %s",
                    symbol,
                    reason,
                )
                continue

            remaining = max(
                0.0,
                VIRTUAL_BUDGET - committed_value(positions),
            )

            max_value = min(
                MAX_POSITION_VALUE,
                remaining,
            )

            quantity = int(
                math.floor(max_value / price_before)
            )

            if quantity < 1:
                continue

            account = trading.get_account()
            buying_power = float(account.buying_power)

            if (
                not math.isfinite(buying_power)
                or buying_power < quantity * price_before
            ):
                log.warning(
                    "Skipping %s: insufficient buying power.",
                    symbol,
                )
                continue

            price_now = latest_price(symbol)

            if (
                price_now
                > price_before * (1 + MAX_PRICE_DRIFT_PERCENT)
            ):
                log.warning(
                    "Skipping %s: price moved above tolerance.",
                    symbol,
                )
                continue

            # Recalculate quantity using the newest price.
            quantity = min(
                quantity,
                int(math.floor(max_value / price_now)),
            )

            if (
                quantity < 1
                or quantity * price_now > MAX_POSITION_VALUE
            ):
                continue

            if has_unresolved_order():
                break

            reason = (
                f"Daily trend signal; "
                f"reference_price={price_now:.4f}; "
                f"RSI={info.get('rsi', 0):.2f}; "
                f"account_equity={equity:.2f}"
            )

            place_market_order(
                symbol,
                OrderSide.BUY,
                quantity,
                reason,
            )

            log.info(
                "Paper BUY submitted: %s quantity=%s "
                "estimated_value=%.2f",
                symbol,
                quantity,
                quantity * price_now,
            )

            # One new entry per scan.
            break

        except Exception as exc:
            log.exception(
                "Scan failed for %s",
                symbol,
            )

            log_db(
                "ERROR",
                "Symbol scan failed",
                {
                    "symbol": symbol,
                    "error": str(exc),
                },
            )

            if has_unresolved_order():
                break

    set_status(
        "Scan completed",
        last_scan=utc_now(),
        mode="paper",
    )


# ============================================================
# WORKER LOOP
# ============================================================

def main():
    set_status(
        "Worker starting; paper mode enforced",
        mode="paper",
    )

    log.info(
        "Paper worker started; scan interval %s seconds.",
        SCAN_INTERVAL_SECONDS,
    )

    while True:
        try:
            scan_once()

        except Exception as exc:
            log.exception("Scan cycle failed")

            log_db(
                "ERROR",
                "Scan cycle failed",
                {
                    "error": str(exc),
                },
            )

            set_status(
                f"Scan cycle error: {str(exc)[:800]}",
                mode="paper",
            )

        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
