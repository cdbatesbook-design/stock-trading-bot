import json
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

# SAFETY: Alpaca paper trading is enforced below.
# The bot also requires enabled=True in Supabase before
# it can evaluate or submit orders.

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

VIRTUAL_BUDGET = 10000.00
MAX_POSITION_VALUE = 900.00
MAX_OPEN_POSITIONS = 5

STOP_LOSS_PERCENT = 0.03
TAKE_PROFIT_PERCENT = 0.06

SCAN_INTERVAL_SECONDS = 300
MAX_PRICE_DRIFT_PERCENT = 0.01
RSI_PERIOD = 14

# Broker order statuses that mean the broker considers
# an order terminal. Partial fills must still be accounted for.
BROKER_TERMINAL_STATUSES = {
    "filled",
    "canceled",
    "cancelled",
    "expired",
    "rejected",
}

# Application statuses that no longer require reconciliation.
FINAL_ORDER_STATUSES = {
    "synced",
    "canceled",
    "cancelled",
    "expired",
    "rejected",
}

# Trade records included in the managed-position ledger.
FILL_ORDER_STATUSES = {
    "filled",
    "partial_filled",
    "partially_filled",
    "synced",
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

log = logging.getLogger("stock_trading_bot")


# ============================================================
# ENVIRONMENT AND CONNECTIONS
# ============================================================

def required_env(name):
    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(
            f"Missing required environment variable: {name}"
        )

    return value.strip()


ALPACA_KEY = required_env("ALPACA_API_KEY")
ALPACA_SECRET = required_env("ALPACA_SECRET_KEY")

SUPABASE_URL = required_env("SUPABASE_URL")
SUPABASE_KEY = required_env("SUPABASE_SERVICE_ROLE_KEY")

# SAFETY: paper=True is hardcoded.
# Do not change this to live trading.
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


def finite_positive(value):
    try:
        number = float(value)
        return math.isfinite(number) and number > 0
    except (TypeError, ValueError):
        return False


def log_db(level, message, details=None):
    """
    Write an informational or error message to Supabase.

    Expected bot_logs columns:
      level
      message
      created_at

    If the insert fails, the error remains visible in Render.
    """

    try:
        if details is not None:
            message = (
                f"{message} | "
                f"{json.dumps(details, default=str)}"
            )

        supabase.table("bot_logs").insert({
            "level": str(level).upper(),
            "message": str(message)[:8000],
            "created_at": utc_now(),
        }).execute()

    except Exception:
        log.exception(
            "Could not write message to bot_logs"
        )


def set_status(message, **extra):
    """
    Update bot_status.

    Paper mode is always forced, regardless of extra arguments.
    """

    payload = {
        "id": 1,
        "last_heartbeat": utc_now(),
        "status_message": str(message)[:1000],
        "updated_at": utc_now(),
        "mode": "paper",
    }

    payload.update(extra)

    # Prevent callers from changing the recorded mode.
    payload["mode"] = "paper"

    supabase.table("bot_status").upsert(
        payload,
        on_conflict="id",
    ).execute()


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
            "bot_status row id=1 was not found in Supabase"
        )

    return rows[0]


def fetch_rows(table):
    """
    Read all rows from a Supabase table in pages.
    """

    results = []
    page_size = 500
    start = 0

    while True:
        response = (
            supabase.table(table)
            .select("*")
            .range(start, start + page_size - 1)
            .execute()
        )

        batch = response.data or []
        results.extend(batch)

        if len(batch) < page_size:
            break

        start += page_size

    return results


def get_trade_rows():
    return fetch_rows("trades")


def get_position_rows():
    return fetch_rows("positions")


# ============================================================
# MARKET DATA AND STRATEGY
# ============================================================

def latest_price(symbol):
    request = StockLatestTradeRequest(
        symbol_or_symbols=[symbol]
    )

    response = market_data.get_stock_latest_trade(request)
    trade = response.get(symbol)

    if trade is None:
        raise RuntimeError(
            f"No latest trade returned for {symbol}"
        )

    price = float(trade.price)

    if not finite_positive(price):
        raise RuntimeError(
            f"Invalid latest price for {symbol}: {price}"
        )

    return price


def calculate_rsi(close, period=RSI_PERIOD):
    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    relative_strength = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + relative_strength))

    # No losses means RSI is 100, unless there were also
    # no gains or losses, in which case RSI is 50.
    rsi = rsi.where(avg_loss != 0, 100.0)

    no_gain_or_loss = (
        (avg_gain == 0) & (avg_loss == 0)
    )

    rsi = rsi.where(~no_gain_or_loss, 50.0)

    return rsi


def get_signal(symbol):
    """
    BUY when all conditions are true:
      - Last completed daily close > SMA20
      - SMA20 > SMA50
      - RSI is between 45 and 68

    Otherwise HOLD.
    """

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
            return "HOLD", {
                "reason": "No historical price data"
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

        # Exclude the current daily candle because it may
        # still be incomplete.
        if len(close) > 1:
            close = close.iloc[:-1]

        if len(close) < 60:
            return "HOLD", {
                "reason": "Not enough completed daily candles"
            }

        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        rsi = calculate_rsi(close)

        last_close = float(close.iloc[-1])
        last_sma20 = float(sma20.iloc[-1])
        last_sma50 = float(sma50.iloc[-1])
        last_rsi = float(rsi.iloc[-1])

        values = [
            last_close,
            last_sma20,
            last_sma50,
            last_rsi,
        ]

        if not all(math.isfinite(value) for value in values):
            return "HOLD", {
                "reason": "Invalid strategy indicator values"
            }

        info = {
            "close": last_close,
            "sma20": last_sma20,
            "sma50": last_sma50,
            "rsi": last_rsi,
        }

        if (
            last_close > last_sma20
            and last_sma20 > last_sma50
            and 45 <= last_rsi <= 68
        ):
            return "BUY", info

        return "HOLD", info

    except Exception as exc:
        log.exception(
            "Signal calculation failed for %s",
            symbol,
        )

        return "HOLD", {
            "reason": str(exc)[:500]
        }


# ============================================================
# TRADE AND POSITION RECORDS
# ============================================================

def order_status(order):
    status = getattr(order, "status", "")
    return str(status).lower()


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
    """
    Insert or update a trade using the unique order_id.

    The trades table must have a unique constraint on order_id.
    """

    payload = {
        "order_id": str(order_id),
        "symbol": str(symbol),
        "side": str(side).lower(),
        "quantity": float(quantity or 0),
        "price": float(price or 0),
        "status": str(status).lower(),
        "strategy": str(strategy),
        "reason": str(reason),
        "updated_at": utc_now(),
    }

    if total_value is not None:
        payload["total_value"] = float(total_value)

    existing = (
        supabase.table("trades")
        .select("order_id")
        .eq("order_id", str(order_id))
        .limit(1)
        .execute()
    )

    if not existing.data:
        payload["created_at"] = utc_now()

    supabase.table("trades").upsert(
        payload,
        on_conflict="order_id",
    ).execute()


def upsert_position(
    symbol,
    quantity,
    entry_price,
    current_price,
    status,
):
    """
    Update the latest open position for this symbol,
    or insert a position when none exists.
    """

    response = (
        supabase.table("positions")
        .select("*")
        .eq("symbol", symbol)
        .eq("status", "open")
        .order("updated_at", desc=True)
        .limit(1)
        .execute()
    )

    now = utc_now()

    payload = {
        "symbol": symbol,
        "quantity": float(quantity),
        "entry_price": float(entry_price),
        "current_price": float(current_price),
        "status": str(status).lower(),
        "updated_at": now,
    }

    if str(status).lower() == "open":
        payload["closed_at"] = None

    if response.data:
        position_id = response.data[0].get("id")

        supabase.table("positions").update(
            payload
        ).eq("id", position_id).execute()

    else:
        payload["opened_at"] = now

        if str(status).lower() == "closed":
            payload["closed_at"] = now

        supabase.table("positions").insert(
            payload
        ).execute()


def active_bot_positions():
    rows = get_position_rows()
    active = []

    for row in rows:
        try:
            quantity = float(row.get("quantity") or 0)
        except (TypeError, ValueError):
            continue

        if (
            str(row.get("status", "")).lower() == "open"
            and quantity > 0
        ):
            active.append(row)

    return active


def unresolved_status(status):
    return (
        str(status or "").lower()
        not in FINAL_ORDER_STATUSES
    )


def has_unresolved_order(symbol=None):
    for row in get_trade_rows():
        if symbol and row.get("symbol") != symbol:
            continue

        if not row.get("order_id"):
            continue

        if unresolved_status(row.get("status")):
            return True

    return False


# ============================================================
# BROKER ACCOUNT AND POSITIONS
# ============================================================

def account_equity():
    account = trading.get_account()
    equity = float(account.equity)

    if not finite_positive(equity):
        raise RuntimeError(
            f"Invalid account equity: {equity}"
        )

    return equity


def broker_positions_by_symbol():
    result = {}

    for position in trading.get_all_positions():
        symbol = str(position.symbol)

        try:
            quantity = float(position.qty)
        except (TypeError, ValueError):
            quantity = 0.0

        result[symbol] = {
            "quantity": quantity,
            "market_value": float(
                getattr(position, "market_value", 0) or 0
            ),
            "entry_price": float(
                getattr(position, "avg_entry_price", 0) or 0
            ),
        }

    return result


def committed_value(positions):
    total = 0.0

    for position in positions:
        try:
            quantity = float(
                position.get("quantity") or 0
            )

            current_price = float(
                position.get("current_price")
                or position.get("entry_price")
                or 0
            )

            if (
                math.isfinite(quantity)
                and math.isfinite(current_price)
            ):
                total += quantity * current_price

        except (TypeError, ValueError):
            continue

    return total


def calculate_managed_position(trades, symbol):
    """
    Calculate bot-managed net long quantity and average cost
    from trade records that contain actual fills.
    """

    symbol_trades = [
        row for row in trades
        if row.get("symbol") == symbol
        and str(row.get("status", "")).lower()
        in FILL_ORDER_STATUSES
    ]

    symbol_trades.sort(
        key=lambda row: str(row.get("created_at") or "")
    )

    quantity = 0.0
    cost_basis = 0.0

    for row in symbol_trades:
        side = str(row.get("side", "")).lower()

        try:
            trade_quantity = float(
                row.get("quantity") or 0
            )
            price = float(row.get("price") or 0)
        except (TypeError, ValueError):
            continue

        if (
            not math.isfinite(trade_quantity)
            or not math.isfinite(price)
            or trade_quantity <= 0
            or price <= 0
        ):
            continue

        if side == "buy":
            cost_basis += trade_quantity * price
            quantity += trade_quantity

        elif side == "sell":
            if quantity <= 0:
                continue

            average_cost = cost_basis / quantity
            sold_quantity = min(
                trade_quantity,
                quantity,
            )

            quantity -= sold_quantity
            cost_basis -= sold_quantity * average_cost

            if quantity <= 0.000001:
                quantity = 0.0
                cost_basis = 0.0

    average_cost = (
        cost_basis / quantity
        if quantity > 0
        else 0.0
    )

    return quantity, average_cost


def sync_position_from_broker(symbol):
    """
    Rebuild a bot-managed position from the trade ledger.

    Unknown broker holdings are not automatically adopted.
    A mismatch raises an error rather than silently creating
    a position from assumptions.
    """

    trades = get_trade_rows()

    quantity, entry_price = calculate_managed_position(
        trades,
        symbol,
    )

    if quantity <= 0:
        response = (
            supabase.table("positions")
            .select("*")
            .eq("symbol", symbol)
            .eq("status", "open")
            .execute()
        )

        for row in response.data or []:
            supabase.table("positions").update({
                "quantity": 0,
                "status": "closed",
                "updated_at": utc_now(),
                "closed_at": utc_now(),
            }).eq("id", row["id"]).execute()

        return

    broker_positions = broker_positions_by_symbol()
    broker_position = broker_positions.get(symbol)

    if broker_position is None:
        raise RuntimeError(
            f"Trade ledger shows {symbol}, but Alpaca "
            "has no corresponding position"
        )

    broker_quantity = float(
        broker_position.get("quantity") or 0
    )

    if broker_quantity + 0.000001 < quantity:
        raise RuntimeError(
            f"Managed quantity for {symbol} ({quantity}) "
            f"exceeds Alpaca quantity ({broker_quantity})"
        )

    price = latest_price(symbol)

    upsert_position(
        symbol=symbol,
        quantity=quantity,
        entry_price=entry_price,
        current_price=price,
        status="open",
    )


# ============================================================
# ORDER RECONCILIATION
# ============================================================

def reconcile_orders():
    """
    Reconcile recorded orders with Alpaca.

    Orders are not automatically resubmitted when a response
    is missing or ambiguous.

    Filled quantities are recorded separately from requested
    quantities. A terminal order with a partial fill is recorded
    as synced only after its actual filled quantity and price
    are available.
    """

    rows = get_trade_rows()

    for row in rows:
        current_status = str(
            row.get("status") or ""
        ).lower()

        if current_status in FINAL_ORDER_STATUSES:
            continue

        client_order_id = row.get("order_id")

        if not client_order_id:
            continue

        try:
            order = trading.get_order_by_client_id(
                client_order_id
            )

        except Exception as exc:
            log.warning(
                "Order reconciliation deferred for %s: %s",
                client_order_id,
                exc,
            )

            log_db(
                "WARNING",
                "Order reconciliation deferred",
                {
                    "order_id": client_order_id,
                    "error": str(exc)[:500],
                },
            )

            continue

        broker_status = order_status(order)

        try:
            filled_quantity = float(
                getattr(order, "filled_qty", 0) or 0
            )

            filled_price = float(
                getattr(order, "filled_avg_price", 0) or 0
            )
        except (TypeError, ValueError):
            filled_quantity = 0.0
            filled_price = 0.0

        has_fill = (
            math.isfinite(filled_quantity)
            and math.isfinite(filled_price)
            and filled_quantity > 0
            and filled_price > 0
        )

        terminal = (
            broker_status in BROKER_TERMINAL_STATUSES
        )

        if has_fill:
            total_value = (
                filled_quantity * filled_price
            )

            # For a terminal order with a fill, retain the
            # actual filled quantity and price. A partially
            # filled cancellation still represents a real fill.
            record_status = (
                "synced" if terminal else broker_status
            )

            write_trade(
                order_id=client_order_id,
                symbol=row.get("symbol"),
                side=row.get("side"),
                quantity=filled_quantity,
                price=filled_price,
                status=record_status,
                strategy=row.get("strategy", ""),
                reason=(
                    f"{row.get('reason', '')}; "
                    f"broker_status={broker_status}"
                ),
                total_value=total_value,
            )

            if terminal:
                sync_position_from_broker(
                    row.get("symbol")
                )

                log_db(
                    "INFO",
                    "Terminal order reconciled",
                    {
                        "order_id": client_order_id,
                        "symbol": row.get("symbol"),
                        "broker_status": broker_status,
                        "filled_quantity": filled_quantity,
                        "filled_price": filled_price,
                    },
                )

        else:
            # No confirmed fill: preserve the order's broker
            # status and requested quantity for later review.
            write_trade(
                order_id=client_order_id,
                symbol=row.get("symbol"),
                side=row.get("side"),
                quantity=float(row.get("quantity") or 0),
                price=float(row.get("price") or 0),
                status=broker_status,
                strategy=row.get("strategy", ""),
                reason=row.get("reason", ""),
                total_value=row.get("total_value"),
            )

            if terminal:
                log_db(
                    "INFO",
                    "Terminal order has no confirmed fill",
                    {
                        "order_id": client_order_id,
                        "symbol": row.get("symbol"),
                        "broker_status": broker_status,
                    },
                )


# ============================================================
# ORDER PLACEMENT
# ============================================================

def place_market_order(symbol, side, qty, reason):
    """
    Record order intent before submitting it.

    If submission returns an ambiguous result, do not
    automatically retry the order.
    """

    if not isinstance(qty, int) or qty < 1:
        raise ValueError(
            f"Order quantity must be a positive integer: {qty}"
        )

    if side not in (OrderSide.BUY, OrderSide.SELL):
        raise ValueError("Invalid order side")

    if has_unresolved_order(symbol):
        raise RuntimeError(
            f"An unresolved order already exists for {symbol}"
        )

    client_order_id = (
        f"bot-{symbol.lower()}-{side.value}-"
        f"{uuid.uuid4().hex[:20]}"
    )

    # Record intent before submission so an uncertain result
    # can be reconciled by client_order_id.
    write_trade(
        order_id=client_order_id,
        symbol=symbol,
        side=side.value,
        quantity=qty,
        price=0,
        status="pending",
        strategy="SMA20>SMA50_RSI45-68",
        reason=reason,
    )

    request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
        client_order_id=client_order_id,
    )

    try:
        order = trading.submit_order(
            order_data=request
        )

    except Exception as exc:
        log.exception(
            "Order submission result may be ambiguous for %s",
            symbol,
        )

        log_db(
            "ERROR",
            "Order submission exception; manual reconciliation "
            "required before any retry",
            {
                "symbol": symbol,
                "client_order_id": client_order_id,
                "error": str(exc)[:500],
            },
        )

        raise RuntimeError(
            "Order submission failed or has an ambiguous result. "
            "The order was not automatically retried."
        ) from exc

    broker_status = order_status(order)

    write_trade(
        order_id=client_order_id,
        symbol=symbol,
        side=side.value,
        quantity=qty,
        price=0,
        status=broker_status,
        strategy="SMA20>SMA50_RSI45-68",
        reason=reason,
    )

    log.info(
        "Submitted %s order: %s shares of %s; status=%s",
        side.value,
        qty,
        symbol,
        broker_status,
    )

    log_db(
        "INFO",
        "Order submitted",
        {
            "order_id": client_order_id,
            "symbol": symbol,
            "side": side.value,
            "quantity": qty,
            "status": broker_status,
        },
    )

    return order


# ============================================================
# EXIT MANAGEMENT
# ============================================================

def manage_exits():
    positions = active_bot_positions()

    for position in positions:
        symbol = position.get("symbol")

        if not symbol:
            continue

        if has_unresolved_order(symbol):
            log.info(
                "Skipping exit check for %s: unresolved order",
                symbol,
            )
            continue

        try:
            quantity_value = float(
                position.get("quantity") or 0
            )

            entry_price = float(
                position.get("entry_price") or 0
            )

            if (
                not math.isfinite(quantity_value)
                or quantity_value < 1
                or not finite_positive(entry_price)
            ):
                continue

            quantity = int(quantity_value)

            price = latest_price(symbol)

            profit_percent = (
                (price - entry_price) / entry_price
            )

            if profit_percent <= -STOP_LOSS_PERCENT:
                reason = (
                    f"Stop-loss trigger observed: "
                    f"{profit_percent:.2%} return"
                )

                place_market_order(
                    symbol,
                    OrderSide.SELL,
                    quantity,
                    reason,
                )

                log.info(
                    "Stop-loss sell submitted for %s",
                    symbol,
                )

            elif profit_percent >= TAKE_PROFIT_PERCENT:
                reason = (
                    f"Take-profit trigger observed: "
                    f"{profit_percent:.2%} return"
                )

                place_market_order(
                    symbol,
                    OrderSide.SELL,
                    quantity,
                    reason,
                )

                log.info(
                    "Take-profit sell submitted for %s",
                    symbol,
                )

            else:
                supabase.table("positions").update({
                    "current_price": price,
                    "updated_at": utc_now(),
                }).eq("id", position["id"]).execute()

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
                    "error": str(exc)[:500],
                },
            )


# ============================================================
# BUY SAFETY CHECKS
# ============================================================

def can_buy(symbol, price, positions, broker_positions):
    if not finite_positive(price):
        return False, "Invalid price"

    if has_unresolved_order(symbol):
        return False, "Unresolved order exists"

    for position in positions:
        if position.get("symbol") == symbol:
            return False, "Already have a bot-managed position"

    if symbol in broker_positions:
        return False, "Broker already holds this symbol"

    if len(positions) >= MAX_OPEN_POSITIONS:
        return False, "Maximum open positions reached"

    invested = committed_value(positions)
    remaining_budget = VIRTUAL_BUDGET - invested

    if remaining_budget < price:
        return False, "Virtual budget exhausted"

    if min(
        MAX_POSITION_VALUE,
        remaining_budget,
    ) < price:
        return False, "Price exceeds allowed position value"

    return True, "Approved"


# ============================================================
# MAIN SCAN
# ============================================================

def scan_once():
    """
    Run one worker cycle.

    Update heartbeat and last_scan at cycle start.
    Keep the bot paused unless Supabase explicitly enables it.
    """

    cycle_time = utc_now()

    try:
        set_status(
            "Worker check started; paper mode enforced",
            last_scan=cycle_time,
            mode="paper",
        )

    except Exception:
        log.exception(
            "Could not update worker heartbeat at cycle start"
        )
        raise

    config = get_bot_config()

    # FIX: Record paused status in Supabase as well as Render.
    if config.get("enabled") is not True:
        message = "Paused: bot_status.enabled is not true"

        set_status(
            message,
            last_scan=cycle_time,
            mode="paper",
        )

        log.info(message)

        log_db(
            "INFO",
            message,
            {
                "enabled": config.get("enabled"),
                "mode": "paper",
                "last_scan": cycle_time,
            },
        )

        return

    # SAFETY: This worker never permits live trading.
    if (
        not PAPER_MODE
        or str(config.get("mode", "paper")).lower() != "paper"
    ):
        message = (
            "SAFETY STOP: only paper trading is permitted"
        )

        set_status(
            message,
            last_scan=cycle_time,
            mode="paper",
        )

        log_db("ERROR", message)

        raise RuntimeError(
            "Live trading is disabled by this worker"
        )

    # Reconcile existing orders before considering new ones.
    reconcile_orders()

    try:
        clock = trading.get_clock()
        equity = account_equity()
        broker_positions = broker_positions_by_symbol()

    except Exception as exc:
        message = (
            "Safety stop: broker/account check failed: "
            f"{str(exc)[:400]}"
        )

        set_status(
            message,
            last_scan=cycle_time,
            mode="paper",
        )

        log.exception("Broker safety check failed")

        log_db(
            "ERROR",
            message,
            {"error": str(exc)[:500]},
        )

        return

    if not clock.is_open:
        message = "Waiting: market is closed"

        set_status(
            message,
            last_scan=cycle_time,
            mode="paper",
        )

        log.info(message)
        log_db("INFO", message)

        return

    manage_exits()

    if has_unresolved_order():
        message = (
            "Waiting: unresolved order requires reconciliation"
        )

        set_status(
            message,
            last_scan=cycle_time,
            mode="paper",
        )

        log_db("WARNING", message)

        return

    for symbol in WATCHLIST:
        positions = active_bot_positions()

        if len(positions) >= MAX_OPEN_POSITIONS:
            log.info(
                "Maximum open positions reached; ending scan"
            )
            break

        if any(
            position.get("symbol") == symbol
            for position in positions
        ):
            continue

        if symbol in broker_positions:
            continue

        signal, info = get_signal(symbol)

        if signal != "BUY":
            log.info(
                "%s: HOLD; details=%s",
                symbol,
                info,
            )
            continue

        try:
            price_before = latest_price(symbol)

            allowed, reason = can_buy(
                symbol,
                price_before,
                positions,
                broker_positions,
            )

            if not allowed:
                log.info(
                    "%s: purchase skipped: %s",
                    symbol,
                    reason,
                )
                continue

            invested = committed_value(positions)
            remaining_budget = (
                VIRTUAL_BUDGET - invested
            )

            max_value = min(
                MAX_POSITION_VALUE,
                remaining_budget,
            )

            qty = math.floor(
                max_value / price_before
            )

            if qty < 1:
                log.info(
                    "%s: price is above the permitted "
                    "position amount",
                    symbol,
                )
                continue

            account = trading.get_account()

            buying_power = float(
                getattr(account, "buying_power", 0) or 0
            )

            if (
                not math.isfinite(buying_power)
                or buying_power <= 0
            ):
                log.warning(
                    "%s: invalid or unavailable broker buying power",
                    symbol,
                )
                continue

            if qty * price_before > buying_power:
                log.info(
                    "%s: insufficient broker buying power",
                    symbol,
                )
                continue

            price_now = latest_price(symbol)

            drift = abs(
                price_now - price_before
            ) / price_before

            if drift > MAX_PRICE_DRIFT_PERCENT:
                log.info(
                    "%s: skipped because price drift was %.2f%%",
                    symbol,
                    drift * 100,
                )
                continue

            qty = math.floor(
                max_value / price_now
            )

            if qty < 1:
                continue

            estimated_value = qty * price_now

            if estimated_value > MAX_POSITION_VALUE:
                log.info(
                    "%s: skipped because position exceeds limit",
                    symbol,
                )
                continue

            if estimated_value > remaining_budget:
                log.info(
                    "%s: skipped because virtual budget "
                    "would be exceeded",
                    symbol,
                )
                continue

            if has_unresolved_order():
                log.warning(
                    "Unresolved order detected; ending scan"
                )
                break

            reason = (
                f"Strategy entry: close={info.get('close')}, "
                f"SMA20={info.get('sma20')}, "
                f"SMA50={info.get('sma50')}, "
                f"RSI={info.get('rsi')}; "
                f"reference_price={price_now}; "
                f"account_equity={equity}"
            )

            place_market_order(
                symbol,
                OrderSide.BUY,
                qty,
                reason,
            )

            log.info(
                "BUY submitted: %s shares of %s at reference "
                "price %.4f",
                qty,
                symbol,
                price_now,
            )

            # At most one new buy order per scan.
            break

        except Exception as exc:
            log.exception(
                "Scan failed while evaluating %s",
                symbol,
            )

            log_db(
                "ERROR",
                "Symbol scan failed",
                {
                    "symbol": symbol,
                    "error": str(exc)[:500],
                },
            )

            # Stop the scan after any order/data exception.
            break

    set_status(
        "Scan completed",
        last_scan=cycle_time,
        mode="paper",
    )

    log_db(
        "INFO",
        "Scan completed",
        {
            "watchlist_count": len(WATCHLIST),
            "last_scan": cycle_time,
            "mode": "paper",
        },
    )


# ============================================================
# WORKER LOOP
# ============================================================

def main():
    set_status(
        "Worker starting; paper mode enforced",
        last_scan=utc_now(),
        mode="paper",
    )

    log.info(
        "Paper worker started; scan interval %s seconds",
        SCAN_INTERVAL_SECONDS,
    )

    log_db(
        "INFO",
        "Paper worker started; trading remains controlled "
        "by bot_status.enabled",
        {
            "mode": "paper",
            "scan_interval_seconds": SCAN_INTERVAL_SECONDS,
            "watchlist_count": len(WATCHLIST),
        },
    )

    while True:
        try:
            scan_once()

        except Exception as exc:
            log.exception("Scan cycle failed")

            log_db(
                "ERROR",
                "Scan cycle failed",
                {"error": str(exc)[:500]},
            )

            try:
                set_status(
                    f"Scan cycle error: {str(exc)[:700]}",
                    last_scan=utc_now(),
                    mode="paper",
                )

            except Exception:
                log.exception(
                    "Could not update Supabase after scan error"
                )

        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
