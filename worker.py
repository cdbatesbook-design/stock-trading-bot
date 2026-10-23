import os
import time
import math
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from supabase import create_client
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus


# ==========================================================
# AUTOTRADER AI — PAPER TRADING WORKER
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger("autotrader")

# Risk settings
VIRTUAL_BUDGET = 10000.00
MAX_POSITION_VALUE = 900.00

= 5

STOP_LOSS_PERCENT = 0.03
TAKE_PROFIT_PERCENT = 0.06

SCAN_INTERVAL_SECONDS = 300

EASTERN = ZoneInfo("America/New_York")

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AMD", "AVGO", "NFLX",
    "JPM", "BAC", "WMT", "COST", "DIS",
    "KO", "PEP", "XOM", "F", "INTC",
]


# ==========================================================
# CONFIGURATION
# ==========================================================

def required_env(name):
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Missing environment variable: {name}"
        )

    return value


ALPACA_KEY = required_env("ALPACA_API_KEY")
ALPACA_SECRET = required_env("ALPACA_SECRET_KEY")

ALPACA_URL = os.getenv(
    "ALPACA_BASE_URL",
    "https://paper-api.alpaca.markets",
).rstrip("/")

SUPABASE_URL = required_env("SUPABASE_URL")
SUPABASE_KEY = required_env("SUPABASE_SERVICE_ROLE_KEY")

# Fail closed unless the configured URL is exactly the paper endpoint.
if ALPACA_URL.lower() != "https://paper-api.alpaca.markets":
    raise RuntimeError(
        "SAFETY STOP: Only the Alpaca paper endpoint is allowed."
    )

db = create_client(SUPABASE_URL, SUPABASE_KEY)

trading = TradingClient(
    ALPACA_KEY,
    ALPACA_SECRET,
    paper=True,
)


# ==========================================================
# DATABASE HELPERS
# ==========================================================

def log_event(message, level="INFO", symbol=None):
    log_method = getattr(
        logger,
        level.lower(),
        logger.info,
    )

    log_method(message)

    try:
        db.table("bot_logs").insert({
            "level": level,
            "message": str(message)[:1000],
            "symbol": symbol,
        }).execute()

    except Exception as exc:
        logger.error("Database logging failed: %s", exc)


def update_status(message, scan_completed=False):
    now = datetime.now(EASTERN).isoformat()

    values = {
        "last_heartbeat": now,
        "status_message": str(message)[:500],
        "updated_at": now,
    }

    if scan_completed:
        values["last_scan"] = now

    try:
        db.table("bot_status").update(
            values
        ).eq("id", 1).execute()

    except Exception as exc:
        logger.error("Status update failed: %s", exc)


def bot_is_enabled():
    result = (
        db.table("bot_status")
        .select("enabled, mode")
        .eq("id", 1)
        .single()
        .execute()
    )

    row = result.data or {}

    return (
        row.get("enabled") is True
        and row.get("mode") == "paper"
    )


def get_bot_trades():
    result = (
        db.table("trades")
        .select("*")
        .execute()
    )

    return result.data or []


def get_tracked_positions():
    result = (
        db.table("positions")
        .select("*")
        .eq("status", "open")
        .execute()
    )

    return result.data or []


# ==========================================================
# BROKER HELPERS
# ==========================================================

def get_broker_positions():
    return trading.get_all_positions()


def get_open_orders():
    request = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        limit=500,
    )

    return trading.get_orders(filter=request)


def get_price_history(symbol):
    data = yf.download(
        symbol,
        period="6mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        try:
            close_data = data["Close"]

            if isinstance(close_data, pd.DataFrame):
                if symbol in close_data.columns:
                    close = close_data[symbol]
                else:
                    close = close_data.iloc[:, 0]
            else:
                close = close_data

        except Exception:
            return None

    else:
        if "Close" not in data.columns:
            return None

        close = data["Close"]

    close = pd.to_numeric(
        close,
        errors="coerce",
    ).dropna()

    if len(close) < 55:
        return None

    return close


def calculate_signal(symbol):
    close = get_price_history(symbol)

    if close is None:
        return None

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    changes = close.diff()

    gains = changes.clip(lower=0).rolling(14).mean()
    losses = (-changes.clip(upper=0)).rolling(14).mean()

    relative_strength = gains / losses.replace(
        0,
        float("nan"),
    )

    rsi = 100 - (100 / (1 + relative_strength))

    price = float(close.iloc[-1])
    fast = float(sma20.iloc[-1])
    slow = float(sma50.iloc[-1])

    current_rsi = (
        float(rsi.iloc[-1])
        if pd.notna(rsi.iloc[-1])
        else 50.0
    )

    if not all(
        math.isfinite(value)
        for value in [price, fast, slow, current_rsi]
    ):
        return None

    return {
        "symbol": symbol,
        "price": price,
        "sma20": fast,
        "sma50": slow,
        "rsi": current_rsi,
        "buy": (
            price > fast
            and fast > slow
            and 45 <= current_rsi <= 68
        ),
    }


# ==========================================================
# ORDER RECONCILIATION
# ==========================================================

TERMINAL_STATUSES = {
    "filled",
    "canceled",
    "cancelled",
    "rejected",
    "expired",
    "done_for_day",
}


def find_order_by_client_id(client_id):
    try:
        return trading.get_order_by_client_id(client_id)
    except Exception:
        return None


def sync_position_from_broker(symbol):
    """
    Synchronize a bot-tracked position with the broker's
    actual position. This function never creates a new order.
    """
    broker_positions = get_broker_positions()

    broker_position = next(
        (
            p for p in broker_positions
            if p.symbol == symbol
        ),
        None,
    )

    existing_result = (
        db.table("positions")
        .select("*")
        .eq("symbol", symbol)
        .eq("status", "open")
        .execute()
    )

    existing = existing_result.data or []

    if broker_position is None:
        for row in existing:
            db.table("positions").update({
                "status": "closed",
                "closed_at": datetime.now(EASTERN).isoformat(),
                "updated_at": datetime.now(EASTERN).isoformat(),
            }).eq("id", row["id"]).execute()

        return

    quantity = abs(float(broker_position.qty))
    entry_price = float(broker_position.avg_entry_price)

    if quantity <= 0 or entry_price <= 0:
        raise RuntimeError(
            f"Invalid broker position for {symbol}"
        )

    current_price_value = getattr(
        broker_position,
        "current_price",
        None,
    )

    current_price = (
        float(current_price_value)
        if current_price_value is not None
        else None
    )

    values = {
        "symbol": symbol,
        "quantity": quantity,
        "entry_price": entry_price,
        "current_price": current_price,
        "status": "open",
        "updated_at": datetime.now(EASTERN).isoformat(),
    }

    if existing:
        # Update instead of adding the same position again.
        db.table("positions").update(
            values
        ).eq("id", existing[0]["id"]).execute()

        # Close any accidental duplicate tracking rows.
        for duplicate in existing[1:]:
            db.table("positions").update({
                "status": "closed",
                "closed_at": datetime.now(EASTERN).isoformat(),
                "updated_at": datetime.now(EASTERN).isoformat(),
            }).eq("id", duplicate["id"]).execute()

    else:
        db.table("positions").insert(values).execute()


def reconcile_orders():
    """
    Check unresolved order records against Alpaca.
    Never assume a submitted order has filled.
    """
    trades = get_bot_trades()

    for trade in trades:
        client_id = trade.get("order_id")
        old_status = str(
            trade.get("status", "")
        ).lower()

        if not client_id or old_status in TERMINAL_STATUSES:
            continue

        order = find_order_by_client_id(client_id)

        if order is None:
            # A missing order cannot safely be assumed to have
            # failed. Leave it unresolved and block new buys
            # for the same symbol until investigated.
            continue

        status = str(order.status).lower()

        filled_quantity = float(
            getattr(order, "filled_qty", 0) or 0
        )

        fill_price_value = getattr(
            order,
            "filled_avg_price",
            None,
        )

        fill_price = (
            float(fill_price_value)
            if fill_price_value is not None
            else None
        )

        if filled_quantity > 0:
            # Synchronize the real broker position before
            # marking the order as reconciled.
            sync_position_from_broker(order.symbol)

        update_values = {
            "status": status,
        }

        if fill_price is not None:
            update_values["price"] = fill_price
            update_values["total_value"] = round(
                filled_quantity * fill_price,
                2,
            )

        # Keep requested quantity for reservation accounting.
        # Actual filled quantity remains available from Alpaca.
        db.table("trades").update(
            update_values
        ).eq("order_id", client_id).execute()

        if status in TERMINAL_STATUSES:
            log_event(
                f"Order reconciled: {order.symbol}, "
                f"status={status}, filled={filled_quantity}",
                symbol=order.symbol,
            )


# ==========================================================
# ORDER SUBMISSION
# ==========================================================

def submit_paper_order(symbol, quantity, side, price, reason):
    if quantity <= 0:
        return False

    if side not in ("buy", "sell"):
        return False

    # Refresh broker orders immediately before submission.
    if any(
        order.symbol == symbol
        for order in get_open_orders()
    ):
        log_event(
            f"Skipped {symbol}: an order is already open.",
            symbol=symbol,
        )
        return False

    # Block a second order if a previous intent or submission
    # for this symbol has not been reconciled.
    for trade in get_bot_trades():
        if (
            trade.get("symbol") == symbol
            and str(trade.get("status", "")).lower()
            not in TERMINAL_STATUSES
        ):
            log_event(
                f"Skipped {symbol}: an earlier order "
                "has not been reconciled.",
                symbol=symbol,
            )
            return False

    now = datetime.now(EASTERN)
    minute_bucket = (now.minute // 5) * 5

    client_id = (
        f"ata-{side}-{symbol.lower()}-"
        f"{now.strftime('%Y%m%d%H')}{minute_bucket:02d}"
    )

    # Write the order intent before contacting the broker.
    # The client ID is stored in order_id because the current
    # database schema has no separate client_order_id column.
    try:
        db.table("trades").insert({
            "order_id": client_id,
            "symbol": symbol,
            "side": side,
            "quantity": float(quantity),
            "price": float(price),
            "total_value": round(
                float(quantity) * float(price),
                2,
            ),
            "strategy": "SMA20_SMA50_RSI",
            "reason": reason,
            "status": "intent",
        }).execute()

    except Exception as exc:
        log_event(
            f"Could not record order intent for {symbol}; "
            f"no order submitted: {exc}",
            level="ERROR",
            symbol=symbol,
        )
        return False

    try:
        order = trading.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol,
                qty=float(quantity),
                side=(
                    OrderSide.BUY
                    if side == "buy"
                    else OrderSide.SELL
                ),
                time_in_force=TimeInForce.DAY,
                client_order_id=client_id,
            )
        )

        db.table("trades").update({
            "status": str(order.status).lower(),
        }).eq("order_id", client_id).execute()

        log_event(
            f"{side.upper()} submitted for {symbol}; "
            f"order status={order.status}",
            symbol=symbol,
        )

        return True

    except Exception as exc:
        # Do not automatically retry an uncertain submission.
        # First try to recover the order using its client ID.
        order = find_order_by_client_id(client_id)

        if order is not None:
            db.table("trades").update({
                "status": str(order.status).lower(),
            }).eq("order_id", client_id).execute()

            log_event(
                f"Recovered order after submission error: "
                f"{symbol}, status={order.status}",
                symbol=symbol,
            )

            return True

        log_event(
            f"Submission outcome uncertain for {symbol}. "
            f"Order intent retained; investigate before retrying. "
            f"Details: {exc}",
            level="ERROR",
            symbol=symbol,
        )

        return False


# ==========================================================
# BUDGET AND SCANNING
# ==========================================================

def calculate_reserved_exposure(broker_positions, trades):
    """
    Conservatively reserve budget for existing broker positions
    and unresolved buy orders.
    """
    exposure = 0.0

    for position in broker_positions:
        market_value = getattr(
            position,
            "market_value",
            None,
        )

        if market_value is not None:
            exposure += abs(float(market_value))

    for trade in trades:
        status = str(
            trade.get("status", "")
        ).lower()

        if trade.get("side") != "buy":
            continue

        if status in TERMINAL_STATUSES:
            continue

        quantity = float(trade.get("quantity", 0) or 0)
        price = float(trade.get("price", 0) or 0)

        exposure += quantity * price

    return exposure


def scan_once():
    if not bot_is_enabled():
        update_status("Worker running; trading disabled.")
        return

    # Use Alpaca's official market clock.
    clock = trading.get_clock()

    if not clock.is_open:
        update_status("Waiting for the market to open.")
        return

    # Reconcile orders before making new decisions.
    reconcile_orders()

    broker_positions = get_broker_positions()
    open_orders = get_open_orders()
    tracked_positions = get_tracked_positions()
    trades = get_bot_trades()

    tracked_symbols = {
        row["symbol"]
        for row in tracked_positions
    }

    position_symbols = {
        position.symbol
        for position in broker_positions
    }

    pending_symbols = {
        order.symbol
        for order in open_orders
    }

    # Count existing positions and pending buys conservatively.
    reserved_symbols = set(position_symbols)

    for order in open_orders:
        if str(order.side).lower().endswith("buy"):
            reserved_symbols.add(order.symbol)

    exposure = calculate_reserved_exposure(
        broker_positions,
        trades,
    )

    log_event(
        f"Scan started. Reserved exposure=${exposure:.2f}; "
        f"broker positions={len(broker_positions)}."
    )

    for symbol in WATCHLIST:
        try:
            # Refresh broker state before each decision.
            broker_positions = get_broker_positions()
            open_orders = get_open_orders()
            tracked_positions = get_tracked_positions()
            trades = get_bot_trades()

            broker_map = {
                position.symbol: position
                for position in broker_positions
            }

            tracked_map = {
                row["symbol"]: row
                for row in tracked_positions
            }

            pending_symbols = {
                order.symbol
                for order in open_orders
            }

            signal = calculate_signal(symbol)

            if signal is None:
                continue

            broker_position = broker_map.get(symbol)
            tracked_position = tracked_map.get(symbol)

            # Only manage positions the bot has recorded.
            # Untracked positions are not automatically sold.
            if tracked_position is not None:
                if broker_position is None:
                    log_event(
                        f"Tracked position {symbol} is missing "
                        "at the broker; manual review required.",
                        level="ERROR",
                        symbol=symbol,
                    )
                    continue

                if symbol in pending_symbols:
                    continue

                quantity = float(broker_position.qty)

                if quantity <= 0:
                    continue

                entry_price = float(
                    broker_position.avg_entry_price
                )

                current_price_value = getattr(
                    broker_position,
                    "current_price",
                    None,
                )

                if (
                    current_price_value is None
                    or entry_price <= 0
                ):
                    continue

                current_price = float(current_price_value)
                change = (current_price - entry_price) / entry_price

                reason = None

                if change <= -STOP_LOSS_PERCENT:
                    reason = "Stop-loss threshold reached"

                elif change >= TAKE_PROFIT_PERCENT:
                    reason = "Take-profit threshold reached"

                elif current_price < signal["sma20"]:
                    reason = "Price below 20-day moving average"

                if reason:
                    submit_paper_order(
                        symbol,
                        quantity,
                        "sell",
                        current_price,
                        reason,
                    )

                continue

            # Do not buy a symbol already held at the broker.
            if broker_position is not None:
                continue

            if symbol in pending_symbols:
                continue

            # Avoid buying if an unresolved database order exists.
            unresolved = any(
                row.get("symbol") == symbol
                and str(row.get("status", "")).lower()
                not in TERMINAL_STATUSES
                for row in trades
            )

            if unresolved:
                continue

            if not signal["buy"]:
                continue

            broker_positions = get_broker_positions()
            trades = get_bot_trades()

            exposure = calculate_reserved_exposure(
                broker_positions,
                trades,
            )

            reserved_symbols = {
                position.symbol
                for position in broker_positions
            }

            for order in get_open_orders():
                if str(order.side).lower().endswith("buy"):
                    reserved_symbols.add(order.symbol)

            if len(reserved_symbols) >= MAX_OPEN_POSITIONS:
                continue

            remaining_budget = VIRTUAL_BUDGET - exposure

            if remaining_budget < 10:
                continue

            reference_price = signal["price"]

            if reference_price <= 0:
                continue

            allocation = min(
                MAX_POSITION_VALUE,
                remaining_budget,
            )

            # Whole shares only; never intentionally exceed
            # the estimated allocation.
            quantity = math.floor(
                allocation / reference_price
            )

            if quantity < 1:
                continue

            submit_paper_order(
                symbol,
                quantity,
                "buy",
                reference_price,
                "SMA20 above SMA50 and RSI entry criteria met",
            )

        except Exception as exc:
            log_event(
                f"Scan failed for {symbol}: {exc}",
                level="ERROR",
                symbol=symbol,
            )

    update_status(
        "Scan completed; paper mode only.",
        scan_completed=True,
    )


# ==========================================================
# STARTUP
# ==========================================================

def main():
    log_event(
        "AutoTrader AI worker starting in PAPER mode."
    )

    account = trading.get_account()

    if getattr(account, "trading_blocked", False):
        raise RuntimeError(
            "Alpaca reports that trading is blocked."
        )

    log_event(
        "Connected to Alpaca paper trading."
    )

    while True:
        try:
            scan_once()

        except Exception as exc:
            log_event(
                f"Worker cycle failed: {exc}",
                level="ERROR",
            )

            update_status(
                f"Worker error: {str(exc)[:400]}"
            )

        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
