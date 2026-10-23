"""
Paper-only automated stock trading worker.

Required environment variables:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY

IMPORTANT:
- Paper trading only.
- Run one worker instance at a time.
- Keep bot_status.enabled = false until testing is complete.
"""

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


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)

log = logging.getLogger("paper-trading-worker")


# ============================================================
# ENVIRONMENT AND CONNECTIONS
# ============================================================

def required_env(name: str) -> str:
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


# Hard-code paper=True so this worker cannot select live trading.
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

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_db(
    level: str,
    message: str,
    details: dict | None = None,
) -> None:
    """
    Save logs using columns confirmed in the database.

    bot_logs has no details column, so additional information
    is appended to message instead.
    """

    try:
        text_message = str(message)

        if details:
            context = json.dumps(
                details,
                default=str,
                separators=(",", ":"),
            )

            text_message += f" | context={context}"

        supabase.table("bot_logs").insert(
            {
                "level": str(level).upper()[:50],
                "message": text_message[:1000],
                "created_at": utc_now(),
            }
        ).execute()

    except Exception:
        log.exception("Could not write bot_logs record")


def set_status(message: str, **extra) -> None:
    """
    Update the single bot_status row.

    The database requires id=1 and mode='paper'.
    """

    row = {
        "id": 1,
        "last_heartbeat": utc_now(),
        "status_message": str(message)[:1000],
        "updated_at": utc_now(),
        **extra,
    }

    # Never allow a caller to change the mode to live.
    row["mode"] = "paper"

    try:
        supabase.table("bot_status").upsert(
            row,
            on_conflict="id",
        ).execute()

    except Exception:
        log.exception("Could not update bot_status")


def get_bot_config() -> dict:
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
            "bot_status row id=1 is missing"
        )

    return rows[0]


def fetch_rows(table: str) -> list[dict]:
    """
    Fetch table rows in pages rather than relying on one response.

    This avoids silently relying on a single default page size.
    """

    all_rows = []
    page_size = 500
    start = 0

    while True:
        response = (
            supabase.table(table)
            .select("*")
            .range(start, start + page_size - 1)
            .execute()
        )

        page = response.data or []
        all_rows.extend(page)

        if len(page) < page_size:
            break

        start += page_size

    return all_rows


def get_trade_rows() -> list[dict]:
    return fetch_rows("trades")


def get_position_rows() -> list[dict]:
    return fetch_rows("positions")


# ============================================================
# MARKET DATA
# ============================================================

def latest_price(symbol: str) -> float:
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


def get_signal(symbol: str) -> tuple[str, dict]:
    """
    Basic daily trend strategy.

    BUY when:
    - Latest completed close is above SMA20.
    - SMA20 is above SMA50.
    - RSI is between 45 and 68.

    This is a rule-based signal, not a guarantee of profitability.
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
            "reason": "daily close history unavailable"
        }

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    if "Close" not in frame.columns:
        return "HOLD", {
            "reason": "daily close history unavailable"
        }

    close = pd.to_numeric(
        frame["Close"],
        errors="coerce",
    ).dropna()

    if len(close) < 61:
        return "HOLD", {
            "reason": "insufficient daily history"
        }

    # Exclude the latest bar because it may be incomplete.
    close = close.iloc[:-1]

    if len(close) < 50:
        return "HOLD", {
            "reason": "insufficient completed daily bars"
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

    rsi.loc[normal] = 100 - (
        100 / (1 + rs)
    )

    values = [
        close.iloc[-1],
        sma20.iloc[-1],
        sma50.iloc[-1],
        rsi.iloc[-1],
    ]

    if any(pd.isna(value) for value in values):
        return "HOLD", {
            "reason": "indicator values unavailable"
        }

    last_close, last_sma20, last_sma50, last_rsi = (
        map(float, values)
    )

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
# ORDER AND TRADE HELPERS
# ============================================================

def order_status(order) -> str:
    status = getattr(order, "status", "")
    return str(
        getattr(status, "value", status)
    ).lower()


def write_trade(
    order_id: str,
    symbol: str,
    side: str,
    quantity: float,
    price: float | None,
    status: str,
    strategy: str,
    reason: str,
    total_value: float | None = None,
) -> None:
    """
    Insert or update a trade using the confirmed unique
    trades.order_id constraint.
    """

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

    existing = (
        supabase.table("trades")
        .select("id")
        .eq("order_id", order_id)
        .limit(1)
        .execute()
        .data
        or []
    )

    if not existing:
        row["created_at"] = utc_now()

    supabase.table("trades").upsert(
        row,
        on_conflict="order_id",
    ).execute()


# ============================================================
# POSITION TRACKING
# ============================================================

def upsert_position(
    symbol: str,
    quantity: float,
    entry_price: float,
    current_price: float,
    status: str,
) -> None:
    """
    Update the current open position by id or insert a new one.

    The confirmed database has no UNIQUE(symbol) constraint.
    Therefore this function does not use ON CONFLICT(symbol).

    Closed rows are retained as history.
    """

    now = utc_now()

    row = {
        "symbol": symbol,
        "quantity": float(quantity),
        "entry_price": float(entry_price),
        "current_price": float(current_price),
        "status": status,
        "updated_at": now,
    }

    if status == "open":
        row["closed_at"] = None

        existing_rows = (
            supabase.table("positions")
            .select("id,opened_at")
            .eq("symbol", symbol)
            .eq("status", "open")
            .order("id", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )

        if existing_rows:
            existing = existing_rows[0]

            row["opened_at"] = (
                existing.get("opened_at") or now
            )

            (
                supabase.table("positions")
                .update(row)
                .eq("id", existing["id"])
                .execute()
            )

        else:
            row["opened_at"] = now

            supabase.table("positions").insert(
                row
            ).execute()

    elif status == "closed":
        row["closed_at"] = now

        open_rows = (
            supabase.table("positions")
            .select("id")
            .eq("symbol", symbol)
            .eq("status", "open")
            .execute()
            .data
            or []
        )

        for existing in open_rows:
            (
                supabase.table("positions")
                .update(row)
                .eq("id", existing["id"])
                .execute()
            )

    else:
        raise ValueError(
            f"Unsupported position status: {status}"
        )


# ============================================================
# ORDER RECONCILIATION
# ============================================================

def reconcile_orders() -> None:
    """
    Check unresolved orders with Alpaca.

    Ambiguous API errors are not treated as proof that an order
    failed. Never blindly resubmit an uncertain order.
    """

    for row in get_trade_rows():
        order_id = str(row.get("order_id") or "")
        status_in_db = str(
            row.get("status") or ""
        ).lower()

        if (
            not order_id
            or status_in_db in FINAL_ORDER_STATUSES
        ):
            continue

        try:
            order = trading.get_order_by_client_id(
                order_id
            )

        except Exception as exc:
            log.warning(
                "Order %s still unresolved: %s",
                order_id,
                exc,
            )
            continue

        broker_status = order_status(order)

        filled_qty = float(
            getattr(order, "filled_qty", 0) or 0
        )

        avg = getattr(
            order,
            "filled_avg_price",
            None,
        )

        fill_price = (
            float(avg)
            if avg not in (None, "")
            else None
        )

        if filled_qty > 0:
            if broker_status == "filled":
                recorded_status = "filled"
            else:
                recorded_status = "partial_filled"

            recorded_qty = filled_qty

            recorded_price = (
                fill_price
                if fill_price is not None
                else row.get("price")
            )

        else:
            recorded_status = broker_status
            recorded_qty = float(
                row.get("quantity") or 0
            )
            recorded_price = row.get("price")

        try:
            write_trade(
                order_id=order_id,
                symbol=str(row["symbol"]),
                side=str(row["side"]),
                quantity=recorded_qty,
                price=recorded_price,
                status=recorded_status,
                strategy=str(
                    row.get("strategy") or "daily-trend"
                ),
                reason=str(row.get("reason") or ""),
                total_value=(
                    filled_qty * fill_price
                    if (
                        filled_qty > 0
                        and fill_price is not None
                    )
                    else row.get("total_value")
                ),
            )

        except Exception:
            log.exception(
                "Failed to save broker status for %s",
                order_id,
            )
            continue

        if filled_qty > 0:
            try:
                sync_position_from_broker(
                    str(row["symbol"])
                )

                if broker_status in {
                    "filled",
                    "canceled",
                    "cancelled",
                    "expired",
                    "rejected",
                }:
                    (
                        supabase.table("trades")
                        .update({"status": "synced"})
                        .eq("order_id", order_id)
                        .execute()
                    )

            except Exception:
                log.exception(
                    "Position sync failed for %s; "
                    "will retry next scan",
                    order_id,
                )


# ============================================================
# POSITION LEDGER AND ACCOUNTING
# ============================================================

def calculate_managed_position(
    trades: list[dict],
    symbol: str,
) -> tuple[float, float]:
    """
    Calculate bot-managed long quantity and average cost.

    Requires filled trades to have reliable timestamps and
    actual fill quantities/prices for accurate accounting.
    """

    relevant = [
        trade
        for trade in trades
        if (
            str(trade.get("symbol") or "") == symbol
            and str(
                trade.get("status") or ""
            ).lower() in FILL_ORDER_STATUSES
            and str(
                trade.get("side") or ""
            ).lower() in {"buy", "sell"}
        )
    ]

    relevant.sort(
        key=lambda trade: str(
            trade.get("created_at") or ""
        )
    )

    quantity = 0.0
    basis = 0.0

    for trade in relevant:
        side = str(
            trade.get("side") or ""
        ).lower()

        qty = float(
            trade.get("quantity") or 0
        )

        price = float(
            trade.get("price") or 0
        )

        if qty <= 0 or price <= 0:
            continue

        if side == "buy":
            if quantity <= 1e-8:
                quantity = 0.0
                basis = 0.0

            new_qty = quantity + qty

            basis = (
                (quantity * basis) + (qty * price)
            ) / new_qty

            quantity = new_qty

        elif side == "sell":
            sold = min(qty, quantity)
            quantity -= sold

            if quantity <= 1e-8:
                quantity = 0.0
                basis = 0.0

    return quantity, basis


def sync_position_from_broker(symbol: str) -> None:
    """
    Rebuild the bot-managed position from the trade ledger
    and compare it with Alpaca's paper-account holdings.
    """

    trades = []
    page_size = 500
    start = 0

    while True:
        page = (
            supabase.table("trades")
            .select("*")
            .eq("symbol", symbol)
            .range(start, start + page_size - 1)
            .execute()
            .data
            or []
        )

        trades.extend(page)

        if len(page) < page_size:
            break

        start += page_size

    managed_qty, entry_price = (
        calculate_managed_position(
            trades,
            symbol,
        )
    )

    if managed_qty <= 1e-8:
        upsert_position(
            symbol,
            0,
            0,
            0,
            "closed",
        )
        return

    broker_positions = trading.get_all_positions()

    broker_pos = next(
        (
            position
            for position in broker_positions
            if position.symbol == symbol
        ),
        None,
    )

    if broker_pos is None:
        raise RuntimeError(
            f"Trade ledger has {managed_qty} shares "
            f"of {symbol}, but Alpaca reports none"
        )

    broker_qty = float(broker_pos.qty)

    if broker_qty + 1e-8 < managed_qty:
        raise RuntimeError(
            f"Ledger has {managed_qty} shares of "
            f"{symbol}, Alpaca reports {broker_qty}"
        )

    upsert_position(
        symbol,
        managed_qty,
        entry_price,
        float(broker_pos.current_price),
        "open",
    )


def active_bot_positions() -> list[dict]:
    rows = get_position_rows()

    return [
        position
        for position in rows
        if (
            str(position.get("status") or "").lower()
            == "open"
            and float(
                position.get("quantity") or 0
            ) > 0
        )
    ]


def unresolved_status(status: str) -> bool:
    return status not in FINAL_ORDER_STATUSES


def has_unresolved_order(
    symbol: str | None = None,
) -> bool:
    for row in get_trade_rows():
        if not row.get("order_id"):
            continue

        if (
            symbol is not None
            and row.get("symbol") != symbol
        ):
            continue

        status = str(
            row.get("status") or ""
        ).lower()

        if unresolved_status(status):
            return True

    return False


# ============================================================
# ACCOUNT AND BUDGET HELPERS
# ============================================================

def account_equity() -> float:
    account = trading.get_account()
    equity = float(account.equity)

    if not math.isfinite(equity) or equity <= 0:
        raise RuntimeError(
            f"Invalid account equity: {equity}"
        )

    return equity


def broker_positions_by_symbol() -> dict:
    return {
        position.symbol: position
        for position in trading.get_all_positions()
    }


def committed_value(positions: list[dict]) -> float:
    total = 0.0

    for position in positions:
        qty = float(
            position.get("quantity") or 0
        )

        price = float(
            position.get("current_price")
            or position.get("entry_price")
            or 0
        )

        total += qty * price

    return total


# ============================================================
# ORDER SUBMISSION
# ============================================================

def place_market_order(
    symbol: str,
    side: OrderSide,
    qty: int,
    reason: str,
) -> str:
    if qty < 1:
        raise ValueError(
            "Order quantity must be at least one whole share"
        )

    if side not in {
        OrderSide.BUY,
        OrderSide.SELL,
    }:
        raise ValueError("Invalid order side")

    if has_unresolved_order(symbol):
        raise RuntimeError(
            f"An unresolved order already exists for {symbol}"
        )

    client_order_id = (
        f"bot-{symbol.lower()}-{side.value}-"
        f"{uuid.uuid4().hex[:20]}"
    )

    # Persist an intent before submitting.
    # If the API call has an ambiguous outcome, do not resubmit.
    write_trade(
        client_order_id,
        symbol,
        side.value,
        qty,
        None,
        "intent",
        "daily-trend",
        reason,
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
            "Submission outcome uncertain for %s",
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
            qty,
            None,
            order_status(order),
            "daily-trend",
            reason,
        )

    except Exception:
        log.exception(
            "Order submitted but local status update failed: %s",
            client_order_id,
        )

    return client_order_id


# ============================================================
# EXIT MANAGEMENT
# ============================================================

def manage_exits() -> None:
    for position in active_bot_positions():
        symbol = str(position["symbol"])

        qty_float = float(
            position.get("quantity") or 0
        )

        qty = int(
            math.floor(qty_float + 1e-8)
        )

        entry = float(
            position.get("entry_price") or 0
        )

        if (
            qty < 1
            or entry <= 0
            or has_unresolved_order(symbol)
        ):
            continue

        try:
            price = latest_price(symbol)
            pnl = (price - entry) / entry

            if pnl <= -STOP_LOSS_PERCENT:
                place_market_order(
                    symbol,
                    OrderSide.SELL,
                    qty,
                    (
                        "stop loss threshold; "
                        f"return={pnl:.2%}"
                    ),
                )

            elif pnl >= TAKE_PROFIT_PERCENT:
                place_market_order(
                    symbol,
                    OrderSide.SELL,
                    qty,
                    (
                        "take profit threshold; "
                        f"return={pnl:.2%}"
                    ),
                )

            else:
                (
                    supabase.table("positions")
                    .update(
                        {
                            "current_price": price,
                            "updated_at": utc_now(),
                        }
                    )
                    .eq("symbol", symbol)
                    .eq("status", "open")
                    .execute()
                )

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
# BUY VALIDATION
# ============================================================

def can_buy(
    symbol: str,
    price: float,
    positions: list[dict],
    broker_positions: dict,
) -> tuple[bool, str]:

    if not math.isfinite(price) or price <= 0:
        return False, "invalid price"

    if has_unresolved_order(symbol):
        return False, "unresolved order exists for symbol"

    if any(
        str(position.get("symbol")) == symbol
        for position in positions
    ):
        return False, "already have a bot-managed position"

    # Do not adopt or add to unknown broker holdings automatically.
    if symbol in broker_positions:
        return False, (
            "symbol already exists in Alpaca holdings; "
            "manual reconciliation required"
        )

    if len(positions) >= MAX_OPEN_POSITIONS:
        return False, "maximum open positions reached"

    remaining = max(
        0.0,
        VIRTUAL_BUDGET - committed_value(positions),
    )

    allowed_value = min(
        MAX_POSITION_VALUE,
        remaining,
    )

    if allowed_value < price:
        return False, "budget too small for one share"

    return True, f"allowed_value={allowed_value:.2f}"


# ============================================================
# MAIN SCAN
# ============================================================

def scan_once() -> None:
    config = get_bot_config()

    if config.get("enabled") is not True:
        set_status(
            "Paused: bot_status.enabled is not true",
            mode="paper",
        )

        log.info("Bot disabled in Supabase")
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
            "Refusing to trade unless paper mode is enabled"
        )

    reconcile_orders()

    try:
        clock = trading.get_clock()
        equity = account_equity()
        broker_positions = broker_positions_by_symbol()

    except Exception:
        set_status(
            "Safety stop: broker account, clock, "
            "or positions unavailable",
            mode="paper",
        )

        log.exception(
            "Broker checks failed; no new entries"
        )

        return

    if not clock.is_open:
        set_status(
            "Waiting: market is closed",
            mode="paper",
        )
        return

    manage_exits()

    # Wait while any order remains unresolved.
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
                "Skipping %s: broker already holds this symbol",
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

            qty = int(
                math.floor(max_value / price_before)
            )

            if qty < 1:
                continue

            # Check buying power again before submission.
            account = trading.get_account()
            buying_power = float(account.buying_power)

            if (
                not math.isfinite(buying_power)
                or buying_power < qty * price_before
            ):
                log.warning(
                    "Skipping %s: insufficient buying power",
                    symbol,
                )
                continue

            price_now = latest_price(symbol)

            if (
                price_now
                > price_before * (1 + MAX_PRICE_DRIFT_PERCENT)
            ):
                log.warning(
                    "Skipping %s: price moved more than tolerance",
                    symbol,
                )
                continue

            # Recalculate the share count at the latest price.
            qty = min(
                qty,
                int(math.floor(max_value / price_now)),
            )

            if (
                qty < 1
                or qty * price_now > MAX_POSITION_VALUE
            ):
                continue

            # Do not submit if another order became unresolved.
            if has_unresolved_order():
                break

            order_reason = (
                "daily trend signal; "
                f"reference_price={price_now:.4f}; "
                f"RSI={info.get('rsi', 0):.2f}; "
                f"account_equity={equity:.2f}"
            )

            place_market_order(
                symbol,
                OrderSide.BUY,
                qty,
                order_reason,
            )

            log.info(
                "Submitted paper BUY %s qty=%s "
                "estimated_value=%.2f",
                symbol,
                qty,
                qty * price_now,
            )

            # Submit no more than one new entry per scan.
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

def main() -> None:
    set_status(
        "Worker starting; paper mode enforced",
        mode="paper",
    )

    log.info(
        "Paper worker started; scan interval %ss",
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
                {"error": str(exc)},
            )

            set_status(
                f"Scan cycle error: {str(exc)[:800]}",
                mode="paper",
            )

        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
