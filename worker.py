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

#============================================================

CONFIGURATION

#============================================================

SAFETY: Alpaca paper trading is enforced below.

The bot also requires enabled=True in Supabase before

it can evaluate or submit orders.

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

Broker order statuses that mean the broker considers

an order terminal. Partial fills must still be accounted for.

BROKER_TERMINAL_STATUSES = {
"filled",
"canceled",
"cancelled",
"expired",
"rejected",
}

Application statuses that no longer require reconciliation.

FINAL_ORDER_STATUSES = {
"synced",
"canceled",
"cancelled",
"expired",
"rejected",
}

Trade records included in the managed-position ledger.

FILL_ORDER_STATUSES = {
"filled",
"partial_filled",
"partially_filled",
"synced",
}

#============================================================

LOGGING

#============================================================

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s %(levelname)s %(message)s",
)

log = logging.getLogger("stock_trading_bot")

#============================================================

ENVIRONMENT AND CONNECTIONS

#============================================================

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

SAFETY: paper=True is hardcoded.

Do not change this to live trading.

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

#============================================================

GENERAL HELPERS

#============================================================

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

#============================================================

MARKET DATA AND STRATEGY

#============================================================

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

#============================================================

TRADE AND POSITION RECORDS

#============================================================

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

#============================================================

BROKER ACCOUNT AND POSITIONS

#============================================================

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

#============================================================

ORDER RECONCILIATION

#============================================================

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
            side=row
