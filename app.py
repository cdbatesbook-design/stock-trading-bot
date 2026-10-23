import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date
from io import StringIO

st.set_page_config(
    page_title="AutoTrader AI",
    page_icon="📈",
    layout="wide"
)

# ---------------- CONFIGURATION ----------------

STARTING_CASH = 10000.00
MAX_POSITION = 1000.00
MAX_POSITIONS = 5
STOP_LOSS = 0.03
TAKE_PROFIT = 0.06
BATCH_SIZE = 40

st.title("📈 AutoTrader AI")
st.caption("Automated stock scanning and simulated trading — no real money")

# ---------------- SESSION STATE ----------------

defaults = {
    "cash": STARTING_CASH,
    "positions": {},
    "trades": [],
    "watchlist": ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"],
    "auto_mode": False,
    "symbols": [],
    "scan_index": 0,
    "last_scan": "Not scanned yet",
    "scan_results": pd.DataFrame(),
    "processed_signals": set(),
    "initial_cash": STARTING_CASH,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------- STOCK UNIVERSE ----------------

@st.cache_data(ttl=86400, show_spinner=False)
def get_stock_universe():
    """Fetch listed U.S. stock symbols from Nasdaq Trader."""

    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]

    symbols = set()

    for url in urls:
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()

            df = pd.read_csv(StringIO(response.text), sep="|")

            if "Symbol" not in df.columns:
                continue

            for _, row in df.iterrows():
                symbol = str(row.get("Symbol", "")).strip()

                if not symbol or symbol.startswith("File Creation"):
                    continue

                if symbol in ("Symbol", "Test Issue"):
                    continue

                test_issue = str(row.get("Test Issue", "N")).upper()
                if test_issue == "Y":
                    continue

                # Skip symbols that often fail basic price-data queries.
                if any(char in symbol for char in ["$", "^", "/", " "]):
                    continue

                # Yahoo Finance commonly uses dashes instead of dots.
                symbol = symbol.replace(".", "-")
                symbols.add(symbol)

        except Exception:
            continue

    return sorted(symbols)


# ---------------- TECHNICAL ANALYSIS ----------------

def calculate_indicators(data, fast_period, slow_period):
    if data is None or data.empty or "Close" not in data:
        return None

    close = data["Close"].dropna()

    if len(close) < slow_period + 5:
        return None

    result = pd.DataFrame(index=close.index)
    result["Close"] = close
    result["Fast_MA"] = close.rolling(fast_period).mean()
    result["Slow_MA"] = close.rolling(slow_period).mean()

    change = close.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result["RSI"] = 100 - (100 / (1 + rs))
    result["RSI"] = result["RSI"].fillna(50)

    result = result.dropna()

    if result.empty:
        return None

    return result


def determine_signal(indicators):
    if indicators is None or len(indicators) < 2:
        return "HOLD"

    previous = indicators.iloc[-2]
    current = indicators.iloc[-1]

    bullish_cross = (
        previous["Fast_MA"] <= previous["Slow_MA"]
        and current["Fast_MA"] > current["Slow_MA"]
    )

    bearish_cross = (
        previous["Fast_MA"] >= previous["Slow_MA"]
        and current["Fast_MA"] < current["Slow_MA"]
    )

    if bullish_cross and current["RSI"] < 70:
        return "BUY"

    if bearish_cross:
        return "SELL"

    return "HOLD"


# ---------------- PORTFOLIO ----------------

def get_current_price(symbol):
    try:
        data = yf.download(
            symbol,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )

        if data is None or data.empty:
            return None

        close = data["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        price = float(close.dropna().iloc[-1])

        if price <= 0 or not np.isfinite(price):
            return None

        return price

    except Exception:
        return None


def portfolio_value():
    total = float(st.session_state.cash)

    for symbol, position in st.session_state.positions.items():
        price = get_current_price(symbol)

        if price is None:
            price = position["entry_price"]

        total += position["shares"] * price

    return total


def record_trade(symbol, action, shares, price, reason):
    st.session_state.trades.append({
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": symbol,
        "Action": action,
        "Shares": round(float(shares), 6),
        "Price": round(float(price), 2),
        "Value": round(float(shares * price), 2),
        "Reason": reason,
    })


def buy_stock(symbol, price, reason="Manual purchase"):
    symbol = symbol.strip().upper()

    if not symbol or price is None or price <= 0:
        return False, "Invalid symbol or price."

    if symbol in st.session_state.positions:
        return False, "You already own this stock."

    if len(st.session_state.positions) >= MAX_POSITIONS:
        return False, "Maximum open positions reached."

    amount = min(MAX_POSITION, st.session_state.cash)

    if amount < price:
        return False, "Not enough simulated cash to buy one share."

    shares = int(amount // price)

    if shares < 1:
        return False, "Not enough simulated cash to buy one whole share."

    cost = shares * price

    st.session_state.cash -= cost
    st.session_state.positions[symbol] = {
        "shares": shares,
        "entry_price": float(price),
        "entry_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    record_trade(symbol, "BUY", shares, price, reason)

    return True, f"Bought {shares} shares of {symbol} at ${price:.2f}."


def sell_stock(symbol, price, reason="Manual sale"):
    symbol = symbol.strip().upper()

    if symbol not in st.session_state.positions:
        return False, "You do not own this stock."

    if price is None or price <= 0:
        return False, "Invalid stock price."

    position = st.session_state.positions.pop(symbol)
    shares = position["shares"]
    proceeds = shares * price

    st.session_state.cash += proceeds

    record_trade(symbol, "SELL", shares, price, reason)

    profit = (price - position["entry_price"]) * shares

    return True, (
        f"Sold {shares} shares of {symbol} at ${price:.2f}. "
        f"Trade profit/loss: ${profit:.2f}."
    )


# ---------------- AUTOMATED TRADING ----------------

def execute_signal(symbol, signal, price, indicators):
    if price is None or indicators is None:
        return

    signal_date = str(indicators.index[-1].date())
    signal_key = f"{symbol}-{signal}-{signal_date}"

    if signal_key in st.session_state.processed_signals:
        return

    st.session_state.processed_signals.add(signal_key)

    if symbol in st.session_state.positions:
        position = st.session_state.positions[symbol]
        entry = position["entry_price"]

        change = (price - entry) / entry

        if change <= -STOP_LOSS:
            sell_stock(symbol, price, "Automatic 3% stop-loss")
            return

        if change >= TAKE_PROFIT:
            sell_stock(symbol, price, "Automatic 6% take-profit")
            return

        if signal == "SELL":
            sell_stock(symbol, price, "Moving-average sell signal")
            return

    elif signal == "BUY":
        buy_stock(symbol, price, "Automatic moving-average buy signal")


def scan_symbols(symbols, fast_period, slow_period):
    if not symbols:
        return pd.DataFrame()

    rows = []

    try:
        downloaded = yf.download(
            tickers=" ".join(symbols),
            period="6mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

    except Exception as exc:
        st.warning(f"Market data download failed: {exc}")
        return pd.DataFrame()

    for symbol in symbols:
        try:
            if len(symbols) == 1:
                data = downloaded
            else:
                if symbol not in downloaded.columns.get_level_values(0):
                    continue
                data = downloaded[symbol]

            if data is None or data.empty:
                continue

            if "Close" not in data.columns:
                continue

            close = data["Close"].dropna()

            if len(close) < slow_period + 5:
                continue

            indicators = calculate_indicators(
                data, fast_period, slow_period
            )

            if indicators is None or indicators.empty:
                continue

            price = float(indicators["Close"].iloc[-1])
            signal = determine_signal(indicators)

            if st.session_state.auto_mode:
                execute_signal(symbol, signal, price, indicators)

            rows.append({
                "Symbol": symbol,
                "Price": round(price, 2),
                "RSI": round(float(indicators["RSI"].iloc[-1]), 1),
                "Fast MA": round(float(indicators["Fast_MA"].iloc[-1]), 2),
                "Slow MA": round(float(indicators["Slow_MA"].iloc[-1]), 2),
                "Signal": signal,
            })

        except Exception:
            continue

    if not rows:
        return pd.DataFrame(
            columns=["Symbol", "Price", "RSI", "Fast MA", "Slow MA", "Signal"]
        )

    return pd.DataFrame(rows).sort_values(
        by=["Signal", "Symbol"]
    ).reset_index(drop=True)


# ---------------- SETTINGS ----------------

st.sidebar.header("⚙️ Trading Settings")

fast_period = st.sidebar.slider(
    "Fast moving average",
    min_value=3,
    max_value=30,
    value=10,
)

slow_period = st.sidebar.slider(
    "Slow moving average",
    min_value=20,
    max_value=100,
    value=50,
)

if fast_period >= slow_period:
    st.sidebar.error("Fast moving average must be less than slow moving average.")

st.sidebar.write(f"Starting paper cash: ${STARTING_CASH:,.2f}")
st.sidebar.write(f"Maximum per position: ${MAX_POSITION:,.2f}")
st.sidebar.write(f"Maximum open positions: {MAX_POSITIONS}")
st.sidebar.write(f"Stop-loss: {STOP_LOSS:.0%}")
st.sidebar.write(f"Take-profit: {TAKE_PROFIT:.0%}")

st.sidebar.warning(
    "Paper trading only. This app does not place real orders. "
    "Hosted sessions can sleep, and trades may not persist after a reset."
)
# ---------------- DASHBOARD ----------------

st.header("💰 Paper Trading Dashboard")

open_positions = st.session_state.positions
cash = st.session_state.cash

market_value = 0.0
position_details = []

for symbol, position in open_positions.items():
    price = get_current_price(symbol)

    if price is None:
        price = position["entry_price"]

    shares = position["shares"]
    entry = position["entry_price"]
    value = shares * price
    pnl = (price - entry) * shares

    market_value += value

    position_details.append({
        "Symbol": symbol,
        "Shares": shares,
        "Entry Price": round(entry, 2),
        "Current Price": round(price, 2),
        "Market Value": round(value, 2),
        "Unrealized P/L": round(pnl, 2),
        "Return %": round(((price - entry) / entry) * 100, 2),
    })

total_value = cash + market_value
total_profit = total_value - STARTING_CASH
total_return = (total_profit / STARTING_CASH) * 100

c1, c2, c3, c4 = st.columns(4)

c1.metric("Available Cash", f"${cash:,.2f}")
c2.metric("Positions Value", f"${market_value:,.2f}")
c3.metric(
    "Total Portfolio",
    f"${total_value:,.2f}",
    f"{total_profit:+,.2f}",
)
c4.metric("Total Return", f"{total_return:+.2f}%")

st.progress(
    min(len(open_positions) / MAX_POSITIONS, 1.0),
    text=f"Open positions: {len(open_positions)} / {MAX_POSITIONS}",
)

st.divider()


# ---------------- AUTOMATION CONTROLS ----------------

st.header("🤖 Automated Paper Trading")

auto_enabled = st.toggle(
    "Enable automatic paper trading",
    value=st.session_state.auto_mode,
    help=(
        "When enabled, the app can simulate buys and sells "
        "during stock scans. It never submits real brokerage orders."
    ),
)

st.session_state.auto_mode = auto_enabled

if auto_enabled:
    st.success("Automatic paper trading is enabled for scans performed in this session.")
else:
    st.info("Automatic paper trading is off. You can still scan and trade manually.")

st.warning(
    "Important: this app does not run continuously in the background. "
    "The scan only runs when you trigger it or the app reruns. "
    "Streamlit sessions may reset, so this is not a dependable unattended trading service."
)


# ---------------- MARKET SCANNER ----------------

st.header("🔎 Stock Market Scanner")

scan_mode = st.radio(
    "Choose scanning range",
    ["Next batch of stocks", "My watchlist"],
    horizontal=True,
)

scan_button = st.button(
    "🔄 Scan Stocks Now",
    type="primary",
    use_container_width=True,
)

if scan_button:
    with st.spinner("Loading stock symbols and market data..."):
        if scan_mode == "My watchlist":
            symbols_to_scan = list(
                dict.fromkeys(
                    s.strip().upper()
                    for s in st.session_state.watchlist
                    if s.strip()
                )
            )
        else:
            universe = get_stock_universe()

            if universe:
                st.session_state.symbols = universe
                total_symbols = len(universe)

                start = st.session_state.scan_index
                end = start + BATCH_SIZE

                symbols_to_scan = universe[start:end]

                if not symbols_to_scan:
                    st.session_state.scan_index = 0
                    symbols_to_scan = universe[:BATCH_SIZE]
                    end = BATCH_SIZE

                st.session_state.scan_index = end

                if st.session_state.scan_index >= total_symbols:
                    st.session_state.scan_index = 0

                st.caption(
                    f"Scanning {len(symbols_to_scan)} symbols from a universe "
                    f"of approximately {total_symbols:,} symbols."
                )
            else:
                st.warning(
                    "Could not download the stock-symbol list. "
                    "Try the watchlist scan instead."
                )
                symbols_to_scan = []

        if symbols_to_scan:
            results = scan_symbols(
                symbols_to_scan,
                fast_period,
                slow_period,
            )

            st.session_state.scan_results = results
            st.session_state.last_scan = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

if st.session_state.last_scan != "Not scanned yet":
    st.caption(f"Last scan: {st.session_state.last_scan}")

if not st.session_state.scan_results.empty:
    results = st.session_state.scan_results

    buy_signals = results[results["Signal"] == "BUY"]
    sell_signals = results[results["Signal"] == "SELL"]

    a, b, c = st.columns(3)
    a.metric("Stocks Scanned", len(results))
    b.metric("Buy Signals", len(buy_signals))
    c.metric("Sell Signals", len(sell_signals))

    signal_filter = st.selectbox(
        "Filter scan results",
        ["All", "BUY", "SELL", "HOLD"],
    )

    display_results = results.copy()

    if signal_filter != "All":
        display_results = display_results[
            display_results["Signal"] == signal_filter
        ]

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download scan results CSV",
        data=csv_data,
        file_name="autotrader_scan_results.csv",
        mime="text/csv",
    )
else:
    st.info("Run a scan to see stock prices and trading signals.")


# ---------------- WATCHLIST ----------------

st.divider()
st.header("⭐ My Watchlist")

watchlist_text = st.text_input(
    "Enter stock symbols separated by commas",
    value=", ".join(st.session_state.watchlist),
    help="Example: AAPL, MSFT, NVDA, TSLA, AMZN",
)

if st.button("Save Watchlist"):
    symbols = [
        symbol.strip().upper()
        for symbol in watchlist_text.split(",")
        if symbol.strip()
    ]

    # Remove duplicates while preserving order.
    st.session_state.watchlist = list(dict.fromkeys(symbols))

    st.success(
        f"Saved {len(st.session_state.watchlist)} watchlist symbols."
    )


# ---------------- MANUAL TRADING ----------------

st.divider()
st.header("🛒 Manual Paper Trading")

buy_tab, sell_tab = st.tabs(["Buy Stock", "Sell Stock"])

with buy_tab:
    st.subheader("Buy a stock")

    buy_symbol = st.text_input(
        "Stock ticker to buy",
        value="AAPL",
        key="manual_buy_symbol",
    ).strip().upper()

    if st.button("Get Buy Price"):
        price = get_current_price(buy_symbol)

        if price is None:
            st.error("Could not retrieve a price for that ticker.")
        else:
            st.session_state["manual_buy_price"] = price
            st.success(f"{buy_symbol}: ${price:.2f}")

    buy_price = st.session_state.get("manual_buy_price")

    if buy_price is not None:
        st.write(f"Latest retrieved price: **${buy_price:.2f}**")

        if st.button("Confirm Paper Buy"):
            success, message = buy_stock(
                buy_symbol,
                buy_price,
                "Manual paper purchase",
            )

            if success:
                st.success(message)
                st.session_state["manual_buy_price"] = None
                st.rerun()
            else:
                st.error(message)

with sell_tab:
    st.subheader("Sell a stock you own")

    if open_positions:
        sell_symbol = st.selectbox(
            "Choose position",
            list(open_positions.keys()),
        )

        if st.button("Get Sell Price"):
            price = get_current_price(sell_symbol)

            if price is None:
                st.error("Could not retrieve a price.")
            else:
                st.session_state["manual_sell_price"] = price
                st.session_state["manual_sell_symbol"] = sell_symbol
                st.success(f"{sell_symbol}: ${price:.2f}")

        sell_price = st.session_state.get("manual_sell_price")
        saved_sell_symbol = st.session_state.get("manual_sell_symbol")

        if sell_price is not None and saved_sell_symbol in open_positions:
            st.write(
                f"Retrieved price for {saved_sell_symbol}: "
                f"**${sell_price:.2f}**"
            )

            if st.button("Confirm Paper Sell"):
                success, message = sell_stock(
                    saved_sell_symbol,
                    sell_price,
                    "Manual paper sale",
                )

                if success:
                    st.success(message)
                    st.session_state["manual_sell_price"] = None
                    st.session_state["manual_sell_symbol"] = None
                    st.rerun()
                else:
                    st.error(message)
    else:
        st.info("You have no open positions to sell.")


# ---------------- OPEN POSITIONS ----------------

st.divider()
st.header("📊 Open Positions")

if position_details:
    positions_df = pd.DataFrame(position_details)

    st.dataframe(
        positions_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download positions CSV",
        data=positions_df.to_csv(index=False).encode("utf-8"),
        file_name="autotrader_positions.csv",
        mime="text/csv",
    )
else:
    st.info("No open positions yet.")


# ---------------- TRADE HISTORY ----------------

st.divider()
st.header("📜 Trade History")

if st.session_state.trades:
    history_df = pd.DataFrame(st.session_state.trades)

    st.dataframe(
        history_df.iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download trade history CSV",
        data=history_df.to_csv(index=False).encode("utf-8"),
        file_name="autotrader_trade_history.csv",
        mime="text/csv",
    )

    if st.button("Clear Trade History"):
        st.session_state.trades = []
        st.rerun()
else:
    st.info("No simulated trades have been recorded.")


# ---------------- BACKTEST ----------------

st.divider()
st.header("🧪 Simple Historical Backtest")

st.write(
    "This backtest simulates a moving-average crossover strategy on one stock. "
    "It is historical analysis only and does not predict future results."
)

backtest_symbol = st.text_input(
    "Ticker for backtest",
    value="AAPL",
    key="backtest_symbol",
).strip().upper()

backtest_period = st.selectbox(
    "Historical period",
    ["1y", "2y", "5y"],
    index=0,
)

if st.button("Run Backtest"):
    with st.spinner("Running historical simulation..."):
        try:
            history = yf.download(
                backtest_symbol,
                period=backtest_period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if history is None or history.empty:
                st.error("No historical data was returned.")
            else:
                close = history["Close"]

                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]

                close = close.dropna()

                if len(close) < slow_period + 5:
                    st.error("Not enough historical data for this strategy.")
                else:
                    fast_ma = close.rolling(fast_period).mean()
                    slow_ma = close.rolling(slow_period).mean()

                    # Long-only strategy, fully invested while signal is positive.
                    signal = (fast_ma > slow_ma).astype(int)
                    daily_returns = close.pct_change().fillna(0)

                    # Shift signal to avoid using today's closing price
                    # to claim a trade happened before that price was known.
                    strategy_returns = signal.shift(1).fillna(0) * daily_returns

                    equity = (1 + strategy_returns).cumprod()
                    buy_hold = (1 + daily_returns).cumprod()

                    strategy_return = (equity.iloc[-1] - 1) * 100
                    buy_hold_return = (buy_hold.iloc[-1] - 1) * 100

                    running_peak = equity.cummax()
                    drawdown = equity / running_peak - 1
                    max_drawdown = drawdown.min() * 100

                    x, y, z = st.columns(3)
                    x.metric(
                        "Strategy Return",
                        f"{strategy_return:+.2f}%",
                    )
                    y.metric(
                        "Buy & Hold Return",
                        f"{buy_hold_return:+.2f}%",
                    )
                    z.metric(
                        "Strategy Max Drawdown",
                        f"{max_drawdown:.2f}%",
                    )

                    chart_df = pd.DataFrame({
                        "Strategy": equity,
                        "Buy and Hold": buy_hold,
                    })

                    st.line_chart(chart_df)

                    st.caption(
                        "Backtest excludes taxes, slippage, transaction costs, "
                        "and real-world execution delays. Past performance "
                        "does not guarantee future results."
                    )

        except Exception as exc:
            st.error(f"Backtest failed: {exc}")


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "AutoTrader AI • Paper trading prototype • "
    "Market data provided by Yahoo Finance via yfinance."
)

st.caption(
    "This is educational software, not financial advice. "
    "Signals can be wrong or delayed. No profits are guaranteed."
)
