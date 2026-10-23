import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AutoTrader AI",
    page_icon="📈",
    layout="wide"
)

APP_VERSION = "2.0"

st.title("📈 AutoTrader AI")
st.caption(
    f"Stock Analysis & Paper Trading | Version {APP_VERSION}"
)

st.warning(
    "SIMULATION ONLY. No real trades are placed. "
    "Signals and historical results do not guarantee profits."
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "cash": 10000.0,
    "starting_cash": 10000.0,
    "positions": {},
    "trade_history": [],
    "watchlist": ["AAPL", "MSFT", "NVDA", "TSLA"],
    "last_ticker": "AAPL",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Trading Settings")

ticker = st.sidebar.text_input(
    "Stock Symbol",
    value=st.session_state.last_ticker
).upper().strip()

period = st.sidebar.selectbox(
    "Historical Data",
    ["3mo", "6mo", "1y", "2y", "5y"],
    index=2
)

fast_period = st.sidebar.slider(
    "Fast Moving Average (days)",
    min_value=5,
    max_value=50,
    value=10
)

slow_period = st.sidebar.slider(
    "Slow Moving Average (days)",
    min_value=20,
    max_value=200,
    value=50
)

trade_amount = st.sidebar.number_input(
    "Paper Trade Amount ($)",
    min_value=10.0,
    max_value=100000.0,
    value=500.0,
    step=50.0
)

if fast_period >= slow_period:
    st.error(
        "The fast moving average must be shorter "
        "than the slow moving average."
    )
    st.stop()

st.sidebar.caption(
    "Stock prices may be delayed. Trades use the latest "
    "available daily closing price, not a live execution price."
)

if st.sidebar.button("Reset Paper Portfolio"):
    st.session_state.cash = 10000.0
    st.session_state.starting_cash = 10000.0
    st.session_state.positions = {}
    st.session_state.trade_history = []

    st.rerun()


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl=300)
def load_stock_data(symbol, history_period):
    """
    Download daily historical data.

    Cached for five minutes to reduce repeated requests.
    """

    if not symbol:
        return pd.DataFrame()

    result = yf.download(
        symbol,
        period=history_period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if result is None or result.empty:
        return pd.DataFrame()

    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)

    if "Close" not in result.columns:
        return pd.DataFrame()

    result = result.dropna(
        subset=["Close"]
    ).copy()

    result = result[
        ~result.index.duplicated(keep="last")
    ]

    return result


def add_indicators(data, fast, slow):
    """
    Calculate moving averages, RSI, and crossover signals.
    """

    result = data.copy()

    result["Fast MA"] = result["Close"].rolling(
        window=fast,
        min_periods=fast
    ).mean()

    result["Slow MA"] = result["Close"].rolling(
        window=slow,
        min_periods=slow
    ).mean()

    delta = result["Close"].diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    result["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # Handle a period with gains but no losses.
    result.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "RSI"
    ] = 100.0

    # Handle a period with neither gains nor losses.
    result.loc[
        (avg_loss == 0) & (avg_gain == 0),
        "RSI"
    ] = 50.0

    previous_fast = result["Fast MA"].shift(1)
    previous_slow = result["Slow MA"].shift(1)

    bullish_cross = (
        (result["Fast MA"] > result["Slow MA"])
        & (previous_fast <= previous_slow)
        & (result["RSI"] < 70)
    )

    bearish_cross = (
        (result["Fast MA"] < result["Slow MA"])
        & (previous_fast >= previous_slow)
    )

    result["Signal"] = "HOLD"

    result.loc[
        bullish_cross.fillna(False),
        "Signal"
    ] = "BUY"

    result.loc[
        bearish_cross.fillna(False),
        "Signal"
    ] = "SELL"

    return result


# ============================================================
# PORTFOLIO PRICE REFRESH
# ============================================================

def get_latest_price(symbol):
    """
    Return the latest available daily closing price.

    None is returned when data is unavailable.
    """

    try:
        recent = load_stock_data(symbol, "5d")

        if recent.empty:
            return None

        value = float(recent["Close"].iloc[-1])

        if not np.isfinite(value) or value <= 0:
            return None

        return value

    except Exception:
        return None


# Refresh prices for existing positions.

for symbol, position in list(
    st.session_state.positions.items()
):
    fresh_price = get_latest_price(symbol)

    if fresh_price is not None:
        position["last_price"] = fresh_price


# ============================================================
# PORTFOLIO SUMMARY
# ============================================================

st.subheader("💰 Paper Trading Portfolio")

holdings_value = sum(
    position["shares"] * position["last_price"]
    for position in st.session_state.positions.values()
)

total_equity = (
    st.session_state.cash + holdings_value
)

profit_loss = (
    total_equity - st.session_state.starting_cash
)

profit_percent = (
    (profit_loss / st.session_state.starting_cash) * 100
    if st.session_state.starting_cash > 0
    else 0.0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Available Cash",
    f"${st.session_state.cash:,.2f}"
)

col2.metric(
    "Holdings Value",
    f"${holdings_value:,.2f}"
)

col3.metric(
    "Total Portfolio Value",
    f"${total_equity:,.2f}"
)

col4.metric(
    "Total Profit / Loss",
    f"${profit_loss:,.2f}",
    delta=f"{profit_percent:.2f}%"
)


# ============================================================
# STOCK ANALYSIS
# ============================================================

st.divider()
st.subheader("🔎 Stock Analysis")

analyze_clicked = st.button(
    "Analyze Stock",
    type="primary"
)

if analyze_clicked:
    if not ticker:
        st.error("Enter a valid stock symbol.")
    else:
        st.session_state.last_ticker = ticker

selected_ticker = st.session_state.last_ticker

try:
    with st.spinner(
        f"Loading market data for {selected_ticker}..."
    ):
        raw_data = load_stock_data(
            selected_ticker,
            period
        )

    if raw_data.empty:
        st.error(
            f"No market data found for {selected_ticker}. "
            "Check the symbol or try again later."
        )

    else:
        data = add_indicators(
            raw_data,
            fast_period,
            slow_period
        )

        valid_data = data.dropna(
            subset=["Fast MA", "Slow MA"]
        )

        if valid_data.empty:
            st.error(
                "There is not enough price history for "
                "the selected moving averages. Choose "
                "a longer historical period."
            )

        else:
            latest = valid_data.iloc[-1]

            latest_price = float(latest["Close"])
            fast_ma = float(latest["Fast MA"])
            slow_ma = float(latest["Slow MA"])

            latest_rsi = latest["RSI"]

            # Determine the current trend separately
            # from the latest crossover event.

            if fast_ma > slow_ma:
                trend = "BULLISH"
            elif fast_ma < slow_ma:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"

            # Display a BUY or SELL only when a crossover
            # occurred on the latest available data row.

            signal = latest["Signal"]

            a, b, c, d = st.columns(4)

            a.metric(
                "Latest Available Close",
                f"${latest_price:,.2f}"
            )

            b.metric(
                f"{fast_period}-Day Average",
                f"${fast_ma:,.2f}"
            )

            c.metric(
                f"{slow_period}-Day Average",
                f"${slow_ma:,.2f}"
            )

            d.metric(
                "14-Day RSI",
                f"{latest_rsi:.1f}"
                if pd.notna(latest_rsi)
                else "N/A"
            )

            st.write(f"**Current Trend:** {trend}")

            if signal == "BUY":
                st.success(
                    "🟢 BUY SIGNAL: A bullish moving-average "
                    "crossover was detected in the latest "
                    "available data, with RSI below 70."
                )

            elif signal == "SELL":
                st.error(
                    "🔴 SELL SIGNAL: A bearish moving-average "
                    "crossover was detected in the latest "
                    "available data."
                )

            else:
                st.info(
                    "🟡 HOLD: No new moving-average crossover "
                    "was detected in the latest data."
                )

            st.caption(
                "A HOLD signal does not mean the stock is safe "
                "or that you should necessarily keep a position. "
                "Signals are based on historical daily data."
            )

            # ------------------------------------------------
            # PRICE CHART
            # ------------------------------------------------

            st.subheader("📈 Price and Moving Averages")

            chart_data = data[
                ["Close", "Fast MA", "Slow MA"]
            ].dropna()

            st.line_chart(chart_data)

            # ------------------------------------------------
            # RECENT SIGNALS
            # ------------------------------------------------

            st.subheader("Recent Market Data")

            display_data = data[
                [
                    "Close",
                    "Fast MA",
                    "Slow MA",
                    "RSI",
                    "Signal"
                ]
            ].tail(10).copy()

            display_data.index = (
                display_data.index.strftime("%Y-%m-%d")
            )

            st.dataframe(
                display_data,
                use_container_width=True
            )

            # ------------------------------------------------
            # PAPER TRADING
            # ------------------------------------------------

            st.divider()
            st.subheader("🛒 Simulated Trading")

            st.write(
                f"Selected stock: **{selected_ticker}**"
            )

            st.write(
                f"Reference price: **${latest_price:,.2f}**"
            )

            shares_to_buy = trade_amount / latest_price

            st.write(
                f"A ${trade_amount:,.2f} simulated purchase "
                f"would represent approximately "
                f"{shares_to_buy:.4f} shares."
            )

            buy_col, sell_col = st.columns(2)

            with buy_col:
                if st.button(
                    f"Paper Buy {selected_ticker}"
                ):
                    if trade_amount > st.session_state.cash:
                        st.error(
                            "Insufficient paper cash."
                        )

                    else:
                        position = (
                            st.session_state.positions.get(
                                selected_ticker,
                                {
                                    "shares": 0.0,
                                    "last_price": latest_price
                                }
                            )
                        )

                        position["shares"] += shares_to_buy
                        position["last_price"] = latest_price

                        st.session_state.positions[
                            selected_ticker
                        ] = position

                        st.session_state.cash -= trade_amount

                        st.session_state.trade_history.append({
                            "Date": str(
                                data.index[-1].date()
                            ),
                            "Symbol": selected_ticker,
                            "Action": "BUY",
                            "Shares": shares_to_buy,
                            "Price": latest_price,
                            "Amount": trade_amount
                        })

                        st.success(
                            "Simulated purchase recorded."
                        )

                        st.rerun()

            with sell_col:
                if st.button(
                    f"Paper Sell {selected_ticker}"
                ):
                    position = (
                        st.session_state.positions.get(
                            selected_ticker
                        )
                    )

                    if (
                        position is None
                        or position["shares"] <= 0
                    ):
                        st.error(
                            "You do not own this stock "
                            "in your paper portfolio."
                        )

                    else:
                        shares_to_sell = min(
                            position["shares"],
                            trade_amount / latest_price
                        )

                        proceeds = (
                            shares_to_sell * latest_price
                        )

                        position["shares"] -= shares_to_sell
                        position["last_price"] = latest_price

                        st.session_state.cash += proceeds

                        st.session_state.trade_history.append({
                            "Date": str(
                                data.index[-1].date()
                            ),
                            "Symbol": selected_ticker,
                            "Action": "SELL",
                            "Shares": shares_to_sell,
                            "Price": latest_price,
                            "Amount": proceeds
                        })

                        if position["shares"] < 0.000001:
                            del st.session_state.positions[
                                selected_ticker
                            ]

                        st.success(
                            "Simulated sale recorded."
                        )

                        st.rerun()

            # ------------------------------------------------
            # HISTORICAL BACKTEST
            # ------------------------------------------------

            st.divider()
            st.subheader("📊 Historical Strategy Test")

            backtest = valid_data.copy()

            # Hold the stock when the fast average is above
            # the slow average. Otherwise remain in cash.
            #
            # Shift the position by one day so that a signal
            # calculated at today's close is not applied to
            # today's already-completed return.

            backtest["Position"] = (
                backtest["Fast MA"] > backtest["Slow MA"]
            ).astype(int)

            backtest["Market Return"] = (
                backtest["Close"].pct_change().fillna(0)
            )

            backtest["Strategy Return"] = (
                backtest["Position"].shift(1).fillna(0)
                * backtest["Market Return"]
            )

            backtest["Strategy Equity"] = (
                1 + backtest["Strategy Return"]
            ).cumprod()

            backtest["Buy and Hold Equity"] = (
                1 + backtest["Market Return"]
            ).cumprod()

            strategy_return = (
                backtest["Strategy Equity"].iloc[-1] - 1
            ) * 100

            buy_hold_return = (
                backtest["Buy and Hold Equity"].iloc[-1] - 1
            ) * 100

            bt1, bt2 = st.columns(2)

            bt1.metric(
                "Historical Strategy Return",
                f"{strategy_return:.2f}%"
            )

            bt2.metric(
                "Buy-and-Hold Return",
                f"{buy_hold_return:.2f}%"
            )

            st.line_chart(
                backtest[
                    [
                        "Strategy Equity",
                        "Buy and Hold Equity"
                    ]
                ]
            )

            st.caption(
                "Illustrative backtest only. It excludes "
                "transaction costs, taxes, slippage, and other "
                "real-world costs. Historical performance does "
                "not predict future returns."
            )

except Exception as error:
    st.error(
        f"Analysis failed: {error}"
    )


# ============================================================
# WATCHLIST
# ============================================================

st.divider()
st.subheader("⭐ Watchlist")

new_symbol = st.text_input(
    "Add a stock symbol",
    key="new_watchlist_symbol"
).upper().strip()

if st.button("Add to Watchlist"):
    if not new_symbol:
        st.error("Enter a stock symbol first.")

    elif new_symbol in st.session_state.watchlist:
        st.info("That symbol is already in your watchlist.")

    else:
        st.session_state.watchlist.append(new_symbol)
        st.success(f"{new_symbol} added to your watchlist.")
        st.rerun()

watchlist_rows = []

with st.spinner("Refreshing watchlist prices..."):
    for symbol in st.session_state.watchlist:
        try:
            watch_data = load_stock_data(symbol, "5d")

            if watch_data.empty:
                continue

            close_prices = watch_data["Close"].dropna()

            if close_prices.empty:
                continue

            current_price = float(close_prices.iloc[-1])

            daily_change = 0.0

            if len(close_prices) >= 2:
                previous_price = float(
                    close_prices.iloc[-2]
                )

                if previous_price != 0:
                    daily_change = (
                        (current_price / previous_price) - 1
                    ) * 100

            watchlist_rows.append({
                "Symbol": symbol,
                "Latest Available Close": current_price,
                "Daily Change (%)": daily_change
            })

        except Exception:
            continue

if watchlist_rows:
    st.dataframe(
        pd.DataFrame(watchlist_rows),
        use_container_width=True
    )

else:
    st.info(
        "No watchlist prices are currently available."
    )

if st.session_state.watchlist:
    remove_symbol = st.selectbox(
        "Select a symbol to remove",
        ["None"] + st.session_state.watchlist
    )

    if st.button("Remove from Watchlist"):
        if remove_symbol != "None":
            st.session_state.watchlist.remove(
                remove_symbol
            )
            st.rerun()


# ============================================================
# CURRENT POSITIONS
# ============================================================

st.divider()
st.subheader("📦 Current Paper Positions")

if st.session_state.positions:
    position_rows = []

    for symbol, position in st.session_state.positions.items():
        shares = position["shares"]
        price = position["last_price"]

        position_rows.append({
            "Symbol": symbol,
            "Shares": round(shares, 6),
            "Latest Recorded Price": round(price, 2),
            "Market Value": round(shares * price, 2)
        })

    st.dataframe(
        pd.DataFrame(position_rows),
        use_container_width=True
    )

else:
    st.info(
        "You do not have any simulated stock positions yet."
    )


# ============================================================
# TRADE HISTORY
# ============================================================

st.divider()
st.subheader("📜 Paper Trade History")

if st.session_state.trade_history:
    history = pd.DataFrame(
        st.session_state.trade_history
    )

    st.dataframe(
        history,
        use_container_width=True
    )

    csv_data = history.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Trade History CSV",
        data=csv_data,
        file_name="autotrader_paper_trades.csv",
        mime="text/csv"
    )

else:
    st.info(
        "Your simulated transactions will appear here."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AutoTrader AI | Educational use | Paper trading only"
)
