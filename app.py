import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(
    page_title="AutoTrader AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AutoTrader AI")
st.caption("Automated Stock Analysis & Paper Trading")

st.warning(
    "Paper trading only. Signals are estimates, not guarantees "
    "of future performance. No real trades are placed."
)

st.sidebar.header("Trading Settings")

ticker = st.sidebar.text_input(
    "Stock Symbol",
    value="AAPL"
).upper().strip()

period = st.sidebar.selectbox(
    "Historical Data",
    ["1mo", "3mo", "6mo", "1y", "2y"],
    index=2
)

fast_period = st.sidebar.slider(
    "Fast Moving Average",
    5, 50, 10
)

slow_period = st.sidebar.slider(
    "Slow Moving Average",
    20, 200, 50
)

starting_cash = st.sidebar.number_input(
    "Paper Trading Starting Cash ($)",
    min_value=100.0,
    value=10000.0,
    step=500.0
)

if fast_period >= slow_period:
    st.error("Fast moving average must be shorter than slow moving average.")
    st.stop()

if st.button("Analyze Stock", type="primary"):
    st.session_state["analyze"] = True

if st.session_state.get("analyze", False):

    with st.spinner("Downloading market data..."):
        try:
            data = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if data.empty:
                st.error("No market data found. Check the stock symbol.")
                st.stop()

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data.dropna(subset=["Close"]).copy()

            if len(data) < slow_period:
                st.error(
                    "Not enough price history for these moving averages. "
                    "Choose a longer historical period."
                )
                st.stop()

            data["Fast MA"] = data["Close"].rolling(
                fast_period
            ).mean()

            data["Slow MA"] = data["Close"].rolling(
                slow_period
            ).mean()

            data["RSI"] = None

            delta = data["Close"].diff()
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
            data["RSI"] = 100 - (100 / (1 + rs))

            data["Signal"] = 0

            data.loc[
                data["Fast MA"] > data["Slow MA"],
                "Signal"
            ] = 1

            data.loc[
                data["Fast MA"] < data["Slow MA"],
                "Signal"
            ] = -1

            data["Position Change"] = data["Signal"].diff()

            latest = data.iloc[-1]
            previous = data.iloc[-2]

            price = float(latest["Close"])
            rsi = latest["RSI"]
            fast_ma = float(latest["Fast MA"])
            slow_ma = float(latest["Slow MA"])

            st.subheader(f"{ticker} Analysis")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Latest Available Close",
                f"${price:,.2f}"
            )

            c2.metric(
                f"{fast_period}-Day Average",
                f"${fast_ma:,.2f}"
            )

            c3.metric(
                f"{slow_period}-Day Average",
                f"${slow_ma:,.2f}"
            )

            c4.metric(
                "14-Day RSI",
                f"{float(rsi):.1f}" if pd.notna(rsi) else "N/A"
            )

            if latest["Signal"] == 1:
                st.success(
                    "Trend Signal: BULLISH — fast moving average "
                    "is above the slow moving average."
                )
            elif latest["Signal"] == -1:
                st.error(
                    "Trend Signal: BEARISH — fast moving average "
                    "is below the slow moving average."
                )
            else:
                st.info("No clear moving-average trend.")

            st.caption(
                "This
