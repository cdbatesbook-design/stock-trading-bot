"""
Multi-Theory Trading Engine
===========================

11 independent day-trading and swing-trading strategies.

DAY TRADING:
1. 5-minute opening-range breakout
2. 15-minute opening-range breakout
3. 30-minute opening-range breakout
4. Intraday momentum
5. EMA trend following
6. VWAP mean reversion

SWING TRADING:
7. 20-day momentum
8. 55-day trend following
9. Trend pullback
10. Mean reversion with 200-day trend context
11. 20-day breakout

This module generates research signals only.
It never submits orders.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd


EASTERN_TIMEZONE = "America/New_York"


# ============================================================
# SIGNAL STRUCTURE
# ============================================================

@dataclass(frozen=True)
class Signal:
    strategy_id: str
    horizon: str
    action: str = "HOLD"
    score: float = 0.0
    reason: str = "No qualifying setup"
    price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    timestamp: Optional[str] = None

    def to_dict(self):
        return asdict(self)


# ============================================================
# MARKET DATA NORMALIZATION
# ============================================================

def normalize_bars(frame):
    """
    Validate and standardize OHLCV data.

    Expected columns:
    Open, High, Low, Close, Volume

    Requires a DatetimeIndex.
    """

    columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)

    df = frame.copy()

    df.columns = [
        str(column).title()
        for column in df.columns
    ]

    if not all(
        column in df.columns
        for column in columns
    ):
        raise ValueError(
            "Market data must contain Open, High, Low, Close, Volume."
        )

    if not isinstance(
        df.index,
        pd.DatetimeIndex,
    ):
        raise ValueError(
            "Market data requires a DatetimeIndex."
        )

    # Interpret naive timestamps as UTC to match the worker's data convention.
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df.index = df.index.tz_convert(
        EASTERN_TIMEZONE
    )

    df = df[columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    df = df[
        ~df.index.duplicated(keep="last")
    ].sort_index()

    df = df[
        (df["Open"] > 0)
        & (df["High"] > 0)
        & (df["Low"] > 0)
        & (df["Close"] > 0)
    ]

    df["Volume"] = (
        df["Volume"]
        .fillna(0)
        .clip(lower=0)
    )

    # Reject malformed candles.
    valid_ohlc = (
        (df["High"] >= df["Low"])
        & (df["High"] >= df["Open"])
        & (df["High"] >= df["Close"])
        & (df["Low"] <= df["Open"])
        & (df["Low"] <= df["Close"])
    )

    df = df[valid_ohlc]

    return df


def current_session(df):
    """
    Return bars belonging to the latest date in the data.

    Timestamps are normalized to Eastern time first.
    """

    if df is None or df.empty:
        return df

    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")

    df = df.copy()

    df.index = df.index.tz_convert(
        EASTERN_TIMEZONE
    )

    latest_date = df.index[-1].date()

    return df[
        df.index.date == latest_date
    ]


def completed_intraday_bars(frame):
    """
    Remove the latest intraday bar because it may still be forming.

    This is a conservative safeguard. Historical feeds can still have
    delays or gaps, so timestamps must also be checked by the caller.
    """

    df = normalize_bars(frame)

    if len(df) < 2:
        return df.iloc[0:0]

    return df.iloc[:-1].copy()


def hold_signal(
    strategy_id,
    horizon,
    reason="No qualifying setup",
):
    return Signal(
        strategy_id=strategy_id,
        horizon=horizon,
        action="HOLD",
        score=0.0,
        reason=reason,
    )


# ============================================================
# INDICATORS
# ============================================================

def calculate_rsi(close, period=14):
    """Calculate the Relative Strength Index."""

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

    relative_strength = (
        avg_gain
        / avg_loss.replace(0, np.nan)
    )

    rsi = 100 - (
        100 / (1 + relative_strength)
    )

    # If there are gains but no losses, RSI is 100.
    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100.0,
    )

    # If neither gains nor losses exist, use neutral RSI.
    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain == 0),
        50.0,
    )

    return rsi.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def calculate_atr(df, period=14):
    """Calculate Average True Range."""

    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (
                df["High"] - previous_close
            ).abs(),
            (
                df["Low"] - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()


def calculate_vwap(df):
    """Calculate cumulative session VWAP."""

    if df is None or df.empty:
        return pd.Series(
            dtype=float,
            index=getattr(df, "index", None),
        )

    typical_price = (
        df["High"]
        + df["Low"]
        + df["Close"]
    ) / 3

    volume = df["Volume"].clip(lower=0)

    cumulative_volume = volume.cumsum()

    cumulative_value = (
        typical_price * volume
    ).cumsum()

    vwap = (
        cumulative_value
        / cumulative_volume.replace(0, np.nan)
    )

    return vwap


# ============================================================
# SIGNAL CONSTRUCTION
# ============================================================

def make_buy_signal(
    strategy_id,
    horizon,
    df,
    reason,
    score=0.5,
    atr_multiple=2.0,
):
    """
    Create a hypothetical long-entry signal.

    Stop and target are estimates, not guaranteed execution prices.
    """

    if df is None or df.empty:
        return hold_signal(
            strategy_id,
            horizon,
            "No valid market data",
        )

    price = float(
        df["Close"].iloc[-1]
    )

    if not np.isfinite(price) or price <= 0:
        return hold_signal(
            strategy_id,
            horizon,
            "Invalid market price",
        )

    atr_series = calculate_atr(df)

    atr = (
        atr_series.iloc[-1]
        if not atr_series.empty
        else np.nan
    )

    if not np.isfinite(atr) or atr <= 0:
        atr = price * 0.02

    stop_price = max(
        0.01,
        price - atr_multiple * float(atr),
    )

    if stop_price >= price:
        return hold_signal(
            strategy_id,
            horizon,
            "Could not calculate a valid stop price",
        )

    risk_per_share = price - stop_price

    target_price = (
        price + 2.0 * risk_per_share
    )

    return Signal(
        strategy_id=strategy_id,
        horizon=horizon,
        action="BUY",
        score=float(
            np.clip(score, 0.0, 1.0)
        ),
        reason=reason,
        price=price,
        stop_price=stop_price,
        target_price=target_price,
        timestamp=df.index[-1].isoformat(),
    )


# ============================================================
# DAY STRATEGY 1: OPENING-RANGE BREAKOUT
# ============================================================

def day_opening_range(
    frame,
    minutes=5,
):
    """
    Evaluate a 5-, 15-, or 30-minute opening-range breakout.

    Opening-range boundaries are based on elapsed clock time,
    not simply the number of rows received.
    """

    strategy_id = (
        f"DAY_ORB_{minutes}M"
    )

    if minutes not in (5, 15, 30):
        return hold_signal(
            strategy_id,
            "day",
            "Unsupported opening-range duration",
        )

    df = normalize_bars(frame)

    if len(df) < 2:
        return hold_signal(
            strategy_id,
            "day",
            "Not enough intraday data",
        )

    today = current_session(df)

    if today.empty:
        return hold_signal(
            strategy_id,
            "day",
            "No bars for the latest session",
        )

    market_open = today.index[-1].replace(
        hour=9,
        minute=30,
        second=0,
        microsecond=0,
    )

    opening_end = (
        market_open
        + pd.Timedelta(minutes=minutes)
    )

    opening_range = today[
        (today.index >= market_open)
        & (today.index < opening_end)
    ]

    later_bars = today[
        today.index >= opening_end
    ]

    if opening_range.empty or later_bars.empty:
        return hold_signal(
            strategy_id,
            "day",
            "Opening range is incomplete",
        )

    # Require multiple observations where expected.
    expected_observations = max(
        1,
        minutes // 2,
    )

    if len(opening_range) < expected_observations:
        return hold_signal(
            strategy_id,
            "day",
            "Too few opening-range observations",
        )

    opening_high = float(
        opening_range["High"].max()
    )

    latest = later_bars.iloc[-1]

    # Compare breakout volume with recent completed bars.
    previous_volumes = today["Volume"].iloc[-21:-1]

    baseline_volume = (
        float(previous_volumes.median())
        if len(previous_volumes) >= 10
        else 0.0
    )

    volume_confirmed = (
        baseline_volume <= 0
        or float(latest["Volume"])
        >= 1.2 * baseline_volume
    )

    if (
        float(latest["Close"]) > opening_high
        and volume_confirmed
    ):
        return make_buy_signal(
            strategy_id,
            "day",
            today,
            (
                f"Price broke above the "
                f"{minutes}-minute opening range; "
                f"volume confirmation={volume_confirmed}"
            ),
            score=0.65,
            atr_multiple=1.5,
        )

    return hold_signal(
        strategy_id,
        "day",
        "No confirmed opening-range breakout",
    )


# ============================================================
# DAY STRATEGY 2: INTRADAY MOMENTUM
# ============================================================

def day_intraday_momentum(frame):
    strategy_id = "DAY_INTRADAY_MOMENTUM"

    df = normalize_bars(frame)
    df = current_session(df)

    if len(df) < 30:
        return hold_signal(
            strategy_id,
            "day",
            "Not enough intraday bars",
        )

    close = df["Close"]
    volume = df["Volume"]

    vwap = calculate_vwap(df)

    if vwap.empty or pd.isna(vwap.iloc[-1]):
        return hold_signal(
            strategy_id,
            "day",
            "VWAP unavailable",
        )

    five_bar_return = (
        close.iloc[-1]
        / close.iloc[-6]
    ) - 1

    baseline_volume = (
        volume.iloc[-21:-1].median()
    )

    volume_ratio = (
        float(
            volume.iloc[-1]
            / baseline_volume
        )
        if baseline_volume > 0
        else 0.0
    )

    if (
        five_bar_return >= 0.004
        and close.iloc[-1] > vwap.iloc[-1]
        and volume_ratio >= 1.3
    ):
        return make_buy_signal(
            strategy_id,
            "day",
            df,
            (
                f"Five-bar return={five_bar_return:.2%}; "
                f"relative volume={volume_ratio:.2f}x"
            ),
            score=0.65,
            atr_multiple=1.5,
        )

    return hold_signal(
        strategy_id,
        "day",
        "Momentum, volume, or VWAP conditions not met",
    )


# ============================================================
# DAY STRATEGY 3: EMA TREND FOLLOWING
# ============================================================

def day_ema_trend(frame):
    strategy_id = "DAY_EMA_TREND"

    df = normalize_bars(frame)
    df = current_session(df)

    if len(df) < 30:
        return hold_signal(
            strategy_id,
            "day",
            "Not enough intraday bars",
        )

    close = df["Close"]

    ema_9 = close.ewm(
        span=9,
        adjust=False,
    ).mean()

    ema_21 = close.ewm(
        span=21,
        adjust=False,
    ).mean()

    vwap = calculate_vwap(df)

    if vwap.empty or pd.isna(vwap.iloc[-1]):
        return hold_signal(
            strategy_id,
            "day",
            "VWAP unavailable",
        )

    if (
        ema_9.iloc[-1] > ema_21.iloc[-1]
        and close.iloc[-1] > vwap.iloc[-1]
        and close.iloc[-1] > close.iloc[-2]
    ):
        return make_buy_signal(
            strategy_id,
            "day",
            df,
            (
                "EMA 9 above EMA 21; price above VWAP "
                "and rising"
            ),
            score=0.60,
            atr_multiple=1.5,
        )

    return hold_signal(
        strategy_id,
        "day",
        "Trend confirmation missing",
    )


# ============================================================
# DAY STRATEGY 4: VWAP MEAN REVERSION
# ============================================================

def day_vwap_reversion(frame):
    strategy_id = "DAY_VWAP_REVERSION"

    df = normalize_bars(frame)
    df = current_session(df)

    if len(df) < 30:
        return hold_signal(
            strategy_id,
            "day",
            "Not enough intraday bars",
        )

    close = df["Close"]
    vwap = calculate_vwap(df)

    deviation = close - vwap

    deviation_std = deviation.rolling(
        20
    ).std()

    rsi = calculate_rsi(
        close,
        14,
    )

    previous_std = deviation_std.iloc[-2]

    if (
        pd.notna(previous_std)
        and previous_std > 0
        and pd.notna(vwap.iloc[-2])
    ):
        below_lower_band = (
            close.iloc[-2]
            < vwap.iloc[-2]
            - 1.5 * previous_std
        )
    else:
        below_lower_band = False

    recovering = (
        close.iloc[-1]
        > close.iloc[-2]
    )

    if (
        below_lower_band
        and recovering
        and pd.notna(rsi.iloc[-1])
        and rsi.iloc[-1] < 40
    ):
        return make_buy_signal(
            strategy_id,
            "day",
            df,
            (
                "Price is recovering from an intraday "
                "oversold deviation"
            ),
            score=0.52,
            atr_multiple=1.25,
        )

    return hold_signal(
        strategy_id,
        "day",
        "No confirmed VWAP-reversion setup",
    )


# ============================================================
# SWING STRATEGY 1: 20-DAY MOMENTUM
# ============================================================

def swing_momentum_20(
    frame,
    benchmark=None,
):
    strategy_id = "SWING_MOMENTUM_20D"

    df = normalize_bars(frame)

    if len(df) < 25:
        return hold_signal(
            strategy_id,
            "swing",
            "Not enough daily bars",
        )

    close = df["Close"]

    return_20d = float(
        close.iloc[-1]
        / close.iloc[-21]
        - 1
    )

    relative_strength = return_20d

    if benchmark is not None and not benchmark.empty:
        benchmark_df = normalize_bars(
            benchmark
        )

        if len(benchmark_df) >= 21:
            benchmark_return = float(
                benchmark_df["Close"].iloc[-1]
                / benchmark_df["Close"].iloc[-21]
                - 1
            )

            relative_strength = (
                return_20d - benchmark_return
            )

    if (
        return_20d >= 0.03
        and relative_strength > 0
    ):
        return make_buy_signal(
            strategy_id,
            "swing",
            df,
            (
                f"20-day return={return_20d:.2%}; "
                f"relative strength={relative_strength:.2%}"
            ),
            score=0.62,
            atr_multiple=2.0,
        )

    return hold_signal(
        strategy_id,
        "swing",
        "Momentum or relative-strength condition missing",
    )


# ============================================================
# SWING STRATEGY 2: 55-DAY TREND FOLLOWING
# ============================================================

def swing_trend_55(frame):
    strategy_id = "SWING_TREND_55D"

    df = normalize_bars(frame)

    if len(df) < 60:
        return hold_signal(
            strategy_id,
            "swing",
            "Not enough daily bars",
        )

    close = df["Close"]

    sma_20 = close.rolling(
        20
    ).mean()

    sma_55 = close.rolling(
        55
    ).mean()

    if (
        close.iloc[-1] > sma_55.iloc[-1]
        and sma_20.iloc[-1] > sma_55.iloc[-1]
    ):
        return make_buy_signal(
            strategy_id,
            "swing",
            df,
            (
                "Price and 20-day average are above "
                "the 55-day average"
            ),
            score=0.62,
            atr_multiple=2.0,
        )

    return hold_signal(
        strategy_id,
        "swing",
        "Long-term trend filter not met",
    )


# ============================================================
# SWING STRATEGY 3: TREND PULLBACK
# ============================================================

def swing_pullback(frame):
    strategy_id = "SWING_TREND_PULLBACK"

    df = normalize_bars(frame)

    if len(df) < 55:
        return hold_signal(
            strategy_id,
            "swing",
            "Not enough daily bars",
        )

    close = df["Close"]

    sma_50 = close.rolling(
        50
    ).mean()

    rsi = calculate_rsi(
        close,
        14,
    )

    rsi_recovered = (
        pd.notna(rsi.iloc[-2])
        and pd.notna(rsi.iloc[-1])
        and rsi.iloc[-2] < 40
        and rsi.iloc[-1] >= 40
    )

    if (
        close.iloc[-1] > sma_50.iloc[-1]
        and rsi_recovered
    ):
        return make_buy_signal(
            strategy_id,
            "swing",
            df,
            (
                "Price above the 50-day average; "
                "RSI recovering from a pullback"
            ),
            score=0.56,
            atr_multiple=2.0,
        )

    return hold_signal(
        strategy_id,
        "swing",
        "No confirmed trend-pullback recovery",
    )


# ============================================================
# SWING STRATEGY 4: MEAN REVERSION
# ============================================================

def swing_mean_reversion(frame):
    strategy_id = "SWING_MEAN_REVERSION"

    df = normalize_bars(frame)

    if len(df) < 205:
        return hold_signal(
            strategy_id,
            "swing",
            "Requires 200-day trend context",
        )

    close = df["Close"]

    rsi = calculate_rsi(
        close,
        14,
    )

    sma_200 = close.rolling(
        200
    ).mean()

    if (
        close.iloc[-1] > sma_200.iloc[-1]
        and pd.notna(rsi.iloc[-1])
        and rsi.iloc[-1] < 30
    ):
        return make_buy_signal(
            strategy_id,
            "swin
