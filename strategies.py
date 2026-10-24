"""
Multi-Theory Trading Engine
Independent day-trading and swing-trading strategies.

This module generates research signals only.
It does not place trades.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd


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


def normalize_bars(frame):
    """Validate and standardize OHLCV market data."""

    columns = ["Open", "High", "Low", "Close", "Volume"]

    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)

    df = frame.copy()
    df.columns = [str(c).title() for c in df.columns]

    if not all(c in df.columns for c in columns):
        raise ValueError("Market data must contain Open, High, Low, Close, Volume")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Market data requires a DatetimeIndex")

    df = df[columns].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["Close"] > 0]
    df["Volume"] = df["Volume"].fillna(0).clip(lower=0)

    return df


def calculate_rsi(close, period=14):
    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    relative_strength = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + relative_strength))

    return rsi.replace([np.inf, -np.inf], np.nan)


def calculate_atr(df, period=14):
    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()


def calculate_vwap(df):
    typical_price = (
        df["High"] + df["Low"] + df["Close"]
    ) / 3

    volume = df["Volume"].replace(0, np.nan)

    return (
        (typical_price * volume).cumsum()
        / volume.cumsum()
    )


def make_buy_signal(
    strategy_id,
    horizon,
    df,
    reason,
    score=0.5,
    atr_multiple=2.0,
):
    """Create a candidate signal with estimated stop and target."""

    if df.empty:
        return hold_signal(strategy_id, horizon)

    price = float(df["Close"].iloc[-1])

    atr = calculate_atr(df).iloc[-1]

    if pd.isna(atr) or atr <= 0:
        atr = price * 0.02

    stop_price = max(0.01, price - atr_multiple * float(atr))

    risk_per_share = price - stop_price

    target_price = price + (2.0 * risk_per_share)

    return Signal(
        strategy_id=strategy_id,
        horizon=horizon,
        action="BUY",
        score=float(np.clip(score, 0, 1)),
        reason=reason,
        price=price,
        stop_price=stop_price,
        target_price=target_price,
        timestamp=df.index[-1].isoformat(),
    )


def hold_signal(strategy_id, horizon, reason="No qualifying setup"):
    return Signal(
        strategy_id=strategy_id,
        horizon=horizon,
        action="HOLD",
        score=0.0,
        reason=reason,
    )


def current_session(df):
    """Return bars belonging to the latest date in the data."""

    if df.empty:
        return df

    latest_date = df.index[-1].date()

    return df[df.index.date == latest_date]


# ============================================================
# DAY-TRADING STRATEGY 1: OPENING-RANGE BREAKOUT
# ============================================================

def day_opening_range(frame, minutes=5):
    strategy_id = f"DAY_ORB_{minutes}M"

    df = normalize_bars(frame)

    if len(df) < max(minutes + 2, 20):
        return hold_signal(
            strategy_id,
            "day",
            "Not enough intraday bars",
        )

    today = current_session(df)

    if len(today) <= minutes:
        return hold_signal(
            strategy_id,
            "day",
            "Opening range is not complete",
        )

    opening_range = today.iloc[:minutes]
    later_bars = today.iloc[minutes:]

    opening_high = float(opening_range["High"].max())

    latest = later_bars.iloc[-1]

    previous_volumes = df["Volume"].iloc[-21:-1]

    baseline_volume = (
        float(previous_volumes.median())
        if len(previous_volumes) >= 10
        else 0.0
    )

    volume_confirmed = (
        baseline_volume <= 0
        or float(latest["Volume"]) >= 1.2 * baseline_volume
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
                f"Price broke above the {minutes}-bar opening range; "
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
# DAY-TRADING STRATEGY 2: INTRADAY MOMENTUM
# ============================================================

def day_intraday_momentum(frame):
    strategy_id = "DAY_INTRADAY_MOMENTUM"

    df = normalize_bars(frame)
    df = current_session(df)

    if len(df) < 30:
        return hold_signal(
            strategy_id,
            "day",
            "Not enough completed intraday bars",
        )

    close = df["Close"]
    volume = df["Volume"]

    vwap = calculate_vwap(df)

    five_bar_return = (
        close.iloc[-1] / close.iloc[-6]
    ) - 1

    baseline_volume = volume.iloc[-21:-1].median()

    if baseline_volume > 0:
        volume_ratio = float(
            volume.iloc[-1] / baseline_volume
        )
    else:
        volume_ratio = 0.0

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
# DAY-TRADING STRATEGY 3: EMA TREND FOLLOWING
# ============================================================

def day_ema_trend(frame):
    strategy_id = "DAY_EMA_TREND"

    df = normalize_bars(frame)
    df = current_session(df)

    if len(df) < 30:
        return hold_signal(
            strategy_id,
            "day",
            "Not enough completed intraday bars",
        )

    close = df["Close"]

    ema_9 = close.ewm(
        span=9,
        adjust=False
    ).mean()

    ema_21 = close.ewm(
        span=21,
        adjust=False
    ).mean()

    vwap = calculate_vwap(df)

    if (
        ema_9.iloc[-1] > ema_21.iloc[-1]
        and close.iloc[-1] > vwap.iloc[-1]
        and close.iloc[-1] > close.iloc[-2]
    ):
        return make_buy_signal(
            strategy_id,
            "day",
            df,
            "EMA 9 above EMA 21, price above VWAP, price rising",
            score=0.60,
            atr_multiple=1.5,
        )

    return hold_signal(
        strategy_id,
        "day",
        "Trend confirmation missing",
    )


# ============================================================
# DAY-TRADING STRATEGY 4: VWAP MEAN REVERSION
# ============================================================

def day_vwap_reversion(frame):
    strategy_id = "DAY_VWAP_REVERSION"

    df = normalize_bars(frame)
    df = current_session(df)

    if len(df) < 30:
        return hold_signal(
            strategy_id,
            "day",
            "Not enough completed intraday bars",
        )

    close = df["Close"]

    vwap = calculate_vwap(df)

    deviation = close - vwap

    deviation_std = deviation.rolling(20).std()

    rsi = calculate_rsi(close, 14)

    previous_std = deviation_std.iloc[-2]

    if pd.notna(previous_std) and previous_std > 0:
        below_lower_band = (
            close.iloc[-2]
            < vwap.iloc[-2] - 1.5 * previous_std
        )
    else:
        below_lower_band = False

    recovering = close.iloc[-1] > close.iloc[-2]

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
            "Intraday price deviation is recovering from an oversold level",
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

def swing_momentum_20(frame, benchmark=None):
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
        close.iloc[-1] / close.iloc[-21] - 1
    )

    relative_strength = return_20d

    if benchmark is not None and not benchmark.empty:
        benchmark_df = normalize_bars(benchmark)

        if len(benchmark_df) >= 21:
            benchmark_return = float(
                benchmark_df["Close"].iloc[-1]
                / benchmark_df["Close"].iloc[-21]
                - 1
            )

            relative_strength = return_20d - benchmark_return

    if return_20d >= 0.03 and relative_strength > 0:
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

    sma_20 = close.rolling(20).mean()
    sma_55 = close.rolling(55).mean()

    if (
        close.iloc[-1] > sma_55.iloc[-1]
        and sma_20.iloc[-1] > sma_55.iloc[-1]
    ):
        return make_buy_signal(
            strategy_id,
            "swing",
            df,
            "Price and 20-day average are above the 55-day average",
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

    sma_50 = close.rolling(50).mean()
    rsi = calculate_rsi(close, 14)

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
            "Price above 50-day average; RSI recovering from a pullback",
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

    rsi = calculate_rsi(close, 14)
    sma_200 = close.rolling(200).mean()

    if (
        close.iloc[-1] > sma_200.iloc[-1]
        and pd.notna(rsi.iloc[-1])
        and rsi.iloc[-1] < 30
    ):
        return make_buy_signal(
            strategy_id,
            "swing",
            df,
            "Oversold RSI while price remains above the 200-day average",
            score=0.52,
            atr_multiple=1.75,
        )

    return hold_signal(
        strategy_id,
        "swing",
        "Oversold and long-term trend conditions not both met",
    )


# ============================================================
# SWING STRATEGY 5: 20-DAY BREAKOUT
# ============================================================

def swing_breakout_20(frame):
    strategy_id = "SWING_BREAKOUT_20D"

    df = normalize_bars(frame)

    if len(df) < 22:
        return hold_signal(
            strategy_id,
            "swing",
            "Not enough daily bars",
        )

    previous_high = float(
        df["High"].iloc[-21:-1].max()
    )

    latest = df.iloc[-1]

    previous_volumes = df["Volume"].iloc[-21:-1]

    baseline_volume = (
        float(previous_volumes.median())
        if len(previous_volumes)
        else 0.0
    )

    volume_confirmed = (
        baseline_volume <= 0
        or float(latest["Volume"]) >= 1.2 * baseline_volume
    )

    if (
        float(latest["Close"]) > previous_high
        and volume_confirmed
    ):
        return make_buy_signal(
            strategy_id,
            "swing",
            df,
            (
                "Price broke above the previous 20-day high; "
                f"volume confirmation={volume_confirmed}"
            ),
            score=0.64,
            atr_multiple=2.0,
        )

    return hold_signal(
        strategy_id,
        "swing",
        "No confirmed 20-day breakout",
    )


# ============================================================
# STRATEGY REGISTRY
# ============================================================

def evaluate_all(intraday, daily, benchmark_daily=None):
    """
    Evaluate each theory independently.

    Returns one Signal per strategy, including HOLD signals.
    """

    results = [
        day_opening_range(intraday, 5),
        day_opening_range(intraday, 15),
        day_opening_range(intraday, 30),
        day_intraday_momentum(intraday),
        day_ema_trend(intraday),
        day_vwap_reversion(intraday),
        swing_momentum_20(daily, benchmark_daily),
        swing_trend_55(daily),
        swing_pullback(daily),
        swing_mean_reversion(daily),
        swing_breakout_20(daily),
    ]

    return results
