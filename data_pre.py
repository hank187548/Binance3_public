"""Prepare market data, engineer features, and generate triple-barrier labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta


def fetch_historical_data(symbol: str = "BTC/USDT", timeframe: str = "4h", days_to_fetch: int = 5000) -> pd.DataFrame:
    """Fetch historical OHLCV candles from Binance."""
    print(f"Step 1/4: fetching {days_to_fetch} days of {symbol} {timeframe} data...")
    exchange = ccxt.binance()
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=days_to_fetch)).isoformat())
    all_ohlcv: list[list[float]] = []

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
        except Exception as exc:  # noqa: BLE001
            print(f"Data fetch error: {exc}")
            break

    df = pd.DataFrame(all_ohlcv, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    df.set_index("datetime", inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    print(f"Fetched {len(df)} rows.")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators and drop warm-up NaN rows."""
    print("Step 2/4: calculating technical indicators...")
    df.ta.macd(close="close", fast=12, slow=26, signal=9, append=True)
    df.ta.obv(close="close", volume="volume", append=True)
    df.ta.rsi(close="close", length=14, append=True)
    df.ta.bbands(close="close", length=20, append=True)
    df.ta.atr(high="high", low="low", close="close", length=14, append=True)
    df.dropna(inplace=True)
    print("Feature calculation complete.")
    return df


def apply_triple_barrier(df: pd.DataFrame, profit_take_pct: float, stop_loss_pct: float, time_limit: int) -> pd.DataFrame:
    """Generate labels with triple-barrier logic.

    Label 2: take-profit hit first
    Label 0: stop-loss hit first
    Label 1: neither barrier hit before time_limit
    """
    print("Step 3/4: generating triple-barrier labels...")
    labels = pd.Series(np.nan, index=df.index)

    for i in range(len(df) - time_limit):
        entry_price = df["close"].iloc[i]
        upper_barrier = entry_price * (1 + profit_take_pct)
        lower_barrier = entry_price * (1 - stop_loss_pct)

        for j in range(1, time_limit + 1):
            future_price = df["close"].iloc[i + j]
            if future_price >= upper_barrier:
                labels.iloc[i] = 2
                break
            if future_price <= lower_barrier:
                labels.iloc[i] = 0
                break
            if j == time_limit:
                labels.iloc[i] = 1
                break

    df["label"] = labels
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)
    print("Label generation complete.")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data preprocessing pipeline for BTC triple-barrier labels")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--days", type=int, default=5000)
    parser.add_argument("--profit-take", type=float, default=0.04)
    parser.add_argument("--stop-loss", type=float, default=0.02)
    parser.add_argument("--time-barrier", type=int, default=12)
    parser.add_argument("--output", default="btc_labeled_data.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Starting data preprocessing")
    print("=" * 60)

    raw_df = fetch_historical_data(args.symbol, args.timeframe, args.days)
    featured_df = add_features(raw_df.copy())
    labeled_df = apply_triple_barrier(
        featured_df.copy(),
        profit_take_pct=args.profit_take,
        stop_loss_pct=args.stop_loss,
        time_limit=args.time_barrier,
    )

    print("Step 4/4: validating and saving data...")
    print("\nFinal dataset overview")
    print(f"Shape: {labeled_df.shape}")
    print("\nLabel distribution (0=stop-loss, 1=timeout, 2=take-profit):")
    print(labeled_df["label"].value_counts(normalize=True))

    labeled_df.to_csv(args.output, index=True)
    print(f"\nSaved dataset to '{args.output}'")
    print("=" * 60)
    print("Data preprocessing completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
