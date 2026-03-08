"""Scheduled BTC prediction with a GRU+Attention model bundle."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from datetime import datetime, timedelta

import ccxt
import pandas as pd
import pandas_ta as ta
import pytz
import torch

try:
    from GRU_attention_modified import GRU_Attention_Model
except ImportError as exc:
    raise ImportError(
        "Failed to import GRU_Attention_Model. Ensure GRU_attention_modified.py is in the same directory."
    ) from exc

try:
    from notification import send_to_telegram
except ImportError:
    def send_to_telegram(message: str, image_path: str | None = None) -> bool:  # type: ignore[override]
        print("Telegram module not available. Printing message instead.")
        print(message)
        return False


DEFAULT_MODEL_DIR = os.getenv(
    "MODEL_DIR",
    "brute_force_results_hl_barrier/lr0.0005-hd32-layers2-drop0.5-wd0.01-bs32-ts36-cv3-ep100",
)


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    timeframe = timeframe.strip().lower()
    if timeframe.endswith("m"):
        return timedelta(minutes=int(timeframe[:-1]))
    if timeframe.endswith("h"):
        return timedelta(hours=int(timeframe[:-1]))
    if timeframe.endswith("d"):
        return timedelta(days=int(timeframe[:-1]))
    if timeframe.endswith("w"):
        return timedelta(weeks=int(timeframe[:-1]))
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def fetch_historical_data(symbol: str = "BTC/USDT", timeframe: str = "4h", days_to_fetch: int = 90) -> pd.DataFrame | None:
    print(f"Fetching {days_to_fetch} days of {symbol} {timeframe} data...")
    exchange = ccxt.binance()
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=days_to_fetch)).isoformat())
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
    except Exception as exc:  # noqa: BLE001
        print(f"Data fetch failed: {exc}")
        return None

    if not ohlcv:
        print("No data returned from exchange.")
        return None

    df = pd.DataFrame(ohlcv, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    df.set_index("datetime", inplace=True)
    df.index = df.index.tz_localize("UTC")
    df = df[~df.index.duplicated(keep="first")]
    print(f"Fetched {len(df)} rows.")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame | None:
    print("Calculating indicators (MACD, OBV, RSI, BBands, ATR)...")
    try:
        df.ta.macd(close="close", fast=12, slow=26, signal=9, append=True)
        df.ta.obv(close="close", volume="volume", append=True)
        df.ta.rsi(close="close", length=14, append=True)
        df.ta.bbands(close="close", length=20, append=True)
        df.ta.atr(high="high", low="low", close="close", length=14, append=True)
        df.dropna(inplace=True)
        print("Feature engineering complete.")
    except Exception as exc:  # noqa: BLE001
        print(f"Feature engineering failed: {exc}")
        return None
    return df


def load_model_bundle(model_dir: str, device: torch.device) -> dict[str, object]:
    model_path = os.path.join(model_dir, "best_model_final.pth")
    scaler_path = os.path.join(model_dir, "final_scaler.pkl")
    feature_cols_path = os.path.join(model_dir, "feature_columns.pkl")
    metadata_path = os.path.join(model_dir, "metadata.json")

    meta: dict[str, object] = {
        "timeframe": "4h",
        "label_horizon_bars": 12,
        "params": {
            "hidden_dim": 32,
            "num_layers": 1,
            "dropout_prob": 0.5,
            "time_steps": 60,
        },
    }

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            for key in ("timeframe", "label_horizon_bars"):
                if key in loaded:
                    meta[key] = loaded[key]
            if "params" in loaded and isinstance(loaded["params"], dict):
                params = meta["params"]
                if isinstance(params, dict):
                    params.update(loaded["params"])
            print(f"Loaded metadata from {metadata_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to parse metadata.json, using defaults: {exc}")
    else:
        print("metadata.json not found, using defaults.")

    with open(feature_cols_path, "rb") as handle:
        feature_columns: list[str] = pickle.load(handle)

    params = meta["params"]
    if not isinstance(params, dict):
        raise TypeError("metadata params is not a dictionary")

    model = GRU_Attention_Model(
        input_dim=len(feature_columns),
        hidden_dim=int(params["hidden_dim"]),
        num_layers=int(params["num_layers"]),
        output_dim=3,
        dropout_prob=float(params["dropout_prob"]),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(scaler_path, "rb") as handle:
        scaler = pickle.load(handle)

    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "time_steps": int(params["time_steps"]),
        "timeframe": str(meta["timeframe"]),
        "label_horizon_bars": int(meta["label_horizon_bars"]),
        "hyperparams": params,
    }


def build_prediction_message(
    label: str,
    probabilities: torch.Tensor,
    last_close_price: float,
    base_close_local: str,
    horizon_end_local: str,
    horizon_bars: int,
    horizon_delta: timedelta,
) -> str:
    return (
        "BTC/USDT Prediction (GRU+Attention)\n\n"
        f"Base candle close (Asia/Taipei): {base_close_local}\n"
        f"Close price: ${last_close_price:,.2f}\n"
        "-----------------------------------\n"
        f"Prediction window: next {horizon_bars} bars ({horizon_delta})\n"
        f"Predicted outcome: {label}\n"
        f"Window end (approx): {horizon_end_local}\n"
        "-----------------------------------\n"
        "Probability distribution:\n"
        f"- take-profit: {probabilities[2].item():.2%}\n"
        f"- timeout: {probabilities[1].item():.2%}\n"
        f"- stop-loss: {probabilities[0].item():.2%}"
    )


def predict_latest(model_dir: str, send_notification: bool = True) -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        bundle = load_model_bundle(model_dir, device)
    except FileNotFoundError as exc:
        print(f"Required file is missing: {exc.filename}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load model bundle: {exc}")
        return False

    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_columns = bundle["feature_columns"]
    time_steps = int(bundle["time_steps"])
    timeframe = str(bundle["timeframe"])
    label_horizon_bars = int(bundle["label_horizon_bars"])

    print(f"Config: timeframe={timeframe}, time_steps={time_steps}, horizon_bars={label_horizon_bars}")
    bar_delta = timeframe_to_timedelta(timeframe)

    btc_df = fetch_historical_data(timeframe=timeframe, days_to_fetch=90)
    if btc_df is None:
        return False

    btc_df_featured = add_features(btc_df.copy())
    if btc_df_featured is None:
        return False

    latest_data = btc_df_featured.tail(time_steps)
    if len(latest_data) < time_steps:
        print(f"Not enough rows after feature engineering. Need {time_steps}, got {len(latest_data)}.")
        return False

    try:
        features_to_scale = latest_data[feature_columns]
    except KeyError as exc:
        print(f"Missing feature columns in predict data: {exc}")
        return False

    scaled = scaler.transform(features_to_scale)
    input_tensor = torch.from_numpy(scaled).float().unsqueeze(0).to(device)

    print("Running inference...")
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze()
        predicted_class = int(torch.argmax(probabilities).item())

    label_map = {0: "stop-loss", 1: "timeout", 2: "take-profit"}

    last_bar_open_utc = latest_data.index[-1]
    last_bar_close_utc = last_bar_open_utc + bar_delta
    horizon_end_utc = last_bar_close_utc + bar_delta * label_horizon_bars

    tz_taipei = pytz.timezone("Asia/Taipei")
    base_close_local = last_bar_close_utc.astimezone(tz_taipei).strftime("%Y-%m-%d %H:%M")
    horizon_end_local = horizon_end_utc.astimezone(tz_taipei).strftime("%Y-%m-%d %H:%M")
    last_close_price = float(latest_data["close"].iloc[-1])

    result_text = build_prediction_message(
        label=label_map[predicted_class],
        probabilities=probabilities,
        last_close_price=last_close_price,
        base_close_local=base_close_local,
        horizon_end_local=horizon_end_local,
        horizon_bars=label_horizon_bars,
        horizon_delta=bar_delta * label_horizon_bars,
    )

    print("\n" + result_text)

    if send_notification:
        print("Sending Telegram notification...")
        send_to_telegram(result_text)
    return True


def compute_next_run_time_utc(timeframe: str, offset_minutes: int = 5) -> datetime:
    """Return the next candle close time plus offset in UTC."""
    delta = timeframe_to_timedelta(timeframe)
    now = datetime.utcnow().replace(tzinfo=pytz.UTC)
    epoch = datetime(1970, 1, 1, tzinfo=pytz.UTC)
    step = int(delta.total_seconds())

    current_open = epoch + timedelta(seconds=(int((now - epoch).total_seconds()) // step) * step)
    next_close = current_open + delta
    return next_close + timedelta(minutes=offset_minutes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scheduled BTC predictions")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Model folder with weights/scaler/features")
    parser.add_argument("--run-once", action="store_true", help="Run once and exit")
    parser.add_argument("--offset-minutes", type=int, default=5, help="Minutes after candle close to run")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram notifications")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notify = not args.no_telegram

    try:
        bundle_preview = load_model_bundle(args.model_dir, torch.device("cpu"))
        timeframe = str(bundle_preview["timeframe"])
        label_horizon_bars = int(bundle_preview["label_horizon_bars"])
        print(f"Startup config: timeframe={timeframe}, horizon_bars={label_horizon_bars}")
    except Exception as exc:  # noqa: BLE001
        print(f"Metadata preload failed ({exc}), fallback timeframe=4h")
        timeframe = "4h"

    print("=" * 60)
    print(f"Initial prediction run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    predict_latest(args.model_dir, send_notification=notify)

    if args.run_once:
        return

    next_run_time_utc = compute_next_run_time_utc(timeframe, offset_minutes=args.offset_minutes)

    while True:
        now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)
        if now_utc >= next_run_time_utc:
            print("\n" + "=" * 60)
            print(f"Scheduled run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            predict_latest(args.model_dir, send_notification=notify)
            next_run_time_utc = compute_next_run_time_utc(timeframe, offset_minutes=args.offset_minutes)

        sleep_seconds = (next_run_time_utc - now_utc).total_seconds()
        if sleep_seconds > 0:
            print(f"\nSleeping for about {sleep_seconds / 60:.1f} minutes.")
            print(
                "Next run (Asia/Taipei): "
                + next_run_time_utc.astimezone(pytz.timezone("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")
            )
            time.sleep(min(sleep_seconds, 60))


if __name__ == "__main__":
    main()
