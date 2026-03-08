# Binance3

GRU + Attention workflow for BTC/USDT data preparation, training, brute-force search, and scheduled inference.

## Project Files

- `data_pre.py`: Fetches OHLCV data, builds indicators, and generates triple-barrier labels.
- `GRU_attention_modified.py`: Model definition and training routine.
- `brute_force_tester.py`: Hyperparameter search runner.
- `predict.py`: Loads model bundle, runs inference, supports scheduled execution.
- `notification.py`: Sends Telegram notifications via environment variables.
- `run_predict.sh`: Convenience launcher that loads `.env.local` automatically.

## 1) Environment Setup

```bash
cd /home/nas2/Personal/Hank/Binance3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch ccxt pandas pandas_ta numpy scikit-learn matplotlib pytz requests
```

## 2) Secrets Setup (`.env.local`)

1. Copy template:

```bash
cp .env.example .env.local
```

2. Edit `.env.local` and set real values:

```env
TELEGRAM_API_TOKEN=your_real_token
TELEGRAM_CHAT_ID=your_real_chat_id
```

3. Restrict local file permissions:

```bash
chmod 600 .env.local
```

`notification.py` reads `TELEGRAM_API_TOKEN` and `TELEGRAM_CHAT_ID` from environment only.

## 3) Data Preparation

```bash
source .venv/bin/activate
python data_pre.py --symbol BTC/USDT --timeframe 4h --days 5000 --profit-take 0.04 --stop-loss 0.02 --time-barrier 12 --output btc_labeled_data.csv
```

## 4) Brute-Force Training

```bash
source .venv/bin/activate
python brute_force_tester.py
```

Outputs are written under `brute_force_results_hl_barrier/`.

## 5) Prediction

### Run once

```bash
./run_predict.sh --run-once --model-dir brute_force_results_hl_barrier/<your_best_run>
```

### Run scheduler (default behavior)

```bash
./run_predict.sh --model-dir brute_force_results_hl_barrier/<your_best_run>
```

Useful flags:

- `--no-telegram`: disable Telegram notifications.
- `--offset-minutes 5`: run N minutes after candle close.

## 6) Git Workflow (Safe Push)

### Standard push

```bash
git status
git add .gitignore .env.example README.md notification.py data_pre.py brute_force_tester.py predict.py GRU_attention_modified.py run_predict.sh
git commit -m "Harden secrets handling and clean code comments"
git push origin main
```

### Important security note

If a token was already committed and pushed in the past, removing it from the current code is not enough.

1. Rotate/revoke the leaked token immediately.
2. Optionally rewrite Git history and force-push to remove old exposure.

## 7) Recommended Ignore Rules

This repo now ignores local secrets and result folders via `.gitignore`:

- `.env`, `.env.*`, `.env.local` (except `.env.example`)
- `brute_force_results_bs/`
- `brute_force_results_hl_barrier/`
- `brute_force_results_no_features/`

