import os
from pathlib import Path

import requests


def _load_local_env(env_path: str = ".env.local") -> None:
    """Load key-value pairs from a local env file into process env."""
    path = Path(__file__).resolve().parent / env_path
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def send_to_telegram(message: str, image_path: str | None = None) -> bool:
    """Send a text message (and optional image) to Telegram."""
    _load_local_env()

    api_token = os.getenv("TELEGRAM_API_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not api_token or not chat_id:
        print("Missing TELEGRAM_API_TOKEN or TELEGRAM_CHAT_ID.")
        return False

    try:
        if image_path and os.path.exists(image_path):
            api_url = f"https://api.telegram.org/bot{api_token}/sendPhoto"
            with open(image_path, "rb") as image_file:
                files = {"photo": image_file}
                data = {"chat_id": chat_id, "caption": message}
                response = requests.post(api_url, data=data, files=files, timeout=30)
        else:
            api_url = f"https://api.telegram.org/bot{api_token}/sendMessage"
            json_data = {"chat_id": chat_id, "text": message}
            response = requests.post(api_url, json=json_data, timeout=30)

        response.raise_for_status()
        print(f"Telegram API response: {response.status_code}")
        return True
    except requests.exceptions.RequestException as exc:
        print(f"Telegram request failed: {exc}")
        return False
