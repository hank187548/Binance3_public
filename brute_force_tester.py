"""Brute-force hyperparameter search driver."""

from __future__ import annotations

import csv
import gc
import itertools
import json
import os
from datetime import datetime

import torch

from GRU_attention_modified import run_training


PARAM_GRID = {
    "learning_rate": [0.001, 0.0005],
    "hidden_dim": [32, 64],
    "num_layers": [1, 2],
    "dropout_prob": [0.3, 0.5],
    "weight_decay": [0.01],
    "batch_size": [16, 32, 64, 128],
    "time_steps": [24, 36, 48, 60],
    "n_splits": [3],
    "epochs": [100],
}

ALIASES = {
    "learning_rate": "lr",
    "hidden_dim": "hd",
    "num_layers": "layers",
    "dropout_prob": "drop",
    "weight_decay": "wd",
    "batch_size": "bs",
    "time_steps": "ts",
    "n_splits": "cv",
    "epochs": "ep",
}


def fmt(value: object) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def main() -> None:
    base_output_dir = "brute_force_results_hl_barrier"
    os.makedirs(base_output_dir, exist_ok=True)

    summary_path = os.path.join(base_output_dir, "summary.csv")
    write_header = not os.path.exists(summary_path)

    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_runs = len(combinations)
    print(f"Starting brute-force search with {total_runs} combinations.")

    best_loss = float("inf")
    best_params: dict[str, object] | None = None
    best_folder: str | None = None

    for index, params in enumerate(combinations, start=1):
        folder_name = "-".join(f"{ALIASES.get(k, k)}{fmt(v)}" for k, v in params.items())
        run_output_dir = os.path.join(base_output_dir, folder_name)

        print("\n" + "=" * 80)
        print(f"Run {index}/{total_runs}")
        print(f"Params: {params}")
        print(f"Output: {run_output_dir}")
        print("=" * 80 + "\n")

        try:
            current_loss = run_training(
                output_dir=run_output_dir,
                data_path="btc_labeled_data.csv",
                **params,
            )

            if current_loss is not None:
                print(f"Run {index}/{total_runs} finished. Avg val loss: {current_loss:.6f}")

                metadata = {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "trainer": "GRU_attention_modified.run_training",
                    "data_path": "btc_labeled_data.csv",
                    "timeframe": "4h",
                    "label_horizon_bars": 12,
                    "label_horizon_hours": 48,
                    "params": params,
                }
                os.makedirs(run_output_dir, exist_ok=True)
                with open(os.path.join(run_output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, ensure_ascii=False, indent=2)

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_params = params
                    best_folder = run_output_dir
                    print(f"New best run found. Loss: {best_loss:.6f}")

                with open(summary_path, "a", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    if write_header:
                        writer.writerow(["timestamp", "run_idx", "avg_val_loss", *keys, "output_dir"])
                        write_header = False
                    writer.writerow([
                        datetime.now().isoformat(timespec="seconds"),
                        index,
                        current_loss,
                        *[params[k] for k in keys],
                        run_output_dir,
                    ])

            del current_loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:  # noqa: BLE001
            print(f"Run {index}/{total_runs} failed")
            print(f"Params: {params}")
            print(f"Error: {exc}")
            with open(os.path.join(base_output_dir, "error_log.txt"), "a", encoding="utf-8") as handle:
                handle.write(
                    f"[{datetime.now().isoformat(timespec='seconds')}] Error for params {params}:\n{exc}\n\n"
                )
            print("Continuing to next run...")

    print("\n" + "=" * 80)
    print("Brute-force search completed")
    print("=" * 80)
    if best_params is not None:
        print("\nBest model summary")
        print(f"  Lowest validation loss: {best_loss:.6f}")
        print(f"  Params: {best_params}")
        print(f"  Output folder: {best_folder}")
        print(f"  Summary CSV: {summary_path}")
    else:
        print("No successful runs. Check error_log.txt")
    print("=" * 80)


if __name__ == "__main__":
    main()
