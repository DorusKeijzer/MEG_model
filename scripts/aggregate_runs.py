import json
import os
import glob
import numpy as np
from collections import defaultdict

def parse_model_task_from_filename(filename):
    parts = filename.split('_')
    task = None
    if 'intra' in parts:
        task = 'intra'
    elif 'cross' in parts:
        task = 'cross'
    else:
        task = 'unknown'
    if task in parts:
        idx = parts.index(task)
        name = '_'.join(parts[:idx])
    else:
        name = 'unknown'
    return name, task

def aggregate_results(base_dir):
    print(f"looking in {base_dir}")
    result_files = glob.glob(os.path.join(base_dir, "*.json"))
    print(f"files: {result_files}")
    grouped_results = defaultdict(list)

    for filepath in result_files:
        with open(filepath, "r") as f:
            data = json.load(f)
        
        filename = os.path.basename(filepath)
        name, task = parse_model_task_from_filename(filename)
        key = f"{name}_{task}"

        # Try multiple locations for test metrics
        test_metrics = data.get("final_test") or data.get("test_metrics") or {}
        # Try multiple locations for final epoch
        final_train_epoch = data.get("training", {}).get("epoch")
        if final_train_epoch is None and "train_loss" in data:
            final_train_epoch = len(data["train_loss"])

        print(f"File: {filename}")
        print(f"Parsed model: {name}, task: {task}")
        print("Test metrics keys:", list(test_metrics.keys()))
        print("Final train epoch:", final_train_epoch)
        print("---")

        if test_metrics and final_train_epoch is not None:
            grouped_results[key].append({
                "metrics": test_metrics,
                "epochs": final_train_epoch
            })

    aggregated_table = []

    for key, runs in grouped_results.items():
        if len(runs) == 0:
            continue
        metrics_keys = runs[0]["metrics"].keys()
        metric_arrays = {k: [] for k in metrics_keys}
        epoch_counts = []

        for run in runs:
            for k in metrics_keys:
                metric_arrays[k].append(run["metrics"].get(k, 0.0))
            epoch_counts.append(run["epochs"])

        avg_metrics = {k: np.mean(metric_arrays[k]) for k in metrics_keys}
        ci_metrics = {k: 1.96 * np.std(metric_arrays[k]) / np.sqrt(len(metric_arrays[k])) if len(metric_arrays[k]) > 1 else 0.0 for k in metrics_keys}
        avg_epochs = np.mean(epoch_counts)
        ci_epochs = 1.96 * np.std(epoch_counts) / np.sqrt(len(epoch_counts)) if len(epoch_counts) > 1 else 0.0

        row = {
            "model_task": key,
            **{f"{k}_avg": round(avg_metrics[k], 4) for k in metrics_keys},
            **{f"{k}_ci": round(ci_metrics[k], 4) for k in metrics_keys},
            "avg_epochs": round(avg_epochs, 2),
            "epochs_ci": round(ci_epochs, 2)
        }

        aggregated_table.append(row)

    return aggregated_table


if __name__ == "__main__":
    import pandas as pd

    base_dir = "./ablation_study/ablation_configs_20250618_150837/metrics/"
    table = aggregate_results(base_dir)
    if len(table) == 0:
        print("No results found to aggregate.")
    else:
        df = pd.DataFrame(table)
        print(df.to_string(index=False))
        df.to_csv("aggregated_metrics.csv", index=False)
