import json
import os
import glob
import numpy as np
from collections import defaultdict


def aggregate_results(base_dir):
    result_files = glob.glob(os.path.join(base_dir, "*", "json", "*.json"))
    grouped_results = defaultdict(list)

    for filepath in result_files:
        with open(filepath, "r") as f:
            data = json.load(f)

        name = data.get("config", {}).get("name", "unknown")
        task = data.get("config", {}).get("task_group_choice", "unknown")
        key = f"{name}_{task}"

        test_metrics = data.get("test_metrics", {})
        final_train_epoch = data.get("training", {}).get("epoch", None)

        if test_metrics and final_train_epoch is not None:
            grouped_results[key].append({
                "metrics": test_metrics,
                "epochs": final_train_epoch
            })

    aggregated_table = []

    for key, runs in grouped_results.items():
        metrics_keys = runs[0]["metrics"].keys()
        metric_arrays = {k: [] for k in metrics_keys}
        epoch_counts = []

        for run in runs:
            for k in metrics_keys:
                metric_arrays[k].append(run["metrics"].get(k, 0.0))
            epoch_counts.append(run["epochs"])

        avg_metrics = {k: np.mean(metric_arrays[k]) for k in metrics_keys}
        ci_metrics = {k: 1.96 * np.std(metric_arrays[k]) / np.sqrt(len(metric_arrays[k])) for k in metrics_keys}
        avg_epochs = np.mean(epoch_counts)
        ci_epochs = 1.96 * np.std(epoch_counts) / np.sqrt(len(epoch_counts))

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

    base_dir = "results/ablation_study"
    table = aggregate_results(base_dir)
    df = pd.DataFrame(table)
    print(df.to_string(index=False))
    df.to_csv("aggregated_metrics.csv", index=False)

