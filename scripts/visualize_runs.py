import json
import matplotlib.pyplot as plt

def plot_training_metrics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    metrics = ['loss', 'acc', 'f1', 'prec', 'recall']
    fig, axs = plt.subplots(len(metrics), 1, figsize=(8, 15))

    for i, metric in enumerate(metrics):
        train = data.get(f"train_{metric}", [])
        val = data.get(f"val_{metric}", [])
        axs[i].plot(train, label=f"Train {metric}")
        axs[i].plot(val, label=f"Val {metric}")
        axs[i].set_title(metric.upper())
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel(metric)
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training_metrics("results/CNNTransformer_Intra_metrics_20250618_193400.json")  # change this path

