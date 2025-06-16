import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys

def print_progress_bar(epoch, total_epochs, train_loss, val_acc, bar_len=30):
    progress = (epoch + 1) / total_epochs
    filled_len = int(bar_len * progress)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write(f"\rEpoch {epoch+1}/{total_epochs} [{bar}] - Train Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")
    sys.stdout.flush()
    if epoch + 1 == total_epochs:
        print()

def plot_and_save(train_losses, val_accuracies, config_name):
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{config_name}_{timestamp}.png"

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(train_losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Val Accuracy', color=color)
    ax2.plot(val_accuracies, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"Training Loss & Validation Accuracy for {config_name}")
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved training plot to {filename}")

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

def eval_epoch(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def ablation_runner(configs, train_dataset, val_dataset, device, epochs=10, batch_size=32):
    results = {}
    for config_name, cfg in configs.items():
        print(f"\nStarting experiment: {config_name}")

        model = CNNSeqModel(
            cnn_type=cfg.get('cnn_type', 'basic'),
            seq_model_type=cfg.get('seq_model_type', 'lstm'),
            lstm_hidden=cfg.get('lstm_hidden', 64),
            num_classes=4
        ).to(device)

        if 'pretrained_weights' in cfg:
            state_dict = torch.load(cfg['pretrained_weights'], map_location=device)
            model.feature_extractor.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {cfg['pretrained_weights']}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = optim.Adam(model.parameters(), lr=cfg.get('lr', 1e-3))

        best_val_acc = 0
        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_acc = eval_epoch(model, val_loader, device)
            train_losses.append(train_loss)
            val_accuracies.append(val_acc)

            print_progress_bar(epoch, epochs, train_loss, val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"best_model_{config_name}.pt")

        plot_and_save(train_losses, val_accuracies, config_name)
        results[config_name] = best_val_acc
        print(f"Best val accuracy for {config_name}: {best_val_acc:.4f}")

    return results


# === Example configs ===
configs = {
    "basic_lstm_no_pretrain": {
        "cnn_type": "basic",
        "seq_model_type": "lstm",
        "lstm_hidden": 64,
        "lr": 1e-3
    },
    "attention_transformer_with_pretrain": {
        "cnn_type": "attention",
        "seq_model_type": "transformer",
        "lr": 5e-4,
        "pretrained_weights": "cnnae_best.pt"
    },
}

# === Usage example ===
if __name__ == "__main__":
    from your_dataset_module import train_dataset, val_dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = ablation_runner(configs, train_dataset, val_dataset, device, epochs=15, batch_size=64)
    print("\n=== Ablation Experiment Summary ===")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")

