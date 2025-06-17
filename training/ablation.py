import os
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from data.dataloader import MEGVolumeDataset 
from torch.utils.data import random_split

from models.spatial_models import CNNFrameAutoencoder 
from models.sequence_models import TemporalTransformer 




from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np


def evaluate(model_cnn, model_transformer, dataloader, criterion, device):
    model_cnn.eval()
    model_transformer.eval()

    all_preds, all_labels = [], []
    val_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            _, z = model_cnn(x)
            logits = model_transformer(z)

            loss = criterion(logits, y)
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    val_loss /= len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return val_loss, acc, recall, f1, precision

def load_weights(model_cnn, model_transformer, cnn_path=None, transformer_path=None):
    if cnn_path and os.path.exists(cnn_path):
        print(f"Loading CNN weights from {cnn_path}")
        model_cnn.load_state_dict(torch.load(cnn_path))
    if transformer_path and os.path.exists(transformer_path):
        print(f"Loading Transformer weights from {transformer_path}")
        model_transformer.load_state_dict(torch.load(transformer_path))

def save_metrics_json(metrics, save_path):
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/ablation_configs.json',
                        help="Path to ablation config JSON")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        experiment_configs = json.load(f)

    for run_idx, config in enumerate(experiment_configs["experiments"]):
        print(f"\nRunning experiment {run_idx+1}/{len(experiment_configs['experiments'])}: {config['name']}")

        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model + Load Pretrained Weights if specified
        model_cnn = CNNFrameAutoencoder(embed_dim=config["embed_dim"]).to(device)
        model_transformer = TemporalTransformer(embed_dim=config["embed_dim"], num_classes=config["num_classes"]).to(device)
        load_weights(model_cnn, model_transformer, config.get("cnn_weights"), config.get("transformer_weights"))

        # Dataset
        dataset = MEGVolumeDataset(config["dataset_path"], mode='train')
        train_size = int(config["train_split"] * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=config["batch_size"], num_workers=2, pin_memory=True)

        # Optimizer & Criterion
        optimizer = optim.Adam(list(model_cnn.parameters()) + list(model_transformer.parameters()), lr=config["lr"])
        criterion = nn.CrossEntropyLoss()

        # Train loop (can extract this into a func later)
        metrics = {
            "train_loss": [], "train_acc": [], "train_recall": [], "train_f1": [], "train_prec": [],
            "val_loss": [], "val_acc": [], "val_recall": [], "val_f1": [], "val_prec": []
        }

        best_val_loss = float('inf')
        wait, patience = 0, config.get("patience", 5)

        for epoch in range(config["num_epochs"]):
            print(f"\nEpoch {epoch+1}/{config['num_epochs']} for run: {config['name']}")
            model_cnn.train()
            model_transformer.train()

            train_loss, all_preds, all_labels = 0.0, [], []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                _, z = model_cnn(x)
                logits = model_transformer(z)

                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

            # Log Train Metrics
            train_loss /= len(train_loader)
            metrics["train_loss"].append(train_loss)
            metrics["train_acc"].append(accuracy_score(all_labels, all_preds))
            metrics["train_prec"].append(precision_score(all_labels, all_preds, average="macro"))
            metrics["train_recall"].append(recall_score(all_labels, all_preds, average="macro"))
            metrics["train_f1"].append(f1_score(all_labels, all_preds, average="macro"))

            # Eval
            val_loss, val_acc, val_recall, val_f1, val_prec = evaluate(
                model_cnn, model_transformer, val_loader, criterion, device
            )
            metrics["val_loss"].append(val_loss)
            metrics["val_acc"].append(val_acc)
            metrics["val_recall"].append(val_recall)
            metrics["val_prec"].append(val_prec)
            metrics["val_f1"].append(val_f1)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

            # Early Stop
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                torch.save({
                    'cnn_state_dict': model_cnn.state_dict(),
                    'transformer_state_dict': model_transformer.state_dict(),
                }, os.path.join(config["save_dir"], f"{config['name']}_best_model.pt"))
                print("Best model saved!")
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered.")
                    break

        # Save final metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_save_path = os.path.join(config["save_dir"], f"{config['name']}_metrics_{timestamp}.json")
        save_metrics_json(metrics, json_save_path)

        print(f"Finished run: {config['name']}, results saved to {json_save_path}")

