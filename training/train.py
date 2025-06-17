import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from data.dataloader import MEGVolumeDataset 
from torch.utils.data import random_split


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

def plot_metrics(metrics):
    epochs = len(metrics["train_loss"])
    x = range(1, epochs + 1)

    fig, axs = plt.subplots(2, 3, figsize=(12, 10))
    axs = axs.ravel()

    axs[0].plot(x, metrics["train_loss"], label="Train")
    axs[0].plot(x, metrics["val_loss"], label="Validation")
    axs[0].set_title("Loss")
    axs[0].legend()

    axs[1].plot(x, metrics["train_acc"], label="Train")
    axs[1].plot(x, metrics["val_acc"], label="Validation")
    axs[1].set_title("Accuracy")
    axs[1].legend()
    

    axs[2].plot(x, metrics["train_prec"], label="Train")
    axs[2].plot(x, metrics["val_prec"], label="Validation")
    axs[2].set_title("P/tcision")
    axs[2].legend()
    
    axs[3].plot(x, metrics["train_recall"], label="Train")
    axs[3].plot(x, metrics["val_recall"], label="Validation")
    axs[3].set_title("Recall")
    axs[3].legend()

    axs[4].plot(x, metrics["train_f1"], label="Train")
    axs[4].plot(x, metrics["val_f1"], label="Validation")
    axs[4].set_title("F1 Score")
    axs[4].legend()

    plt.tight_layout()
    plt.show()



from models.spatial_models import CNNFrameAutoencoder 
from models.sequence_models import TemporalTransformer 


def train(model_cnn, model_transformer, dataloader, optimizer, criterion, device):
    model_cnn.train()
    model_transformer.train()

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        _, z = model_cnn(x)             # z: (B, T, D)
        logits = model_transformer(z)   # logits: (B, num_classes)

        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"[{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")




def main():
    import os
    from sklearn.metrics import accuracy_score, recall_score, f1_score

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    embed_dim = 256
    num_classes = 4  # adjust this if using only 4 tasks
    batch_size = 1
    num_epochs = 50
    lr = 1e-4

    # Models
    model_cnn = CNNFrameAutoencoder(embed_dim=embed_dim).to(device)
    model_transformer = TemporalTransformer(embed_dim=embed_dim, num_classes=num_classes).to(device)

    # Dataset + Loader
    dataset = MEGVolumeDataset("./data/processed_data/", mode='train')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=2, pin_memory=True)

    # Optimizer + Criterion
    optimizer = optim.Adam(list(model_cnn.parameters()) + list(model_transformer.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Tracking & early stopping
    best_val_loss = float('inf')
    patience = 5
    wait = 0
    metrics = {
        "train_loss": [], "train_acc": [], "train_recall": [], "train_f1": [], "train_prec": [], 
        "val_loss": [], "val_acc": [], "val_recall": [], "val_f1": [],  "val_prec": []
    }

    print("Training started ...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model_cnn.train()
        model_transformer.train()

        train_loss = 0.0
        all_preds, all_labels = [], []

        for batch_idx, (x, y) in enumerate(train_loader):
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

        train_loss /= len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_prec = precision_score(all_labels, all_preds, average="macro")
        train_recall = recall_score(all_labels, all_preds, average='macro')
        train_f1 = f1_score(all_labels, all_preds, average='macro')

        # Validation
        val_loss, val_acc, val_recall, val_f1, val_prec = evaluate(
            model_cnn, model_transformer, val_loader, criterion, device
        )

        # Logging
        print(f"Train | Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Precison : {train_prec:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"Val   | Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        # Store metrics
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["train_recall"].append(train_recall)
        metrics["train_prec"].append(train_prec)
        metrics["train_f1"].append(train_f1)

        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)
        metrics["val_recall"].append(val_recall)
        metrics["val_prec"].append(val_prec)
        metrics["val_f1"].append(val_f1)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save({
                'cnn_state_dict': model_cnn.state_dict(),
                'transformer_state_dict': model_transformer.state_dict(),
            }, 'results/model_weights/classification_no_pretraining/best_model.pt')
            print("Saved best model!")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # Final Plot
    plot_metrics(metrics)

if __name__ == "__main__":
    main()
