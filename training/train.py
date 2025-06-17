import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import os

from models.models import CNNSeqModel, 
from scripts.dataloader import MEGDataset  # replace with your actual module name

# === Config ===
DATA_DIR = '/path/to/meg/files'
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 4  # <-- adjust if needed

# === CNN and sequence model types ===
cnn_types = ['basic', 'attention', 'transformer']
seq_types = ['lstm', 'transformer']

# === Split your dataset ===
def split_dataset(dataset, val_ratio=0.2):
    total = len(dataset)
    val_size = int(total * val_ratio)
    train_size = total - val_size
    return torch.utils.data.random_split(dataset, [train_size, val_size])

# === Training loop ===
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for x in loader:
        x = x.to(DEVICE)  # (B, 20, 21, T)
        x = x.permute(0, 3, 1, 2).unsqueeze(2)  # (B, T, 1, 20, 21)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)  # (B*T, 1, 20, 21)
        labels = torch.randint(0, NUM_CLASSES, (B,), device=DEVICE)  # dummy labels

        optimizer.zero_grad()
        out = model(x.view(B, T, C, H, W))  # (B, num_classes)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc

# === Evaluation loop ===
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x in loader:
            x = x.to(DEVICE)
            x = x.permute(0, 3, 1, 2).unsqueeze(2)
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            labels = torch.randint(0, NUM_CLASSES, (B,), device=DEVICE)

            out = model(x.view(B, T, C, H, W))
            loss = criterion(out, labels)
            total_loss += loss.item()

            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc

# === Main training loop ===
def run_all_combinations():
    dataset = MEGDataset(DATA_DIR)
    train_set, val_set = split_dataset(dataset)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    for cnn_type in cnn_types:
        for seq_type in seq_types:
            print(f"\n>>> Training combo: CNN={cnn_type}, SEQ={seq_type}")
            model = CNNSeqModel(cnn_type=cnn_type, seq_model_type=seq_type, num_classes=NUM_CLASSES)
            model.to(DEVICE)

            criterion = nn.CrossEntropyLoss()
            optimizer = Adam(model.parameters(), lr=LR)

            best_val_acc = 0
            for epoch in range(EPOCHS):
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
                val_loss, val_acc = evaluate(model, val_loader, criterion)

                print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), f"best_model_{cnn_type}_{seq_type}.pt")
                    print(f"  --> Saved best model for combo CNN={cnn_type}, SEQ={seq_type}")

if __name__ == "__main__":
    run_all_combinations()

