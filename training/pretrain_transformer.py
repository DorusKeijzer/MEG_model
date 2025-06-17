import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np

from data.dataloader import MaskedMEGSequenceDataset
from models.spatial_models import CNNFrameAutoencoder
from models.sequence_models import TemporalTransformer
from models.sequence_models import TransformerWithDecoder

BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 1
VAL_SPLIT = 0.2

full_dataset = MaskedMEGSequenceDataset("./data/processed_data/", seq_len=100, mask_ratio=0.3)
val_len = int(VAL_SPLIT * len(full_dataset))
train_len = len(full_dataset) - val_len
train_set, val_set = random_split(full_dataset, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

autoencoder = CNNFrameAutoencoder(embed_dim=256).to(DEVICE)
transformer = TemporalTransformer(embed_dim=256).to(DEVICE)
model = TransformerWithDecoder(transformer, embed_dim=256).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

train_losses = []
val_losses = []
best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
        inputs = batch['input'].to(DEVICE)       # (B, T, 1, 20, 21)
        masks = batch['mask'].to(DEVICE)         # (B, T)
        targets = batch['target'].to(DEVICE)     # (B, T, 1, 20, 21)

        _, embeddings = autoencoder(inputs)      # (B, T, D)
        preds, true = model.forward_for_pretraining(embeddings, masks)

        loss = criterion(preds, true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            inputs = batch['input'].to(DEVICE)
            masks = batch['mask'].to(DEVICE)
            targets = batch['target'].to(DEVICE)

            _, embeddings = autoencoder(inputs)
            preds, true = model.forward_for_pretraining(embeddings, masks)

            loss = criterion(preds, true)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "/results/model_weights/transformer/best_pretrained_transformer.pt")
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title("Pretraining Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("pretraining_loss.png")
plt.show()

