
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import numpy as np
import random

from data.dataloader import MaskedMEGSequenceDataset
from models.spatial_models import CNNFrameAutoencoder
from models.sequence_models import TemporalTransformer, TransformerWithDecoder

BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-3
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 1
VAL_SPLIT = 0.2

def collate_fn_padded(batch):
    max_len = max(item['seq_len'] for item in batch)
    batch_size = len(batch)

    inputs_padded = torch.zeros((batch_size, max_len, 1, 20, 21), dtype=torch.float32)
    targets_padded = torch.zeros_like(inputs_padded)
    masks_padded = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        inputs_padded[i, :seq_len] = item['input']
        targets_padded[i, :seq_len] = item['target']
        masks_padded[i, :seq_len] = item['mask']

    return {
        'input': inputs_padded,
        'target': targets_padded,
        'mask': masks_padded,
        'seq_len': torch.tensor([item['seq_len'] for item in batch], dtype=torch.int)
    }

from matplotlib import pyplot as plt


def save_visualization(preds, targets, mask, epoch, batch_idx, save_dir='./visualizations'):
    os.makedirs(save_dir, exist_ok=True)

    preds_np = preds.detach().cpu().numpy()       # (N_masked, D)
    targets_np = targets.detach().cpu().numpy()   # (N_masked, D)

    # Just pick a few samples to visualize
    num_samples = min(8, preds_np.shape[0])  # max 8 frames for clarity

    fig, axes = plt.subplots(num_samples, 2, figsize=(6, num_samples * 3))

    for i in range(num_samples):
        ax_pred = axes[i, 0]
        ax_true = axes[i, 1]

        ax_pred.imshow(preds_np[i].reshape(16, 16), cmap='viridis')  # assuming D=256
        ax_pred.set_title("Predicted")
        ax_pred.axis('off')

        ax_true.imshow(targets_np[i].reshape(16, 16), cmap='viridis')  # reshape to visual 2D
        ax_true.set_title("Ground Truth")
        ax_true.axis('off')

    plt.tight_layout()
    out_path = os.path.join(save_dir, f'epoch{epoch}_batch{batch_idx}_viz.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved visualization to {out_path}")

# Load files
all_files = []
root_dir = "./data/processed_data/"
for task_group in ['Intra', 'Cross']:
    task_group_path = os.path.join(root_dir, task_group)
    for subfolder in ['train']:
        folder_path = os.path.join(task_group_path, subfolder)
        if not os.path.exists(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith('.npy'):
                all_files.append(os.path.join(folder_path, file))

random.shuffle(all_files)
split_idx = int(len(all_files) * 0.8)
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

train_set = MaskedMEGSequenceDataset(files=train_files, max_seq_len=3000, min_seq_len=100, max_mask_blocks=1, mask_ratio=0.7)
val_set = MaskedMEGSequenceDataset(files=val_files, max_seq_len=3000, min_seq_len=100, max_mask_blocks=1, mask_ratio=0.7)

print(f"Train: {len(train_set)} samples\nVal: {len(val_set)} samples")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_padded)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=collate_fn_padded)

autoencoder = CNNFrameAutoencoder(embed_dim=256).to(DEVICE)
for p in autoencoder.parameters():
    p.requires_grad = False
autoencoder.eval()

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

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training")):
        inputs = batch['input'].to(DEVICE)       # (B, T, 1, 20, 21)
        masks = batch['mask'].to(DEVICE)         # (B, T)
        targets = batch['target'].to(DEVICE)     # (B, T, 1, 20, 21)

        with torch.no_grad():
            _, embeddings = autoencoder(inputs)  # (B, T, D)

        preds, true = model.forward_for_pretraining(embeddings, masks)

        # Sanity check: make sure we have masked tokens, else skip
        if preds.shape[0] == 0:
            print(f"Warning: No masked tokens in epoch {epoch} batch {batch_idx}, skipping loss.")
            continue

        loss = criterion(preds, true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        # Save first batch every 5 epochs for visualization
        if batch_idx == 40: 
            save_visualization(preds, true, masks.view(-1), epoch, batch_idx)

    avg_train_loss = total_train_loss / (batch_idx + 1)
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for val_batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
            inputs = batch['input'].to(DEVICE)
            masks = batch['mask'].to(DEVICE)
            targets = batch['target'].to(DEVICE)

            _, embeddings = autoencoder(inputs)
            preds, true = model.forward_for_pretraining(embeddings, masks)

            if preds.shape[0] == 0:
                continue

            loss = criterion(preds, true)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / (val_batch_idx + 1)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "./results/model_weights/transformer/best_pretrained_transformer.pt")
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print("Early stopping triggered.")
            break
