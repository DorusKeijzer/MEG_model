import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from data.dataloader import get_masked_cnn_pretrain_dataset, get_masked_noisy_cnn_pretrain_dataset, get_denoising_cnn_pretrain_dataset
from utils import available_device, noise_mask
from sys import argv

lr = 1e-3
print(f"learning rate is {lr}")

def evaluate(model, val_loader, device=available_device):
    criterion = torch.nn.MSELoss()
    model.eval()

    total_loss = 0
    with torch.no_grad():  # <<< Add this
        for clean, noisy in val_loader:
            noisy = noisy.to(device).float()
            clean = clean.to(device).float()

            reconstructed, _ = model(noisy)
            loss = criterion(reconstructed, clean)
            total_loss += loss.item() * noisy.size(0)  # Use .item() to keep it on CPU

    avg_loss = total_loss / len(val_loader.dataset)  # Use dataset size for average
    return avg_loss



def train_cnn_denoising_autoencoder(
    model,
    train_loader,
    val_loader,
    task,
    epochs=20,
    lr=1e-1,
    device=available_device,
    patience=5  # early stopping patience in epochs
):
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = np.inf
    epochs_no_improve = 0
    train_losses = []

    total_batches = len(train_loader)
    print(f"Total batches per epoch: {total_batches}")
    print(f"Device: {device}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch, (clean, noisy) in enumerate(train_loader):
            if batch == 0 and epoch == 0:
                print("Clean min/max:", clean.min().item(), clean.max().item())
                print("Noisy min/max:", noisy.min().item(), noisy.max().item())

            noisy = noisy.to(device).float()
            clean = clean.to(device).float()
            optimizer.zero_grad()
            reconstructed,  _ = model(noisy)
            loss = criterion(reconstructed, clean)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * noisy.size(0)

            if (batch + 1) % max(1, total_batches // 4) == 0 or batch == total_batches - 1:
                percent = 100 * (batch + 1) / total_batches
                print(f"\tEpoch {epoch+1}/{epochs} - Batch {batch+1}/{total_batches} ({percent:.1f}%)", end="\r")

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        print(f"Epoch [{epoch+1}/{epochs}] Training Loss: {epoch_loss:.6f}")

        val_loss = evaluate(model, val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] Validation Loss: {val_loss:.6f}")

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            # Save best model checkpoint
            torch.save(model.state_dict(), f"results/model_weights/{task}/best_cnn_denoising_autoencoder.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Plot loss curve()
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', color='blue')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig("cnn_denoising_loss_curve.png")
    plt.show()

    print("Training complete. Best loss: {:.6f}".format(best_loss))

if __name__ == "__main__":
    from sys import argv

    specified_dataset = "noise"

    if len(argv) == 2:
        specified_dataset = argv[1]

    from models.spatial_models import CNNFrameAutoencoder
    from data.dataloader import get_denoising_cnn_pretrain_dataset
    from torch.utils.data import random_split

    model = CNNFrameAutoencoder()

    if specified_dataset == "masking":
        dataset = get_masked_cnn_pretrain_dataset(noise_mask=noise_mask)
    if specified_dataset == "both":
        dataset = get_masked_noisy_cnn_pretrain_dataset(noise_mask=noise_mask) 
    else:
        dataset = get_denoising_cnn_pretrain_dataset(noise_mask=noise_mask)


    train_size = int(0.8 *len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=128, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=64, num_workers=2, pin_memory=False)

    train_cnn_denoising_autoencoder(model, train_loader, val_loader, specified_dataset, epochs=40)
