import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from utils import available_device
from sys import argv

if len(argv) == 1:
    lr = 1e-1
else:
    lr = float(argv[1]) 


print(f"learning rate is {lr}")

def train_cnn_denoising_autoencoder(
    model,
    dataloader,
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

    total_batches = len(dataloader)
    print(f"Total batches per epoch: {total_batches}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch, (noisy, clean) in enumerate(dataloader):
            noisy = noisy.to(device).float()
            clean = clean.to(device).float()
            optimizer.zero_grad()
            reconstructed,  _ = model(noisy)
            loss = criterion(reconstructed, clean)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * noisy.size(0)

            # Progress print every 10% of batches
            if (batch + 1) % max(1, total_batches // 10) == 0 or batch == total_batches - 1:
                percent = 100 * (batch + 1) / total_batches
                print(f"Epoch {epoch+1}/{epochs} - Batch {batch+1}/{total_batches} ({percent:.1f}%)")

        epoch_loss = running_loss / len(dataloader.dataset)
        train_losses.append(epoch_loss)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.6f}")

        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            # Save best model checkpoint
            torch.save(model.state_dict(), "results/model_weights/denoising/best_cnn_denoising_autoencoder.pth")
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
    from models.spatial_models import CNNFrameAutoencoder
    from data.dataloader import get_denoising_cnn_pretrain_loader

    model = CNNFrameAutoencoder()
    dataloader = get_denoising_cnn_pretrain_loader()

    train_cnn_denoising_autoencoder(model, dataloader, 40)
