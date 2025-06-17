import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def train_cnn_denoising_autoencoder(
    model,
    dataset,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    noise_std=0.1,
    device='cuda',
    patience=5  # early stopping patience in epochs
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = np.inf
    epochs_no_improve = 0
    train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * noisy.size(0)

        epoch_loss = running_loss / len(dataset)
        train_losses.append(epoch_loss)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.6f}")

        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            # Save best model checkpoint
            torch.save(model.state_dict(), "best_cnn_denoising_autoencoder.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Plot loss curve
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
    from data.dataloader import get_cnn_pretrain_loader

    model = CNNFrameAutoencoder
    dataloader = get_cnn_pretrain_loader("hier komt geprocessde data.npy")

    train_cnn_denoising_autoencoder(model, dataloader, 40)
