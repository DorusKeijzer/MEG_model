import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import os

def train_autoencoder(model, dataloader, epochs, device, save_path, lr=1e-3):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            inputs = batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Outputs can be tuple for CNNAE; handle accordingly
            if isinstance(outputs, tuple):
                recon = outputs[0]
            else:
                recon = outputs

            loss = F.mse_loss(recon, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f}")

        # Save best checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

def load_checkpoint(model, checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")

# === USAGE EXAMPLE ===
if __name__ == "__main__":
    # Assuming you have a Dataset `meg_dataset` for (B, 1, 20, 20) or (B, T, 1, 20, 20)
    from your_dataset_module import meg_dataset  # Replace with your dataset import

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For CNNAutoencoder:
    from your_autoencoder_module import CNNAutoencoder
    ae_model = CNNAutoencoder()
    checkpoint_path = "cnnae_best.pt"
    load_checkpoint(ae_model, checkpoint_path, device)
    dataloader = DataLoader(meg_dataset, batch_size=32, shuffle=True)
    train_autoencoder(ae_model, dataloader, epochs=20, device=device, save_path=checkpoint_path)

    # For CNNLSTMAutoencoder:
    # from your_autoencoder_module import CNNLSTMAutoencoder
    # ae_seq_model = CNNLSTMAutoencoder()
    # checkpoint_path_seq = "cnn_lstm_ae_best.pt"
    # load_checkpoint(ae_seq_model, checkpoint_path_seq, device)
    # dataloader_seq = DataLoader(meg_seq_dataset, batch_size=16, shuffle=True)
    # train_autoencoder(ae_seq_model, dataloader_seq, epochs=20, device=device, save_path=checkpoint_path_seq)

