import torch
import torch.nn as nn


class BasicCNNEncoder(nn.Module):
    def __init__(self, embed_dim=256, out_features=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((10, 10)),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(128 * 4 * 4, embed_dim)
        self.out = nn.Linear(embed_dim, out_features)


    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(B, T, -1)
        return self.out(x)  # (B, T, D) — each frame gets a feature

class CNNFrameAutoencoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc_enc = nn.Linear(128 * 4 * 4, embed_dim)

        # Decoder
        self.fc_dec = nn.Linear(embed_dim, 128 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1), nn.ReLU(),  # 4x4 -> 9x9
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1), nn.ReLU(),   # 9x9 -> ~18x18
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid(),                          # 18x18 -> 18x18
            nn.Upsample(size=(20, 21), mode='bilinear', align_corners=False)       # Upsample to 20x21
        )




    # Supports (B, T, 1, H, W)
    def forward(self, x):
        if x.dim() == 4:
            # If input is (B, 1, H, W), add T=1
            x = x.unsqueeze(1)  # (B, 1, 1, H, W)
        # print(x.shape)

        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.fc_enc(x)
        x = self.fc_dec(z).view(B * T, 128, 4, 4)
        x = self.decoder(x)
        x = x.view(B, T, 1, 20, 21)
        z = z.view(B, T, -1)
        return x, z


if __name__ == "__main__":
    import torch

    B, T, C, H, W = 4, 100, 1, 20, 21  # Batch, Time, Channel, Height, Width
    dummy_input = torch.randn(B, T, C, H, W)

    print("=== Testing BasicCNNEncoder ===")
    encoder = BasicCNNEncoder(embed_dim=128, out_features=64)
    out = encoder(dummy_input)
    print("Output shape (BasicCNNEncoder):", out.shape)
    assert out.shape == (B, 64), "BasicCNNEncoder output shape mismatch"

    autoencoder = CNNFrameAutoencoder(embed_dim=128)
    recon, latents = autoencoder(dummy_input)
    print("Reconstructed shape:", recon.shape)
    print("Latent shape:", latents.shape)
    assert recon.shape == (B, T, 1, H, W), "Autoencoder recon shape mismatch"
    assert latents.shape == (B, T, 128), "Autoencoder latent shape mismatch"

    # print("\n=== Testing AttentionCNNEncoder ===")
    # attn_cnn = AttentionCNNEncoder(embed_dim=128, out_features=64)
    # out = attn_cnn(dummy_input)
    # print("Output shape (AttentionCNNEncoder):", out.shape)
    # assert out.shape == (B, 64), "AttentionCNNEncoder output shape mismatch"

    # print("\n=== Testing TransformerFeatureExtractor ===")
    # transformer = TransformerFeatureExtractor(input_dim=H * W, embed_dim=128)
    # out = transformer(dummy_input.squeeze(2))  # Remove channel dim for this model
    # print("Output shape (TransformerFeatureExtractor):", out.shape)
    # assert out.shape == (B, 128), "TransformerFeatureExtractor output shape mismatch"

