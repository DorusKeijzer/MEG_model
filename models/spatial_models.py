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
        x = x.mean(dim=1)  # average over time
        return self.out(x)


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ff_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        ff_out = self.ff(x)
        return self.ff_norm(x + ff_out)

class AttentionCNN(nn.Module):
    def __init__(self, embed_dim=256, out_features=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((10, 10)),  # Downsample (20x21) to ~10x10
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(128 * 4 * 4, embed_dim)
        self.attn_block = AttentionBlock(embed_dim)
        self.out_fc = nn.Linear(embed_dim, out_features)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)         # Merge batch + time
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(B, T, -1)               # Reshape to (B, T, embed_dim)
        x = self.attn_block(x)
        return self.out_fc(x.mean(dim=1)) # Global average over time

class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_hw=(20, 21), embed_dim=256, num_layers=3, num_heads=4):
        super().__init__()
        H, W = input_hw
        self.embed = nn.Linear(H * W, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        B, T, C, H, W = x.shape  # (B, T, 1, 20, 21)
        x = x.view(B, T, -1)     # Flatten spatial dims → (B, T, 420)
        x = self.embed(x)        # → (B, T, embed_dim)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)    # → (B, embed_dim, T)
        return self.pool(x).squeeze(-1)  # (B, embed_dim)

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



    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.fc_enc(x)
        x = self.fc_dec(z).view(B * T, 128, 4, 4)
        x = self.decoder(x)
        return x.view(B, T, 1, 20, 21), z.view(B, T, -1)  # Reconstructed & latent


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

