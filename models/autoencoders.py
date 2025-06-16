import torch
from torch import nn

class CNNAutoencoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Encoder (same as BasicCNN)
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc_enc = nn.Linear(128 * 4 * 4, embed_dim)

        # Decoder
        self.fc_dec = nn.Linear(embed_dim, 128 * 4 * 4)
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1), nn.ReLU(),  # -> 9x9
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1), nn.ReLU(),   # -> ~18x18
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid()  # -> 20x20
        )

    def forward(self, x):
        # x: (B, 1, 20, 20)
        enc = self.encoder_cnn(x)
        enc_flat = enc.view(x.size(0), -1)
        z = self.fc_enc(enc_flat)

        dec_flat = self.fc_dec(z)
        dec = dec_flat.view(x.size(0), 128, 4, 4)
        out = self.decoder_cnn(dec)
        return out, z  # output and encoded representation

