import torch
import torch.nn as nn

class CNNFrameAutoencoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((8, 8)),
            
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc_enc = nn.Linear(128 * 4 * 4, embed_dim)

        # Decoder
        self.fc_dec = nn.Linear(embed_dim, 128 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 1, 3, padding=1),  
            nn.Upsample(size=(20, 21), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, 1, H, W)

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
