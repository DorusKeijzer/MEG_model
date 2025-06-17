import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2, num_classes=10, dropout=0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # Important: (B, T, D) ordering
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):
        """
        x: Tensor of shape (B, T, D)
        """
        x = self.transformer(x)             # (B, T, D)
        final_token = x[:, -1, :]           # take final timestep, nigga!
        logits = self.classifier(final_token)
        return logits
