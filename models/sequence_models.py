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



class TransformerWithDecoder(nn.Module):
    def __init__(self, base_transformer, embed_dim=256):
        super().__init__()
        self.base_transformer = base_transformer  # instance of TemporalTransformer
        self.decoder_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward_for_pretraining(self, embeddings, mask):
        """
        embeddings: Tensor (B, T, D) from CNN encoder
        mask: BoolTensor (B, T), True where input was masked
        """
        B, T, D = embeddings.shape
        x = self.base_transformer.transformer(embeddings)  # (B, T, D)

        decoded = self.decoder_head(x)  # (B, T, D)

        mask = mask.bool()
        masked_preds = decoded[mask]      # (N_masked, D)
        masked_targets = embeddings[mask] # (N_masked, D)

        return masked_preds, masked_targets

    def forward(self, x):
        return self.base_transformer(x)  # classification

