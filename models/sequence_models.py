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
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x, padding_mask=None):
        """
        x: Tensor of shape (B, T, D)
        padding_mask: BoolTensor of shape (B, T) where True indicates PAD tokens to be ignored
        """
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # (B, T, D)

        if padding_mask is not None:
            padding_mask_inverted = ~padding_mask  # (B, T)
            masked_sum = (x * padding_mask_inverted.unsqueeze(-1)).sum(dim=1)  # (B, D)
            lengths = padding_mask_inverted.sum(dim=1).clamp(min=1).unsqueeze(-1)  # (B, 1)
            pooled = masked_sum / lengths  # (B, D)
        else:
            pooled = x.mean(dim=1)

        logits = self.classifier(pooled)
        return logits

class TransformerWithDecoder(nn.Module):
    def __init__(self, base_transformer, embed_dim=256):
        super().__init__()
        self.base_transformer = base_transformer
        self.decoder_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward_for_pretraining(self, embeddings, mask, padding_mask=None):
        """
        embeddings: Tensor (B, T, D)
        mask: BoolTensor (B, T), True where input was masked
        padding_mask: BoolTensor (B, T), True where input is padding
        """
        x = self.base_transformer.transformer(embeddings, src_key_padding_mask=padding_mask)  # (B, T, D)
        decoded = self.decoder_head(x)  # (B, T, D)

        masked_preds = decoded[mask]         # (N_masked, D)
        masked_targets = embeddings[mask]    # (N_masked, D)
        return masked_preds, masked_targets

    def forward(self, x, padding_mask=None):
        return self.base_transformer(x, padding_mask=padding_mask)
