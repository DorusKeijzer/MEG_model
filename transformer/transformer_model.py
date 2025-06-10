import torch
from torch import nn

class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        seq_length: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)