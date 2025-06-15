import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class CNNLSTMModel(nn.Module):
    def __init__(self, lstm_hidden: int = 64, num_classes: int = 4, dropout: float = 0.2):
        super().__init__()

        # CNN to extract features from each 20x20 image
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc_cnn = nn.Linear(128 * 4 * 4, 256)

        # Temporal compression: CNN over time axis (1D conv)
        self.temporal_compression = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )

        # LSTM input will be reduced sequence length with same 256-dim features
        self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def lstm_forward_checkpointed(self, x):
        """Memory-efficient LSTM forward pass"""
        def custom_forward(*inputs):
            return self.lstm(inputs[0])
        return checkpoint(custom_forward, x, use_reentrant=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 20, 20)
        B, T, H, W = x.shape

        # Prepare CNN input: (B*T, 1, H, W)
        x = x.view(B * T, 1, H, W)
        x = self.cnn(x)
        x = x.view(B * T, -1)
        x = self.fc_cnn(x)  # Shape: (B*T, 256)

        # Back to temporal format: (B, T, 256)
        x = x.view(B, T, 256)

        # Temporal compression: reshape for Conv1D
        x = x.transpose(1, 2)  # (B, 256, T)
        x = self.temporal_compression(x)  # (B, 256, T_reduced)
        x = x.transpose(1, 2)  # (B, T_reduced, 256)

        # Memory-efficient LSTM
        lstm_out, (h_n, _) = self.lstm_forward_checkpointed(x)

        out = self.dropout(h_n[-1])  # Last layer's final hidden state
        return self.fc(out)

