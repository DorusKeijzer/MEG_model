import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class BasicCNN(nn.Module):
    def __init__(self, out_features=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8,8)),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)),
        )
        self.fc = nn.Linear(128 * 4 * 4, out_features)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.ReLU(),
            nn.Linear(embed_dim*4, embed_dim)
        )
        self.ff_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        ff_out = self.ff(x)
        x = self.ff_norm(x + ff_out)
        return x

class AttentionCNN(nn.Module):
    def __init__(self, out_features=256, embed_dim=256):
        super().__init__()
        # same basic CNN layers as BasicCNN but smaller output for attn
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8,8)),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)),
        )
        self.fc = nn.Linear(128 * 4 * 4, embed_dim)
        self.attn_block = AttentionBlock(embed_dim)
        self.out_fc = nn.Linear(embed_dim, out_features)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x).unsqueeze(1)  # Add seq_len=1 for attn block
        x = self.attn_block(x)       # output shape (batch, 1, embed_dim)
        x = x.squeeze(1)
        return self.out_fc(x)

class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim=400, embed_dim=256, num_layers=3, num_heads=4):
        super().__init__()
        # input_dim = 20*20 = 400
        self.embed = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # pool seq_len dimension to 1

    def forward(self, x):
        # x: (batch, seq_len, 20, 20)
        B, T, H, W = x.shape
        x = x.view(B*T, H*W)  # flatten each image
        x = self.embed(x)     # (B*T, embed_dim)
        x = x.view(B, T, -1)  # (B, T, embed_dim)
        x = self.transformer_encoder(x)  # (B, T, embed_dim)
        x = x.transpose(1,2)  # (B, embed_dim, T)
        x = self.pool(x).squeeze(-1)  # (B, embed_dim)
        return x

class LSTMSequenceModel(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def lstm_forward_checkpointed(self, x):
        def custom_forward(*inputs):
            return self.lstm(inputs[0])
        return checkpoint(custom_forward, x, use_reentrant=False)

    def forward(self, x):
        # x shape: (batch, seq_len, feat_dim)
        lstm_out, (h_n, _) = self.lstm_forward_checkpointed(x)
        out = self.dropout(h_n[-1])
        return out  # (batch, hidden_dim)

class TransformerSequenceModel(nn.Module):
    def __init__(self, embed_dim=256, num_layers=3, num_heads=4, dropout=0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        x = self.transformer_encoder(x)  # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)             # (batch, embed_dim, seq_len)
        x = self.pool(x).squeeze(-1)     # (batch, embed_dim)
        return x

class CNNSeqModel(nn.Module):
    def __init__(self, cnn_type='basic', seq_model_type='lstm', lstm_hidden=64, num_classes=4):
        super().__init__()

        # Feature extractor selection
        if cnn_type == 'basic':
            self.feature_extractor = BasicCNN()
        elif cnn_type == 'attention':
            self.feature_extractor = AttentionCNN()
        elif cnn_type == 'transformer':
            # TransformerFeatureExtractor expects raw images with shape (B, T, 20, 20)
            self.feature_extractor = TransformerFeatureExtractor()
        else:
            raise ValueError("cnn_type must be one of ['basic', 'attention', 'transformer']")

        # Sequence model selection
        if seq_model_type == 'lstm':
            self.sequence_model = LSTMSequenceModel(input_dim=256, hidden_dim=lstm_hidden)
            seq_out_dim = lstm_hidden
        elif seq_model_type == 'transformer':
            self.sequence_model = TransformerSequenceModel(embed_dim=256)
            seq_out_dim = 256
        else:
            raise ValueError("seq_model_type must be one of ['lstm', 'transformer']")

        self.fc = nn.Linear(seq_out_dim, num_classes)

    def forward(self, x):
        # If transformer feature extractor: input shape (B, T, 20, 20)
        # else: iterate over time steps and stack features

        if isinstance(self.feature_extractor, TransformerFeatureExtractor):
            features = self.feature_extractor(x)  # shape (B, embed_dim)
            # We skip sequence model, treat whole sequence as input to transformer
            seq_out = features
        else:
            B, T, H, W = x.shape
            # flatten time dim for CNN
            x_reshaped = x.view(B*T, 1, H, W)
            feats = self.feature_extractor(x_reshaped)  # (B*T, feat_dim)
            feats = feats.view(B, T, -1)  # (B, T, feat_dim)
            seq_out = self.sequence_model(feats)

        return self.fc(seq_out)

