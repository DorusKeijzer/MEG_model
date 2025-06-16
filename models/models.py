import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

# === Basic CNN Encoder ===
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

# === CNN with Self-Attention ===
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
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        ff_out = self.ff(x)
        return self.ff_norm(x + ff_out)

class AttentionCNN(nn.Module):
    def __init__(self, out_features=256, embed_dim=256):
        super().__init__()
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
        x = self.fc(x).unsqueeze(1)
        x = self.attn_block(x)
        x = x.squeeze(1)
        return self.out_fc(x)

# === Transformer for 2D image sequences ===
class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim=400, embed_dim=256, num_layers=3, num_heads=4):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        B, T, H, W = x.shape
        x = x.view(B*T, H*W)
        x = self.embed(x)
        x = x.view(B, T, -1)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        return self.pool(x).squeeze(-1)

# === LSTM Sequence Model ===
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
        lstm_out, (h_n, _) = self.lstm_forward_checkpointed(x)
        return self.dropout(h_n[-1])

# === Transformer Sequence Model ===
class TransformerSequenceModel(nn.Module):
    def __init__(self, embed_dim=256, num_layers=3, num_heads=4, dropout=0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        return self.pool(x).squeeze(-1)

# === Full Modular Model for Classification ===
class CNNSeqModel(nn.Module):
    def __init__(self, cnn_type='basic', seq_model_type='lstm', lstm_hidden=64,
                 num_classes=4, pretrained_cfg=None):
        super().__init__()
        self.pretrained_cfg = pretrained_cfg or {}

        self.feature_extractor = self._init_feature_extractor(cnn_type)
        self.sequence_model, seq_out_dim = self._init_sequence_model(seq_model_type, lstm_hidden)
        self.fc = nn.Linear(seq_out_dim, num_classes)

    def _init_feature_extractor(self, cnn_type):
        if cnn_type == 'basic':
            model = BasicCNN()
        elif cnn_type == 'attention':
            model = AttentionCNN()
        elif cnn_type == 'transformer':
            model = TransformerFeatureExtractor()
        else:
            raise ValueError("cnn_type must be one of ['basic', 'attention', 'transformer']")

        if 'cnn' in self.pretrained_cfg:
            print(f"[INFO] Loading pretrained CNN weights from {self.pretrained_cfg['cnn']}")
            model.load_state_dict(torch.load(self.pretrained_cfg['cnn']))

        return model

    def _init_sequence_model(self, seq_model_type, lstm_hidden):
        if seq_model_type == 'lstm':
            model = LSTMSequenceModel(input_dim=256, hidden_dim=lstm_hidden)
            if 'lstm' in self.pretrained_cfg:
                print(f"[INFO] Loading pretrained LSTM weights from {self.pretrained_cfg['lstm']}")
                model.load_state_dict(torch.load(self.pretrained_cfg['lstm']))
            return model, lstm_hidden
        elif seq_model_type == 'transformer':
            return TransformerSequenceModel(embed_dim=256), 256
        else:
            raise ValueError("seq_model_type must be one of ['lstm', 'transformer']")

    def forward(self, x):
        if isinstance(self.feature_extractor, TransformerFeatureExtractor):
            features = self.feature_extractor(x)
            seq_out = features
        else:
            B, T, H, W = x.shape
            x_reshaped = x.view(B*T, 1, H, W)
            feats = self.feature_extractor(x_reshaped)
            feats = feats.view(B, T, -1)
            seq_out = self.sequence_model(feats)

        return self.fc(seq_out)
