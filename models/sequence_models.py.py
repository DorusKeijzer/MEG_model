
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

