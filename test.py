import torch
from CNN.cnn_lstm_model import CNNSeqModel

# Define all options
cnn_types = ['basic', 'attention', 'transformer']
seq_models = ['lstm', 'transformer']

batch_size = 2
height = 20
width = 20
seq_length = 3563

x = torch.randn(batch_size, seq_length, height, width)

for cnn_type in cnn_types:
    for seq_model_type in seq_models:
        # Skip invalid combo: transformer cnn_type uses internal sequence processing, seq_model ignored
        if cnn_type == 'transformer' and seq_model_type != 'lstm':
            continue

        print(f"\nTesting model with CNN='{cnn_type}' and Sequence Model='{seq_model_type}'")
        model = CNNSeqModel(cnn_type=cnn_type, seq_model_type=seq_model_type)
        output = model(x)
        print(f"Output shape: {output.shape}")

