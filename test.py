import torch
from CNN.cnn_lstm_model import CNNLSTMModel
from transformer.transformer_model import TransformerModel

# Example input
batch_size = 2
height = 20
width = 20
seq_length = 3563
cnn_lstm_model = CNNLSTMModel()
x = torch.randn(batch_size, seq_length, height, width)
output = cnn_lstm_model(x)
print("Output shape:", output.shape)

# transformer_model = TransformerModel(input_size=n_sensors, seq_length=n_timesteps)
# print("Output shape:", output.shape)
