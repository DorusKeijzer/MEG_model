import torch
from CNN.cnn_lstm_model import CNNLSTMModel
from transformer.transformer_model import TransformerModel

# Example input
batch_size = 2
n_sensors = 248
n_timesteps = 3563
cnn_lstm_model = CNNLSTMModel(input_size=n_sensors)
x = torch.randn(batch_size, n_sensors, n_timesteps)
output = cnn_lstm_model(x)
print("Output shape:", output.shape)

transformer_model = TransformerModel(input_size=n_sensors, seq_length=n_timesteps)
x = torch.randn(batch_size, n_sensors, n_timesteps)
output = transformer_model(x)
print("Output shape:", output.shape)
