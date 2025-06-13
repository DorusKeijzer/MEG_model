import torch
from torch import nn


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size: int, cnn_channels: list = [64, 64], kernel_size: int = 3, 
                 lstm_hidden: int = 64, num_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        self.cnn_channels = cnn_channels
        self.kernel_size = kernel_size
        self.lstm_hidden = lstm_hidden
        self.num_classes = num_classes

        self.conv_layers = nn.ModuleList()

        # Convolutional layers
        self.cnn_l1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding= 'same')
        self.cnn_l2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding='same')
        self.cnn_l3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding='same')
        
        # Adaptive pooling layer
        self.adapool = nn.AdaptiveAvgPool2d((4, 4))  # Output size is (4, 4)
        
        # Fully connected layers
        self.fc_cnn = nn.Linear(128 * 4 * 4, 256)  # maybe tune? or add more layers?

        self.lstm = nn.LSTM(cnn_channels[-1], lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    @property
    def model_parameters(self):
        return f"cnn={self.cnn_channels}, kernel={self.kernel_size}, lstm={self.lstm_hidden}, classes={self.num_classes}"

    @property
    def name(self):
        return f"CNN-LSTM_{self.model_parameters}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        #CNN section
        x = self.cnn_l1(x.unsqueeze(1)) 
        x = self.cnn_l2(x)
        x = self.cnn_l3(x)
        x = self.adapool(x)         
        x = x.view(batch_size, -1) # Flatten
        x = self.fc_cnn(x)            # one FC, maybe we need more and/or dropout?

        #LSTM section
        x = x.view(batch_size, -1, self.cnn_channels[-1])
        _, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])
        return self.fc(out)
