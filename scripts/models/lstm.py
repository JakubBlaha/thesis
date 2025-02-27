import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
            self, input_size, hidden_size, num_layers, num_classes,
            dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


input_size = 64
hidden_size = 128
num_layers = 2
learning_rate = 0.001
batch_size = 32
epochs = 10
