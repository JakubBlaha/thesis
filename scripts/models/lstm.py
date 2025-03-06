import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
            self, *, input_size, num_classes, dropout):
        super(LSTMClassifier, self).__init__()
        # First LSTM layer with 64 units
        self.lstm1 = nn.LSTM(
            input_size, 64, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        # Second LSTM layer with 32 units
        self.lstm2 = nn.LSTM(
            64, 32, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        # Dense layers using Sequential
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        # First LSTM layer (return_sequences=True in Keras)
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)

        # Second LSTM layer (return_sequences=False in Keras)
        lstm2_out, _ = self.lstm2(lstm1_out)
        # Take only the last time step (equivalent to return_sequences=False)
        lstm2_out = lstm2_out[:, -1, :]
        lstm2_out = self.dropout2(lstm2_out)

        # Pass through the sequential classifier
        out = self.classifier(lstm2_out)

        return out
