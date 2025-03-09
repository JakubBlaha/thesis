import torch
import torch.nn as nn


class EEG_LSTMClassifier(nn.Module):
    def __init__(
            self,
            *,
            input_size,        # Number of channels or features per time step
            num_classes,       # Number of output classes
            dropout=0.3,       # Increased default dropout for EEG noise
            hidden_sizes=[64, 32, 16],  # Configurable LSTM hidden units
            bidirectional=False,        # Option for bidirectional LSTM
            use_attention=False         # Option for attention mechanism
    ):
        super(EEG_LSTMClassifier, self).__init__()

        # Validate input
        assert len(hidden_sizes) >= 1, "At least one hidden size must be provided"

        # Store parameters
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Define LSTM layers dynamically
        lstm_layers = []
        in_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            lstm = nn.LSTM(
                in_size,
                hidden_size,
                batch_first=True,
                bidirectional=bidirectional
            )
            lstm_layers.append(lstm)
            # Double size if bidirectional
            in_size = hidden_size * (2 if bidirectional else 1)
            if i < len(hidden_sizes) - 1:  # Dropout between layers
                lstm_layers.append(nn.Dropout(dropout))

        self.lstm_layers = nn.ModuleList(lstm_layers)

        # Optional attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=in_size,
                num_heads=2,  # Adjust based on in_size divisibility
                batch_first=True
            )

        # Dense classifier
        final_lstm_size = in_size
        self.classifier = nn.Sequential(
            nn.Linear(final_lstm_size, final_lstm_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_lstm_size // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, input_size, seq_len)

        # Swap seq_len and input_size dimensions to match LSTM
        x = x.permute(0, 2, 1)

        # Pass through LSTM layers
        lstm_out = x
        for i, layer in enumerate(self.lstm_layers):
            if isinstance(layer, nn.LSTM):
                lstm_out, _ = layer(lstm_out)
            else:  # Dropout
                lstm_out = layer(lstm_out)

        # Optional attention
        if self.use_attention:
            # Shape for attention: (batch_size, seq_len, embed_dim)
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Take the last time step or pool across time
            attn_out = attn_out[:, -1, :]
            features = attn_out
        else:
            # Take the last time step
            features = lstm_out[:, -1, :]

        # Pass through classifier
        out = self.classifier(features)
        return out
