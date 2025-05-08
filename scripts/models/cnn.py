import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, num_classes, dropout=0.5, num_channels=14):
        super(EEGNet, self).__init__()

        print("N classes: ", num_classes)

        n_feats_base = 10
        feat_grow_rate = 2

        n_feats_a = int(n_feats_base)
        n_feats_b = int(n_feats_base * feat_grow_rate)
        n_feats_c = int(n_feats_base * (feat_grow_rate ** 2))
        n_feats_d = int(n_feats_base * (feat_grow_rate ** 3))

        self.model = nn.Sequential(
            # Layer 1
            nn.Conv1d(
                in_channels=num_channels, out_channels=n_feats_a,
                kernel_size=7, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(n_feats_a),
            nn.Dropout(p=dropout),

            # Layer 2
            nn.Conv1d(in_channels=n_feats_a, out_channels=n_feats_b,
                      kernel_size=3, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(n_feats_b),
            nn.AvgPool1d(kernel_size=2),

            # Layer 3
            nn.Conv1d(in_channels=n_feats_b, out_channels=n_feats_c,
                      kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            # Layer 4
            nn.Conv1d(in_channels=n_feats_c, out_channels=n_feats_d,
                      kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(n_feats_d),
            nn.AvgPool1d(kernel_size=2),

            # Global Average Pooling
            nn.AdaptiveAvgPool1d(1),

            # Flatten
            nn.Flatten(),

            # Linear Layer
            nn.Linear(n_feats_d, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)
