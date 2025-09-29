import torch
import torch.nn as nn

class SimpleCNNModel(nn.Module):
    def __init__(self, input_channels, num_agents=50, seq_len=50, hidden_size=64, dropout=0.3):
        super(SimpleCNNModel, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(num_agents, hidden_size, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()  # shape: (B, hidden_size * new_seq_len * input_channels)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * ((seq_len // 4) + 1) * input_channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2 * 60)
        )

    def forward(self, x):
        x = x.reshape(-1, 50, 50, 6)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x.view(x.size(0), 60, 2)  # (B, 60, 2)