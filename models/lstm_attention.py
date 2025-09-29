import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, output_dim=60*2, num_agents=50):
        super(LSTMAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        
        # Shared LSTM for processing all agents' trajectories
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Attention mechanism to combine agents' influences
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Final prediction layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # x2 for ego + context

    def forward(self, x):
        x = x.reshape(-1, 50, 6)
        
        lstm_out, _ = self.lstm(x)  # (B*N, 50, H)
        last_hidden = lstm_out[:, -1, :].view(x.shape[0] // 50, self.num_agents, self.hidden_dim)
        
        # Separate ego features and other agents
        ego_hidden = last_hidden[:, 0, :].unsqueeze(1)  # (B, 1, H)
        others_hidden = last_hidden[:, 1:, :]  # (B, N-1, H)
        
        context, _ = self.attention(
            ego_hidden, others_hidden, others_hidden,
            need_weights=False
        )
        
        # Combine ego features with context
        combined = torch.cat([ego_hidden.squeeze(1), context.squeeze(1)], dim=1)
        
        # Final prediction
        out = self.fc(combined)
        
        return out.view(-1, 60, 2)