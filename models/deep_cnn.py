import torch
import torch.nn as nn

class DeepCNNModel(nn.Module):
    def __init__(self, input_channels, num_agents=50, seq_len=50, hidden_size=128, dropout=0.3):
        super(DeepCNNModel, self).__init__()
        
        # Stage 1: Compress seq_len 50 → 25
        self.conv1 = nn.Conv2d(
            in_channels=num_agents, 
            out_channels=hidden_size, 
            kernel_size=(5, 1),      
            stride=(2, 1),
            padding=(2,0)
        )
        self.bn_conv1 = nn.BatchNorm2d(hidden_size)
        
        # Stage 2: Compress seq_len 25 → 12
        self.conv2_downsample = nn.Conv2d(
            in_channels=hidden_size, 
            out_channels=hidden_size, 
            kernel_size=(5, 1), 
            stride=(2, 1),
            padding=(1,0)
        )
        self.bn_conv2_ds = nn.BatchNorm2d(hidden_size)
        
        self.conv3_downsample = nn.Conv2d(
            in_channels=hidden_size, 
            out_channels=hidden_size, 
            kernel_size=(5, 1),     # Reduce from 12 → 6
            stride=(2, 1),          
            padding=(2,0)
        )
        self.bn_conv3_ds = nn.BatchNorm2d(hidden_size)

        self.conv4_downsample = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=(6, 1),
            stride=(1, 1)
        )
        self.bn_conv4_ds = nn.BatchNorm2d(hidden_size)

        self.conv2 = nn.Conv2d(
            in_channels=hidden_size, 
            out_channels=hidden_size, 
            kernel_size=(1, 1)       
        )
        self.bn_conv2 = nn.BatchNorm2d(hidden_size)
        
        self.conv_to_fc = nn.Linear(hidden_size * input_channels, hidden_size)
        self.bn_conv_to_fc = nn.BatchNorm1d(hidden_size)
        
        self.fc1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.bn1 = nn.BatchNorm1d(2 * hidden_size)
        
        self.fc2 = nn.Linear(2 * hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.fc3 = nn.Linear(hidden_size, 2 * 60)
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.reshape(-1, 50, 50, 6)
        
        # Stage 1: seq_len 50 → 25
        x = self.conv1(x)           # (B, hidden_size, 25, input_channels)
        x = self.bn_conv1(x)
        x = torch.relu(x)
        
        # Stage 2: seq_len 25 → 12
        x = self.conv2_downsample(x) # (B, hidden_size, 12, input_channels)
        x = self.bn_conv2_ds(x)
        x = torch.relu(x)
        
        x = self.conv3_downsample(x)  # (B, hidden_size, 6, input_channels)
        x = self.bn_conv3_ds(x)
        x = torch.relu(x)
        
        # Stage 4: seq_len 6 → 1
        x = self.conv4_downsample(x)  # (B, hidden_size, 1, input_channels)
        residual = x
        x = self.bn_conv4_ds(x)
        x = torch.relu(x)
        
        x = self.conv2(x)           # (B, hidden_size, 1, input_channels)
        x = self.bn_conv2(x)
        x += residual
        x = torch.relu(x)

        x = x.squeeze(2)            # (B, hidden_size, input_channels)
        x = x.reshape(x.size(0), -1) # (B, hidden_size * input_channels)
        
        # Continue with original FC layers
        x = self.conv_to_fc(x)
        residual = x
        x = self.bn_conv_to_fc(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = torch.relu(out)
        out = self.fc3(out)
        
        return out.view(out.size(0), 60, 2)  # (B, 60, 2)