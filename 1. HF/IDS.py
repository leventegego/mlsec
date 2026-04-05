import torch.nn as nn

# Model architecture
class IntrusionDetector(nn.Module):
    """Simple feedforward neural network for intrusion detection."""
    
    def __init__(self, input_dim):
        super(IntrusionDetector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)