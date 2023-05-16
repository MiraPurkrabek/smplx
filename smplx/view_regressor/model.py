import numpy as np
import torch
import torch.nn as nn

from data_processing import angular_distance

class RegressionModel(nn.Module):
    def __init__(self, input_size=51, output_size=3):
        super(RegressionModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 51),
            nn.ReLU(),
            # nn.Linear(51, 51),
            # nn.ReLU(),
            # nn.Linear(51, 51),
            # nn.ReLU(),
            # nn.Linear(51, 51),
            # nn.ReLU(),
            nn.Linear(51, 32),
            nn.ReLU(),
            # nn.Linear(32, 32),
            # nn.ReLU(),
            # nn.Linear(32, 32),
            # nn.ReLU(),
            # nn.Linear(32, 32),
            # nn.ReLU(),
            # nn.Linear(32, 32),
            # nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )
        
    def forward(self, x):
        x = self.layers(x)
        # x[:, 0] = torch.clamp(x[:, 0], min=-np.pi/2, max=np.pi/2)
        # x[:, 1] = torch.clamp(x[:, 1], min=-np.pi, max=np.pi)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SphericalDistanceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SphericalDistanceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        
        distance = angular_distance(pred, target, use_torch=True)

        # Apply reduction
        if self.reduction == 'mean':
            loss = distance.mean()
        elif self.reduction == 'sum':
            loss = distance.sum()
        else:
            loss = distance

        return loss
