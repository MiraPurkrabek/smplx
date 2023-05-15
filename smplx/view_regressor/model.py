import numpy as np
import torch
import torch.nn as nn

class RegressionModel(nn.Module):
    def __init__(self, output_size=3):
        super(RegressionModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(51, 51),
            nn.ReLU(),
            nn.Linear(51, 51),
            nn.ReLU(),
            nn.Linear(51, 51),
            nn.ReLU(),
            nn.Linear(51, 51),
            nn.ReLU(),
            nn.Linear(51, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
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

    def forward(self, pred_rad, target_rad):
        
        # Subtract pi/2 from theta to simulate latitude
        pred_rad[:, 0] = pred_rad[:, 0] - np.pi/2
        target_rad[:, 0] = target_rad[:, 0] - np.pi/2

        print("---")
        print("pred", torch.min(pred_rad, dim=0).values, torch.max(pred_rad, dim=0).values)
        print("target", torch.min(target_rad, dim=0).values, torch.max(target_rad, dim=0).values)

        # Calculate spherical distance
        delta = pred_rad - target_rad
        a = torch.sin(delta[:, 0] / 2) ** 2 + torch.cos(pred_rad[:, 0]) * torch.cos(target_rad[:, 0]) * torch.sin(delta[:, 1] / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        distance = c

        # Apply reduction
        if self.reduction == 'mean':
            loss = distance.mean()
        elif self.reduction == 'sum':
            loss = distance.sum()
        else:
            loss = distance

        return loss
