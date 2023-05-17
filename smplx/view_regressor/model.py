import numpy as np
import torch
import torch.nn as nn

from data_processing import angular_distance

class RegressionModel(nn.Module):
    def __init__(self, input_size=51, output_size=3, width=32, depth=3):
        super(RegressionModel, self).__init__()
        if width == -1:
            width = 2 ** np.log2(input_size).astype(int)

        if depth < 2:
            depth = 2

        # Width is always a power of 2
        width = 2 ** np.log2(width).astype(int)

        layers_list = [
            nn.Linear(input_size, width),
            nn.ReLU(),
        ]
        hidden_size = 2 ** np.linspace(np.log2(width).astype(int), 4, depth).astype(int)
        for i in range(depth-1):
            in_size = hidden_size[i]
            out_size = hidden_size[i+1]
            layers_list.append(nn.Linear(in_size, out_size))
            layers_list.append(nn.ReLU())

        # Add the last layer
        layers_list.append(nn.Linear(hidden_size[-1], output_size))  

        self.layers = nn.Sequential(*layers_list)

        hidden_size = hidden_size.tolist()
        hidden_size.insert(0, input_size)
        hidden_size.append(output_size)
        print("Created model with layers: {}".format(hidden_size))
        
    def forward(self, x):
        x = self.layers(x)
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
