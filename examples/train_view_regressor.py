import os
import argparse
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the model
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(51, 51),
            nn.ReLU(),
            nn.Linear(51, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument('--views-filename', type=str, default="views_w_oks.json",
                        help='Filename of the views file')
    parser.add_argument('--coco-filename', type=str, default="person_keypoints_val2017.json",
                        help='Filename of the coco annotations file')
    return parser.parse_args()


def main(args):
    views_filepath = os.path.join(args.folder, args.views_filename)
    coco_filepath = os.path.join(args.folder, args.coco_filename)

    views_dict = json.load(open(views_filepath, "r"))
    coco_dict = json.load(open(coco_filepath, "r"))

    keypoints = []
    positions = []

    for annot in coco_dict["annotations"]:
        image_id = annot["image_id"]
        image_name = "{:d}.jpg".format(image_id)
        kpts = np.array(annot["keypoints"])

        # Resahpe keypoints to Nx3
        # kpts = kpts.reshape(-1, 3)

        view = views_dict[image_name]
        camera_pos = view["camera_position"]

        keypoints.append(kpts)
        positions.append(camera_pos)

    keypoints = np.array(keypoints)
    positions = np.array(positions)

    # Split into train and test
    train_idx = np.random.choice(len(keypoints), int(0.8*len(keypoints)), replace=False)
    test_idx = np.setdiff1d(np.arange(len(keypoints)), train_idx)
    train_keypoints = torch.from_numpy(keypoints[train_idx, :]).type(torch.float32)
    train_positions = torch.from_numpy(positions[train_idx, :]).type(torch.float32)
    test_keypoints = torch.from_numpy(keypoints[test_idx, :]).type(torch.float32)
    test_positions = torch.from_numpy(positions[test_idx, :]).type(torch.float32)

    # Define the model, loss function, and optimizer
    model = RegressionModel()
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    print("Number of parameters: {}".format(model.count_parameters()))
    print("Number of training samples: {}".format(len(train_keypoints)))
    print("Ratio pf training samples to parameters: {:.2f}".format(len(train_keypoints)/model.count_parameters()))
    print("Number of test samples: {}".format(len(test_keypoints)))

    # Train the model
    num_epochs = 300
    for epoch in range(num_epochs):
        
        # Forward pass
        y_pred = model(train_keypoints)
        loss = criterion(y_pred, train_positions)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch+1) % int(num_epochs/10) == 0:
            print("+---------------------------+")
            print("Epoch [{}/{}]".format(epoch+1, num_epochs))
            print("Loss: {:.4f}".format(loss.item()))

            y_test_pred = model(test_keypoints)
            test_loss = criterion(y_test_pred, test_positions)
            print("Test loss: {:.4f}".format(test_loss.item()))
            
    # Test the model on new data
    print("=================================")
    print("=================================")
    y_test_pred = model(test_keypoints)
    # criterion = nn.MSELoss(reduction="none")
    # test_loss = criterion(y_test_pred, test_positions).detach().numpy()
    test_loss = y_test_pred.detach().numpy() - test_positions.detach().numpy()
    test_dist = np.linalg.norm(test_loss, axis=1)
    print("Test loss:")
    print("min: {:.4f}".format(np.min(test_loss)))
    print("max: {:.4f}".format(np.max(test_loss)))
    print("mean: {:.4f}".format(np.mean(test_loss)))
    print("---\nTest dist:")
    print("min: {:.4f}".format(np.min(test_dist)))
    print("max: {:.4f}".format(np.max(test_dist)))
    print("mean: {:.4f}".format(np.mean(test_dist)))

    
if __name__ == "__main__":
    args = parse_args()
    main(args)
