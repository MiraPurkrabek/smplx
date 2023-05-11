import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
class RegressionModel(nn.Module):
    def __init__(self, output_size=3):
        super(RegressionModel, self).__init__()
        self.layers = nn.Sequential(
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
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def cartesian_to_spherical(pts):
    # pts: Nx3
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    radius = np.linalg.norm(pts, axis=1)

    theta = np.arctan2(y, x)
    phi = np.arctan2(np.sqrt(x * x + y * y), z)

    return np.stack([radius, theta, phi], axis=1)


def spherical_to_cartesian(pts):
    # pts: Nx3
    radius = pts[:, 0]
    theta = pts[:, 1]
    phi = pts[:, 2]

    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(phi)

    return np.stack([x, y, z], axis=1)


def plot_training_data(epochs, lr, train_loss_log, test_loss_log, test_positions, y_test_pred):
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # Plot the training and test loss
    if not train_loss_log == [] and not test_loss_log == []:
        ax1.plot(np.arange(epochs), train_loss_log, label="Train loss")
        ax1.plot(list(range(0, epochs, int(epochs/10))), test_loss_log, label="Test loss")
        ax1.legend()
        ax1.grid()
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

    # Plot the distance of test samples
    test_dist = np.linalg.norm(test_positions - y_test_pred, axis=1)
    sorted_test_dist = np.sort(test_dist)
    ax2.plot(np.arange(len(sorted_test_dist)), sorted_test_dist, label="Distance of test samples")
    ax2.legend()
    ax2.grid()
    ax2.set_ylabel("L2 Distance")
    
    fig.suptitle("Training data (lr={:.4f}, epochs={:d})".format(lr, epochs))
    plt.show()

    # Plot the predicted positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(test_positions[:, 0], test_positions[:, 1], test_positions[:, 2], label="True positions")
    # ax.scatter(y_test_pred[:, 0], y_test_pred[:, 1], y_test_pred[:, 2], label="Predicted positions")

    # Draw arrows
    for start, end, dist in zip(test_positions, y_test_pred, test_dist):
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color="black", 
                linestyle="dashed", linewidth=dist)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Predicted positions (lr={:.4f})".format(lr))
    ax.legend()

    max_value = np.max(np.abs(np.stack([test_positions, y_test_pred], axis=0)))
    axis_size = 1.5*max_value
    x_line = np.array([[0, axis_size], [0, 0], [0, 0]])
    ax.plot(x_line[0, :], x_line[1, :], x_line[2, :], c='r', linewidth=5)
    y_line = np.array([[0, 0], [0, axis_size], [0, 0]])
    ax.plot(y_line[0, :], y_line[1, :], y_line[2, :], c='g', linewidth=5)
    z_line = np.array([[0, 0], [0, 0], [0, axis_size]])
    ax.plot(z_line[0, :], z_line[1, :], z_line[2, :], c='b', linewidth=5)

    plt.show()


def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument('--views-filename', type=str, default="views_w_oks.json",
                        help='Filename of the views file')
    parser.add_argument('--coco-filename', type=str, default="person_keypoints_val2017.json",
                        help='Filename of the coco annotations file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--spherical', action="store_true", default=False,
                        help='If True, will train the regressor on spherical coordinates ignoring the radius')
    parser.add_argument('--load', action="store_true", default=False,
                        help='If True, will load the model from the checkpoint file')
    
    return parser.parse_args()


def main(args):
    views_filepath = os.path.join(args.folder, args.views_filename)
    coco_filepath = os.path.join(args.folder, args.coco_filename)

    views_dict = json.load(open(views_filepath, "r"))
    coco_dict = json.load(open(coco_filepath, "r"))

    image_ids = []
    keypoints = []
    positions = []

    for annot in coco_dict["annotations"]:
        image_id = annot["image_id"]
        image_ids.append(image_id)
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
    image_ids = np.array(image_ids)

    if args.spherical:
        positions = cartesian_to_spherical(positions)
        positions = positions[:, 1:]  # Ignore the radius

    # Split into train and test
    train_idx = np.random.choice(len(keypoints), int(0.95*len(keypoints)), replace=False)
    test_idx = np.setdiff1d(np.arange(len(keypoints)), train_idx)
    train_keypoints = torch.from_numpy(keypoints[train_idx, :]).type(torch.float32)
    train_positions = torch.from_numpy(positions[train_idx, :]).type(torch.float32)
    train_images = image_ids[train_idx]
    test_keypoints = torch.from_numpy(keypoints[test_idx, :]).type(torch.float32)
    test_positions = torch.from_numpy(positions[test_idx, :]).type(torch.float32)
    test_images = image_ids[test_idx]

    # Define the model, loss function, and optimizer
    model = RegressionModel(
        output_size = 2 if args.spherical else 3,
    )
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

    print("Number of parameters: {}".format(model.count_parameters()))
    print("Number of training samples: {}".format(len(train_keypoints)))
    print("Ratio pf training samples to parameters: {:.2f}".format(len(train_keypoints)/model.count_parameters()))
    print("Number of test samples: {}".format(len(test_keypoints)))

    num_epochs = args.epochs
    train_loss_log = []
    test_loss_log = []
    
    if args.load:
        model.load_state_dict(torch.load("regression_model.pt"))
    else:
        # Train the model
        for epoch in range(num_epochs):
            
            # Forward pass
            y_pred = model(train_keypoints)
            loss = criterion(y_pred, train_positions)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_log.append(loss.item())

            # Print progress
            if (epoch+1) % int(num_epochs/10) == 0:
                print("+---------------------------+")
                print("Epoch [{}/{}]".format(epoch+1, num_epochs))
                print("Loss: {:.4f}".format(loss.item()))

                y_test_pred = model(test_keypoints)
                test_loss = criterion(y_test_pred, test_positions)
                print("Test loss: {:.4f}".format(test_loss.item()))
                test_loss_log.append(test_loss.item())
            
    # Test the model on new data
    print("=================================")
    y_test_pred = model(test_keypoints)
    test_loss = y_test_pred.detach().numpy() - test_positions.detach().numpy()
    test_dist = np.linalg.norm(test_loss, axis=1)
    print("Test dist:")
    print("min: {:.4f}".format(np.min(test_dist)))
    print("max: {:.4f}".format(np.max(test_dist)))
    print("mean: {:.4f}".format(np.mean(test_dist)))
    if args.spherical:
        angle_dist = np.linalg.norm(test_loss[:, 1:], axis=1)
        print("---\nTest dist (last two coordinates):")
        print("min: {:.4f}".format(np.min(angle_dist)))
        print("max: {:.4f}".format(np.max(angle_dist)))
        print("mean: {:.4f}".format(np.mean(angle_dist)))

    sort_idx = np.argsort(test_dist)
    sorted_test_dist = test_dist[sort_idx]
    sorted_test_images = test_images[sort_idx]

    print("---\nBest images:")
    for i in range(10):
        print("Image ID: {:d}, dist: {:.4f}".format(sorted_test_images[i], sorted_test_dist[i]))
    
    print("---\nWorst images:")
    for i in range(1, 11):
        print("Image ID: {:d}, dist: {:.4f}".format(sorted_test_images[-i], sorted_test_dist[-i]))

    plot_training_data(
        args.epochs,
        args.lr,
        train_loss_log,
        test_loss_log,
        test_positions,
        y_test_pred.detach().numpy(),
    )

    # Save the model
    model_filename = "regression_model.pt"
    torch.save(model.state_dict(), model_filename)

    
if __name__ == "__main__":
    args = parse_args()
    main(args)
