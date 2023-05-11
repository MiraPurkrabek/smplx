import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
from scipy.interpolate import LinearNDInterpolator


from train_view_regressor import RegressionModel, cartesian_to_spherical, spherical_to_cartesian

def plot_testing_data(y_test_pred):
    
    # Plot the predicted positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(y_test_pred[:, 0], y_test_pred[:, 1], y_test_pred[:, 2], label="Predicted positions (n={:d})".format(len(y_test_pred)))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    max_value = np.max(np.abs(y_test_pred))
    axis_size = 1.*max_value
    x_line = np.array([[0, axis_size], [0, 0], [0, 0]])
    ax.plot(x_line[0, :], x_line[1, :], x_line[2, :], c='r', linewidth=5)
    y_line = np.array([[0, 0], [0, axis_size], [0, 0]])
    ax.plot(y_line[0, :], y_line[1, :], y_line[2, :], c='g', linewidth=5)
    z_line = np.array([[0, 0], [0, 0], [0, axis_size]])
    ax.plot(z_line[0, :], z_line[1, :], z_line[2, :], c='b', linewidth=5)

    plt.show()


def plot_heatmap(pts):

    spherical = cartesian_to_spherical(pts)

    radiuses = spherical[:, 0]
    radius = np.mean(radiuses)

    data_theta = spherical[:, 1].squeeze()
    data_phi = spherical[:, 2].squeeze()

    print("Mean theta: {:.3f}".format(np.mean(data_theta)))
    print("Mean phi: {:.3f}".format(np.mean(data_phi)))

    # Print theta and phi shape
    print("Theta shape: {}".format(data_theta.shape))
    print("Phi shape: {}".format(data_phi.shape))

    significant_points = {
        "TOP": (np.pi/2, np.pi/2, "ro"),
        "BOTTOM": (np.pi/2, -np.pi/2, "rx"),
        "FRONT": (0, 0, "bo"),
        # "FRONT": (0, np.pi/2),
        # "FRONT": (0, np.pi),
        # "FRONT": (0, -np.pi/2),
        "BACK": (np.pi, 0, "bx"),
        "LEFT": (np.pi/2, 0, "co"),
        "RIGHT": (np.pi/2, np.pi, "cx"),
    }

    # plt.hexbin(data_phi, data_theta, gridsize=100, bins="log")
    plt.hist2d(data_phi, data_theta, bins=100)
    # plt.scatter(data_phi, data_theta, label="data")
    for key, sp in significant_points.items():
        mkr = sp[2]
        plt.plot(sp[0], sp[1], mkr, label=key)
    plt.plot()
    plt.legend()
    plt.colorbar()
    plt.axis("equal")
    plt.xlabel("phi")
    plt.ylabel("theta")
    plt.title("Distribution of samples, average distance = {:.2f}".format(radius))
    plt.savefig(os.path.join(
        "images",
        "heatmaps",
        "heatmap_distance_{:.1f}.png".format(radius)
    ))
    plt.show()


def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('coco_filepath', type=str,
                        help='Filename of the coco annotations file')
    parser.add_argument('--load-from', type=str, default="regression_model.pt",
                        help='Path to the model to load')
    parser.add_argument('--num-images', type=int, default=-1,
                        help='Number of images to evaluate. Default is all images')
    parser.add_argument('--plot-3d', action='store_true', default=False,
                        help='Whether to plot 3D coordinates')
    
    return parser.parse_args()


def main(args):

    coco_dict = json.load(open(args.coco_filepath, "r"))

    image_ids = []
    keypoints = []

    for annot in coco_dict["annotations"]:
        image_id = annot["image_id"]
        image_ids.append(image_id)
        kpts = np.array(annot["keypoints"])

        # Resahpe keypoints to Nx3
        kpts_readable = kpts.reshape(-1, 3)
        if np.sum(kpts_readable[:, 2]) < 5:
            continue

        keypoints.append(kpts)

    keypoints = np.array(keypoints)
    image_ids = np.array(image_ids)

    keypoints = torch.from_numpy(keypoints).float()

    # If the number of images is specified, only use that many random images
    if args.num_images > 0:
        select_idx = np.random.choice(len(keypoints), size=args.num_images, replace=False)
        keypoints = keypoints[select_idx, :]
        image_ids = image_ids[select_idx]
    

    # Define the model, loss function, and optimizer
    model = RegressionModel(output_size = 3)

    model.load_state_dict(torch.load(args.load_from))
            
    # Test the model on new data
    print("=================================")
    y_test_pred = model(keypoints).detach().numpy()
    test_radius = np.linalg.norm(y_test_pred, axis=1)
    print("Test positions:")
    print("min: {}".format(np.min(y_test_pred, axis=0)))
    print("max: {}".format(np.max(y_test_pred, axis=0)))
    print("mean: {}".format(np.mean(y_test_pred, axis=0)))
    print("---\nTest radiuses:")
    print("min: {}".format(np.min(test_radius)))
    print("max: {}".format(np.max(test_radius)))
    print("mean: {}".format(np.mean(test_radius)))

    if args.plot_3d:
        plot_testing_data(y_test_pred)
    else:
        plot_heatmap(y_test_pred)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
