import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
from scipy.interpolate import LinearNDInterpolator

from smplx.view_regressor.data_processing import s2c, c2s, process_keypoints, load_data_from_coco_file
from model import RegressionModel
from visualizations import plot_testing_data, plot_heatmap

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

    # Load the data
    keypoints, bboxes_xywh, image_ids = load_data_from_coco_file(args.coco_filepath)
    keypoints = process_keypoints(keypoints, bboxes_xywh)

    keypoints = torch.from_numpy(keypoints).float()

    # If the number of images is specified, only use that many random images
    if args.num_images > 0:
        select_idx = np.random.choice(len(keypoints), size=args.num_images, replace=False)
        keypoints = keypoints[select_idx, :]
        image_ids = image_ids[select_idx]

    # Define the model, loss function, and optimizer
    try:
        model = RegressionModel(output_size = 3)
        model.load_state_dict(torch.load(args.load_from))
        is_spherical = False
    except RuntimeError:
        model = RegressionModel(output_size = 2)
        model.load_state_dict(torch.load(args.load_from))
        is_spherical = True

    # Test the model on new data
    print("=================================")
    y_test_pred = model(keypoints).detach().numpy()

    print("Test positions:")
    print("min: {}".format(np.min(y_test_pred, axis=0)))
    print("max: {}".format(np.max(y_test_pred, axis=0)))
    print("mean: {}".format(np.mean(y_test_pred, axis=0)))
    
    if not is_spherical:
        test_radius = np.linalg.norm(y_test_pred, axis=1)
        print("---\nTest radiuses:")
        print("min: {}".format(np.min(test_radius)))
        print("max: {}".format(np.max(test_radius)))
        print("mean: {}".format(np.mean(test_radius)))

    if args.plot_3d:
        plot_testing_data(y_test_pred, is_spherical)
    else:
        plot_heatmap(y_test_pred, is_spherical)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
