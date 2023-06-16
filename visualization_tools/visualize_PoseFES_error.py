import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, interp1d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str)
    parser.add_argument("--heatmap", action="store_true", default=False)
    return parser.parse_args()


def show_heatmap(bbox_centers, scores, heatmap=True):
    data_x = bbox_centers[:, 0]
    data_y = bbox_centers[:, 1]
    
    # Normalize x and y between 0 and 1
    data_x -= np.min(data_x)
    data_x /= np.max(data_x)
    data_y -= np.min(data_y)
    data_y /= np.max(data_y)
    data_y = 1-data_y           # Flip y-axis to correspond to image coordinates

    # Filter out invalid scores
    valid_scores = ~ np.isnan(scores)
    data_x = data_x[valid_scores]
    data_y = data_y[valid_scores]
    scores = scores[valid_scores]

    if heatmap:
        mesh_x = np.linspace(0, 1, 100)
        mesh_y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(mesh_x, mesh_y)  # 2D grid for interpolation
        interp = LinearNDInterpolator(list(zip(data_x, data_y)), scores)
        SCORE = interp(X, Y)    

        plt.pcolormesh(X, Y, SCORE, shading='auto', cmap="jet", vmin=0.0, vmax=1)
        # plt.plot()
        plt.colorbar()
        plt.axis("equal")
    else:
        print(np.min(scores), np.max(scores))
        plt.scatter(data_x, data_y, c=scores, cmap="jet", vmin=0.0, vmax=1)
        # Or if you want different settings for the grids:
        plt.colorbar()
        
    
    plt.grid(which='both', alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Interpolated heatmap, mean = {:.2f}".format(np.mean(scores)))


def main(args):

    coco_dict = json.load(open(args.filepath, "r"))

    bbox_centers = []
    scores = []

    # For each annotation in the COCO dataset, find the center of its bounding box
    # and corresponding OKS score
    for annotation in coco_dict["annotations"]:
        bbox_wh = np.array(annotation["bbox"])
        bbox_center = bbox_wh[:2] + bbox_wh[2:] / 2
        bbox_centers.append(bbox_center)
        if "oks_score" in annotation.keys():
            oks = annotation["oks_score"]
        elif "oks" in annotation.keys():
            oks = annotation["oks"]
        else:
            oks = 1
        scores.append(oks)

    bbox_centers = np.array(bbox_centers)
    scores = np.array(scores)
    show_heatmap(bbox_centers, scores, heatmap=args.heatmap)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)