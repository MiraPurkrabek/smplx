import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, interp1d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("trained", type=str)
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--heatmap", action="store_true", default=False)
    args = parser.parse_args()

    if args.baseline is None:
        args.baseline = os.path.join(
            os.path.dirname(args.trained),
            "coco_dict_with_oks_baseline.json"
        )

    return args


def show_heatmap(bbox_centers, scores, heatmap=False):
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
        mesh_x = np.linspace(0, 1, 50)
        mesh_y = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(mesh_x, mesh_y)  # 2D grid for interpolation
        interp = LinearNDInterpolator(list(zip(data_x, data_y)), scores)
        SCORE = interp(X, Y)    

        plt.pcolormesh(X, Y, SCORE, shading='auto', cmap="jet")
        # plt.plot()
        plt.colorbar()
        plt.axis("equal")
    else:
        plt.scatter(data_x, data_y, c=scores, cmap="jet")#, vmin=-1, vmax=1)
        plt.colorbar()

    plt.grid(which='both', alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Interpolated heatmap, mean = {:.2f}".format(np.mean(scores)))


def main(args):

    coco_dict_baseline = json.load(open(args.baseline, "r"))
    coco_dict_trained = json.load(open(args.trained, "r"))

    image_ids = []
    bbox_ids = []
    bbox_centers = []
    scores = []

    # For each annotation in the COCO dataset, find the center of its bounding box
    # and corresponding OKS score
    for a1, a2 in zip(coco_dict_baseline["annotations"], coco_dict_trained["annotations"]):
        assert a1["image_id"] == a2["image_id"]
        assert a1["id"] == a2["id"]

        image_ids.append(a1["image_id"])
        bbox_ids.append(a1["id"])
        
        bbox_wh = np.array(a1["bbox"])
        bbox_center = bbox_wh[:2] + bbox_wh[2:] / 2
        bbox_centers.append(bbox_center)
        
        if "oks_score" in a1.keys():
            oks1 = a1["oks_score"]
            oks2 = a2["oks_score"]
        elif "oks" in a1.keys():
            oks1 = a1["oks"]
            oks2 = a2["oks"]
        else:
            oks1 = 1
            oks2 = 1
        scores.append(oks2 - oks1)
        # scores.append(int(a1["image_id"]))

    bbox_centers = np.array(bbox_centers)
    scores = np.array(scores)

    nan_mask = np.isnan(scores)
    bbox_centers = bbox_centers[~nan_mask, :]
    scores = scores[~nan_mask]

    print("Max - the images with best improvement")
    max_idx = np.argsort(scores)[::-1]
    for idx in max_idx[:10]:
        print("{:4d}-{:4d}\t{:4.2f}".format(image_ids[idx], bbox_ids[idx], scores[idx]))
    
    print("Min - the images with worse improvement")
    min_idx = np.argsort(scores)
    for idx in min_idx[:10]:
        print("{:4d}-{:4d}\t{:4.2f}".format(image_ids[idx], bbox_ids[idx], scores[idx]))

    show_heatmap(bbox_centers, scores, heatmap=args.heatmap)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)