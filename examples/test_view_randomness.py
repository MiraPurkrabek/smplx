import argparse
import json
import numpy as np

from pose_and_view_generation import random_camera_pose
from visualizations import draw_points_on_sphere


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default=None)
    return parser.parse_args()


def main(args):

    pts = []
    score = []
    have_score = True
    if args.filepath is None:
        for _ in range(1000):
            _, pt, _ = random_camera_pose(distance=1.5, view_preference=None, return_vectors=True)
            pts.append(pt)
    else:
        input_dict = json.load(open(args.filepath, "r"))
        for img_name in input_dict.keys():
            pts.append(input_dict[img_name]["camera_position"])
            if "oks_score" in input_dict[img_name].keys():
                score.append(input_dict[img_name]["oks_score"])
            else:
                have_score = False

    pts = np.array(pts)
    score = np.array(score).squeeze()
    
    if have_score:
        draw_points_on_sphere(pts, score=score)
    else:
        draw_points_on_sphere(pts)


if __name__ == "__main__":
    args = parse_args()
    main(args)