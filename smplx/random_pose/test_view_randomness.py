import os
import argparse
import json
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
import matplotlib.pyplot as plt

from smplx.random_pose.pose_and_view_generation import random_camera_pose
from smplx.random_pose.visualizations import draw_points_on_sphere

from smplx.view_regressor.data_processing import c2s
from smplx.view_regressor.visualizations import plot_heatmap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default=None)
    parser.add_argument('--distance', action="store_true", default=False,
                        help='If True, will plot distance from origin instead of position on sphere')
    parser.add_argument('--histogram', action="store_true", default=False,
                        help='If True, will plot distance from origin instead of position on sphere')
    parser.add_argument('--heatmap', action="store_true", default=False,
                        help='If True, will plot 2D heatmap instead of 3D scatter plot')
    return parser.parse_args()


def interpolate_sphere(pts, score):

    pts_sph = c2s(pts)
    radiuses = pts_sph[:, 0]
    radius = np.mean(radiuses)
    data_theta = pts_sph[:, 1]
    data_phi = pts_sph[:, 2]

    theta = np.linspace(0, np.pi, 250)
    phi = np.linspace(-np.pi, np.pi, 500)
    PHI, THETA = np.meshgrid(phi, theta)  # 2D grid for interpolation
    interp = LinearNDInterpolator(list(zip(data_phi, data_theta)), score)
    SCORE = interp(PHI, THETA)    

    significant_points = {
        "TOP": (np.pi/2, np.pi/2, "ro"),
        "BOTTOM": (np.pi/2, -np.pi/2, "rx"),
        "FRONT": (0, 0, "bo"),
        "BACK": (np.pi, 0, "bx"),
        "LEFT": (np.pi/2, 0, "co"),
        "RIGHT": (np.pi/2, np.pi, "cx"),
    }

    plt.pcolormesh(THETA, PHI, SCORE, shading='auto')
    for key, sp in significant_points.items():
        mkr = sp[2]
        plt.plot(sp[0], sp[1], mkr, label=key)
    plt.plot()
    plt.legend()
    plt.colorbar()
    plt.axis("equal")
    plt.xlabel("phi")
    plt.ylabel("theta")
    plt.title("Interpolated heatmap, average distance = {}".format(radius))
    plt.savefig(os.path.join(
        "images",
        "heatmaps",
        "heatmap_distance_{:.1f}.png".format(radius)
    ))
    plt.show()


def main(args):

    pts = []
    score = []
    have_score = True
    if args.filepath is None:
        have_score = False
        for _ in range(50000):
            _, pt, _ = random_camera_pose(distance=-1, view_preference=None, return_vectors=True)
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
    score = np.clip(score, 0, 1)
    
    if have_score:
        if args.distance:
            dist = np.linalg.norm(pts, axis=1)
            sort_idx = np.argsort(dist)
            sorted_dist = dist[sort_idx]
            sorted_score = score[sort_idx]
            window_size = 50
            tmp = np.convolve(sorted_score, np.ones(window_size)/window_size, mode='valid')
            tmp_x = np.linspace(np.min(sorted_dist), np.max(sorted_dist), len(tmp))
            plt.scatter(dist, score)
            plt.xlabel("Distance from origin")
            plt.ylabel("OKS score")
            plt.plot(tmp_x, tmp, "r-")
            plt.legend(["OKS score", "Moving average"])
            plt.grid()
            plt.show()
        elif args.heatmap:
            interpolate_sphere(pts, score)
        else:
            draw_points_on_sphere(pts, score=score)
    else:
        if args.histogram:
            dist = np.linalg.norm(pts, axis=1)
            spherical = cartesian_to_spherical(pts)
            theta = spherical[:, 1]
            phi = spherical[:, 2]
            
            fig, (ax0, ax1, ax2) = plt.subplots(1, 3)

            # Histogram of distance
            ax0.hist(dist, bins=100)
            ax0.set_xlabel("Distance from origin")
            ax0.set_ylabel("Number of points in bin")
            ax0.grid()

            # Histogram of theta
            ax1.hist(theta, bins=100)
            ax1.set_xlabel("Theta")
            ax1.grid()

            # Histogram of phi
            ax2.hist(phi, bins=100)
            ax2.set_xlabel("Phi")
            ax2.grid()

            plt.show()
        elif args.heatmap:
            plot_heatmap(pts)
        else:
            draw_points_on_sphere(pts)


if __name__ == "__main__":
    args = parse_args()
    main(args)