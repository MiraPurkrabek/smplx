import os
import argparse
import json
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
import matplotlib.pyplot as plt

from pose_and_view_generation import random_camera_pose
from visualizations import draw_points_on_sphere


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default=None)
    parser.add_argument('--distance', action="store_true", default=False,
                        help='If True, will plot distance from origin instead of position on sphere')
    parser.add_argument('--heatmap', action="store_true", default=False,
                        help='If True, will plot 2D heatmap instead of 3D scatter plot')
    return parser.parse_args()


def interpolate_sphere(pts, score):

    radius = np.mean(np.linalg.norm(pts, axis=1))

    # data_phi = np.arccos(pts[:, 1] / radius)
    # data_theta = np.arctan2(pts[:, 2], pts[:, 0])
    # xy = pts[:, 0]**2 + pts[:, 1]**2
    # data_phi = np.arctan2(pts[:, 1], pts[:, 0])
    # data_theta = np.arctan2(pts[:, 2], np.sqrt(xy)) # for elevation angle defined from XY-plane up]    

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    data_theta = np.arctan2(y, x)
    data_phi = np.arctan2(np.sqrt(x * x + y * y), z)
    
    # print("Data theta:", np.min(data_theta), np.max(data_theta))
    # print("Data phi:", np.min(data_phi), np.max(data_phi))
    # print("Score:", np.min(score), np.max(score))
    # print("=====================================")

    theta = np.linspace(-np.pi, np.pi, 500)
    phi = np.linspace(0, np.pi, 250)
    PHI, THETA = np.meshgrid(phi, theta)  # 2D grid for interpolation
    interp = LinearNDInterpolator(list(zip(data_phi, data_theta)), score)
    SCORE = interp(PHI, THETA)

    # print("Theta:", np.min(theta), np.max(theta))
    # print("Phi:", np.min(phi), np.max(phi))
    # print("Score:", np.min(SCORE), np.max(SCORE))

    # y = radius * np.sin(beta)
    # a = radius * np.cos(beta)
    # x = a * np.sin(alpha)
    # z = a * np.cos(alpha)
    xx = radius * np.cos(THETA) * np.sin(PHI)
    yy = radius * np.sin(THETA) * np.sin(PHI)
    zz = radius * np.cos(PHI)

    # print(x.shape, y.shape, z.shape, SCORE.shape)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_surface(xx, yy, zz, facecolors=plt.cm.jet(SCORE))
    # plt.show()
    

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

    plt.pcolormesh(PHI, THETA, SCORE, shading='auto')
    # plt.plot(data_phi, data_theta, "ok", label="input point")
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
        for _ in range(1000):
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
            f = interp1d(dist, score)
            x = np.linspace(np.min(dist), np.max(dist), 1000)
            y = f(x)
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
        if args.distance:
            dist = np.linalg.norm(pts, axis=1)
            sorted_dist = np.sort(dist)
            plt.hist(sorted_dist, bins=100)
            plt.xlabel("Distance from origin")
            plt.ylabel("Number of points in bin")
            plt.grid()
            plt.show()
        else:
            draw_points_on_sphere(pts)


if __name__ == "__main__":
    args = parse_args()
    main(args)