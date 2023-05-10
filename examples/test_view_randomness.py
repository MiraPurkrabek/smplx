import argparse
import json
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt

from pose_and_view_generation import random_camera_pose
from visualizations import draw_points_on_sphere


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default=None)
    parser.add_argument('--distance', action="store_true", default=False,
                        help='If True, will plot distance from origin instead of position on sphere')
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

    data_theta = np.arctan2(np.sqrt(x * x + y * y), z)
    data_phi = np.arctan2(y, x)
    
    phi = np.linspace(0, 2*np.pi, 200)
    theta = np.linspace(-np.pi, np.pi, 200)
    PHI, THETA = np.meshgrid(phi, theta)  # 2D grid for interpolation
    interp = LinearNDInterpolator(list(zip(data_phi, data_theta)), score)
    SCORE = interp(PHI, THETA)

    # y = radius * np.sin(beta)
    # a = radius * np.cos(beta)
    # x = a * np.sin(alpha)
    # z = a * np.cos(alpha)
    x = radius * np.cos(THETA) * np.sin(PHI)
    y = radius * np.sin(THETA) * np.sin(PHI)
    z = radius * np.cos(PHI)

    print(x.shape, y.shape, z.shape, SCORE.shape)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=plt.cm.jet(SCORE))
    plt.show()

    
    # plt.pcolormesh(PHI, THETA, SCORE, shading='auto')
    # # plt.plot(x, y, "ok", label="input point")
    # plt.legend()
    # plt.colorbar()
    # plt.axis("equal")
    # plt.show()


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
    
    if args.distance:
        dist = np.linalg.norm(pts, axis=1)
        # sort_idx = np.argsort(dist)
        # pts_sorted = pts[sort_idx, :]
        # score_sorted = score[sort_idx]
        # dist_sorted = dist[sort_idx]
        plt.scatter(dist, score)
        plt.grid()
        plt.show()
    else:
        if have_score:
            draw_points_on_sphere(pts, score=score)
        else:
            draw_points_on_sphere(pts)
        # interpolate_sphere(pts, score)


if __name__ == "__main__":
    args = parse_args()
    main(args)