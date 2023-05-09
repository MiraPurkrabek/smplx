import numpy as np

from pose_and_view_generation import random_camera_pose
from visualizations import draw_points_on_sphere


def main():

    pts = []
    for _ in range(1000):
        _, pt, _ = random_camera_pose(distance=1.5, view_preference=None, return_vectors=True)
        pts.append(pt)

    pts = np.array(pts)
    
    draw_points_on_sphere(pts)


if __name__ == "__main__":
    main()