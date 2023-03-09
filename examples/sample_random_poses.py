# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import math
import os.path as osp
import os
import shutil
import argparse

import pyrender
import trimesh            

import numpy as np
import torch
import cv2
from tqdm import tqdm

import json

import smplx
from smplx.joint_names import COCO_JOINTS, COCO_SKELETON, OPENPOSE_SKELETON, OPENPOSE_COLORS

from psbody.mesh import Mesh, MeshViewers

import mesh_to_depth as m2d

TSHIRT_PARTS = ["spine1", "spine2", "leftShoulder", "rightShoulder", "rightArm", "spine", "hips", "leftArm"]
SHIRT_PARTS = ["spine1", "spine2", "leftShoulder", "rightShoulder", "rightArm", "spine", "hips", "leftArm", "leftForeArm", "rightForeArm"]
SHORTS_PARTS = ["rightUpLeg", "leftUpLeg"]
PANTS_PARTS = ["rightUpLeg", "leftUpLeg", "leftLeg", "rightLeg"]
SHOES_PARTS = ["leftToeBase", "rightToeBase", "leftFoot", "rightFoot", ]
SKIN_COLOR = np.array([1.0, 0.66, 0.28, 1.0]) # RGB format


def generate_pose(typical_pose=None, simplicity=5):
    # Front rotation
    # Counter-clockwise rotation
    # Side bend

    joints = {
        "Left leg": 0,
        "Right leg": 1,
        "Torso": 2,
        "Left knee": 3,
        "Right knee": 4,
        "Mid-torso": 5,
        "Left ankle": 6,
        "Right ankle": 7,
        "Chest": 8,
        "Left leg fingers": 9,
        "Right leg fingers": 10,
        "Neck": 11,
        "Left neck": 12,
        "Right neck": 13,
        "Upper neck": 14,
        "Left shoulder": 15,
        "Right shoulder": 16,
        "Left elbow": 17,
        "Right elbow": 18,
        "Left wrist": 19,
        "Right wrist": 20,
    }
    body_pose = torch.zeros((1, len(joints)*3))

    limits_deg = {
        "Left leg": [
            (-90, 45), (-50, 90), (-30, 90),
        ],
        "Right leg": [
            (-90, 45), (-90, 50), (-90, 30),
        ],
        "Torso": [
            (0, 0), (-40, 40), (-20, 20),
        ],
        "Left knee": [
            (0, 150), (0, 0), (0, 0)
        ],
        "Right knee": [
            (0, 150), (0, 0), (0, 0)
        ],
        "Mid-torso": [
            (0, 20), (-30, 30), (-30, 30)
        ],
        "Left ankle": [
            (-20, 70), (-15, 5), (0, 20)
        ],
        "Right ankle": [
            (-20, 70), (-5, 15), (-20, 0)
        ],
        "Chest": [
            (0, 0), (-20, 20), (-10, 10),
        ],
        "Left leg fingers": [
            (0, 0), (0, 0), (0, 0)
        ],
        "Right leg fingers": [
            (0, 0), (0, 0), (0, 0)
        ],
        "Neck": [
            (-45, 45), (-40, 40), (-20, 20)
        ],
        "Left neck": [
            (0, 0), (0, 0), (0, 0)
        ],
        "Right neck": [
            (0, 0), (0, 0), (0, 0)
        ],
        "Upper neck": [
            (-20, 20), (-30, 30), (-10, 10)
        ],
        "Left shoulder": [
            (-30, 30), (-90, 20), (-60, 90)
        ],
        "Right shoulder": [
            (-30, 30), (-20, 90), (-90, 60)
        ],
        "Left elbow": [
            (0, 0), (-120, 0), (0, 0)
        ],
        "Right elbow": [
            (0, 0), (0, 120), (0, 0)
        ],
        "Left wrist": [
            (-30, 30), (-10, 10), (-70, 90)
        ],
        "Right wrist": [
            (-30, 30), (-10, 10), (-90, 70)
        ],
    }
    
    if typical_pose is None:
        # Generate a completely random pose
        min_limits = []
        max_limits = []

        for _, lims in limits_deg.items():
            if len(lims) > 0:
                for l in lims:
                    min_limits.append(l[0])
                    max_limits.append(l[1])
            else:
                min_limits += [0, 0, 0]
                max_limits += [0, 0, 0]

        min_limits = torch.Tensor(min_limits) / simplicity
        max_limits = torch.Tensor(max_limits) / simplicity
        joints_rng = max_limits - min_limits

        random_angles = torch.rand((len(joints)*3)) * joints_rng + min_limits

        # Transfer to radians
        random_angles = random_angles / 180 * np.pi

        body_pose = random_angles.reshape((1, len(joints)*3))
        
    elif typical_pose.lower() == "min":
        for joint, lims in limits_deg.items():
            for li, l in enumerate(lims):
                body_pose[0, joints[joint]*3+li] = l[0] / 180 * np.pi
    
    elif typical_pose.lower() == "max":
        for joint, lims in limits_deg.items():
            for li, l in enumerate(lims):
                body_pose[0, joints[joint]*3+li] = l[1] / 180 * np.pi

    elif typical_pose.lower() == "sit":
        body_pose[0, joints["Right knee"]*3+0] = np.pi / 2
        body_pose[0, joints["Left knee"]*3+0] = np.pi / 2
        body_pose[0, joints["Right leg"]*3+0] = - np.pi / 2
        body_pose[0, joints["Left leg"]*3+0] = - np.pi / 2
        body_pose[0, joints["Left shoulder"]*3+2] = - np.pi / 2 * 4/5
        body_pose[0, joints["Right shoulder"]*3+2] = np.pi / 2 * 4/5

    return body_pose


def random_camera_pose(distance=3):
    t = np.random.rand(3) * 2 - 1

    t_norm = t / np.linalg.norm(t)
    # t = t_norm
    t = t_norm * distance

    a = [0, 0, 1]
    b = t_norm
    v = np.cross(a, b)
    c = np.dot(a, b)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])

    R = np.eye(3) + vx + np.dot(vx, vx) * 1/(1+c)

    pose = np.array([
        [R[0, 0], R[0, 1], R[0, 2], t[0]],
        [R[1, 0], R[1, 1], R[1, 2], t[1]],
        [R[2, 0], R[2, 1], R[2, 2], t[2]],
        [      0,       0,       0,    1],
    ])
    return pose


def get_joints_vertices(joints, vertices, joints_range=None):
    idxs = []

    for ji, j in enumerate(joints):
        dist_to_vs = np.linalg.norm(vertices - j, axis=1)
        sort_min_idxs = np.argsort(dist_to_vs)
        rng = joints_range[ji] if joints_range is not None else 100
        v_idx = sort_min_idxs[:rng]
        idxs.append(v_idx)
        
    return idxs


def get_joints_visibilities(joint_vertices, visibilities):
    n_joints = len(joint_vertices)
    vis = np.zeros((n_joints))
    for ji in range(n_joints):
        vis[ji] = np.any(visibilities[joint_vertices[ji]])
    return vis.astype(bool)


def generate_color(alpha=1.0):
    col = np.random.rand(3)
    col = np.concatenate([col, np.ones(1)*alpha], axis=0)

    return col


def project_to_2d(pts, K, T):
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    points_2d = (K @ np.linalg.inv(T) @ pts_h.T).T
    points_2d[:, 0] = (points_2d[:, 0] / points_2d[:, 3]) * 1024 / 2 + 1024 / 2
    points_2d[:, 1] = (points_2d[:, 1] / -points_2d[:, 3]) * 1024 / 2 + 1024 / 2
    points_2d = points_2d.astype(np.int32)
    points_2d = points_2d[:, :2]
    
    return points_2d


def draw_pose(img, kpts, joints_vis, draw_style="custom"):

    assert draw_style in [
        "custom",
        "openpose",
    ]

    skeleton = COCO_SKELETON

    if draw_style == "openpose":
        # Reorder kpts to OpenPose order
        kpts = kpts.copy()
        kpts = kpts[[0, 0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3], :]
        
        # Compute pelvis as mean of shoulders
        kpts[1, :] = np.mean(kpts[[2, 5], :], axis=1)

        skeleton = OPENPOSE_SKELETON

    print(kpts.shape, len(skeleton))

    for pi, pt in enumerate(kpts):
        
        if draw_style == "openpose":
            img = cv2.circle(
                img,
                tuple(pt.tolist()),
                radius=4,
                color=OPENPOSE_COLORS[pi],
                thickness=-1
            )
        else:
            marker_color = (0, 0, 255) if joints_vis[pi] else (40, 40, 40)
            thickness = 2 if joints_vis[pi] else 1
            marker_type = cv2.MARKER_CROSS
        
            img = cv2.drawMarker(
                img,
                tuple(pt.tolist()),
                color=marker_color,
                markerType=marker_type,
                thickness=thickness
            )

    for bi, bone in enumerate(skeleton):
        b = np.array(bone) - 1 # COCO_SKELETON is 1-indexed
        start = kpts[b[0], :]
        end = kpts[b[1], :]
        if draw_style == "openpose":
            stickwidth = 4
            current_img = img.copy()
            mX = np.mean(np.array([start[0], end[0]]))
            mY = np.mean(np.array([start[1], end[1]]))
            length = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(start[0] - end[0], start[1] - end[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(current_img, polygon, OPENPOSE_COLORS[bi])
            img = cv2.addWeighted(img, 0.4, current_img, 0.6, 0)

        else:
            if not (joints_vis[b[0]] and joints_vis[b[1]]):
                continue
            
            img = cv2.line(
                img,
                start,
                end,
                thickness=1,
                color=(0, 0, 255)
            )

    return img


def main(args):
    
    shutil.rmtree(args.out_folder, ignore_errors=True)
    os.makedirs(args.out_folder, exist_ok=True)
    
    with open("models/smplx/SMPLX_segmentation.json", "r") as fp:
        seg_dict = json.load(fp)

    gt_coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "supercategory": "person",
                "id": 1,
                "name": "person",
                "keypoints": list(COCO_JOINTS.keys()),
                "skeleton": COCO_SKELETON,
            },
        ]
    }

    print("Generating poses and views...")
    with tqdm(total=args.num_views * args.num_poses) as progress_bar:

        for pose_i in range(args.num_poses):
            if args.gender.upper() == "RANDOM":
                gndr = np.random.choice(["male", "female", "neutral"])
            else:
                gndr = args.gender

            model = smplx.create(args.model_folder, model_type=args.model_type,
                                gender=gndr, use_face_contour=args.use_face_contour,
                                num_betas=args.num_betas,
                                num_expression_coeffs=args.num_expression_coeffs,
                                ext=args.model_ext)
            
            betas, expression = None, None
            if args.sample_shape:
                betas = torch.randn([1, model.num_betas], dtype=torch.float32)
            if args.sample_expression:
                expression = torch.randn(
                    [1, model.num_expression_coeffs], dtype=torch.float32)

            body_pose = generate_pose(simplicity=args.pose_simplicity)

            output = model(betas=betas, expression=expression,
                        return_verts=True, body_pose=body_pose)
            vertices = output.vertices.detach().cpu().numpy().squeeze()
            joints = output.joints.detach().cpu().numpy().squeeze()
            coco_joints = joints[[v["idx"] for _, v in COCO_JOINTS.items()], :]
            joints_range = np.array([v["range"] for _, v in COCO_JOINTS.items()])

            msh = Mesh(vertices, model.faces)

            # Default (= skin) color
            if args is not None and args.show:
                skin_color = SKIN_COLOR
            else:
                skin_color = SKIN_COLOR[[2, 1, 0, 3]]
            vertex_colors = np.ones([vertices.shape[0], 4]) * skin_color

            if np.random.rand(1)[0] < 0.5:
                BOTTOM = PANTS_PARTS
            else:    
                BOTTOM = SHORTS_PARTS
            if np.random.rand(1)[0] < 0.5:
                TOP = TSHIRT_PARTS
            else:    
                TOP = SHIRT_PARTS

            segments = [TOP, BOTTOM, SHOES_PARTS]
            segments_colors = [generate_color() for _ in segments]

            for seg, seg_col in zip(segments, segments_colors):
                for body_part in seg:
                    vertex_colors[seg_dict[body_part], :] = seg_col

            joints_vertices = get_joints_vertices(coco_joints, vertices, joints_range)

            tri_mesh = trimesh.Trimesh(vertices, model.faces,
                                    vertex_colors=vertex_colors)

            mesh = pyrender.Mesh.from_trimesh(tri_mesh)

            scene = pyrender.Scene(bg_color=generate_color())
            scene.add(mesh)

            light = pyrender.DirectionalLight(color=[1,1,1], intensity=5e2)
            for _ in range(5):
                scene.add(light, pose=random_camera_pose(distance=2*args.camera_distance))
            
            if args is not None and args.show:
                # render scene
                for view_idx in range(args.num_views):    
                    progress_bar.update()
                pyrender.Viewer(scene, use_raymond_lighting=True)
            
            else:
                fov = np.pi/2
                camera = pyrender.PerspectiveCamera( yfov=fov, aspectRatio=1)
                last_camera_node = None
                for view_idx in range(args.num_views):    
                    if last_camera_node is not None:
                        scene.remove_node(last_camera_node)
                    
                    cam_pose = random_camera_pose(distance=args.camera_distance)

                    last_camera_node = scene.add(camera, pose=cam_pose)

                    r = pyrender.OffscreenRenderer(1024, 1024)
                    rendered_img, _ = r.render(scene)
                    rendered_img = rendered_img.astype(np.uint8)

                    # Name file differently to avoid confusion
                    if args.plot_gt:
                        img_name = "sampled_pose_{:02d}_view_{:02d}_GT.jpg".format(pose_i, view_idx)
                    else:
                        img_name = "sampled_pose_{:02d}_view_{:02d}.jpg".format(pose_i, view_idx)
                    img_id = int(abs(hash(img_name)))

                    # For COCO compatibility
                    img_name = "{:d}.jpg".format(img_id)
                    
                    camera_position = cam_pose[:3, -1].squeeze().tolist()
                    camera_rotation = cam_pose[:3, :3].squeeze()

                    visibilities = msh.vertex_visibility(
                        camera = camera_position,
                        omni_directional_camera = True
                    )

                    K = camera.get_projection_matrix(1024, 1024)
                    
                    joints_2d = project_to_2d(coco_joints, K, cam_pose)
                    vertices_2d = project_to_2d(vertices, K, cam_pose)

                    in_image = np.all(vertices_2d >= 0, axis=1)
                    in_image = np.all(vertices_2d < 1024, axis=1) & in_image
                    vertices_2d = vertices_2d[in_image, :]
                    
                    if args.gt_type == "depth":
                        cam_up = camera_rotation @ np.array([0, 1, 0])
                        params = [{
                            'cam_pos': camera_position,
                            'cam_lookat': [0, 0, 0],
                            'cam_up': cam_up,
                            'x_fov': fov,  # End-to-end field of view in radians
                            'near': 0.01, 'far': 10,
                            'height': 1024, 'width': 1024,
                            'is_depth': True,  # If false, output a ray displacement map, i.e. from the mesh surface to the camera center.
                        }]
                        depthmap = m2d.mesh2depth(
                            vertices.copy().astype(np.float32),
                            model.faces.astype(np.uint32),
                            params,
                            empty_pixel_value=-1,
                        )[0]
                        depthmap[depthmap < 0] =  1.1 * np.max(depthmap)
                        depthmap = depthmap - np.min(depthmap)
                        depthmap /= np.max(depthmap)
                        depthmap = 1 - depthmap
                        depthmap *= 255
                        cv2.imwrite(
                            osp.join(args.out_folder, "{:d}_depth.jpg".format(img_id)),
                            depthmap.astype(np.uint8)
                        )

                    joints_vis = get_joints_visibilities(joints_vertices, visibilities)
                    joints_vis = np.all(joints_2d >= 0, axis=1) & joints_vis
                    joints_vis = np.all(joints_2d < 1024, axis=1) & joints_vis

                    if args.plot_gt:
                        rendered_img = draw_pose(rendered_img, joints_2d, joints_vis)

                    if args.gt_type == "openpose":
                        posemap = np.zeros((1024, 1024, 3), dtype=np.uint8)
                        posemap = draw_pose(posemap, joints_2d, joints_vis, draw_style="openpose")
                        cv2.imwrite(
                            osp.join(args.out_folder, "{:d}_openpose.jpg".format(img_id)),
                            posemap.astype(np.uint8)
                        )

                    keypoints = np.concatenate([
                        joints_2d,
                        2*joints_vis.astype(np.float32).reshape((-1, 1))
                    ], axis=1)

                    keypoints[~ joints_vis, :] = 0


                    bbox_xy = np.array([
                        np.min(vertices_2d[:, 0]),
                        np.min(vertices_2d[:, 1]),
                        np.max(vertices_2d[:, 0]),
                        np.max(vertices_2d[:, 1]),
                    ], dtype=np.float32)
                    bbox_wh = np.array([
                        bbox_xy[0], bbox_xy[1],
                        bbox_xy[2] - bbox_xy[0],
                        bbox_xy[3] - bbox_xy[1],
                    ], dtype=np.float32)


                    if args.plot_gt:
                        rendered_img = cv2.rectangle(
                            rendered_img,
                            (int(bbox_xy[0]), int(bbox_xy[1])),
                            (int(bbox_xy[2]), int(bbox_xy[3])),
                            color=(0, 255, 0),
                            thickness=1
                        )

                    gt_coco_dict["images"].append({
                        "file_name": img_name,
                        "height": 1024,
                        "width": 1024,
                        "id": img_id,
                    })
                    gt_coco_dict["annotations"].append({
                        "num_keypoints": int(np.sum(joints_vis)),
                        "iscrowd": 0,
                        "area": float(bbox_wh[2] * bbox_wh[3]),
                        "keypoints": keypoints.flatten().tolist(),
                        "image_id": img_id,
                        "bbox": bbox_wh.flatten().tolist(),
                        "category_id": 1,
                        "id": int(abs(hash(img_name + str(view_idx))))
                    })

                    save_path = osp.join(args.out_folder, img_name)
                    cv2.imwrite(save_path, rendered_img)

                    progress_bar.update()
        
        gt_filename = os.path.join(args.out_folder, "coco_annotations.json")
        with open(gt_filename, "w") as fp:
            json.dump(gt_coco_dict, fp, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

    # Original params
    parser.add_argument('--model-folder', default="models", type=str,
                        help='The path to the model folder')
    parser.add_argument('--model-type', default='smplx', type=str,
                        choices=['smpl', 'smplh', 'smplx', 'mano', 'flame'],
                        help='The type of model to load')
    parser.add_argument('--gender', type=str, default='random',
                        help='The gender of the model')
    parser.add_argument('--num-betas', default=10, type=int,
                        dest='num_betas',
                        help='Number of shape coefficients.')
    parser.add_argument('--num-expression-coeffs', default=10, type=int,
                        dest='num_expression_coeffs',
                        help='Number of expression coefficients.')
    parser.add_argument('--model-ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--sample-shape',
                        action="store_true", default=True,
                        help='Sample a random shape')
    parser.add_argument('--sample-expression',
                        action="store_true", default=True,
                        help='Sample a random expression')
    parser.add_argument('--use-face-contour',
                        action="store_true", default=False,
                        help='Compute the contour of the face')
    # Added params
    parser.add_argument('--num-views', default=3, type=int,
                        dest='num_views',
                        help='Number of views for each pose.')
    parser.add_argument('--num-poses', default=1, type=int,
                        dest='num_poses',
                        help='Number of poses to sample.')
    parser.add_argument('--pose-simplicity', default=1, type=float,
                        dest='pose_simplicity',
                        help='Measure of simplicty. The higher the simpler poses')
    parser.add_argument('--camera-distance', default=2, type=float,
                        dest='camera_distance',
                        help='Distance of the camera from the mesh.')
    parser.add_argument('--out-folder', default="sampled_poses",
                        help='Output folder')
    parser.add_argument('--plot-gt',
                        action="store_true", default=False,
                        help='The path to the model folder')
    parser.add_argument('--show',
                        action="store_true", default=False,
                        help='If True, will render and show results instead of saving images')
    parser.add_argument('--gt-type', default='NONE', type=str,
                        choices=['NONE', 'depth', 'openpose', 'cocopose'],
                        help='The type of model to load')

    args = parser.parse_args()
    args.model_folder = osp.expanduser(osp.expandvars(args.model_folder))

    main(args)
