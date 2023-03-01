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

import os.path as osp
import os
import shutil
import argparse

import numpy as np
import torch
import cv2
from tqdm import tqdm

import json

import smplx
from smplx.joint_names import COCO_JOINTS, COCO_SKELETON

from psbody.mesh import Mesh, MeshViewers

TSHIRT_PARTS = ["spine1", "spine2", "leftShoulder", "rightShoulder", "rightArm", "spine", "hips", "leftArm"]
SHIRT_PARTS = ["spine1", "spine2", "leftShoulder", "rightShoulder", "rightArm", "spine", "hips", "leftArm", "leftForeArm", "rightForeArm"]
SHORTS_PARTS = ["rightUpLeg", "leftUpLeg"]
PANTS_PARTS = ["rightUpLeg", "leftUpLeg", "leftLeg", "rightLeg"]
SHOES_PARTS = ["leftToeBase", "rightToeBase", "leftFoot", "rightFoot", ]


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


def main(model_folder,
         model_type='smplx',
         ext='npz',
         gender='neutral',
         plot_joints=False,
         num_betas=10,
         sample_shape=True,
         sample_expression=True,
         num_expression_coeffs=10,
         plotting_module='pyrender',
         use_face_contour=False,
         num_poses=1,
         num_views=1,
         out_folder="sampled_poses",
         args=None):
    
    if args is not None:
        simplicity=args.simplicity
    else:
        simplicity=1
    
    if args is not None:
        camera_distance=args.distance
    else:
        camera_distance=3

    shutil.rmtree(out_folder, ignore_errors=True)
    os.makedirs(out_folder, exist_ok=True)
    
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
    with tqdm(total=num_views * num_poses) as progress_bar:

        for pose_i in range(num_poses):
            if gender.upper() == "RANDOM":
                gndr = np.random.choice(["male", "female", "neutral"])
            else:
                gndr = gender

            model = smplx.create(model_folder, model_type=model_type,
                                gender=gndr, use_face_contour=use_face_contour,
                                num_betas=num_betas,
                                num_expression_coeffs=num_expression_coeffs,
                                ext=ext)
            
            betas, expression = None, None
            if sample_shape:
                betas = torch.randn([1, model.num_betas], dtype=torch.float32)
            if sample_expression:
                expression = torch.randn(
                    [1, model.num_expression_coeffs], dtype=torch.float32)

            body_pose = generate_pose(simplicity=simplicity)

            output = model(betas=betas, expression=expression,
                        return_verts=True, body_pose=body_pose)
            vertices = output.vertices.detach().cpu().numpy().squeeze()
            joints = output.joints.detach().cpu().numpy().squeeze()
            coco_joints = joints[[v["idx"] for _, v in COCO_JOINTS.items()], :]
            joints_range = np.array([v["range"] for _, v in COCO_JOINTS.items()])

            msh = Mesh(vertices, model.faces)

            if plotting_module == 'pyrender':
                import pyrender
                import trimesh
                
                # Default (= skin) color
                SKIN_COLOR = np.array([1.0, 0.66, 0.28, 1.0]) # RGB format
                if args is not None and args.show:
                    vertex_colors = np.ones([vertices.shape[0], 4]) * SKIN_COLOR
                else:
                    vertex_colors = np.ones([vertices.shape[0], 4]) * SKIN_COLOR[[2, 1, 0, 3]]
   
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

                scene = pyrender.Scene()
                scene.add(mesh)

                light = pyrender.DirectionalLight(color=[1,1,1], intensity=5e2)
                for _ in range(3):
                    scene.add(light, pose=random_camera_pose(distance=5))
                
                if args is not None and args.show:
                    # render scene
                    for view_idx in range(num_views):    
                        progress_bar.update()
                    pyrender.Viewer(scene, use_raymond_lighting=True)
                
                else:
                    camera = pyrender.PerspectiveCamera( yfov=np.pi /2, aspectRatio=1)
                    last_camera_node = None
                    for view_idx in range(num_views):    
                        if last_camera_node is not None:
                            scene.remove_node(last_camera_node)
                        
                        T = random_camera_pose(distance=camera_distance)

                        last_camera_node = scene.add(camera, pose=T)

                        r = pyrender.OffscreenRenderer(1024, 1024)
                        color, _ = r.render(scene)
                        color = color.astype(np.uint8)

                        img_name = "sampled_pose_{:02d}_view_{:02d}.jpg".format(pose_i, view_idx)
                        img_id = int(abs(hash(img_name)))
                        save_path = osp.join(out_folder, img_name)
                        cv2.imwrite(save_path.format(view_idx), color)
                        
                        if plot_joints:
                            
                            cam = T[:3, -1].squeeze().tolist()
                            visibilities = msh.vertex_visibility(
                                camera = cam,
                                omni_directional_camera = True
                            )

                            K = camera.get_projection_matrix(1024, 1024)
                            
                            joints_2d = project_to_2d(coco_joints, K, T)
                            vertices_2d = project_to_2d(vertices, K, T)

                            in_image = np.all(vertices_2d >= 0, axis=1)
                            in_image = np.all(vertices_2d < 1024, axis=1) & in_image
                            vertices_2d = vertices_2d[in_image, :]

                            joints_vis = get_joints_visibilities(joints_vertices, visibilities)
                            joints_vis = np.all(joints_2d >= 0, axis=1) & joints_vis
                            joints_vis = np.all(joints_2d < 1024, axis=1) & joints_vis

                            for pi, pt in enumerate(joints_2d):
                                marker_color = (0, 0, 255) if joints_vis[pi] else (40, 40, 40)
                                thickness = 2 if joints_vis[pi] else 1
                                color = cv2.drawMarker(
                                    color,
                                    tuple(pt.tolist()),
                                    color=marker_color,
                                    markerType=cv2.MARKER_CROSS,
                                    thickness=thickness
                                )

                            for bone in COCO_SKELETON:
                                b = np.array(bone) - 1 # COCO_SKELETON is 1-indexed
                                if not (joints_vis[b[0]] and joints_vis[b[1]]):
                                    continue
                                
                                start = joints_2d[b[0], :]
                                end = joints_2d[b[1], :]
                                color = cv2.line(
                                    color,
                                    start,
                                    end,
                                    thickness=1,
                                    color=(0, 0, 255)
                                )

                            keypoints = np.concatenate([
                                joints_2d,
                                2*joints_vis.astype(np.float32).reshape((-1, 1))
                            ], axis=1)

                            keypoints[~ joints_vis, :] = 0

                            bbox = np.array([
                                np.min(vertices_2d[:, 0]),
                                np.min(vertices_2d[:, 1]),
                                np.max(vertices_2d[:, 0]),
                                np.max(vertices_2d[:, 1]),
                            ])

                            color = cv2.rectangle(
                                color,
                                (int(bbox[0]), int(bbox[1])),
                                (int(bbox[2]), int(bbox[3])),
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
                                "keypoints": keypoints.flatten().tolist(),
                                "image_id": img_id,
                                "bbox": bbox.flatten().tolist(),
                                "category_id": 1,
                                "id": int(abs(hash(img_name + str(view_idx))))
                            })

                            save_path = osp.join(out_folder, "sampled_pose_{:02d}_view_{:02d}_gt.jpg".format(pose_i, view_idx))
                            cv2.imwrite(save_path.format(view_idx), color)

                        progress_bar.update()

            elif plotting_module == 'matplotlib':
                from matplotlib import pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                mesh = Poly3DCollection(vertices[model.faces], alpha=0.1)
                face_color = (1.0, 1.0, 0.9)
                edge_color = (0, 0, 0)
                mesh.set_edgecolor(edge_color)
                mesh.set_facecolor(face_color)
                ax.add_collection3d(mesh)
                ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

                if plot_joints:
                    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)
                plt.show()
            elif plotting_module == 'open3d':
                import open3d as o3d

                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(
                    vertices)
                mesh.triangles = o3d.utility.Vector3iVector(model.faces)
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color([0.3, 0.3, 0.3])

                geometry = [mesh]
                if plot_joints:
                    joints_pcl = o3d.geometry.PointCloud()
                    joints_pcl.points = o3d.utility.Vector3dVector(joints)
                    joints_pcl.paint_uniform_color([0.7, 0.3, 0.3])
                    geometry.append(joints_pcl)

                o3d.visualization.draw_geometries(geometry)
            else:
                raise ValueError('Unknown plotting_module: {}'.format(plotting_module))
        
        if plot_joints:
            gt_filename = os.path.join(out_folder, "coco_annotations.json")
            with open(gt_filename, "w") as fp:
                json.dump(gt_coco_dict, fp, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

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
    parser.add_argument('--num-views', default=3, type=int,
                        dest='num_views',
                        help='Number of views for each pose.')
    parser.add_argument('--num-poses', default=1, type=int,
                        dest='num_poses',
                        help='Number of poses to sample.')
    parser.add_argument('--simplicity', default=1, type=int,
                        dest='simplicity',
                        help='Measure of simplicty. The higher the simpler poses')
    parser.add_argument('--distance', default=2, type=int,
                        dest='distance',
                        help='Distance of the camera from the mesh.')
    parser.add_argument('--num-expression-coeffs', default=10, type=int,
                        dest='num_expression_coeffs',
                        help='Number of expression coefficients.')
    parser.add_argument('--plotting-module', type=str, default='pyrender',
                        dest='plotting_module',
                        choices=['pyrender', 'matplotlib', 'open3d'],
                        help='The module to use for plotting the result')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--plot-joints', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='The path to the model folder')
    parser.add_argument('--show', default=False,
                        help='If True, will render and show results instead of saving images')
    parser.add_argument('--sample-shape', default=True,
                        dest='sample_shape',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random shape')
    parser.add_argument('--sample-expression', default=True,
                        dest='sample_expression',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random expression')
    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')
    parser.add_argument('--out-folder', default="sampled_poses",
                        help='Output folder')

    args = parser.parse_args()

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    model_type = args.model_type
    plot_joints = args.plot_joints
    use_face_contour = args.use_face_contour
    gender = args.gender
    ext = args.ext
    plotting_module = args.plotting_module
    num_betas = args.num_betas
    num_expression_coeffs = args.num_expression_coeffs
    sample_shape = args.sample_shape
    sample_expression = args.sample_expression

    main(model_folder, model_type, ext=ext,
         gender=gender, plot_joints=plot_joints,
         num_betas=num_betas,
         num_expression_coeffs=num_expression_coeffs,
         sample_shape=sample_shape,
         sample_expression=sample_expression,
         plotting_module=plotting_module,
         use_face_contour=use_face_contour,
         num_poses = args.num_poses,
         num_views = args.num_views,
         out_folder = args.out_folder,
         args=args)
