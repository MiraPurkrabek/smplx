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

    if simplicity < 0:
        simplicity = rnd(1.0, 1.5)

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
    
    elif typical_pose.lower() == "stretch":
        body_pose[0, joints["Torso"]*3+0] = np.pi / 2
        body_pose[0, joints["Left shoulder"]*3+1] = - np.pi / 2 * 4/5
        body_pose[0, joints["Right shoulder"]*3+1] = np.pi / 2 * 4/5

    return body_pose


def random_3d_position_polar(distance=2.0, view_preference=None):
    
    if distance < 0:
        distance = rnd(1.5, 5.0)
    
    # Noise is 20°
    noise_size = 20 / 180 * np.pi

    # ALPHA - rotation around the sides
    # BETA  - rotation around the top and bottom
    if view_preference is None:
        alpha_mean = 0
        alpha_range = np.pi
        beta_mean = 0
        beta_range = np.pi
    elif view_preference.upper() == "PERIMETER":
        alpha_mean = 0
        alpha_range = np.pi
        beta_mean = 0
        beta_range = noise_size
    elif view_preference.upper() == "FRONT":
        alpha_mean = 0
        alpha_range = noise_size
        beta_mean = 0
        beta_range = 3*noise_size
    elif view_preference.upper() == "BACK":
        alpha_mean = np.pi
        alpha_range = noise_size
        beta_mean = 0
        beta_range = 3*noise_size
    elif view_preference.upper() == "SIDE":
        alpha_mean = random_sgn() * np.pi/2
        alpha_range = noise_size
        beta_mean = 0
        beta_range = 3*noise_size
    elif view_preference.upper() == "TOP":
        alpha_mean = 0
        alpha_range = noise_size
        beta_mean = np.pi/2
        beta_range = noise_size
    elif view_preference.upper() == "BOTTOM":
        alpha_mean = 0
        alpha_range = noise_size
        beta_mean = - np.pi/2
        beta_range = noise_size
    else:
        raise ValueError("Unknown view preference")
    
    # [ 0,  0,  1] - FRONT view
    # [ 0,  0, -1] - BACK view (do not converge)
    # [ 0,  1,  0] - TOP view
    # [ 0, -1,  0] - BOTTOM view
    # [ 1,  0,  0] - LEFT view
    # [-1,  0,  0] - RIGHT view
    alpha = alpha_mean + rnd(-alpha_range, alpha_range)
    beta = beta_mean + rnd(-beta_range, beta_range)
    
    return alpha, beta, distance


def polar_to_cartesian(alpha, beta, distance):
    y = distance * np.sin(beta)
    a = distance * np.cos(beta)
    x = a * np.sin(alpha)
    z = a * np.cos(alpha)
    return np.array([x, y, z])
    

def random_camera_pose(distance=3, view_preference=None, rotation=0, return_vectors=False):
    
    if rotation < 0:
        rotation = rnd(0, 360)

    # Convert to radians
    rotation = rotation / 180 * np.pi

    alpha, beta, distance = random_3d_position_polar(distance, view_preference)
    camera_pos = polar_to_cartesian(alpha, beta, distance)

    # Default camera_up is head up
    camera_up = np.array([0, 1, 0], dtype=np.float32)
    # For TOP and BOTTOM, default camera_up is front
    
    if not view_preference is None and view_preference.upper() in ["TOP", "BOTTOM"]:
        camera_up = np.array([0, 1, 0], dtype=np.float32)

    center = np.array([0, 0, 0], dtype=np.float32)

    f = center - camera_pos
    f /= np.linalg.norm(f)
    
    s = np.cross(f, camera_up)
    s /= np.linalg.norm(s)

    u = np.cross(s, f)
    u /= np.linalg.norm(u)

    R = np.array([
        [s[0], u[0], -f[0]],
        [s[1], u[1], -f[1]],
        [s[2], u[2], -f[2]],
    ])

    # theta = np.random.rand() * 2*rotation - rotation
    theta = random_sgn() * rotation + random_sgn() * rnd(0, np.pi/5)
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [            0,              0, 1],
    ])
    R = R @ rot_z
    camera_up = R @ camera_up

    pose = np.array([
        [R[0, 0], R[0, 1], R[0, 2], camera_pos[0]],
        [R[1, 0], R[1, 1], R[1, 2], camera_pos[1]],
        [R[2, 0], R[2, 1], R[2, 2], camera_pos[2]],
        [      0,       0,       0,    1],
    ])

    if return_vectors:
        return pose, camera_pos, camera_up
    else:
        return pose


def rnd(a_min=0, a_max=1):
    rng = a_max-a_min
    return np.random.rand() * rng + a_min


def random_sgn():
    return -1 if np.random.rand() < 0.5 else 1


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


def draw_depth(img, depthmap):
    overlay_img = img.copy()
    depthmap = depthmap > 0
    overlay_img[depthmap, :] = [0, 0, 255]
    alpha = 0.6
    return cv2.addWeighted(img, alpha, overlay_img, 1-alpha, 0)


def draw_pose(img, kpts, joints_vis, draw_style="custom"):
    img = img.copy()

    assert draw_style in [
        "custom",
        "openpose",
        "openpose_vis",
    ]

    joints_vis = joints_vis.astype(bool)
    skeleton = COCO_SKELETON

    if draw_style.startswith("openpose"):
        # Reorder kpts to OpenPose order
        kpts = kpts.copy()
        kpts = kpts[[0, 0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3], :]
        joints_vis = joints_vis[[0, 0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]]
        
        # Compute pelvis as mean of shoulders
        kpts[1, :] = np.mean(kpts[[2, 5], :], axis=1)
        joints_vis[1] = np.all(joints_vis[[2, 5]])

        skeleton = OPENPOSE_SKELETON

    for pi, pt in enumerate(kpts):
        
        if draw_style == "openpose":
            img = cv2.circle(
                img,
                tuple(pt.tolist()),
                radius=4,
                color=OPENPOSE_COLORS[pi],
                thickness=-1
            )
        elif draw_style == "openpose_vis":
            if joints_vis[pi]:
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
        if draw_style.startswith("openpose"):
            
            if draw_style == "openpose_vis" and not (joints_vis[b[0]] and joints_vis[b[1]]):
                continue
            
            stickwidth = 4
            current_img = img.copy()
            mX = np.mean(np.array([start[0], end[0]]))
            mY = np.mean(np.array([start[1], end[1]]))
            length = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(start[1] - end[1], start[0] - end[0]))
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
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


def get_textured_mesh(vertices, texture_path=None, args=None):
    
    # When no texture path is provided, use a random one
    if texture_path is None:
        texture_folder = np.random.choice(os.listdir("textures"))
        texture_path = os.path.join(
            "textures",
            texture_folder,
            "images",
            np.random.choice(os.listdir(os.path.join("textures", texture_folder, "images")))
        )
    else:
        texture_folder = os.path.dirname(os.path.dirname(texture_path))

    indices_path = os.path.join(
        "textures",
        texture_folder,
        "mappings",
        "SMPLX_MALE_indices.npz",
    )
    obj_path = os.path.join(
        "textures",
        texture_folder,
        "mappings",
        "SMPLX_UV.obj",
    )

    # Load the texture image with OpenCV
    if args is not None and args.show:
        im = cv2.imread(texture_path)[:, :, ::-1]
    else:
        im = cv2.imread(texture_path)

    # Load the OBJ file with UV map
    m = trimesh.load(obj_path)

    # Load the pre-computed mapping from 10475 to 11313 vertices
    indices_to_sort = np.load(indices_path)['indices']
    m.vertices = vertices[indices_to_sort, :]

    # Create the Texture and a trimesh object
    material = trimesh.visual.texture.SimpleMaterial(image=im)
    color_visuals = trimesh.visual.TextureVisuals(uv=m.visual.uv, image=im, material=material)
    tri_mesh = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, visual=color_visuals, validate=True, process=False)

    return tri_mesh


def get_colored_mesh(vertices, faces, args):
    with open("models/smplx/SMPLX_segmentation.json", "r") as fp:
        seg_dict = json.load(fp)
    
    # Default (= skin) color
    if args is not None and args.show:
        skin_color = SKIN_COLOR
    else:
        skin_color = SKIN_COLOR[[2, 1, 0, 3]]
    vertex_colors = np.ones([vertices.shape[0], 4]) * skin_color

    if not args.naked:
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

    tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
    return tri_mesh


def main(args):
    
    shutil.rmtree(args.out_folder, ignore_errors=True)
    os.makedirs(args.out_folder, exist_ok=True)

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

    # Read the background image from the file
    backgrounds_folder = os.path.join("images", "backgrounds")
    background_list = os.listdir(backgrounds_folder)

    with tqdm(total=args.num_views * args.num_poses, ascii=True) as progress_bar:

        for pose_i in range(args.num_poses):
            
            random_background_image = np.random.choice(background_list)
            background_image = cv2.imread(os.path.join(backgrounds_folder, random_background_image))
            background_image = cv2.resize(background_image, (1024, 1024))
            
            if args.save_default_pose:
                gndr = "male"
            elif args.gender.upper() == "RANDOM":
                gndr = np.random.choice(["male", "female", "neutral"])
            else:
                gndr = args.gender

            model = smplx.create(args.model_folder, model_type=args.model_type,
                                gender=gndr, use_face_contour=args.use_face_contour,
                                num_betas=args.num_betas,
                                num_expression_coeffs=args.num_expression_coeffs,
                                ext=args.model_ext)
            
            betas, expression = None, None
            if args.sample_shape and not args.save_default_pose:
                betas = torch.randn([1, model.num_betas], dtype=torch.float32)
            if args.sample_expression and not args.save_default_pose:
                expression = torch.randn(
                    [1, model.num_expression_coeffs], dtype=torch.float32)


            hand_pose = model.left_hand_pose
            if args.save_default_pose:
                body_pose = torch.zeros([1, model.NUM_BODY_JOINTS*3], dtype=torch.float32)
                left_hand_pose = torch.zeros(hand_pose.shape, dtype=torch.float32)
                right_hand_pose = torch.zeros(hand_pose.shape, dtype=torch.float32)
            else:
                body_pose = generate_pose(simplicity=args.pose_simplicity, typical_pose="stretch")
                left_hand_pose = (torch.rand(hand_pose.shape)-0.5) * 3
                right_hand_pose = (torch.rand(hand_pose.shape)-0.5) * 3

            output = model(
                betas=betas,
                expression=expression,
                return_verts=True,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                jaw_pose=None,
                )

            vertices = output.vertices.detach().cpu().numpy().squeeze()
            joints = output.joints.detach().cpu().numpy().squeeze()
            coco_joints = joints[[v["idx"] for _, v in COCO_JOINTS.items()], :]
            joints_range = np.array([v["range"] for _, v in COCO_JOINTS.items()])

            msh = Mesh(vertices, model.faces)

            joints_vertices = get_joints_vertices(coco_joints, vertices, joints_range)
            
            # Add random noise to the vertices
            # vertices += np.random.normal(0, 0.005, vertices.shape)

            if args.save_default_pose:
                np.savez("default_body_pose_vertices.npz", vertices=vertices)

            # Generate the tri mesh with texture or coloring
            if args.not_textured:
                tri_mesh = get_colored_mesh(vertices, model.faces, args)
            else:
                tri_mesh = get_textured_mesh(vertices, texture_path=None, args=args)
 
            if args.uniform_background:
                scene = pyrender.Scene(bg_color=generate_color())
            else:
                scene = pyrender.Scene(bg_color=(255, 255, 255, 255))

            mesh = pyrender.Mesh.from_trimesh(tri_mesh)
            scene.add(mesh)

            # Add lights
            light_intensity = 5e2 if args.not_textured else 3e3
            light = pyrender.DirectionalLight(color=[1,1,1], intensity=light_intensity)
            for _ in range(5):
                scene.add(light, pose=random_camera_pose(distance=abs(2*args.camera_distance)))
            
            if args is not None and args.show:
                # Render scene
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
                    
                    cam_pose, camera_position, _ = random_camera_pose(
                        distance=args.camera_distance,
                        view_preference=args.view_preference,
                        rotation=args.rotation,
                        return_vectors=True
                    )

                    last_camera_node = scene.add(camera, pose=cam_pose)

                    r = pyrender.OffscreenRenderer(1024, 1024)
                    rendered_img, depthmap = r.render(scene)
                    rendered_img = rendered_img.astype(np.uint8)

                    # Process the depthmap
                    depthmap[depthmap <= 0] =  1.1 * np.max(depthmap)
                    depthmap = depthmap - np.min(depthmap)
                    depthmap /= np.max(depthmap)
                    depthmap = 1 - depthmap
                    depthmap *= 255

                    # Add background
                    if not args.uniform_background:
                        rendered_img_w_bckg = background_image.copy()
                        depthmap_mask = depthmap > 0
                        depthmap_mask = cv2.erode(depthmap_mask.astype(np.uint8), np.ones((3, 3)), iterations=1).astype(bool)
                        rendered_img_w_bckg[depthmap_mask, :] = rendered_img[depthmap_mask, :]
                        rendered_img = rendered_img_w_bckg

                    # Name file differently to avoid confusion
                    if args.plot_gt:
                        img_name = "sampled_pose_{:02d}_view_{:02d}_GT.jpg".format(pose_i, view_idx)
                    else:
                        img_name = "sampled_pose_{:02d}_view_{:02d}.jpg".format(pose_i, view_idx)
                    img_id = int(abs(hash(img_name)))
                    
                    # For COCO compatibility
                    img_name = "{:d}.jpg".format(img_id)

                    visibilities = msh.vertex_visibility(
                        camera = camera_position.tolist(),
                        omni_directional_camera = True
                    )

                    K = camera.get_projection_matrix(1024, 1024)
                    
                    joints_2d = project_to_2d(coco_joints, K, cam_pose)
                    vertices_2d = project_to_2d(vertices, K, cam_pose)

                    in_image = np.all(vertices_2d >= 0, axis=1)
                    in_image = np.all(vertices_2d < 1024, axis=1) & in_image
                    vertices_2d = vertices_2d[in_image, :]
                    
                    joints_vis = get_joints_visibilities(joints_vertices, visibilities)
                    joints_vis = np.all(joints_2d >= 0, axis=1) & joints_vis
                    joints_vis = np.all(joints_2d < 1024, axis=1) & joints_vis

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
                    pad = 0.1 * bbox_wh[2:]
                    crop_bbox = np.array([
                        int(max(0, bbox_xy[1] - pad[1])),
                        int(max(0, bbox_xy[0] - pad[0])),
                        int(min(1024, bbox_xy[3] + pad[1])),
                        int(min(1024, bbox_xy[2] + pad[0])),
                    ], dtype=np.int32)

                    if args.crop:
                        depthmap = depthmap[crop_bbox[0]:crop_bbox[2], crop_bbox[1]:crop_bbox[3]]
                    
                    if "DEPTH" in args.gt_type:
                        cv2.imwrite(
                            osp.join(args.out_folder, "{:d}_depth.jpg".format(img_id)),
                            depthmap.astype(np.uint8)
                        )
                    if "OPENPOSE" in args.gt_type:
                        posemap = np.zeros((1024, 1024, 3), dtype=np.uint8)
                        posemap_all = draw_pose(posemap, joints_2d, joints_vis, draw_style="openpose")
                        posemap_vis = draw_pose(posemap, joints_2d, joints_vis, draw_style="openpose_vis")
                        if args.crop:
                            posemap_all = posemap_all[crop_bbox[0]:crop_bbox[2], crop_bbox[1]:crop_bbox[3]]
                            posemap_vis = posemap_vis[crop_bbox[0]:crop_bbox[2], crop_bbox[1]:crop_bbox[3]]
                        cv2.imwrite(
                            osp.join(args.out_folder, "{:d}_openpose_all.jpg".format(img_id)),
                            posemap_all.astype(np.uint8)
                        )
                        cv2.imwrite(
                            osp.join(args.out_folder, "{:d}_openpose_vis.jpg".format(img_id)),
                            posemap_vis.astype(np.uint8)
                        )

                    if args.plot_gt:
                        rendered_img = cv2.rectangle(
                            rendered_img,
                            (int(bbox_xy[0]), int(bbox_xy[1])),
                            (int(bbox_xy[2]), int(bbox_xy[3])),
                            color=(0, 255, 0),
                            thickness=1
                        )

                        if "OPENPOSE" in args.gt_type:
                            rendered_img = draw_pose(rendered_img, joints_2d, joints_vis)

                    annot_height = 1024
                    annot_width = 1024

                    if args.crop:
                        # Recompute annotations
                        annot_height = int(crop_bbox[2] - crop_bbox[0])
                        annot_width = int(crop_bbox[3] - crop_bbox[1])
                        keypoints[joints_vis, 0] -= crop_bbox[1]
                        keypoints[joints_vis, 1] -= crop_bbox[0]
                        bbox_wh[0] -= crop_bbox[1]
                        bbox_wh[1] -= crop_bbox[0]

                        # Crop the image
                        rendered_img = rendered_img[crop_bbox[0]:crop_bbox[2], crop_bbox[1]:crop_bbox[3]]
                    
                    if args.plot_gt:
                        if "DEPTH" in args.gt_type:
                            rendered_img = draw_depth(rendered_img, depthmap)
                
                    save_path = osp.join(args.out_folder, img_name)
                    cv2.imwrite(save_path, rendered_img)

                    gt_coco_dict["images"].append({
                        "file_name": img_name,
                        "height": annot_height,
                        "width": annot_width,
                        "id": img_id,
                    })

                    area = float(np.count_nonzero(depthmap > 0))
                    gt_coco_dict["annotations"].append({
                        "num_keypoints": int(np.sum(joints_vis)),
                        "iscrowd": 0,
                        "area": area,
                        "keypoints": keypoints.flatten().tolist(),
                        "image_id": img_id,
                        "bbox": bbox_wh.flatten().tolist(),
                        "category_id": 1,
                        "id": int(abs(hash(img_name + str(view_idx))))
                    })

                    progress_bar.update()
        
        gt_filename = os.path.join(args.out_folder, "coco_annotations.json")
        with open(gt_filename, "w") as fp:
            json.dump(gt_coco_dict, fp, indent=2)
        
        metadata_filename = os.path.join(args.out_folder, "metadata")
        with open(metadata_filename, "w") as fp:
            json.dump(vars(args), fp, indent=2)


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
    parser.add_argument('--num-views', default=2, type=int,
                        dest='num_views',
                        help='Number of views for each pose.')
    parser.add_argument('--num-poses', default=1, type=int,
                        dest='num_poses',
                        help='Number of poses to sample.')
    parser.add_argument('--pose-simplicity', default=1.5, type=float,
                        dest='pose_simplicity',
                        help='Measure of pose simplicty. The higher number, the simpler poses')
    parser.add_argument('--view-preference', default=None, type=str,
                        dest='view_preference',
                        help='Prefer some specific types of views.')
    parser.add_argument('--rotation', default=0, type=int,
                        dest='rotation',
                        help='Maximal rotation of the image; in degrees')
    parser.add_argument('--camera-distance', default=2, type=float,
                        dest='camera_distance',
                        help='Distance of the camera from the mesh.')
    parser.add_argument('--out-folder', default=None,
                        help='Output folder')
    parser.add_argument('--plot-gt',
                        action="store_true", default=False,
                        help='The path to the model folder')
    parser.add_argument('--show',
                        action="store_true", default=False,
                        help='If True, will render and show results instead of saving images')
    parser.add_argument("--gt-type", nargs="+",)
    parser.add_argument('--crop',
                        action="store_true", default=False,
                        help='If True, will crop the image by the computed bbox (slightly larger)')
    parser.add_argument('--naked',
                        action="store_true", default=False,
                        help='If True, humans will have uniform color (no clothes colors). Works only with "--not-textured" param')
    parser.add_argument('--not-textured',
                        action="store_true", default=False,
                        help='If True, humans will have naive coloring (no textures)')
    parser.add_argument('--save-default-pose',
                        action="store_true", default=False,
                        help='If True, will save pose with default params. Used for development.')
    parser.add_argument('--uniform-background',
                        action="store_true", default=False,
                        help='If True, will draw background with uniform color')
    # parser.add_argument('--gt-type', default='NONE', type=str,
    #                     choices=['NONE', 'depth', 'openpose', 'cocopose'],
    #                     help='The type of model to load')

    args = parser.parse_args()

    if args.gt_type is None:
        args.gt_type = []
    args.gt_type = list(map(lambda x: x.upper(), args.gt_type))

    if args.out_folder is None:
        if args.rotation < 0:
            rotation_str = "RND"
        else:
            rotation_str = "{:03d}".format(args.rotation)

        if args.camera_distance < 0:
            distance_str = "RND"
        else:
            distance_str = "{:.1f}".format(args.camera_distance)
        
        if args.pose_simplicity < 0:
            simplicity_str = "RND"
        else:
            simplicity_str = "{:.1f}".format(args.pose_simplicity)

        naked_str = ""
        if args.naked:
            naked_str = "_naked"

        args.out_folder = os.path.join(
            "sampled_poses",
            "distance_{:s}_simplicity_{:s}_view_{}_rotation_{:s}{:s}".format(
                distance_str,
                simplicity_str,
                args.view_preference,
                rotation_str,
                naked_str,
            )
        )

    args.model_folder = osp.expanduser(osp.expandvars(args.model_folder))

    main(args)
