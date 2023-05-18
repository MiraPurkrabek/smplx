import json
import numpy as np
import torch


def c2s(pts, use_torch=False):
    if use_torch:
        x, y, z = torch.unbind(pts, dim=1)
    else:
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    if use_torch:
        r = torch.norm(pts, dim=1)
        theta = torch.atan2(
            torch.sqrt(x*x + y*y),
            z,
        )
        phi = torch.atan2(y, x)
        spherical = torch.stack([r, theta, phi], dim=1)
    else:
        r = np.linalg.norm(pts, axis=1)
        theta = np.arctan2(
            np.sqrt(x*x + y*y),
            z,
        )
        phi = np.arctan2(y, x)
        spherical = np.stack([r, theta, phi], axis=1)
    return spherical


def s2c(pts, use_torch=False):
    if use_torch:
        fn = torch
    else:
        fn = np

    if pts.shape[1] == 3:
        if use_torch:
            r, theta, phi = torch.unbind(pts, dim=1)
        else:
            r = pts[:, 0]
            theta = pts[:, 1]
            phi = pts[:, 2]
    else:
        if use_torch:
            theta, phi = torch.unbind(pts, dim=1)
            r = torch.ones(pts.shape[0]).to(theta.device) # If no radius given, use 1
        else:
            r = np.ones(pts.shape[0]) # If no radius given, use 1
            theta = pts[:, 0]
            phi = pts[:, 1]

    x = r * fn.sin(theta) * fn.cos(phi)
    y = r * fn.sin(theta) * fn.sin(phi)
    z = r * fn.cos(theta)
    return fn.stack([x, y, z], axis=1)


def load_data_from_coco_file(coco_filepath, views_filepath=None):
    coco_dict = json.load(open(coco_filepath, "r"))
    image_ids = []
    keypoints = []
    bboxes_xywh = []
    
    if not views_filepath is None:
        views_dict = json.load(open(views_filepath, "r"))
        positions = []

    for annot in coco_dict["annotations"]:
        image_id = annot["image_id"]
        image_ids.append(image_id)
        image_name = "{:d}.jpg".format(image_id)
        kpts = np.array(annot["keypoints"])
        bbox = np.array(annot["bbox"])

        # At least 4 keypoints must be visible
        vis_mask = kpts[2::3] > 1
        if np.sum(vis_mask) < 3:
            continue

        keypoints.append(kpts)
        bboxes_xywh.append(bbox)
        if not views_filepath is None:
            view = views_dict[image_name]
            camera_pos = view["camera_position"]
            positions.append(camera_pos)

    keypoints = np.array(keypoints)
    bboxes_xywh = np.array(bboxes_xywh)
    image_ids = np.array(image_ids)
    
    if not views_filepath is None:
        positions = np.array(positions)
        return keypoints, bboxes_xywh, image_ids, positions

    return keypoints, bboxes_xywh, image_ids


def process_keypoints(keypoints, bboxes, add_visibility=False, add_bboxes=True, normalize=True):
    """
    Process the keypoints to minimize the domain gap between synthetic and COCO keypoints.
    1. Normalize the keypoints to be in the range [0, 1] with respect to the bounding box
    2. Remove keypoints with visibility < 2
    """

    num_keypoints = keypoints.shape[0]
    keypoints = np.reshape(keypoints, (-1, 17, 3)).astype(np.float32)
    
    # Normalize the keypoints to be in the range [0, 1] with respect to the bounding box
    bboxes = bboxes[:, None, :]
    if normalize:
        keypoints[:, :, 0] = (keypoints[:, :, 0] - bboxes[:, :, 0]) / bboxes[:, :, 2]
        keypoints[:, :, 1] = (keypoints[:, :, 1] - bboxes[:, :, 1]) / bboxes[:, :, 3]

    # Remove keypoints with visibility < 2
    visibilities = keypoints[:, :, 2].squeeze()
    keypoints[visibilities < 2, :] = 0

    # Remove the visibility flag from the keypoints
    if not add_visibility:
        keypoints = keypoints[:, :, :2]

    # Stack bbox width and height to the keypoints
    if add_bboxes:
        if add_visibility:
            keypoints = np.reshape((num_keypoints, -1))
            keypoints = np.concatenate([keypoints, bboxes[:, :, 2:].squeeze()], axis=1)
        else:
            keypoints = np.concatenate([keypoints, bboxes[:, :, 2:]], axis=1)
    
    # Reshape the keypoints to be a 1D array
    keypoints = np.reshape(keypoints, (num_keypoints, -1))
    print("Keypoints shape:", keypoints.shape)
    
    # for coor in range(keypoints.shape[1]):
    #     print("Coordinate {:d}: min={:.3f}, max={:.3f}".format(coor, np.min(keypoints[:, coor]), np.max(keypoints[:, coor])))
    
    return keypoints


def angular_distance(pts1, pts2, use_torch=False):
    """
    Compute the angular distance between two points on a unit sphere.
    """
    if use_torch:
        acos = torch.arccos
        sin = torch.sin
        cos = torch.cos
        all = torch.all
        any = torch.any
        abs = torch.abs
        clip = torch.clamp
    else:
        acos = np.arccos
        sin = np.sin
        cos = np.cos
        all = np.all
        any = np.any
        abs = np.abs
        clip = np.clip

    if pts1.shape[1] == 3:
        radius1, theta1, phi1 = pts1[:, :1], pts1[:, 1:2], pts1[:, 2:]
        radius2, theta2, phi2 = pts2[:, :1], pts2[:, 1:2], pts2[:, 2:]
    else:
        theta1, phi1 = pts1[:, :1], pts1[:, 1:2]
        theta2, phi2 = pts2[:, :1], pts2[:, 1:2]

    # Clip the input - not sure if this helped any
    # theta1 = clip(theta1, 0, np.pi)
    # theta2 = clip(theta2, 0, np.pi)
    # phi1 = clip(phi1, -np.pi, np.pi)
    # phi2 = clip(phi2, -np.pi, np.pi)
        
    dist = acos(sin(theta1)*sin(theta2) + cos(theta1)*cos(theta2)*cos(phi1 - phi2))

    # Add radius difference - not sure if this helped any
    if pts1.shape[1] == 3:
        dist += 1.0 * abs(radius1 - radius2)

    # if not all(dist >= 0):
    #     print(pts1[:10])
    #     print(pts2[:10])
    #     print(dist[:10])

    assert all(dist >= 0)
    # assert all(dist <= np.pi)

    return dist


if __name__ == "__main__":

    pts1 = np.array([
        [0, 0, -1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [1, 1, 0],
        [1, 0, -1],
    ], dtype=np.float32)
    pts2 = np.zeros(pts1.shape, dtype=np.float32)
    pts2[:, -1] = 1
    d = angular_distance(c2s(pts1), c2s(pts2))
    d = d * 180 / np.pi
    
    print("Distance of selected points on a unit sphere (in degrees):")
    for i in range(pts1.shape[0]):
        print("Points {} x {}:\t\t{:.3f}".format(pts1[i, :], pts2[i, :], d[i]))
    
    pts1 = np.random.normal(size = (100000, 3))
    pts2 = np.random.normal(size = (100000, 3))
    d = angular_distance(c2s(pts1), c2s(pts2))
    d = d * 180 / np.pi

    print()
    print("Distance of random points on a unit sphere (in degrees):")
    print("Min: {:.3f}, Mean: {:.3f}, Max: {:.3f}".format(np.min(d), np.mean(d), np.max(d)))