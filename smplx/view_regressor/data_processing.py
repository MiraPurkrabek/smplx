import json
import numpy as np
import torch


def c2s(pts):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    r = np.linalg.norm(pts, axis=1)
    theta = np.arctan2(
        np.sqrt(x*x + y*y),
        z,
    )
    phi = np.arctan2(y, x)
    return np.stack([r, theta, phi], axis=1)


def s2c(pts):
    if pts.shape[1] == 3:
        r = pts[:, 0]
        theta = pts[:, 1]
        phi = pts[:, 2]
    else:
        r = np.ones(pts.shape[0]) # If no radius given, use 1
        theta = pts[:, 0]
        phi = pts[:, 1]

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1)


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


def process_keypoints(keypoints, bboxes):
    """
    Process the keypoints to minimize the domain gap between synthetic and COCO keypoints.
    1. Normalize the keypoints to be in the range [0, 1] with respect to the bounding box
    2. Remove keypoints with visibility < 2
    """

    keypoints = np.reshape(keypoints, (-1, 17, 3))
    
    # Normalize the keypoints to be in the range [0, 1] with respect to the bounding box
    # bboxes = bboxes[:, None, :]
    # keypoints[:, :, 0] = (keypoints[:, :, 0] - bboxes[:, :, 0]) / bboxes[:, :, 2]
    # keypoints[:, :, 1] = (keypoints[:, :, 1] - bboxes[:, :, 1]) / bboxes[:, :, 3]

    # Remove keypoints with visibility < 2
    visibilities = keypoints[:, :, 2].squeeze()
    keypoints[visibilities < 2, :] = 0

    keypoints = np.reshape(keypoints, (-1, 51))
    
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
    else:
        acos = np.arccos
        sin = np.sin
        cos = np.cos
        all = np.all
    if pts1.shape[1] == 3:
        theta1, phi1 = pts1[:, 1], pts1[:, 2]
        theta2, phi2 = pts2[:, 1], pts2[:, 2]
    else:
        theta1, phi1 = pts1[:, 0], pts1[:, 1]
        theta2, phi2 = pts2[:, 0], pts2[:, 1]

    dist = acos(sin(theta1)*sin(theta2) + cos(theta1)*cos(theta2)*cos(phi1 - phi2))

    assert all(dist >= 0)
    assert all(dist <= np.pi)

    return dist


if __name__ == "__main__":

    pts1 = np.array([[0, 0, -1]], dtype=np.float32)
    pts2 = np.array([[0, 0, 1]], dtype=np.float32)
    pts1 = np.random.normal(size = (100000, 3))
    pts2 = np.random.normal(size = (100000, 3))
    d = angular_distance(c2s(pts1), c2s(pts2))
    d = d * 180 / np.pi

    print(np.min(d), np.mean(d), np.max(d))