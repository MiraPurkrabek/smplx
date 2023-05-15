import json
import numpy as np


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
    bboxes = bboxes[:, None, :]
    keypoints[:, :, 0] = (keypoints[:, :, 0] - bboxes[:, :, 0]) / bboxes[:, :, 2]
    keypoints[:, :, 1] = (keypoints[:, :, 1] - bboxes[:, :, 1]) / bboxes[:, :, 3]

    # Remove keypoints with visibility < 2
    visibilities = keypoints[:, :, 2].squeeze()
    keypoints[visibilities < 2, :] = 0

    keypoints = np.reshape(keypoints, (-1, 51))
    
    return keypoints

# def cartesian_to_spherical(pts):
#     # pts: Nx3
#     x = pts[:, 0]
#     y = pts[:, 1]
#     z = pts[:, 2]

#     radius = np.linalg.norm(pts, axis=1)

#     theta = np.arctan2(y, x)
#     phi = np.arctan2(np.sqrt(x * x + y * y), z)

#     return np.stack([radius, theta, phi], axis=1)
# def spherical_to_cartesian(pts):
#     # pts: Nx3
#     radius = pts[:, 0]
#     theta = pts[:, 1]
#     phi = pts[:, 2]

#     x = radius * np.cos(theta) * np.sin(phi)
#     y = radius * np.sin(theta) * np.sin(phi)
#     z = radius * np.cos(phi)

#     return np.stack([x, y, z], axis=1)