import os
import argparse
import json
import numpy as np
from posevis.visualization import pose_visualization
from tqdm import tqdm
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str)
    return parser.parse_args()


def main(args):

    coco_dict = json.load(open(args.filepath, "r"))

    id2name = {}
    for img in coco_dict["images"]:
        id2name[img["id"]] = img["file_name"]

    new_folder_path = os.path.join(
        os.path.dirname(args.filepath),
        "..",
        "val2017_vis_samples",
    )
    os.makedirs(new_folder_path, exist_ok=True)
    
    for annotation in tqdm(coco_dict["annotations"], ascii=True):
        image_id = annotation["image_id"]
        image_name = id2name[image_id]
        image_path = os.path.join(
            os.path.dirname(args.filepath),
            "..",
            "val2017",
            image_name,
        )
        
        keypoints = np.array(annotation["keypoints"]).reshape(-1, 3)

        new_image_path = os.path.join(
            new_folder_path,
            image_name.replace(".png", "_{:05d}.png".format(annotation["id"])),
        )   
        
        img = pose_visualization(image_path, keypoints)

        # Crop the image by the bounding box
        bbox = annotation["bbox"]
        bbox = np.array(bbox).astype(int)
        img = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :]

        # Save the image
        cv2.imwrite(new_image_path, img)


if __name__ == "__main__":
    args = parse_args()
    main(args)