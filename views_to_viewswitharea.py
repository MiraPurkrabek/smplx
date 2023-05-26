import os
import argparse
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str)
    return parser.parse_args()


def main(args):

    views_dict = json.load(open(args.filepath, "r"))
    coco_dict = json.load(open(os.path.join(
        os.path.dirname(args.filepath),
        "person_keypoints_val2017.json"
    ), "r"))


    cumulavive_dict = {}
    # Requires only one annotation per image!
    for annotation in coco_dict["annotations"]:
        image_id = annotation['image_id']
        image_name = "{:d}.jpg".format(image_id)
        
        visible_keypoints = np.array(annotation["keypoints"])[2::3]
        counts = np.zeros(3)
        for i in range(3):
            counts[i] = np.sum(visible_keypoints == i)
        counts = counts.astype(int)

        cumulavive_dict[image_name] = {
            "camera_position": views_dict[image_name]["camera_position"],
            "area": annotation["area"],
            "visible_keypoints": counts.flatten().tolist(),
            "bbox": annotation["bbox"]
        }
        if "oks_score" in views_dict[image_name].keys():
            cumulavive_dict[image_name]["oks_score"] = views_dict[image_name]["oks_score"]
        
    json.dump(cumulavive_dict, open(os.path.join(
        os.path.dirname(args.filepath),
        "rich_" + os.path.basename(args.filepath)
    ), "w"), indent=2)

if __name__ == "__main__":
    args = parse_args()
    main(args)