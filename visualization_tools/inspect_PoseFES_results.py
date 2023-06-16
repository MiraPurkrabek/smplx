import os
import argparse
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, interp1d

import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cocopath", type=str)
    parser.add_argument("--improvement", action="store_true", default=False)
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--timeline", action="store_true", default=False)
    parser.add_argument("--show-results", action="store_true", default=False)
    args = parser.parse_args()

    if args.baseline is None:
        args.baseline = os.path.join(
            os.path.dirname(args.cocopath),
            "coco_dict_with_oks_baseline.json"
        )

    return args


def show_scatterplot(bbox_centers, scores, image_paths, improvement=False):
    
    # Normalize x and y between 0 and 1
    bbox_centers[:, 0] -= np.min(bbox_centers[:, 0])
    bbox_centers[:, 0] /= np.max(bbox_centers[:, 0])
    bbox_centers[:, 1] -= np.min(bbox_centers[:, 1])
    bbox_centers[:, 1] /= np.max(bbox_centers[:, 1])
    bbox_centers[:, 1] = 1-bbox_centers[:, 1]           # Flip y-axis to correspond to image coordinates
    
    # Create the scatter plot
    fig, ax = plt.subplots()
    print(np.min(scores), np.max(scores))
    vmin = -1 if improvement else 0.0
    vmax = 1 if improvement else 1
    scatter = ax.scatter(bbox_centers[:, 0], bbox_centers[:, 1], c=scores, cmap='jet')#, vmin=vmin, vmax=vmax)
    
    im = OffsetImage(plt.imread(image_paths[0]), zoom=1)
    xybox=(150., 150.)
    ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
            boxcoords="offset points",  pad=0.1,  arrowprops=dict(arrowstyle="->"))
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)

    # Add hover tooltips to show the corresponding image
    def hover_tooltip(event):
        if scatter.contains(event)[0]:
            ind = scatter.contains(event)[1]["ind"][0]

            w,h = fig.get_size_inches()*fig.dpi
            ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
            hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
            ab.xybox = (xybox[0]*ws, xybox[1]*hs)
            
            ab.set_visible(True)
            ab.xy = bbox_centers[ind, :].flatten()

            img = plt.imread(image_paths[ind])
            img = cv2.resize(img, (250, 250))
            image_name = ".".join(os.path.basename(image_paths[ind]).split(".")[:-1])
            image_id = int(image_name.split("_")[-2])
            bbox_id = int(image_name.split("_")[-1])
            img = cv2.putText(
                img, 
                "{:d}-{:d}".format(image_id, bbox_id),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(1, 0, 0),
                thickness=1,
            )
            img = cv2.putText(
                img, 
                "{:.2f}".format(scores[ind]),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(1, 0, 0),
                thickness=1,
            )
            im.set_data(img)

        else:
            ab.set_visible(False)
        fig.canvas.draw_idle()
        
    fig.canvas.mpl_connect('motion_notify_event', hover_tooltip)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Bounding Box Centers')

    # Set the grid
    ax.grid(which='both', alpha=0.5)

    # Add colorbar
    cbar = fig.colorbar(scatter)
    cbar.set_label('Scores')

    # Display the scatter plot
    plt.show()


def extract_score_from_ann(ann, id2name, folder_with_images, timeline=False):
    img_name = id2name[ann["image_id"]]
    image_path = os.path.join(
        folder_with_images,
        img_name.replace(".png", "_{:05d}.png".format(ann["id"])),
    )
            
    bbox_wh = np.array(ann["bbox"])
    bbox_center = bbox_wh[:2] + bbox_wh[2:] / 2
    
    if timeline:
        oks = int(ann["image_id"])
    elif "oks_score" in ann.keys():
        oks = ann["oks_score"]
    elif "oks" in ann.keys():
        oks = ann["oks"]
    else:
        oks = 1

    return bbox_center, oks, image_path


def extract_scores(coco_dict, folder_with_images, baseline_dict=None, timeline=False):
    image_paths = []  
    bbox_centers = []
    scores = []

    id2name = {}
    for img in coco_dict["images"]:
        id2name[img["id"]] = img["file_name"]

    if baseline_dict is None:
        for ann in coco_dict["annotations"]:
            bbox_center, oks, image_path = extract_score_from_ann(
                ann,
                id2name,
                folder_with_images,
                timeline=timeline,
            )
            bbox_centers.append(bbox_center)
            scores.append(oks)
            image_paths.append(image_path)
    else:
        for ann1, ann2 in zip(coco_dict["annotations"], baseline_dict["annotations"]):
            assert ann1["id"] == ann2["id"], "Annotation IDs do not match"
            assert ann1["image_id"] == ann2["image_id"], "Image IDs do not match"

            bbox_center, oks1, image_path = extract_score_from_ann(
                ann1,
                id2name,
                folder_with_images,
                timeline=timeline,
            )
            _, oks2, _ = extract_score_from_ann(
                ann2,
                id2name,
                folder_with_images,
                timeline=timeline,
            )
            bbox_centers.append(bbox_center)
            scores.append(oks1 - oks2)
            image_paths.append(image_path)

    bbox_centers = np.array(bbox_centers)
    scores = np.array(scores)

    nan_mask = np.isnan(scores)
    bbox_centers = bbox_centers[~nan_mask, :]
    scores = scores[~nan_mask]

    return bbox_centers, scores, image_paths


def main(args):

    coco_dict = json.load(open(args.cocopath, "r"))

    baseline_dict = None
    if args.improvement:
        baseline_dict = json.load(open(args.baseline, "r"))

    if args.show_results:
        model = os.path.basename(args.cocopath)
        model = ".".join(model.split(".")[:-1])
        model = model.split("_")[-1]
        folder_with_images = os.path.join(
            os.path.dirname(args.cocopath),
            "..",
            "test_visualization",
            "ViTPose_small_coco_256x192_{:s}".format(model),
        )
    else:
        folder_with_images = os.path.join(
            os.path.dirname(args.cocopath),
            "..",
            "val2017_vis_samples",
        )
    
    assert os.path.exists(folder_with_images), "Folder with images ({:s}) does not exist".format(folder_with_images)

    bbox_centers, scores, image_paths = extract_scores(
        coco_dict, 
        folder_with_images, 
        baseline_dict=baseline_dict,
        timeline=args.timeline,
    )

    show_scatterplot(bbox_centers, scores, image_paths, args.improvement)


if __name__ == "__main__":
    args = parse_args()
    main(args)