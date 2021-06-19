import os
import math

import cv2
import torch
import numpy as np
import matplotlib as plt

from src.yolov3.utils.utils import xywh2xyxy, plot_one_box

def plot_yolo_bbox(
    images, targets, paths=None, names=None, max_size=640, max_subplots=16
):
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness

    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    # Fix class - colour map
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    hex2rgb = lambda h: tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()["color"]]

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y : block_y + h, block_x : block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype("int")
            gt = image_targets.shape[1] == 6  # ground truth if no conf column
            conf = (
                None if gt else image_targets[:, 6]
            )  # check for confidence presence (gt vs pred)

            boxes[[0, 2]] *= w
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] *= h
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = color_lut[cls % len(color_lut)]
                cls = names[cls] if names else cls
                if gt or conf[j] > 0.3:  # 0.3 conf thresh
                    label = "%s" % cls if gt else "%s %.1f" % (cls, conf[j])
                    plot_one_box(
                        box, mosaic, label=label, color=color, line_thickness=tl
                    )

        # Draw image filename labels
        if paths is not None:
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(
                mosaic,
                label,
                (block_x + 5, block_y + t_size[1] + 5),
                0,
                tl / 3,
                [220, 220, 220],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

        # Image border
        cv2.rectangle(
            mosaic,
            (block_x, block_y),
            (block_x + w, block_y + h),
            (255, 255, 255),
            thickness=3,
        )

    mosaic = cv2.resize(
        mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA
    )

    return mosaic
