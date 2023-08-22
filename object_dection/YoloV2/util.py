"""
工具
"""
import typing as t
import random

import cv2
import numpy as np
import torch
from torchvision import transforms

from argument import args_train
from settings import FEATURE_MAP_W, FEATURE_MAP_H


def normalization(*bbox_coordinate, **pic) -> t.Tuple[float, float, float, float]:
    """
    normalize pic_w, pic_h first, then multiply by feature map size.
    """
    pic_h = pic.get('pic_h')
    pic_w = pic.get('pic_w')
    n_w = 1.0 / pic_w * 416
    n_h = 1.0 / pic_h * 416
    mid_x = (bbox_coordinate[0] + bbox_coordinate[2]) / 2.0
    mid_y = (bbox_coordinate[1] + bbox_coordinate[3]) / 2.0
    bbox_w = bbox_coordinate[2] - bbox_coordinate[0]
    bbox_h = bbox_coordinate[3] - bbox_coordinate[1]
    return mid_x * n_w, mid_y * n_h, bbox_w * n_w, bbox_h * n_h


def shuffle(arr: t.List[str], arr_len: int) -> None:
    """
    基于Fisher-Yates的洗牌算法打乱arr
    时间复杂度为O(N)
    """
    for i in range(arr_len - 1):
        idx = random.randint(i, arr_len - 1)
        arr[i], arr[idx] = arr[idx], arr[i]


def trans_anchors(anchors_size: t.List[t.List]):
    """
    转换anchors格式, 由列表转为ndarray
    [[w1, h1], [w2,h2]...] -->

    [[0, 0, w1, h1]
     [0, 0, w2, h2]
     ...
     [0, 0, wn, hn]
    ]
    """
    anchors = np.zeros((len(anchors_size), 4))
    for idx, size in enumerate(anchors_size):
        anchors[idx, 2:] = np.array([size[0], size[1]])
    return anchors


def compute_iou_gt_anchors(anchor_boxes: np.ndarray[np.ndarray], gt_box: np.ndarray) -> np.ndarray:
    """
    Calculate the iou of K (yolo v2=5) anchors and gt box for a gt box (grid)

    Input:
        anchor_boxes : ndarray -> [[0, 0, anchor_w1, anchor_h1], ..., [0, 0, anchor_wn, anchor_hn]].
        gt_box : ndarray -> [0, 0, gt_w, gt_h].
    Output:
        iou : ndarray -> [iou_1, iou_2, ..., iou_m], and m is equal to the number of anchor boxes.

    algorithm idea:
    1. calculate the coordinate of top-left and bottom-right based on anchor_boxes and gt_box respectively
    2. calculate the area of anchor box and ground-truth box respectively
    3. calculate the intersection area and union area, then compute the IOU of anchor boxes and ground-truth box.
    """
    len_a = len(anchor_boxes)
    anchor_x1y1_x2y2 = np.zeros((len_a, 4))
    gt_x1y1_x2y2 = np.zeros((len_a, 4))
    gt_box = gt_box.repeat(len_a, axis=0)

    anchor_x1y1_x2y2[:, 0] = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2  # x_min
    anchor_x1y1_x2y2[:, 1] = anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2  # y_min
    anchor_x1y1_x2y2[:, 2] = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2  # x_max
    anchor_x1y1_x2y2[:, 3] = anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2  # y_max

    gt_x1y1_x2y2[:, 0] = gt_box[:, 0] - gt_box[:, 2] / 2  # x_min
    gt_x1y1_x2y2[:, 1] = gt_box[:, 1] - gt_box[:, 3] / 2  # y_min
    gt_x1y1_x2y2[:, 2] = gt_box[:, 0] + gt_box[:, 2] / 2  # x_min
    gt_x1y1_x2y2[:, 3] = gt_box[:, 1] + gt_box[:, 3] / 2  # y_min

    anchor_areas = anchor_boxes[:, 2] * anchor_boxes[:, 3]
    gt_areas = gt_box[:, 2] * gt_box[:, 3]

    inter_w = np.minimum(anchor_x1y1_x2y2[:, 2], gt_x1y1_x2y2[:, 2]) - np.maximum(anchor_x1y1_x2y2[:, 0],
                                                                                  gt_x1y1_x2y2[:, 0])
    inter_h = np.minimum(anchor_x1y1_x2y2[:, 3], gt_x1y1_x2y2[:, 3]) - np.maximum(anchor_x1y1_x2y2[:, 1],
                                                                                  gt_x1y1_x2y2[:, 1])
    intersection = inter_w * inter_h
    union = anchor_areas + gt_areas - intersection + 1e-15
    return intersection / union


def iou_pred_gt(x1y1_x2y2_pred: torch.Tensor, x1y1_x2y2_gt: torch.Tensor) -> torch.Tensor:
    """
    compute the iou between predicted box and ground-truth box.
    Output:
        iou: tensor -> [B, ws*hs*anchor_num, 1]
    """

    inter_w = torch.minimum(x1y1_x2y2_pred[..., 2], x1y1_x2y2_gt[..., 2]) - torch.maximum(x1y1_x2y2_pred[..., 0],
                                                                                          x1y1_x2y2_gt[..., 0])
    inter_h = torch.minimum(x1y1_x2y2_pred[..., 3], x1y1_x2y2_gt[..., 3]) - torch.maximum(x1y1_x2y2_pred[..., 1],
                                                                                          x1y1_x2y2_gt[..., 1])
    intersection = inter_w * inter_h

    pred_area = torch.prod(x1y1_x2y2_pred[..., 2:] - x1y1_x2y2_pred[..., :2], 2)
    gt_area = torch.prod(x1y1_x2y2_gt[..., 2:] - x1y1_x2y2_gt[..., :2], 2)

    # pred_area = (x1y1_x2y2_pred[..., 2] - x1y1_x2y2_pred[..., 0]) * (x1y1_x2y2_pred[..., 3] - x1y1_x2y2_pred[..., 1])
    # gt_area = (x1y1_x2y2_gt[..., 2] - x1y1_x2y2_gt[..., 0]) * (x1y1_x2y2_gt[..., 3] - x1y1_x2y2_gt[..., 1])

    union = pred_area + gt_area - intersection + 1e-15
    return intersection / union


def divide_samples(
        gt_label: t.List,
        width: int,
        height: int,
        anchors_size: t.List[t.List],
        stride: int = 32
) -> t.List[t.List]:
    """
    divide positive, ignore and negative samples

    Input:
        gt_label: list -> [x_min, y_min, x_max, y_max, cls_id]
        anchors_size: list -> [[w1, h1], [w2, h2],...]
        ...
    Output:
        res: list -> [a_idx, grid_y, grid_x, tx, ty, tw, th, size_weight, x_min, y_min, x_max, y_max]
    Algorithm ideas:

    1.Calculate the center point and width, height of the ground-truth box, which relates to the entire img.
    then map the width, height and the center point of gt-box to the grid, in Yolo-V2, grid is 13x13.

    2.set the center point of gt and anchors to 0, as the anchors of each grid are relative to the center of the grid,
    then calculate iou using length and width of the 5 anchors computed through clustering algorithm.

    3. making positive samples based on IOUs and anchors threshold. the IOUs obtained by anchor boxes and gt label

    4. if multiple IOUs are greater than the threshold, then select the anchor corresponding to the largest IOU as the
    positive sample, ignore the ones that are greater than the threshold but not the largest, and the rest are negative
    samples; Also, if none of the IOUs is greater than threshold, select the anchor according to the largest IOU as the
    positive sample, and the rest as the negative sample.
    """

    x_min, y_min, x_max, y_max = gt_label[:-1]
    c_x_grid = (x_min + x_max) / 2 * width / stride
    c_y_grid = (y_min + y_max) / 2 * height / stride
    w_grid = (x_max - x_min) * width / stride
    h_grid = (y_max - y_min) * height / stride

    grid_x = int(c_x_grid)
    grid_y = int(c_y_grid)

    anchor_boxes = trans_anchors(anchors_size)
    gt_box = np.array([[0, 0, w_grid, h_grid]])

    iou_plural = compute_iou_gt_anchors(anchor_boxes, gt_box)

    iou_positive = (iou_plural > args_train.opts.anchors_thresh)
    res = []
    largest_a_idx = np.argmin(iou_plural)
    for a_idx, iou in enumerate(iou_positive):
        if a_idx == largest_a_idx:
            pw, ph = anchors_size[a_idx]
            tx = c_x_grid - grid_x
            ty = c_y_grid - grid_y
            tw = np.log(w_grid / pw)
            th = np.log(h_grid / ph)
            """
            add weight to the size of the identified target, the weight of the small target is greater than the
            weight of the larger target, in order to increase learning on the small target.
            """
            size_weight = 2.0 - (x_max - x_min) * (y_max - y_min)
            res.append([a_idx, grid_y, grid_x, tx, ty, tw, th, size_weight, x_min, y_min, x_max, y_max])
        else:
            # used to indicate ignored anchors, when we know it, we also know which anchors are negative samples
            res.append([a_idx, grid_y, grid_x, 0., 0., 0., 0., -1.0, 0., 0., 0., 0.])
    return res


def label_generator(
        input_size: int,
        label_list: t.List[t.List],
        anchors_size: t.List[t.List],
        stride: int = 32
) -> np.ndarray:
    """
    Generator for ground truth box
    Input:
        input_size: list -> the size of image in the training stage
        label_list: list -> [[[x_min, y_min, x_max, y_max], ...], [], []]
        anchor_size: list -> [[w1, h1],[w2,h2],[w3,h3] ...]
    Output:
        gt_tensor: ndarray -> [B, ws * hs * anchor_num, 1 + 1 + 4 + 1 + 4]
        1: confidence
        1: class
        4: tx, ty, tw, th
        1: box size weight
        4. x_min, y_min, x_max, y_max

    note:
        when size_weight > 0., it means positive sample.
        when size_weight < 0., it means negative sample.
        when size_weight == 0. it means ignore sample.
    """

    batch_size = len(label_list)
    h = w = input_size
    ws, hs = input_size // stride, input_size // stride
    anchor_number = len(anchors_size)

    gt_tensor = np.zeros([batch_size, hs, ws, anchor_number, 1 + 1 + 4 + 1 + 4])

    for b_idx, batch in enumerate(label_list):
        for gt_label in batch:
            gt_label: t.List
            gt_class = gt_label[-1]
            results = divide_samples(gt_label, w, h, anchors_size, stride)
            for res in results:
                a_idx, grid_x, grid_y, tx, ty, tw, th, size_weight, x_min, y_min, x_max, y_max = res
                # positive sample
                if size_weight > 0.:
                    gt_tensor[b_idx, grid_x, grid_y, a_idx, 0] = 1.
                    gt_tensor[b_idx, grid_x, grid_y, a_idx, 1] = gt_class
                    gt_tensor[b_idx, grid_x, grid_y, a_idx, 2: 6] = np.array([tx, ty, tw, th])
                    gt_tensor[b_idx, grid_x, grid_y, a_idx, 6] = size_weight
                    gt_tensor[b_idx, grid_x, grid_y, a_idx, 7:] = np.array([x_min, y_min, x_max, y_max])
                # negative sample
                else:
                    gt_tensor[b_idx, grid_x, grid_y, a_idx, 0] = 0.
                    gt_tensor[b_idx, grid_x, grid_y, a_idx, 6] = -1.

    gt_tensor = gt_tensor.reshape((batch_size, ws * hs * anchor_number, -1))
    return gt_tensor


def detection_collate(batch: t.Iterable[t.Tuple]):
    """
    custom collate func for dealing with batches of images that have a different number
    of object annotations (bbox).

    by the way, this func is used to customize the content returned by the dataloader.
    """

    labels = []
    images = []
    for img, label in batch:
        images.append(img)
        labels.append(label)
    return torch.stack(images, dim=0), labels


def retrieve_anchors(file: str) -> t.List[t.List[float]]:
    """
    retrieve anchors from file
    Output:
        res: List[Tuple] -> [[w1, h1], [w2,h2]...]
    """
    res = []
    with open(file, 'r') as f:
        anchor_info = f.read().split(' ')
        for i in range(0, len(anchor_info), 2):
            res.append([float(anchor_info[i]), float(anchor_info[i + 1])])
    return res


class Augmentation(object):
    """
    image augmentation
    """

    def __init__(self, size: int = 416):
        self.size = size
        self.augment = transforms.Compose([
            self.resize
        ])

    def __call__(self, img):
        return self.augment(img)

    def resize(self, img) -> np.ndarray:
        return cv2.resize(img, (self.size, self.size))


if __name__ == '__main__':
    anchor_boxes_ = np.array([[0, 0, 50, 50], [0, 0, 25, 25]])
    gt_box_ = np.array([[0, 0, 40, 40]])
    r = compute_iou_gt_anchors(anchor_boxes_, gt_box_)

    labels_ = [[]]
    label_generator(416, [[[0.5, 0.5, 0.5, 0.5, 12], [0.5, 0.5, 0.5, 0.5, 8]]],
                    [[10.5, 9.5], [25.6, 15.5], [150.2, 132.5]])
    print(r)
