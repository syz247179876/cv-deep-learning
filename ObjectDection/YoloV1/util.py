"""
工具函数
"""
import typing as t
import numpy as np
import torch
from settings import CLASSES


def calculate_iou(bbox1: t.Tuple, bbox2: t.Tuple) -> float:
    """
    计算两个bbox的iou
    """
    # 当bbox1和bbox2没有交集时, 返回0
    if bbox1[2] <= bbox2[0] or bbox1[3] <= bbox2[1] or bbox2[2] <= bbox1[0] or bbox2[3] <= bbox1[1]:
        return 0
    # 重合区域的top-left和bottom-right的coordinate
    intersect_bbox = [0., 0., 0., 0.]
    intersect_bbox[0] = max(bbox1[0], bbox2[0])
    intersect_bbox[1] = max(bbox1[1], bbox2[1])
    intersect_bbox[2] = min(bbox1[2], bbox2[2])
    intersect_bbox[3] = min(bbox1[3], bbox2[3])

    # 重合区域面积
    w = max(intersect_bbox[2] - intersect_bbox[0], 0)
    h = max(intersect_bbox[3] - intersect_bbox[1], 0)
    area1 = w * h
    area2 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - area1 + 1e-6
    return area1 / area2


def choice_bbox(nn_output: torch.Tensor, iou_threshold: float) -> np.ndarray:
    """
    根据网络输出的7x7x30的张量, 采用多类别的NMS算法, 筛选出最优的一个bbox
    """
    assert nn_output.size() == (7, 7, 30), 'the dimension of network output is not (7, 7, 30)'
    nn_output: np.ndarray = nn_output.cpu().numpy()

    bboxes = np.zeros((98, 6))
    nn_output = nn_output.reshape(49, -1)

    # 处理前10列, 即包含两个bbox的cx, cy, w, h, confidence
    bbox = nn_output[:, :10].reshape(98, 5)

    # 构造行列grid,便于通过矩阵运算由相对grid边界的中心点cx, cy 推出相对pic的bbox的中心点bx, by
    r_grid, c_grid = np.array(list(range(7))), np.array(list(range(7)))

    # (7, ) -> (98,)
    r_grid = np.repeat(r_grid, repeats=14, axis=0)

    # (7, ) -> (1x14)
    c_grid = np.repeat(c_grid, repeats=2, axis=0)[np.newaxis, :]
    # (1x14) -> (7x14) -> (98, )
    c_grid = np.repeat(c_grid, repeats=7, axis=0).reshape(-1)

    # 计算bx, by, w, h, confidence
    bboxes[:, 0] = np.maximum((bbox[:, 0] + c_grid) / 7.0 - bbox[:, 2] / 2.0, 0)
    bboxes[:, 1] = np.maximum((bbox[:, 1] + r_grid) / 7.0 - bbox[:, 3] / 2.0, 0)
    bboxes[:, 2] = np.minimum((bbox[:, 0] + c_grid) / 7.0 + bbox[:, 2] / 2.0, 1)
    bboxes[:, 3] = np.minimum((bbox[:, 1] + r_grid) / 7.0 + bbox[:, 3] / 2.0, 1)
    # 预测时, 表示正样本候选区的概率, 网络会输出一个confidence值, 隐含包含了IOU
    bboxes[:, 4] = bbox[:, 4]
    # 选出每一个bbox的最大score对应的idx
    cls = np.argmax(nn_output[:, 10:], axis=1)
    cls = np.repeat(cls, repeats=2, axis=0)
    bboxes[:, 5] = cls
    # 对98个bbox执行NMS算法，筛选best bbox, 清楚confidence score较低 以及 iou大于指定阈值的bbox
    best_bboxes = nms_multi(bboxes, iou_threshold)
    return bboxes[best_bboxes]


def nms_single(bboxes: np.ndarray, iou_threshold: float) -> t.List[int]:
    """
    对一个类别所有的bbox使用nms挑选出best bboxes
    算法思想：
    - 数组原地删除算法
    1.首先将scores按照从大到小进行排列
    2.选出score最大的bbox, 将其与剩余的bbox计算iou
    3.将iou小于阈值的bbox保留，其余删除, 原地更新数组。
    4.循环步骤2-3,直至没有多余的bbox待筛选
    5.返回保留下来的best bboxes
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4]
    areas = (x2 - x1 + 1) * (y1 - y2 + 1)
    # 按照执行度降序排列
    order = np.argsort(-scores)

    # 用于保存最后保留的bbox
    keep = []
    while order.size > 0:
        top = order[0]
        keep.append(top)

        xx1 = np.maximum(x1[top], x1[order[1:]])
        yy1 = np.maximum(y1[top], y1[order[1:]])
        xx2 = np.minimum(x2[top], x2[order[1:]])
        yy2 = np.minimum(y2[top], y2[order[1:]])

        # 计算相交面积
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[top] + areas[order[1:]] - inter)
        idx = np.where(iou <= iou_threshold)[0]
        # 没有bbox需要保留了, 跳出循环
        if idx.size == 0:
            break
        m = idx + 1
        order = order[idx + 1]

    return keep


def nms_multi(bboxes: np.ndarray, iou_threshold: float) -> t.List[int]:
    """
    多类别的NMS算法
    算法思想：
    1.遍历所有类别(20个), 找出bboxes所有grid中最大score所对应的类别与当前遍历到的类别相同的bbox下标
    2.对同属一个类别下的这些bbox使用nsm算法。
    """
    res = []
    for i in range(len(CLASSES)):
        bboxes_idx: np.ndarray = np.where(bboxes[:, 5] == i)[0]
        bbox: np.ndarray = bboxes[bboxes_idx, 0: 5]

        # 理论上不存在的图片
        if bbox.shape[0] == 0 or bbox.shape[1] == 0:
            continue

        best_boxes: t.List[int] = nms_single(bbox, iou_threshold)
        res.append(bboxes_idx[best_boxes][0])
    return res


if __name__ == '__main__':
    # a = torch.randn((7, 7, 30))
    # choice_bbox(a, 0.5)
    r = calculate_iou((40, 40, 60, 60), (41, 41, 65, 65))
    print(r)
