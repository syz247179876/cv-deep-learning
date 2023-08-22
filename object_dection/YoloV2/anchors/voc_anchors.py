"""
使用聚类算法生成Anchor Boxes
"""
import os
import numpy
import numpy as np
import typing as t
from settings import *

from argument import Args


class GenerateAnchorBoxes(object):
    """
    基于聚类算法生成Anchor Boxes
    算法K-means步骤：
    1.随机选取K个box作为初始anchor
    2.使用IOU作为距离度量, 将每个box分配给其距离最近的anchor
    3.计算每个簇中所有box的宽和高的均值, 更新anchor
    4.重复2、3步，直到anchor不在变化或者达到了最大迭代次数

    Annotation: 对Box进行归一化后, 可假设所有box都位于左上角
    """

    def __init__(
            self,
            k: t.Optional[int] = None
    ):
        args = Args()
        args.set_process_args()
        self.k = k or args.opts.anchors_num
        self.max_iter = args.opts.max_iter
        self.random_seed = args.opts.random_seed
        self.base_dir = args.opts.base_dir
        self.cur_iter = 0
        self.anchors = None
        self.labels = None  # 记录每个样本所属的anchor簇
        self.iou_plural = None

    def retrieve_bbox(self) -> np.ndarray[np.ndarray]:
        """
        读取预处理后的bbox的文件数据, 构造(nx2)结构的只含w, d的 ndarray
        """
        labels_dir = os.path.join(self.base_dir, f'{MODEL_NAME}_Labels')
        bbox_filename = os.listdir(labels_dir)
        res = []
        for filename in bbox_filename:
            file_path = os.path.join(labels_dir, filename)
            with open(file_path, 'r') as f:
                bboxes_info = f.read().split('\n')[:-1]
            bboxes_info = [bbox.split()[3: 5] for bbox in bboxes_info]
            for box in bboxes_info:
                res.append(box)
            # res.extend(bboxes_info)
        return np.array(res).astype(np.float32)

    @staticmethod
    def iou(boxes: np.ndarray[np.ndarray], anchors: np.ndarray[np.ndarray]):
        """
        计算gt boxes中每个box与所有anchors的IOU, 不考虑box的位置，只考虑box的大小。
        假设所有box的左上顶点都在原点, 因为其实相对于每个grid都要找到相同的K个大小形状的anchors
        boxes: shape(n x 2)
        anchors: shape(k x 2)
        ==> shape(n, k)
        """
        w_min = np.minimum(boxes[:, 0, np.newaxis], anchors[np.newaxis, :, 0])
        h_min = np.minimum(boxes[:, 1, np.newaxis], anchors[np.newaxis, :, 1])
        inter = w_min * h_min  # n x k
        box_area = boxes[:, 0] * boxes[:, 1]
        anchors_area = anchors[:, 0] * anchors[:, 1]
        # 广播, (nx1) + (1xk) ==> (nxk + nxk) => (nxk)
        union = box_area[:, np.newaxis] + anchors_area[np.newaxis]

        return inter / (union - inter)

    def k_means(self, boxes: numpy.ndarray):
        assert self.k < len(boxes), "K must be less than the number of data."

        if self.cur_iter > 0:
            self.cur_iter = 0
        np.random.seed(self.random_seed)
        n = boxes.shape[0]
        self.labels = np.zeros((n,))

        # 随机在boxes中选择K个box作为初始anchors
        self.anchors = boxes[np.random.choice(n, self.k, replace=True)]

        while True:
            self.cur_iter += 1
            if self.cur_iter > self.max_iter:
                break

            self.iou_plural = self.iou(boxes, self.anchors)
            distance = 1 - self.iou_plural
            # 将每一个box归属到距离其最近的anchors box中
            cur_labels = np.argmin(distance, axis=1)

            # anchors不在变化
            if (cur_labels == self.labels).all():
                break

            # 利用每个anchors簇中的所有bbox的长、宽更新每个anchors
            for i in range(self.k):
                self.anchors[i] = np.mean(boxes[cur_labels == i], axis=0)

            self.labels = cur_labels

    def save_anchors(self):
        """
        存储计算出的anchors的位置
        """
        anchors_dir = os.path.join(self.base_dir, ANCHORS_DIR)
        if not os.path.exists(anchors_dir):
            os.mkdir(anchors_dir)
        with open(os.path.join(anchors_dir, 'voc_anchors.txt'), 'w') as f:
            f.write(' '.join([str(c / DOWN_SAMPLING) for anchor in self.anchors for c in anchor]))

    def avg_iou(self):
        """
        针对所有样本, 计算其与所属的簇的IOU的平均值
        筛选区间self.iou_plural[(n, ), (n, )]
        """
        return np.mean(self.iou_plural[np.arange(len(self.labels)), self.labels])


def statistic():
    """
    比较并挑选较好的K值
    """
    _obj = GenerateAnchorBoxes()
    _boxes = _obj.retrieve_bbox()
    for i in range(2, 11):
        _obj.k = i
        _obj.k_means(_boxes)
        # obj.save_anchors()
        print(f'k={i}, avg iou={_obj.avg_iou()}')


if __name__ == "__main__":
    obj = GenerateAnchorBoxes()
    boxes_ = obj.retrieve_bbox()
    obj.k_means(boxes_)
    obj.save_anchors()
    print(f'k = 5, avg_iou={obj.avg_iou()}')
    # statistic()
