"""
单元测试模块, 用于测试模型的各个步骤的正确性
"""
import os
import sys

import cv2
import numpy
import numpy as np
import torch
import argparse
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

from settings import GRID_NUM, CLASSES, STRIDE, COLOR, STATIC_DATA_PATH
from util import calculate_iou, choice_bbox
from data import YOLOV1Dataset
from argument import Args


class YoloV1Test(object):
    """
    测试模型
    """

    def __init__(self, opts: argparse.Namespace):
        self.opts = opts

    def main(self):
        """
        测试入口
        1. 获取命令行参数
        2. 获取测试集, 分为自己的测试级或voc划分的测试集
        3. 加载网络模型
        4. 用网络模型对测试集进行测试，得到测试结果
        5. 计算测试集的评价指标， 或者可视化测试结果
        """

        test_dataset = YOLOV1Dataset(dataset_dir=self.opts.dataset_dir, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=self.opts.batch_size)

        # 加载训练好的网络模型
        model = torch.load(self.opts.model_dir)
        if self.opts.use_gpu:
            model.to(self.opts.gpu_id)

        with torch.no_grad():
            for batch, (x, _) in enumerate(test_loader):
                if self.opts.use_gpu:
                    x = x.to(self.opts.gpu_id)
                print(x.size())
                pred = model(x)
                pred = torch.squeeze(pred, dim=0)
                # 从网络输出中使用NMS筛选最优bbox, 先根据class, 再根据confidence筛选
                bbox: np.ndarray = choice_bbox(pred, self.opts.nms_threshold)
                img_path = test_dataset.use_img_path[batch]
                aim_img = cv2.imread(img_path)
                self.draw_bbox(aim_img, bbox)

    def test_my_dataset(self):
        """
        测试自定义的图片
        """
        my_img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'my_img')
        # my_img_dir = os.path.join(STATIC_DATA_PATH, 'YOLO_V1_ProcessImg')
        img_list = os.listdir(my_img_dir)
        # 加载训练好的网络模型
        model = torch.load(self.opts.model_dir)
        if self.opts.use_gpu:
            model.to(self.opts.gpu_id)
        for img_name in img_list:
            img_path = os.path.join(my_img_dir, img_name)
            img = cv2.imread(img_path)
            h, w, dim = img.shape
            # 先统一padding图像为正方形
            pad_w, pad_h = 0, 0
            if h > w:
                pad_w = (h - w) // 2
                img = np.pad(img, ((0, 0), (pad_w, pad_w), (0, 0)), 'constant', constant_values=0)
            elif h < w:
                pad_h = (w - h) // 2
                img = np.pad(img, ((0, 0), (pad_h, pad_h), (0, 0)), 'constant', constant_values=0)
            img = cv2.resize(img, (448, 448))
            cv2.imwrite(img_path, img)
            trans = transforms.Compose([
                transforms.ToTensor(),
            ])
            img = trans(img)
            img = img.unsqueeze(dim=0)
            if self.opts.use_gpu:
                img = img.to(self.opts.gpu_id)
            with torch.no_grad():
                pred = model(img)
                pred = torch.squeeze(pred, dim=0)
                # 从网络输出中使用NMS筛选最优bbox, 先根据class, 再根据confidence筛选
            bbox: np.ndarray = choice_bbox(pred, self.opts.nms_threshold)
            aim_img = cv2.imread(img_path)
            self.draw_bbox(aim_img, bbox, img_name)

    @staticmethod
    def draw_all_bbox(img: numpy.ndarray, pred: torch.Tensor) -> None:
        """
        绘制98个bbox
        """
        h, w = img.shape[0: 2]
        # 删去batch_size对应的维度, pred为7x7x30
        pred = torch.squeeze(pred, dim=0).cpu().numpy()
        for r in range(GRID_NUM):
            for c in range(GRID_NUM):
                # 构造bbox的top-left和bottom-right坐标
                confidence_1, confidence_2 = pred[r][c][4], pred[r][c][9]
                print(f'confidence_1:{confidence_1}, confidence_2: {confidence_2}')
                if confidence_1 > 0.15:
                    bbox1 = (int((pred[r][c][0] + c) * STRIDE - pred[r][c][2] * 224),
                             int((pred[r][c][1] + r) * STRIDE - pred[r][c][3] * 224),
                             int((pred[r][c][0] + c) * STRIDE + pred[r][c][2] * 224),
                             int((pred[r][c][1] + r) * STRIDE + pred[r][c][3] * 224))
                    # print(bbox1)
                    cv2.rectangle(img, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]), (0, 255, 0))
                if confidence_2 > 0.15:
                    bbox2 = (int((pred[r][c][5] + c) * STRIDE - pred[r][c][7] * 224),
                             int((pred[r][c][6] + r) * STRIDE - pred[r][c][8] * 224),
                             int((pred[r][c][5] + c) * STRIDE + pred[r][c][7] * 224),
                             int((pred[r][c][6] + r) * STRIDE + pred[r][c][8] * 224))
                    cv2.rectangle(img, (bbox2[0], bbox2[1]), (bbox2[2], bbox2[3]), (255, 0, 0))

        cv2.imshow("bbox", img)
        cv2.waitKey(0)

    @staticmethod
    def draw_bbox(img: numpy.ndarray, bbox: numpy.ndarray, *args):
        """
        根据bbox信息在img上绘制矩形框
        """
        h, w = img.shape[0: 2]
        num = bbox.shape[0]
        for i in range(num):
            confidence = bbox[i, 4]
            if confidence < 0.2:
                continue
            top_left = (int(w * bbox[i, 0]), int(h * bbox[i, 1]))
            bottom_right = (int(w * bbox[i, 2]), int(h * bbox[i, 3]))
            cls_name = CLASSES[int(bbox[i, 5])]
            cv2.rectangle(img, top_left, bottom_right, COLOR[int(bbox[i, 5])], 2)
            cv2.putText(img, cls_name, top_left, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0))
            cv2.putText(img, str(confidence), (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0))
        cv2.imshow(f"bbox", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    args = Args()
    args.set_test_args()  # 获取命令行参数
    test = YoloV1Test(args.opts)
    test.main()
    # test.test_my_dataset()
