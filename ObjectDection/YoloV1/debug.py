"""
增加DEBUG调试函数：
1.调试数据预处理过程
"""
import os
import sys
import typing as t
import numpy as np
import cv2
from settings import DEBUG_OPEN

if not DEBUG_OPEN:
    # 关闭print打印
    sys.stdout = open(os.devnull, 'w')


class DebugYOLOv1(object):

    @staticmethod
    def draw_box(
            img: np.ndarray,
            top_left: t.Tuple[int, int],
            bottom_right: t.Tuple[int, int],
            color: t.Tuple[int, int, int] = (0, 0, 255)
    ) -> None:
        """
        根据top-left and bottom_right的 coordinate draw box
        """
        cv2.rectangle(img, top_left, bottom_right, color, 2)


debug_yolo_v1 = DebugYOLOv1()
