"""
定义YOLO v1的损失函数
"""
import torch
import torch.nn as nn
import debug

from settings import *
from util import calculate_iou


class YoloV1Loss(nn.Module):
    """
    定义LOSS
    根据网络输出的7x7x30的张量, 计算出两个prediction bbox和ground truth bbox的top-left和bottom-right
    的坐标, 根据坐标分别计算两个prediction bbox和gt bbox的iou, 选择iou大的那一个bbox作为负责检测obj的bbox,
    并构造它的损失函数, 随后在train过程中对loss function进行bp反向传播。
    """

    def __init__(self):
        super(YoloV1Loss, self).__init__()
        self.coord = 5.  # 对weight, height, x, y的损失赋予的权重
        self.no_obj = 0.5  # 对没有obj的bbox的损失赋予的权重

    def calculate_loss(
            self,
            data: torch.Tensor,
            labels: torch.Tensor,
    ):
        """
        构造损失函数
        """
        data = data.double()
        labels = labels.double()
        batch_size = labels.size(0)
        obj_w_h_loss = 0.  # 包含目标的bbox的width,height损失
        obj_co_loss = 0.  # 包含目标的bbox坐标损失
        obj_confi_loss = 0.  # 包含目标的bbox的置信度损失
        no_obj_confi_loss = 0.  # 不包含目标的bbox的置信度损失
        obj_cls_loss = 0.  # 包含目标的类别概率损失
        for i in range(batch_size):
            for c in range(GRID_NUM):  # x轴
                for r in range(GRID_NUM):  # y轴
                    # 该grid中包含obj
                    if labels[i, r, c, 4] == 1:
                        # 1.将labels中相对grid中心点坐标cx, cy转换成相对bbox的中心点坐标bx,by
                        # 2.根据bx, by计算top-left坐标 和 bottom-right坐标
                        """
                            cx = bx * 448 / 64 - int(bx * 448 / 64)
                            cy = by * 448 / 64 - int(bx * 448 / 64)
                            其中, int(bx * 448 / 64) == c; int(by * 448 / 64) == r
                            
                            ==>
                            bx = (cx + c) / 7
                            by = (cy + r) / 7
                        """
                        pred_bbox1_xy = ((data[i, r, c, 0] + c) / GRID_NUM - data[i, r, c, 2] / 2,
                                         (data[i, r, c, 1] + r) / GRID_NUM - data[i, r, c, 3] / 2,
                                         (data[i, r, c, 0] + c) / GRID_NUM + data[i, r, c, 2] / 2,
                                         (data[i, r, c, 1] + r) / GRID_NUM + data[i, r, c, 3] / 2)
                        pred_bbox2_xy = ((data[i, r, c, 5] + c) / GRID_NUM - data[i, r, c, 7] / 2,
                                         (data[i, r, c, 6] + r) / GRID_NUM - data[i, r, c, 8] / 2,
                                         (data[i, r, c, 5] + c) / GRID_NUM + data[i, r, c, 7] / 2,
                                         (data[i, r, c, 6] + r) / GRID_NUM + data[i, r, c, 8] / 2)
                        gt_bbox_xy = ((labels[i, r, c, 0] + c) / GRID_NUM - labels[i, r, c, 2] / 2,
                                      (labels[i, r, c, 1] + r) / GRID_NUM - labels[i, r, c, 3] / 2,
                                      (labels[i, r, c, 0] + c) / GRID_NUM + labels[i, r, c, 2] / 2,
                                      (labels[i, r, c, 1] + r) / GRID_NUM + labels[i, r, c, 3] / 2)

                        iou1 = calculate_iou(pred_bbox1_xy, gt_bbox_xy)
                        iou2 = calculate_iou(pred_bbox2_xy, gt_bbox_xy)
                        # 选择iou中最大的为正样本, 剩余的标记为负样本, 此处只有2个bbox, 故直接采用if判断
                        if iou1 >= iou2:
                            # 构造iou1对应的bbox的损失函数
                            obj_co_loss = obj_co_loss + self.coord * torch.sum(
                                (data[i, r, c, 0: 2] - labels[i, r, c, 0: 2]) ** 2
                            )
                            obj_w_h_loss = obj_w_h_loss + self.coord * torch.sum(
                                (torch.sqrt(data[i, r, c, 2: 4]) - torch.sqrt(labels[i, r, c, 2: 4])) ** 2
                            )
                            # 对于正样本来说, 置信度损失的标签为正样本对应的iou, 即iou1
                            # 或者给定学习目标, 即置信度标签为1
                            obj_confi_loss = obj_confi_loss + torch.sum(
                                (data[i, r, c, 4] - iou1) ** 2
                            )
                            # 对于负样本来说, 置信度损失的标签为0
                            no_obj_confi_loss = no_obj_confi_loss + self.no_obj * torch.sum(
                                (data[i, r, c, 9]) ** 2
                            )
                        else:
                            # 构造iou2对应的bbox的损失函数
                            obj_co_loss = obj_co_loss + self.coord * torch.sum(
                                (data[i, r, c, 5: 7] - labels[i, r, c, 5: 7]) ** 2
                            )
                            obj_w_h_loss = obj_w_h_loss + self.coord * torch.sum(
                                (torch.sqrt(data[i, r, c, 7: 9]) - torch.sqrt(labels[i, r, c, 7: 9])) ** 2
                            )
                            obj_confi_loss = obj_confi_loss + torch.sum(
                                (data[i, r, c, 9] - iou2) ** 2
                            )
                            no_obj_confi_loss = no_obj_confi_loss + self.no_obj * torch.sum(
                                (data[i, r, c, 4]) ** 2
                            )
                        obj_cls_loss = obj_cls_loss + torch.sum(
                            (data[i, r, c, 10:] - labels[i, r, c, 10:]) ** 2
                        )
                    else:
                        # 不包含物体时, 只计算不包含obj的置信度损失, 对应置信度标签为0
                        no_obj_confi_loss = no_obj_confi_loss + self.no_obj * torch.sum(
                            data[i, r, c, [4, 9]] ** 2
                        )
        loss = obj_co_loss + obj_w_h_loss + obj_confi_loss + no_obj_confi_loss + obj_cls_loss
        # 训练时使用SGD,这里只计算一个样本的损失
        return loss / batch_size
