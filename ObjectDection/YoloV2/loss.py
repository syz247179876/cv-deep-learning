import torch
import torchvision.models as tv_model
import typing as t
from torch import nn

from argument import Args
from settings import VOC_CLASSES_LEN
from util import iou_pred_gt


class YoloV2Loss(nn.Module):
    """
    YoloV2的损失函数
    """

    def __init__(self, input_size: int, stride: int = 32):
        """
        要支持多尺度多尺寸特征的训练
        """
        super(YoloV2Loss, self).__init__()
        args = Args()
        args.set_train_args()
        self.opts = args.opts
        self.input_size = input_size
        self.stride = stride
        self.anchor_size: torch.Tensor = torch.tensor(self.opts.anchors_num)
        self.grid_xy = None
        self.anchors_wh = None
        self.coord = 5.  # 对weight, height, x, y的损失赋予的权重
        self.no_obj = 0.5  # 对没有obj的bbox的损失赋予的权重

    def generate_grid(self) -> None:
        """
        生成grid矩阵
        grid_xy相关维度说明: [B, hs*ws, anchors_num, 4]
        """
        hs, ws = self.input_size // self.stride, self.input_size // self.stride
        # 划分网格, grid_y 按行划分, grid_x按列划分
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        # 用于连接两个大小相同的张量, 并扩张维度
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, ws * hs, 1, 2).to(self.opts.gpu_id)
        anchors_wh = self.anchor_size.repeat(hs * ws, 1, 1).unsqueeze(dim=0).float().to(self.opts.gpu_id)
        self.grid_xy = grid_xy
        self.anchors_wh = anchors_wh

    def decode_xy_wh(self, t_xy_wh_pred: torch.Tensor) -> torch.Tensor:
        """
        decode tx, ty, tw, th

        based on the model learning output tx, ty, tw, ty, selecting the anchors with the largest IOU, compute
        the predicted box that will be used to calculate the loss.

        Input:
            learning_pos: tensor -> [B, hs*ws, anchor_num, 4]
        Output:
            xy_wh_pred: tensor -> [B, hs*ws*anchor_num, 4]
        """
        b, hw, anchor_num, offset = t_xy_wh_pred.shape
        cx_cy_pred = torch.sigmoid(t_xy_wh_pred[..., :2]) + self.grid_xy
        wh_pred = torch.exp(t_xy_wh_pred[..., 2:]) + self.anchors_wh

        # [B, H * W, anchor_num, 4] --> [B, H*W*anchor_num, 4], 便于计算损失
        xy_wh_pred = torch.cat([cx_cy_pred, wh_pred], dim=-1).view(b, hw * anchor_num, offset) * self.stride
        return xy_wh_pred

    def decode_boxes(self, t_xy_wh_pred: torch.Tensor) -> torch.Tensor:
        """
        decode tx, ty, tw, th and cx, cy, w, h,

        based on [cx, cy, w, h], transform these into another structure, [x1, y1, x2, y3]
        Input:
            cx_cy_wh_pred: tensor -> [B, hs*ws, anchor_num, 4]
        Output:
            x1_y1_x2_y2_pred: tensor -> [B, hs*ws*anchor_num, 4]
        """

        xy_wh_pred = self.decode_xy_wh(t_xy_wh_pred)
        x1y1_pred = xy_wh_pred[..., :2] - xy_wh_pred[..., 2:] * 0.5
        x2y2_pred = xy_wh_pred[..., :2] + xy_wh_pred[..., 2:] * 0.5
        return torch.cat([x1y1_pred, x2y2_pred], dim=-1)

    def decode_pred(self, pred: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        decode tensors output by the network.

        the main function is to extract some info, such as confidence, class, coordinate,
        which are used to calculate loss.
        """
        b, h_w, info = pred.size()

        # decode predicted box confidence
        conf_pred: torch.Tensor = pred[:, :, :self.opts.anchors_num].contiguous().view(b, h_w * self.opts.anchors_num,
                                                                                       1)
        # decode predicted box class
        cls_pred: torch.Tensor = pred[:, :, self.opts.anchors_num: self.opts.anchors_num * (
                1 + VOC_CLASSES_LEN)].contiguous().view(
            b, h_w * self.opts.anchors_num, VOC_CLASSES_LEN
        )
        # decode predicted box coordinate offset, such as tx, ty, tw, th
        tx_ty_tw_th_pred: torch.Tensor = pred[:, :, self.opts.anchors_num * (1 + VOC_CLASSES_LEN):].contiguous().view(
            b, h_w, self.opts.anchors_num, 4
        )
        # decode predicted box coordinate, such as x1, y1, x2, y2
        x1y1_x2y2_pred = (self.decode_boxes(tx_ty_tw_th_pred) / torch.tensor(self.input_size))

        tx_ty_tw_th_pred = tx_ty_tw_th_pred.view(b, -1, 4)
        return conf_pred, cls_pred, tx_ty_tw_th_pred, x1y1_x2y2_pred

    def decode_gt(self, gt: torch.Tensor, x1y1_x2y2_pred: torch.Tensor) -> torch.Tensor:
        """
        decode gt boxes
        Input:
            gt: tensor -> [B, ws*hs*anchor_num, 4]
            x1y1_x2y2_pred: tensor -> [B, hs*ws*anchor_num, 4]
        Output:
        """

        x1y1_x2y2_gt = gt[:, :, 7:]
        batch_size, _, _ = gt.size()
        # as a confidence label of goal
        gt_conf = iou_pred_gt(x1y1_x2y2_pred, x1y1_x2y2_gt).view(batch_size, -1, 1)
        label = torch.cat([gt_conf, gt[:, :, :7]], dim=2)
        return label

    def forward(self, pred: torch.Tensor, labels: torch.Tensor):
        """
        forward propagation

        1.the confidence loss using MSE
        2.the coordinate of bbox loss using MSE
        3.the class of bbox loss using CrossEntropyLoss
        """
        conf_loss_function = nn.MSELoss(reduction='none')
        iter_loss_function = nn.MSELoss(reduction='none')
        cls_loss_function = nn.CrossEntropyLoss(reduction='none')
        bbox_loss_function = nn.MSELoss(reduction='none')
        iou_loss = nn.MSELoss(reduction='none')
        self.generate_grid()
        conf_pred, cls_pred, tx_ty_tw_th_pred, x1y1_x2y2_pred = self.decode_pred(pred)
        labels_gt = self.decode_gt(labels, x1y1_x2y2_pred)

        # pred info
        conf_pred = conf_pred[..., 0]
        cls_pred = cls_pred.permute(0, 2, 1)
        batch_size = conf_pred.size(0)

        # gt info
        conf_gt = labels_gt[..., 0].float()
        obj_gt = labels_gt[..., 1].float()
        cls_gt = labels_gt[..., 2].long()
        tx_ty_tw_th_gt = labels_gt[..., 3: 7].float()
        scale_box_gt = labels_gt[..., 7].float()

        negative_sample = (scale_box_gt < 0.).float()
        positive_sample = (scale_box_gt > 0.).float()

        # negative sample only need to calculate the confidence loss
        no_obj_conf_loss = torch.sum(
            conf_loss_function(torch.zeros_like(conf_pred), conf_pred) * negative_sample) / batch_size

        # if there are object in the bbox, add following loss
        # coordinate loss
        obj_coord_loss = torch.sum(torch.sum(iter_loss_function(tx_ty_tw_th_gt, tx_ty_tw_th_pred),
                                   dim=-1) * positive_sample * scale_box_gt) / batch_size

        # iou loss
        obj_conf_loss = torch.sum(conf_loss_function(conf_pred, conf_gt) * positive_sample) / batch_size

        # class loss
        obj_cls_loss = torch.sum(cls_loss_function(cls_pred, cls_gt)) / batch_size

        total_loss = no_obj_conf_loss + obj_cls_loss + obj_conf_loss + obj_coord_loss

        return total_loss


if __name__ == "__main__":
    s = YoloV2Loss(416)
    s.generate_grid()
