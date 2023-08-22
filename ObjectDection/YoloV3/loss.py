import torch.nn as nn
import torch
import typing as t

from argument import args_train
from settings import *
from utils import compute_iou_gt_anchors


class YoloLoss(nn.Module):

    def __init__(
            self,
            anchors: t.List[t.List] = ANCHORS,
            classes_num: int = VOC_CLASS_NUM,
            input_shape: t.Tuple[int, int] = (416, 416),
    ):
        super(YoloLoss, self).__init__()
        self.anchors = anchors
        self.classes_num = classes_num
        self.input_shape = input_shape
        self.opts = args_train.opts
        self.use_gpu = args_train.opts.use_gpu
        self.gpu_id = args_train.opts.gpu_id
        self.balance = [0.5, 1.0, 5.0]
        self.box_ratio = 0.05
        self.cls_ratio = 1 * (5 + self.classes_num) / 80
        self.obj_ratio = 2 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        self.coord_mode = self.opts.coord_loss_mode

    def generate_positive_labels(
            self,
            level: int,
            labels: torch.Tensor,
            scaled_anchors: t.List[t.Tuple],
            f_h: int,
            f_w: int
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        divide positive samples

        divide positive samples strategy:
            *Based on max iou allocation criteria!
            *Calculate IOU for each ground-truth box and 9 anchors.

                The Yolo V3 in the paper used the max iou(single) based on anchors box and ground truth box.
            While following Yolo V4, I used the strategy that I divide all anchors that are greater than
            the positive sample threshold into positive samples.

        Input:
            labels: List[torch.Tensor] -> [[[x_mid, y_mid, w, h, cls_id], ...], [], []]
            x_mid, y_mid, w, h has been normalized

        """
        batch_size = len(labels)

        gt_tensor = torch.zeros(batch_size, ANCHORS_NUM, f_h, f_w, 5 + self.classes_num, requires_grad=False)
        # make net pay attention to small obj
        box_loss_scale = torch.zeros(batch_size, ANCHORS_NUM, f_h, f_w, requires_grad=False)

    def generate_ignore_labels(self):
        """
        divide positive samples

        divide positive samples strategy:
        *Calculate IOU for each ground-truth box and each prediction box
        *When the maximum anchors iou corresponding to all gt box are greater than the ignore threshold, divide it into ignore
        samples
        """
        pass

    def forward(self, level: int, pred: torch.Tensor, labels: torch.Tensor) -> t.Tuple[torch.Tensor, t.Tuple]:
        conf_loss_func = nn.BCELoss(reduction='mean')
        cls_loss_func = nn.BCELoss(reduction='mean')

        batch_size, _, g_h, g_w = pred.size()

        stride_h = self.input_shape[0] / g_h
        stride_w = self.input_shape[1] / g_w

        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]


class YoloV3Loss(nn.Module):

    def __init__(
            self,
            anchors: t.List[t.List] = ANCHORS,
            classes_num: int = VOC_CLASS_NUM,
            input_shape: t.Tuple[int, int] = (416, 416),
    ):
        super(YoloV3Loss, self).__init__()
        self.anchors = anchors
        self.classes_num = classes_num
        self.input_shape = input_shape
        self.opts = args_train.opts
        self.use_gpu = args_train.opts.use_gpu
        self.gpu_id = args_train.opts.gpu_id
        self.balance = [0.5, 1.0, 2.0]

        self.box_ratio = 0.05
        self.no_obj_ratio = 0.5
        self.cls_ratio = 1 * (5 + self.classes_num) / 80
        self.obj_ratio = 2 * (input_shape[0] * input_shape[1]) / (416 ** 2)

        self.coord_mode = self.opts.coord_loss_mode

    def generator_labels(
            self,
            level: int,
            labels: torch.Tensor,
            scaled_anchors: t.List[t.Tuple],
            f_h: int,
            f_w: int
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        divide positive labels, negative labels, ignore labels

        Input:
            labels: List[torch.Tensor] -> [[[x_mid, y_mid, w, h, cls_id], ...], [], []]
            x_mid, y_mid, w, h has been normalized
        """
        batch_size = len(labels)

        gt_tensor = torch.zeros(batch_size, ANCHORS_NUM, f_h, f_w, 5 + self.classes_num + 1, requires_grad=False)
        # make net pay attention to small obj
        box_loss_scale = torch.zeros(batch_size, ANCHORS_NUM, f_h, f_w, requires_grad=False)

        for batch_idx, batch_label in enumerate(labels):
            if len(batch_label) == 0:
                continue
            batch_tensor = torch.zeros_like(batch_label)
            # according to all labels in one batch, trans normalization to feature map size
            # batch_tensor[0:, [0,1,2,3,4]] => n_x_mid, n_y_mid, n_w, n_h , cls_id
            batch_tensor[:, [0, 2]] = batch_label[:, [0, 2]] * f_w
            batch_tensor[:, [1, 3]] = batch_label[:, [1, 3]] * f_h
            batch_tensor[:, 4] = batch_label[:, 4]

            """
            compute iou between anchors and labels, we only pay attention to the shape of w and h, not care center 
            position, as we can learn the offset continuously.
            """
            gt_box = torch.cat((torch.zeros((batch_tensor.size(0), 2)), batch_tensor[:, 2: 4]), 1).type(torch.float32)
            anchors_box = torch.cat((torch.zeros(len(scaled_anchors), 2), torch.tensor(scaled_anchors)), 1).type(
                torch.float32)

            # TODO: can we use the anchors relative to the current feature layer, not all anchors?
            iou_plural = compute_iou_gt_anchors(gt_box, anchors_box)
            # positive samples , dim = [b, 1]
            best_iou_plural = torch.argmax(iou_plural, dim=-1)

            # generate positive, ignore and negative samples' label
            for i, a_idxes in enumerate(iou_plural):
                # best anchor not in the anchors relative to the current level feature map
                best_a_idx = best_iou_plural[i]
                grid_x = torch.floor(batch_tensor[i, 0]).long()
                grid_y = torch.floor(batch_tensor[i, 1]).long()
                for a_id in range(len(a_idxes)):
                    if (a_id == best_a_idx) and a_id in ANCHORS_MASK[level]:
                        # access best anchor by using ANCHORS_MASK[level][aim_a_idx], 0 <= aim_a_dix <= 2
                        aim_a_idx = ANCHORS_MASK[level].index(a_id)
                        gt_tensor[batch_idx, aim_a_idx, grid_y, grid_x, 0] = batch_tensor[i, 0] - grid_x.float()
                        gt_tensor[batch_idx, aim_a_idx, grid_y, grid_x, 1] = batch_tensor[i, 1] - grid_y.float()
                        gt_tensor[batch_idx, aim_a_idx, grid_y, grid_x, 2] = torch.log(
                            batch_tensor[i, 2] / scaled_anchors[a_id][0])
                        gt_tensor[batch_idx, aim_a_idx, grid_y, grid_x, 3] = torch.log(
                            batch_tensor[i, 3] / scaled_anchors[a_id][1])
                        gt_tensor[batch_idx, aim_a_idx, grid_y, grid_x, 4] = 1
                        gt_tensor[batch_idx, aim_a_idx, grid_y, grid_x, 5 + batch_tensor[i, 4].long()] = 1
                        box_loss_scale[batch_idx, aim_a_idx, grid_y, grid_x] = batch_tensor[i, 2] * batch_tensor[
                            i, 3] / f_w / f_h
                        # set positive samples
                        gt_tensor[batch_idx, aim_a_idx, grid_y, grid_x, -1] = 1
                    elif a_idxes[a_id] < self.opts.anchors_negative_thresh and a_id in ANCHORS_MASK[level]:
                        # set negative samples
                        aim_a_idx = ANCHORS_MASK[level].index(a_id)
                        gt_tensor[batch_idx, aim_a_idx, grid_y, grid_x, -1] = -1

        return gt_tensor, box_loss_scale

    def forward(self, level: int, pred: torch.Tensor, labels: torch.Tensor) -> t.Tuple[torch.Tensor, t.Tuple]:
        """
        Input:
            level: it means the level of feature map, there are three level in YOLO v3 net, 13x13, 26x26, 52x52
        """
        conf_loss_func = nn.BCELoss(reduction='none')
        xy_loss_func = nn.BCELoss(reduction='none')
        wh_loss_func = nn.MSELoss(reduction='none')
        cls_loss_func = nn.BCELoss(reduction='none')

        batch_size, _, f_h, f_w = pred.size()
        stride_h = self.input_shape[0] / f_h
        stride_w = self.input_shape[1] / f_w

        scaled_anchors = [(anchor_w / stride_w, anchor_h / stride_h) for anchor_w, anchor_h in ANCHORS_SORT]
        pred = pred.view(batch_size, ANCHORS_NUM, 4 + 1 + self.classes_num, f_h, f_w). \
            permute(0, 1, 3, 4, 2).contiguous()

        tx = torch.sigmoid(pred[..., 0])
        ty = torch.sigmoid(pred[..., 1])
        tw = pred[..., 2]
        th = pred[..., 3]

        pred_conf = torch.sigmoid(pred[..., 4])
        pred_cls = torch.sigmoid(pred[..., 5:])

        gt_tensor, box_loss_scale = self.generator_labels(level, labels, scaled_anchors, f_h, f_w)
        avg_loss = torch.tensor(0.)
        loss_tx = torch.tensor(0.)
        loss_ty = torch.tensor(0.)
        loss_tw = torch.tensor(0.)
        loss_th = torch.tensor(0.)
        loss_cls = torch.tensor(0.)
        loss_conf = torch.tensor(0.)
        no_obj_loss_conf = torch.tensor(0.)
        if self.use_gpu:
            gt_tensor = gt_tensor.to(self.gpu_id)
            tx = tx.to(self.gpu_id)
            ty = ty.to(self.gpu_id)
            tw = tw.to(self.gpu_id)
            th = th.to(self.gpu_id)
            pred_conf = pred_conf.to(self.gpu_id)
            pred_cls = pred_cls.to(self.gpu_id)
            avg_loss = avg_loss.to(self.gpu_id)
            loss_tx = loss_tx.to(self.gpu_id)
            loss_ty = loss_ty.to(self.gpu_id)
            loss_tw = loss_tw.to(self.gpu_id)
            loss_th = loss_th.to(self.gpu_id)
            loss_cls = loss_cls.to(self.gpu_id)
            loss_conf = loss_conf.to(self.gpu_id)
            no_obj_loss_conf = no_obj_loss_conf.to(self.gpu_id)

            # small obj has larger scale weight, larger obj has smaller scale weight
            box_loss_scale = (2 - box_loss_scale).to(self.gpu_id)

        obj_mask = gt_tensor[..., -1] == 1
        no_obj_mask = gt_tensor[..., -1] == -1
        valid_obj_num, valid_no_obj_num = torch.sum(obj_mask), torch.sum(no_obj_mask)
        if valid_obj_num:
            # coordinate offset loss
            loss_tx = torch.mean(
                xy_loss_func(tx[obj_mask],
                             gt_tensor[..., 0][obj_mask]).float() * box_loss_scale[obj_mask] * self.opts.coord_weight)
            loss_ty = torch.mean(
                xy_loss_func(ty[obj_mask],
                             gt_tensor[..., 1][obj_mask]).float() * box_loss_scale[obj_mask] * self.opts.coord_weight)

            loss_tw = torch.mean(
                wh_loss_func(tw[obj_mask],
                             gt_tensor[..., 2][obj_mask]).float() * box_loss_scale[obj_mask] * self.opts.coord_weight)
            loss_th = torch.mean(
                wh_loss_func(th[obj_mask],
                             gt_tensor[..., 3][obj_mask]).float() * box_loss_scale[obj_mask] * self.opts.coord_weight)

            loss_cls = torch.mean(
                cls_loss_func(pred_cls[obj_mask], gt_tensor[..., 5:-1][obj_mask]).float())

            loss_conf = torch.mean(
                conf_loss_func(pred_conf[obj_mask], gt_tensor[..., 4][obj_mask]).float()) * self.balance[level]
            obj_loss = (loss_tx + loss_ty + loss_tw + loss_th) * self.box_ratio + \
                       loss_conf * self.obj_ratio + loss_cls * self.cls_ratio
        else:
            obj_loss = 0.
        if valid_no_obj_num:
            no_obj_loss_conf = torch.mean(
                conf_loss_func(pred_conf[no_obj_mask],
                               torch.zeros_like(pred_conf)[no_obj_mask]).float()) * self.balance[level]
            no_obj_loss = no_obj_loss_conf * self.obj_ratio
        else:
            no_obj_loss = 0.
        avg_loss += obj_loss + no_obj_loss
        return avg_loss, (loss_tx, loss_ty, loss_tw, loss_th, loss_conf, no_obj_loss_conf, loss_cls)
