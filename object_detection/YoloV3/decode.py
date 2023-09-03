"""
Decoding the extracted features
"""
import numpy as np
import torch
import torch.nn as nn
import typing as t
from torchvision.ops import nms
from argument import args_test
from settings import *
from utils import revert_img_box


class DecodeFeature(object):

    def __init__(self, img_size: int, classes_num: int):
        super(DecodeFeature, self).__init__()
        self.opts = args_test.opts
        self.classes_num = classes_num
        self.img_size = img_size

    @staticmethod
    def generator_xy(
            grid_h: int,
            grid_w: int,
            batch_size: int,
            method: int = 1,
            split_res: bool = False,
    ) -> t.Union[torch.Tensor, t.Tuple[torch.Tensor, torch.Tensor]]:
        """
        generator grid according to the grid
        note: I implement two methods to generate grid and w, h

        1. the first method based on broadcast mechanism in Torch.
        2. the second method has the same dimensions as the feature map.
        """
        # method one:
        if method:
            grid_y, grid_x = torch.meshgrid((torch.arange(grid_h)), torch.arange(grid_w))
            # dimension -> [1, 1, 13, 13 ,2], if down sample multiple is 32
            grid_xy = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, grid_h, grid_w, 2)
        else:
            # method two:
            grid_x = torch.linspace(0, grid_h - 1, grid_h).repeat(grid_h, 1). \
                repeat(batch_size * ANCHORS_NUM, 1, 1).view(batch_size, ANCHORS_NUM, grid_h, grid_w, 1).\
                type(torch.FloatTensor)
            grid_y = torch.linspace(0, grid_w - 1, grid_w).repeat(grid_w, 1).t(). \
                repeat(batch_size * ANCHORS_NUM, 1, 1).view(batch_size, ANCHORS_NUM, grid_h, grid_w, 1). \
                type(torch.FloatTensor)
            # dimension -> [B, anchors_nums, 13, 13, 2], if down sample multiple is 32
            grid_xy = torch.cat((grid_x, grid_y), dim=-1)
        return grid_xy if not split_res else (grid_x, grid_y)

    def decode_pred(self, inputs: t.Union[t.List[torch.Tensor], t.Tuple[torch.Tensor, ...]]) -> t.List:
        """
        compute bx, by, bw, bh of bbox based on tx, ty, tw, th, and then replace tx, ty, tw, th in feature map
        Input:
            Tensor -> [B, self.anchors_num * (1 + 4 + self.classes_num), height, width]
        Output:
            Tensor -> [B, self.anchors_num, height, width, (1 + 4 + self.class_num)

            note: 4 -> [bx, by, bw, bn] are bbox's actual coordinate according to grid cell
        """
        outputs = []
        for idx, pred in enumerate(inputs):
            pred: torch.Tensor
            batch_size, _, g_h, g_w = pred.size()

            stride_h = self.img_size / g_h
            stride_w = self.img_size / g_w

            scaled_anchors = [(anchor_w / stride_w, anchor_h / stride_h) for anchor_w, anchor_h in ANCHORS[idx]]

            # trans structure to [B, self.anchor, height, width, (1 + 4 + self.class_num)]

            pred = pred.view(batch_size, ANCHORS_NUM, 4 + 1 + self.classes_num, g_h, g_w). \
                permute(0, 1, 3, 4, 2).contiguous()

            float_ = torch.cuda.FloatTensor if pred.is_cuda else torch.FloatTensor
            long_ = torch.cuda.LongTensor if pred.is_cuda else torch.LongTensor

            # compute bx, by, bw, bh
            grid_x, grid_y = self.generator_xy(g_h, g_w, batch_size, method=0, split_res=True)
            # grid_xy = self.generator_xy(g_h, g_w, batch_size, method=0)
            # if pred.is_cuda:
            #     grid_xy = grid_xy.to(self.opts.gpu_id)
            # b_xy = torch.sigmoid(pred[..., :2]) + grid_xy

            if pred.is_cuda:
                grid_x = grid_x.to(self.opts.gpu_id)
                grid_y = grid_y.to(self.opts.gpu_id)
            _x = torch.sigmoid(pred[..., 0])
            _y = torch.sigmoid(pred[..., 1])
            b_x = _x + grid_x.view(_x.size())
            b_y = _y + grid_y.view(_y.size())

            pw, ph = pred[..., 2], pred[..., 3]
            anchor_w = float_(scaled_anchors).index_select(1, long_([0])).repeat(batch_size, 1). \
                repeat(1, 1, g_h * g_w).view(pw.size())
            anchor_h = float_(scaled_anchors).index_select(1, long_([1])).repeat(batch_size, 1). \
                repeat(1, 1, g_h * g_w).view(ph.size())
            b_w = torch.exp(pw) * anchor_w
            b_h = torch.exp(ph) * anchor_h

            # pred[..., :2] = b_xy
            pred[..., 0] = b_x
            pred[..., 1] = b_y
            pred[..., 2] = b_w
            pred[..., 3] = b_h

            # normalize the results to decimals
            scale = torch.FloatTensor([g_w, g_h, g_w, g_h])
            if pred.is_cuda:
                scale = scale.to(self.opts.gpu_id)
            # concat bx, by, bw, bh, conf, cls
            output = torch.cat((pred[..., :4].view(batch_size, -1, 4) / scale,
                                torch.sigmoid(pred[..., 4]).view(batch_size, -1, 1),
                                torch.sigmoid(pred[..., 5:].view(batch_size, -1, self.classes_num))), dim=-1)
            outputs.append(output.data)
        return outputs

    def execute_nms(
            self,
            predictions: torch.Tensor,
            image_shapes: t.List[t.Tuple],
            letterbox_image: bool,
            input_shape: t.Tuple = INPUT_SHAPE,
    ) -> t.List[np.ndarray]:
        """
        NMS
        1.classify all bboxes by category.
        2.according to each category, executing NMS to seek out best bboxes.

        Input:
            predictions: Tensor -> [B, anchor_num * g_h * g_w, 4 + 1 + self.num_classes]
        Output:
            outputs: t.List -> [[num1, 4 + 1 + 1], [num2, 4 + 1 + 1], ...]
        note:
            in the test, we can define B = 1, so that we can look the effects of prediction and classification
        """
        # pred has the same device and shape as predictions
        pred = predictions.new(predictions.size())
        # ccs = predictions[..., 2:4]
        pred[..., 0] = predictions[..., 0] - predictions[..., 2] / 2
        pred[..., 2] = predictions[..., 0] + predictions[..., 2] / 2
        pred[..., 1] = predictions[..., 1] - predictions[..., 3] / 2
        pred[..., 3] = predictions[..., 1] + predictions[..., 3] / 2
        predictions[:, :, :4] = pred[:, :, :4]

        outputs: t.List[t.Optional[torch.Tensor]] = [None for _ in range(pred.size(0))]
        for idx, image_pred in enumerate(predictions):
            idx: int
            image_pred: torch.Tensor
            # dimension -> [anchor_num * g_h * g_w, 1]
            max_cls_conf, max_cls_idx = torch.max(image_pred[:, 5: 5 + self.classes_num], dim=1, keepdim=True)

            # first filter ----- confidence filter
            # score = conf * cls
            conf_mask = (image_pred[:, 4] * max_cls_conf[:, 0] >= self.opts.conf_thresh).squeeze()
            image_pred = image_pred[conf_mask]
            max_cls_conf = max_cls_conf[conf_mask]
            max_cls_idx = max_cls_idx[conf_mask]

            if image_pred.size(0) == 0:
                continue

            # classify all boxes by category
            filter_pred = torch.cat((image_pred[:, :5], max_cls_conf.float(), max_cls_idx.float()), dim=1)
            unique_cls = filter_pred[:, -1].unique()

            for c in unique_cls:
                pred_box = filter_pred[filter_pred[:, -1] == c]
                keep = nms(pred_box[:, :4],
                           pred_box[:, 4] * pred_box[:, 5],
                           self.opts.iou_thresh)
                best_pred = filter_pred[keep]

                outputs[idx] = best_pred if outputs[idx] is None else torch.concat((outputs[idx], best_pred))
            if outputs[idx] is not None:
                outputs[idx] = outputs[idx].cpu().numpy()
                box_xy, box_wh = (outputs[idx][:, 0: 2] + outputs[idx][:, 2: 4]) / 2, \
                                 outputs[idx][:, 2: 4] - outputs[idx][:, 0: 2]
                outputs[idx][:, :4] = revert_img_box(box_xy, box_wh, image_shapes[idx],
                                                     input_shape, letterbox_image)
        return outputs


if __name__ == "__main__":
    d = DecodeFeature(416, 20)
    in_p = [torch.randn(4, 75, 13, 13), torch.randn(4, 75, 26, 26), torch.randn(4, 75, 52, 52)]
    res = d.decode_pred(in_p)
    print(res[0].size(), res[1].size(), res[2].size())
