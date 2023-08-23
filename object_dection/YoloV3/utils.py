import typing as t
import cv2
import torch
import numpy as np
import colorsys
from PIL import Image, ImageFont
from colorama import Fore


def resize_img_box(
        image: t.Any,
        labels: np.ndarray,
        new_size: t.Tuple[int, int],
        distort: bool
) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    1.resize img, add padding to picture <==> scale and paste.
    2.resize each box in each sample according to new_size
    """
    o_w, o_h = image.size
    n_w, n_h = new_size

    if distort:
        # no deformed conversion
        scale = min(n_w / o_w, n_h / o_h)
        n_w_ = int(o_w * scale)
        n_h_ = int(o_h * scale)
        dw = (n_w - n_w_) // 2
        dh = (n_h - n_h_) // 2
        image = image.resize((n_w_, n_h_), Image.BICUBIC)
        new_image = Image.new('RGB', new_size, (128, 128, 128))
        new_image.paste(image, (dw, dh))

        if len(labels) > 0:
            # np.random.shuffle(labels)
            labels[:, [0, 2]] = labels[:, [0, 2]] * scale + dw
            labels[:, [1, 3]] = labels[:, [1, 3]] * scale + dh
            labels[:, 0: 2][labels[:, 0: 2] < 0] = 0
            labels[:, 2][labels[:, 2] > n_w] = n_w
            labels[:, 3][labels[:, 3] > n_h] = n_h
            box_w = labels[:, 2] - labels[:, 0]
            box_h = labels[:, 3] - labels[:, 1]
            # filter invalid box, which width and height of box less than 1.
            labels = labels[np.logical_and(box_w > 1, box_h > 1)]
    else:
        new_image = image.resize((n_w, n_h), Image.BICUBIC)
        # TODO resize box
    # draw = ImageDraw.Draw(new_image)
    # for label in labels:
    #     draw.rectangle((label[0], label[1], label[2], label[3]), outline="red")
    # del draw
    # new_image.show()
    return np.array(new_image, np.float32), labels


def revert_img_box(
        box_xy: np.ndarray,
        box_wh: np.ndarray,
        image_shape: t.Tuple,
        input_shape: t.Tuple,
        letterbox_image: bool = True
):
    """
        Before training, we resized the image shape to input shape, and their gt boxes, then send it to model.
    because we never save the image that preprocessed, so, while we infer, we should revert the resized image
    to it original shape, and adjust the size of prediction boxes.

    Output:
        box_xy: x and y are center point of the box, its value between [0, 1], dimension -> [num1, 2],
            nums1 <= anchor_num * g_h * g_w
        image_shape: original shape of image
        input_shape: resized shape of image, such as 416x416
        letterbox_image: is the image distorted
    """
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        # scale = np.min(input_shape / image_shape)
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        # this offset is the offset of the effective area of the image according to the top-left corner of the image.
        # you can draw in your draft paper to understand
        # remember offset should be normalized, as the box_xy and box_wh are be normalized!
        offset = (input_shape - new_shape) / 2.0 / input_shape
        scale = input_shape / new_shape

        box_xy = (box_xy - offset) * scale
        box_wh *= scale
    box_t_l = box_xy - box_wh / 2.
    box_b_r = box_xy + box_wh / 2.
    boxes = np.concatenate([box_t_l[..., 0: 1], box_t_l[..., 1: 2], box_b_r[..., 0: 1], box_b_r[..., 1: 2]], axis=-1)
    boxes *= np.concatenate((image_shape, image_shape), axis=-1)

    return boxes


def compute_iou_gt_anchors(
        gt_boxes: torch.Tensor,
        anchor_boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the iou of gt boxes(dimension is [b,4]) and anchors(dimension is [9,4])
    Input:
            gt_boxes: Tensor -> [b, 4]
            anchor_boxes: Tensor -> [9, 4]
    Output:
            iou_plural: Tensor -> [b, 9]

    note: b means the number of box in one sample
    """

    a_x1, a_y1 = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2, anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2
    a_x2, a_y2 = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2, anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2

    gt_x1, gt_y1 = gt_boxes[:, 0] - gt_boxes[:, 2] / 2, gt_boxes[:, 1] - gt_boxes[:, 3] / 2
    gt_x2, gt_y2 = gt_boxes[:, 0] + gt_boxes[:, 2] / 2, gt_boxes[:, 1] + gt_boxes[:, 3] / 2

    # store top-left and bottom-right coordinate
    box_a, box_gt = torch.zeros_like(anchor_boxes), torch.zeros_like(gt_boxes)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = a_x1, a_y1, a_x2, a_y2
    box_gt[:, 0], box_gt[:, 1], box_gt[:, 2], box_gt[:, 3] = gt_x1, gt_y1, gt_x2, gt_y2

    a_size, gt_size = anchor_boxes.size(0), gt_boxes.size(0)

    # compute intersection
    # [b, 2] -> [b, 9, 2]
    inter_t_l = torch.max(box_gt[:, :2].unsqueeze(1).expand(gt_size, a_size, 2),
                          box_a[:, :2].unsqueeze(0).expand(gt_size, a_size, 2))
    inter_b_r = torch.min(box_gt[:, 2:].unsqueeze(1).expand(gt_size, a_size, 2),
                          box_a[:, 2:].unsqueeze(0).expand(gt_size, a_size, 2))
    # compress negative numbers to 0
    inter = torch.clamp(inter_b_r - inter_t_l, min=0)
    inter = inter[..., 0] * inter[..., 1]

    # compute union
    # gt_area = (gt_boxes[:, 2] * gt_boxes[:, 3]).unsqueeze(1).expand_as(inter)
    # a_area = (anchor_boxes[:, 2] * anchor_boxes[:, 3]).unsqueeze(0).expand_as(inter)
    gt_area = ((box_gt[:, 2] - box_gt[:, 0]) * (box_gt[:, 3] - box_gt[:, 1])).unsqueeze(1).expand_as(inter)
    a_area = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(0).expand_as(inter)
    return inter / (gt_area + a_area - inter)


def print_log(txt: str, color: t.Any = Fore.GREEN):
    print(color, txt)


def detection_collate(batch: t.Iterable[t.Tuple]) -> t.Tuple[torch.Tensor, t.List, t.List, t.List]:
    """
    custom collate func for dealing with batches of images that have a different number
    of object annotations (bbox).

    by the way, this func is used to customize the content returned by the dataloader.
    """

    labels = []
    images = []
    img_paths = []
    img_shapes = []
    for img, label, img_path, img_shape in batch:
        images.append(img)
        labels.append(label)
        img_paths.append(img_path)
        img_shapes.append(img_shape)
    return torch.stack(images, dim=0), labels, img_paths, img_shapes


class ImageAugmentation(object):

    def __call__(self, image_path: str, labels: np.ndarray, input_shape=(416, 416)) -> t.Tuple[np.ndarray, np.ndarray]:
        image = Image.open(image_path)
        image.convert('RGB')
        # resize image and add grey on image, modify shape of ground-truth box (label) according to after-resized image
        image_data, labels = resize_img_box(image, labels, input_shape, True)
        return image_data, labels


class Normalization(object):

    def __init__(self, img_shape: t.Tuple[int, int] = (416, 416), mode='simple'):
        self.mode = mode
        self.img_shape = img_shape

    def __call__(self, images: np.ndarray, boxes: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:
        func = getattr(self, self.mode)
        return func(images, boxes)

    def simple(self, images: np.ndarray, boxes: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        simple normalization function, it directly divided the limit upper.
        for example
        1. for images, divide each pixel by 255
        2. for boxes, the center of the box divided by the width and height of the entire image
        """
        images = images / 255.0
        boxes = boxes.astype(np.float32)
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.img_shape[1]
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.img_shape[0]

            # trans to mid_x, mid_y, w, h
            boxes[:, 2: 4] = boxes[:, 2: 4] - boxes[:, 0: 2]
            boxes[:, 0: 2] = boxes[:, 0: 2] + boxes[:, 2: 4] / 2
        return images, boxes


class ComputeMAP(object):
    """
    compute map
    """

    def __init__(self, iou_thresh: float = 0.5):
        self.iou_thresh = iou_thresh

    def calculate_tp(
            self,
            pred_coord: torch.Tensor,
            pred_score: torch.Tensor,
            gt_coord: torch.Tensor,
            gt_difficult: torch.Tensor,
    ) -> t.Tuple[int, t.List, t.List]:
        """

        calculate tp/fp for all predicted bboxes for one class of one image.

        Input:
            pred_coord: Tensor -> [N, 4], coordinates of all prediction boxes for a certain category
                        in a certain image(x0, y0, x1, y1)
            pred_score: Tensor -> [N, 1], score(confidence) of all prediction boxes for a certain category
                        in a certain image
            gt_coord: Tensor -> [M, 4], coordinates of all prediction boxes for a certain category
                        in a certain image(x0, y0, x1, y1)
            gt_difficult: Tensor -> [M, 1] -> whether the value of gt box of a certain category
                        in a certain image is difficult target?
            iou_thresh: the threshold to split TP and FP/FN.

        Output:
            gt_num: the number of gt box for a certain category in a certain image
            tp_list:
            conf_list:
        """

        not_difficult_gt_mask = torch.LongTensor(gt_difficult == 0)
        gt_num = not_difficult_gt_mask.sum()
        if gt_num == 0 or gt_coord.numel() == 0:
            return 0, [], []

        if pred_coord.numel() == 0:
            return len(gt_coord), [], []

        # compute iou of gt-box and pred-box
        gt_size, pred_size = gt_coord.size(0), pred_coord.size(0)

        inter_t_l = torch.max(gt_coord[..., :2].unsqueeze(1).expand(gt_size, pred_size, 2),
                              pred_coord[..., :2].unsqueeze(0).expand(gt_size, pred_size, 2))
        inter_b_r = torch.min(gt_coord[..., 2:].unsqueeze(1).expand(gt_size, pred_size, 2),
                              pred_coord[..., 2:].unsqueeze(0).expand(gt_size, pred_size, 2))
        inter = torch.clamp(inter_b_r - inter_t_l, min=0)
        inter = inter[..., 0] * inter[..., 1]

        area_gt = ((gt_coord[..., 2] - gt_coord[..., 0]) * (gt_coord[..., 3] - gt_coord[..., 2])).unsqueeze(
            1).expand_as(inter)
        area_pred = ((pred_coord[..., 2] - pred_coord[..., 0]) * (pred_coord[..., 3] - pred_coord[..., 2])).unsqueeze(
            1).expand_as(inter)

        iou_plural = inter / (area_pred + area_gt - inter + 1e-20)

        max_iou_val, max_iou_idx = torch.max(iou_plural, dim=0)

        # remove/ignore difficult gt box and the corresponding pred iou
        not_difficult_pb_mask = iou_plural[not_difficult_gt_mask].max(dim=0)[0] == max_iou_val
        max_iou_val, max_iou_idx = max_iou_val[not_difficult_pb_mask], max_iou_idx[not_difficult_pb_mask]
        if max_iou_idx.numel() == 0:
            return gt_num, [], []

        # for different bboxes that match to the same gt, set the highest score tp=1, and the other tp=0
        # score = conf * iou
        conf = pred_score.view(-1)[not_difficult_pb_mask]
        tp_list = torch.zeros_like(max_iou_val)
        for i in max_iou_idx[max_iou_val > self.iou_thresh].unique():
            gt_mask = (max_iou_val > self.iou_thresh) * (max_iou_idx == i)
            idx = (conf * gt_mask.float()).argmax()
            tp_list[idx] = 1
        return gt_num, tp_list.tolist(), conf.tolist()

    @staticmethod
    def calculate_pr(gt_num: int, tp_list: t.List, confidence_score: t.List) -> t.Tuple[t.List, t.List]:
        """
        calculate p-r according to gt number and tp_list for a certain category in a certain image
        """
        if gt_num == 0 or len(tp_list) == 0 or len(confidence_score) == 0:
            return [0], [0]
        if isinstance(tp_list, (tuple, list)):
            tp_list = np.array(tp_list)
        if isinstance(confidence_score, (tuple, list)):
            confidence_score = np.array(confidence_score)

        assert len(tp_list) == len(confidence_score), 'the length of tp_list is not equal to that in confidence score'

        # sort from max to min
        sort_mask = np.argsort(-confidence_score)
        tp_list = tp_list[sort_mask]

        # x = [1,3,1,2,5] -> np.cumsum(x) -> [1,4,5,7,12] ==> prefix sum
        recall = np.cumsum(tp_list) / gt_num
        precision = np.cumsum(tp_list) / (np.arange(len(tp_list)) + 1)

        return recall.tolist(), precision.tolist()


def set_font_thickness(font_filename: str, size: int, thickness: t.Tuple[int]):
    """
    set font and thickness of draw
    """
    font = ImageFont.truetype(font=font_filename, size=size)
    thickness = int(max(*thickness, 1))
    return font, thickness


def generate_colors(classes_num: int) -> t.List:
    """
    generate different kinds of color according to class num
    """
    hsv_tuples = [(x / classes_num, 1., 1.) for x in range(classes_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    return colors


def print_detail(
        cur_epoch: int,
        end_epoch: int,
        batch: int,
        iter_total: int,
        last_loss: float,
        tx_loss: float,
        ty_loss: float,
        tw_loss: float,
        th_loss: float,
        obj_conf_loss: float,
        no_obj_conf_loss: float,
        cls_loss: float,
        avg_loss: float,
        log_f: t.TextIO,
        write: bool = False,
) -> None:
    """
    print log information during epoch in training
    when write is True, write down the detail info.
    """
    info = f'Epoch: {cur_epoch}/{end_epoch} \tIter: {batch}/{iter_total} \ttx_loss: {round(tx_loss, 4)} ' \
           f'\tty_loss: {round(ty_loss, 4)} \ttw_loss: {round(tw_loss, 4)} \tth_loss: {round(th_loss, 4)}' \
           f'\tcls_loss: {round(cls_loss, 4)} \n\tobj_conf_loss: {round(obj_conf_loss, 4)} ' \
           f'\tno_obj_conf_loss: {round(no_obj_conf_loss, 4)} \tlast_loss: {round(last_loss, 4)} ' \
           f'\tavg_loss: {round(avg_loss, 6)}\n'

    print_log(info, color=Fore.RED)
    if write:
        log_f.write(info)
        log_f.flush()


def print_detail_giou(
        cur_epoch: int,
        end_epoch: int,
        batch: int,
        iter_total: int,
        last_loss: float,
        loss_loc: float,
        loss_conf: float,
        loss_cls: float,
        avg_loss: float,
        log_f: t.TextIO,
        write: bool = False,
) -> None:
    info = f'Epoch: {cur_epoch}/{end_epoch} \tIter: {batch}/{iter_total} \tloss_loc: {round(loss_loc, 4)} ' \
           f'\tloss_conf: {round(loss_conf, 4)} \tloss_cls: {round(loss_cls, 4)} ' \
           f'\tlast_loss: {round(last_loss, 4)} \tavg_loss: {round(avg_loss, 6)}\n'

    print_log(info, color=Fore.RED)
    if write:
        log_f.write(info)
        log_f.flush()


def draw_image(
        cls_ids: t.Union[t.List, np.ndarray],
        coords: t.Union[t.List, np.ndarray],
        scores: t.Union[t.List, np.ndarray],
        cur_img: t.Any,
        classes: t.Union[t.List, np.ndarray],
        thickness: int = 1,
) -> None:
    """
    draw box
    """
    colors = generate_colors(len(classes))
    width, height = cur_img.size
    cur_img = np.array(cur_img)
    for cls_id, score, coord in zip(cls_ids, scores, coords):
        pred_cls_name = classes[int(cls_id)]
        top, left, bottom, right = coord
        top = max(0, np.floor(top).astype('int32'))
        left = max(0, np.floor(left).astype('int32'))
        bottom = min(width, np.floor(bottom).astype('int32'))
        right = min(width, np.floor(right).astype('int32'))
        label = f'{pred_cls_name} {round(score, 2)}'
        print_log(f'{label} {top} {left} {bottom} {right}', Fore.BLUE)
        if top - 2 > 0:
            text_pos = np.array((top - 1, left))
        else:
            text_pos = np.array((top + 1, left))

        cv2.rectangle(cur_img, (top, left), (bottom, right), color=colors[cls_id], thickness=thickness)
        cv2.putText(cur_img, label, tuple(text_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color=colors[cls_id], thickness=thickness)
    cv2.imshow(f"bbox", cur_img[:, :, ::-1])
    cv2.waitKey(0)


if __name__ == "__main__":
    file_path = r'C:\Users\24717\Projects\pascal voc2012\VOCdevkit\VOC2012\JPEGImages\2007_000032.jpg'
    image_ = Image.open(file_path)
    print(image_.size)
    image_.show()
