import os.path
import time
import typing as t

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from argument import args_test
from data.voc_data import VOCDataset
from decode import DecodeFeature
from loss2 import YOLOLoss
from settings import VOC_CLASSES, INPUT_SHAPE, ANCHORS_SORT, VOC_CLASS_NUM
from utils import detection_collate, print_log, generate_colors, draw_image


class YoloV3Test(object):
    """
    Yolo Model test
    """

    def __init__(
            self,
            img_size: int,
            classes_num: int,
            letterbox_image: bool = True
    ):
        self.opts = args_test.opts
        self.decode = DecodeFeature(img_size, classes_num)
        self.classes_num = classes_num
        self.letterbox_image = letterbox_image
        self.colors = generate_colors(classes_num)

    def main(self):
        test_dataset = VOCDataset(mode='test')
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.opts.batch_size,
            shuffle=self.opts.shuffle,
            num_workers=self.opts.num_workers,
            drop_last=self.opts.drop_last,
            collate_fn=detection_collate,
        )
        test_num = len(test_dataset)
        assert self.opts.pretrain_file is not None, 'need to load model trained!'
        model = torch.load(self.opts.pretrain_file)
        print_log(f'Load model file {self.opts.pretrain_file} successfully!')
        if self.opts.use_gpu:
            model.to(self.opts.gpu_id)
        model.eval()

        loss_obj = YOLOLoss(ANCHORS_SORT, VOC_CLASS_NUM, INPUT_SHAPE, self.opts.use_gpu,
                            [[6, 7, 8], [3, 4, 5], [0, 1, 2]])
        total_loss = 0.
        detail_list = [0 for _ in range(3)]
        for batch, (x, labels, img_paths, image_shapes) in enumerate(test_loader):
            with torch.no_grad():
                if self.opts.use_gpu:
                    x = x.to(self.opts.gpu_id)
                pred: t.Tuple = model(x)
                cur_loss = torch.tensor(0).float().to(self.opts.gpu_id)
                t1 = time.time()
                for idx, output in enumerate(pred):
                    temp, details = loss_obj(idx, output, labels)
                    cur_loss += temp
                    for _id, detail in enumerate(details):
                        detail_list[_id] += detail.item()
                t2 = time.time()
                total_loss += cur_loss.item()
                print_log(
                    f"cur batch: Average loss: {round(cur_loss.item(), 6)}, Inference time:{int((t2 - t1) * 1000)}ms, "
                    f"cur num: {len(x)}")

                # decode pred and execute multi-nms
                decoded_pred: t.List = self.decode.decode_pred(pred)
                results: t.List = self.decode.execute_nms(
                    torch.cat(decoded_pred, dim=1),
                    image_shapes,
                    self.letterbox_image
                )

                thickness = 1
                if results and results[0] is not None:
                    top_cls_idxs = np.array([result[:, 6] for result in results], dtype='int32')
                    top_scores = np.array([result[:, 4] * result[:, 5] for result in results], dtype='float')
                    top_boxes = np.array([result[:, :4] for result in results], dtype='float')

                    # draw picture
                    for cls_ids, scores, boxes, img_path in zip(top_cls_idxs, top_scores, top_boxes, img_paths):
                        # dimension -> [num1, 1], num1 < anchor_num * g_h * g_w
                        cls_ids: np.ndarray
                        scores: np.ndarray
                        boxes: np.ndarray
                        img_path: str
                        cur_img = Image.open(img_path)
                        cur_img = cur_img.convert('RGB')
                        # draw image
                        draw_image(cls_ids, boxes, scores, cur_img, VOC_CLASSES, thickness)

        avg_loss = total_loss / (len(test_loader) + 1)
        print_log(f"Test set: Average loss: {round(avg_loss, 6)}, Total num: {test_num}")


if __name__ == '__main__':
    test = YoloV3Test(416, 20, )
    test.main()
