import os.path
import time

import torch
from colorama import Fore
from torch.utils.data import DataLoader

from augment import args_test
from data import ViTDataset, FlowerTransform, ImageAugmentation, z_score_
from loss import ViTLoss
from utils import classify_collate, print_log, print_detail, draw_image
from model import ModelFactory, VisionTransformer


class ViTTest(object):

    def __init__(self):
        self.opts = args_test.opts

    def main(
            self
    ):
        test_dataset = ViTDataset(mode='test', anno_transform=FlowerTransform,
                                  img_augmentation=ImageAugmentation, normalization=z_score_, distort=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.opts.batch_size,
            shuffle=self.opts.shuffle,
            num_workers=1,
            drop_last=self.opts.drop_last,
            collate_fn=classify_collate,
        )

        model = torch.load(self.opts.pretrain_file)
        model.eval()
        loss_obj = ViTLoss()
        for batch, (x, labels, img_path) in enumerate(test_loader):
            with torch.no_grad():
                if self.opts.use_gpu:
                    x = x.to(self.opts.gpu_id)
                    labels = labels.to(self.opts.gpu_id)
                pred = model(x)
                cur_loss = loss_obj(pred, labels)
                draw_image(img_path, cur_loss.item(), pred, test_dataset.classes)



