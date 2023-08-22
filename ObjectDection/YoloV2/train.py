import argparse
import os
import time
import torch.nn
import typing as t
from torch.utils.data import DataLoader
from data import VOCDataset
from model import YoloV2
from loss import YoloV2Loss
from argument import args_train
from settings import FEATURE_MAP_W, FEATURE_MAP_H, ANCHORS_DIR
from util import detection_collate, label_generator, retrieve_anchors


class YoloV2Train(object):

    def __init__(self):
        """
        get args
        """
        self.opts = args_train.opts

    def __train_epoch(
            self,
            model: YoloV2,
            loss_obj: YoloV2Loss,
            train_loader: DataLoader,
            anchors: t.List[t.List],
            optimizer: torch.optim.Optimizer,
            epoch: int,
            train_num: int,
            input_size: int,
            stride: int = 32,
    ) -> None:
        """
        complete each round of training
        """

        print(f'------starting training------epoch time: {epoch}')
        model.train()
        w = h = input_size

        avg_loss = 0.
        with open(os.path.join(self.opts.checkpoints_dir, 'log.txt'), 'a+') as log_f:
            for batch, (x, labels) in enumerate(train_loader):

                labels = label_generator(input_size, labels, anchors)
                labels = torch.from_numpy(labels)
                if self.opts.use_gpu:
                    x = x.to(self.opts.gpu_id)
                    labels = labels.to(self.opts.gpu_id)

                # forward propagation
                pred = model(x)
                # compute loss
                loss = loss_obj(pred, labels)
                # gradient initialization
                optimizer.zero_grad()
                # backward propagation
                loss.backward()
                # update params
                optimizer.step()
                # compute avg loss
                avg_loss = (avg_loss * batch + loss.item()) / (batch + 1)

                if batch % self.opts.print_frequency == 0:
                    print("Epoch %d/%d | Iter %d/%d | training loss = %.3f, avg_loss = %.3f" %
                          (epoch, self.opts.end_epoch, batch, train_num // self.opts.batch_size, loss.item(), avg_loss))
                    log_f.write("Epoch %d/%d | Iter %d/%d | training loss = %.3f, avg_loss = %.3f\n" %
                                (epoch, self.opts.end_epoch, batch, train_num // self.opts.batch_size, loss.item(),
                                 avg_loss))
                    log_f.flush()

    def __save_model(
            self,
            model: YoloV2,
            epoch: int,
    ) -> None:
        """
        存储模型
        """
        model_name = f'epoch{epoch}.pkl'
        torch.save(model, os.path.join(self.opts.checkpoints_dir, model_name))

    def main(self) -> None:
        """
        entrance of train
        """
        if not os.path.exists(self.opts.checkpoints_dir):
            os.mkdir(self.opts.checkpoints_dir)
        train_dataset = VOCDataset()
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.opts.batch_size,
            shuffle=self.opts.shuffle,
            num_workers=self.opts.num_workers,
            drop_last=self.opts.drop_last,
            collate_fn=detection_collate,
        )
        train_num = len(train_dataset)

        # continue training at breakpoint
        if not self.opts.pretrain_file:
            model = YoloV2()
        else:
            model = torch.load(self.opts.pretrain_file)
        loss_obj = YoloV2Loss(416, 32)

        # use GPU to train model
        if self.opts.use_gpu:
            model.to(self.opts.gpu_id)
        # use momentum = 0.9 to accelerate convergence
        optimizer = torch.optim.SGD(model.parameters(), lr=self.opts.lr, momentum=0.9,
                                    weight_decay=self.opts.weight_decay)

        anchors_dir = os.path.join(self.opts.base_dir, ANCHORS_DIR)
        anchor_file = os.path.join(anchors_dir, 'voc_anchors.txt')
        anchors = retrieve_anchors(anchor_file)

        for e in range(self.opts.start_epoch, self.opts.end_epoch + 1):
            t1 = time.time()
            self.__train_epoch(model, loss_obj, train_loader, anchors, optimizer, e, train_num, 416, 32)
            t2 = time.time()
            print("Training consumes %.2f second\n" % (t2 - t1))
            with open(os.path.join(self.opts.checkpoints_dir, 'log.txt'), 'a+') as log_f:
                log_f.write(f'Training one epoch consumes %.2f second\n' % (t2 - t1))

                if e % self.opts.save_frequency == 0 or e == self.opts.end_epoch:
                    self.__save_model(model, e)


if __name__ == "__main__":
    yolo_v2 = YoloV2Train()
    yolo_v2.main()
