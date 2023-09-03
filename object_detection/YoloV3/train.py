import os.path
import time
import typing as t
import torch.nn as nn
import torch
from colorama import Fore
from torch.utils.data import DataLoader
from argument import args_train
from data.voc_data import VOCDataset
from loss import YoloV3Loss
from loss2 import YOLOLoss
from settings import ANCHORS, ANCHORS_SORT, INPUT_SHAPE, VOC_CLASS_NUM
from utils import print_log, detection_collate, print_detail, print_detail_giou
from model.darknet_53 import Darknet53


class YoloV3Train(object):

    def __init__(self):
        self.opts = args_train.opts

    def __train_epoch(
            self,
            model: Darknet53,
            loss_obj: YOLOLoss,
            train_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            epoch: int,
            train_num: int,
    ) -> None:
        total_loss = 0
        detail_list = [0 for _ in range(3)]
        with open(os.path.join(self.opts.checkpoints_dir, 'log.txt'), 'a+') as log_f:
            for batch, (x, labels, _, _) in enumerate(train_loader):

                if self.opts.use_gpu:
                    x = x.to(self.opts.gpu_id)
                    # labels = labels.to(self.opts.gpu_id)

                pred: t.Tuple[torch.Tensor] = model(x)
                # calculate loss of different feature level
                cur_loss = torch.tensor(0).float().to(self.opts.gpu_id)
                for idx, output in enumerate(pred):
                    temp, details = loss_obj(idx, output, labels)
                    cur_loss += temp
                    for _id, detail in enumerate(details):
                        detail_list[_id] += detail.item()
                optimizer.zero_grad()
                cur_loss.backward()
                optimizer.step()
                total_loss += cur_loss.item()

                if batch % self.opts.print_frequency == 0:
                    # print_detail(epoch, self.opts.end_epoch, batch, train_num // self.opts.batch_size,
                    #              cur_loss.item(), *[detail / self.opts.print_frequency for detail in detail_list],
                    #              avg_loss=total_loss / (batch + 1), log_f=log_f, write=True)
                    print_detail_giou(epoch, self.opts.end_epoch, batch, train_num // self.opts.batch_size,
                                      cur_loss.item(), *[detail / self.opts.print_frequency for detail in detail_list],
                                      avg_loss=total_loss / (batch + 1), log_f=log_f, write=True)
                    detail_list = [0 for _ in range(3)]

    def __save_model(
            self,
            model: Darknet53,
            epoch: int,
    ) -> None:
        """
        save model
        """
        model_name = f'epoch{epoch}.pkl'
        torch.save(model, os.path.join(self.opts.checkpoints_dir, model_name))

    @property
    def init_lr(self) -> float:
        """
        adjust learning rate dynamically according to batch_size and epoch

        learning rate decrease five times every 30 epoch
        """
        max_lr = self.opts.lr_max
        min_lr = self.opts.lr_min
        batch_size = self.opts.batch_size
        lr = min(max(batch_size / 64 * self.opts.lr_base * 0.5 * 0.5, min_lr), max_lr)
        return lr

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

        if not self.opts.pretrain_file:
            model = Darknet53()
            print_log('Init model successfully!')
        else:
            model = torch.load(self.opts.pretrain_file)
            print_log(f'Load model file {self.opts.pretrain_file} successfully!')
        loss_obj = YOLOLoss(ANCHORS_SORT, VOC_CLASS_NUM, INPUT_SHAPE, self.opts.use_gpu,
                            [[6, 7, 8], [3, 4, 5], [0, 1, 2]])

        if self.opts.use_gpu:
            model.to(self.opts.gpu_id)

        # during train, use model.train() to update the mean and val according to each mini-batch of BN level
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.init_lr, momentum=0.9,
                                    weight_decay=self.opts.weight_decay)

        # adjust learning rate
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.opts.decrease_interval, gamma=0.5)

        for e in range(self.opts.start_epoch, self.opts.end_epoch + 1):
            t1 = time.time()
            self.__train_epoch(model, loss_obj, train_loader, optimizer, e, train_num)
            t2 = time.time()
            scheduler.step()
            print_log(f"learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}", Fore.BLUE)
            print_log("Training consumes %.2f second\n" % (t2 - t1), Fore.RED)
            with open(os.path.join(self.opts.checkpoints_dir, 'log.txt'), 'a+') as log_f:
                log_f.write(f'Training one epoch consumes %.2f second\n' % (t2 - t1))

                if e % self.opts.save_frequency == 0 or e == self.opts.end_epoch:
                    self.__save_model(model, e)


if __name__ == "__main__":
    yolo_v2 = YoloV3Train()
    yolo_v2.main()
