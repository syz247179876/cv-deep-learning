import math
import os.path
import time

import torch
from colorama import Fore
from torch.utils.data import DataLoader

from augment import args_train
from data import MAEDataset, ImageAugmentation
from util import print_detail, mae_collate, print_log
from loss import MAELoss
from model.mae import MAE, model_factory


class MAETrain(object):

    def __init__(self):
        self.opts = args_train.opts

    def __save_model(
            self,
            model: MAE,
            optimizer: torch.optim.Optimizer,
            epoch: int
    ) -> None:
        """
        save model
        """
        model_name = f'epoch{epoch}.pth'
        torch.save({
            'last_epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(self.opts.checkpoints_dir, model_name))

    @property
    def init_lr(self) -> float:
        """
        adjust learning rate dynamically according to batch_size and epoch

        learning rate decrease five times every several epoch
        """
        max_lr = self.opts.lr_max
        min_lr = self.opts.lr_min
        batch_size = self.opts.batch_size
        lr = min(max(batch_size / 64 * self.opts.lr_base, min_lr), max_lr)
        return lr

    def __train_epoch(
            self,
            model: MAE,
            loss_obj: MAELoss,
            train_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            epoch: int,
            train_num: int,
    ) -> None:
        total_loss = 0.
        with open(os.path.join(self.opts.checkpoints_dir, 'log.txt'), 'a+') as f:
            for batch, (x, _) in enumerate(train_loader):
                if self.opts.use_gpu:
                    x = x.to(self.opts.gpu_id)
                pred, mask, target = model(x)
                cur_loss = loss_obj(pred, target, mask)
                optimizer.zero_grad()
                cur_loss.backward()
                optimizer.step()
                total_loss += cur_loss.item()

                if batch % self.opts.print_frequency == 0:
                    print_detail(epoch, self.opts.end_epoch, batch, train_num // self.opts.batch_size,
                                 cur_loss.item(), avg_loss=total_loss / (batch + 1), log_f=f, write=True)
            print(f'epoch-{epoch}')

    def main(self):
        if not os.path.exists(self.opts.checkpoints_dir):
            os.mkdir(self.opts.checkpoints_dir)

        train_dataset = MAEDataset(mode='train', img_augmentation=ImageAugmentation)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.opts.batch_size,
            shuffle=self.opts.shuffle,
            num_workers=self.opts.num_workers,
            drop_last=self.opts.drop_last,
            collate_fn=mae_collate,
        )
        train_num = len(train_dataset)

        model = model_factory(model_name=self.opts.model, load_file=True)
        print_log(f'Init model---MAE-{self.opts.model} successfully!')
        if self.opts.use_gpu:
            model.to(self.opts.gpu_id)
        model.train()

        # optimizer = torch.optim.SGD(model.parameters(), lr=self.init_lr, momentum=0.9,
        #                             weight_decay=self.opts.weight_decay)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.opts.decrease_interval, gamma=0.85)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.opts.lr_base * self.opts.batch_size / 256,
                                  betas=(0.9, 0.95), weight_decay=self.opts.weight_decay)
        lr_func = lambda epoch: min((epoch + 1) / (self.opts.decrease_interval + 1e-8),
                                    0.5 * (math.cos(epoch / self.opts.end_epoch * math.pi) + 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)
        last_epoch = self.opts.start_epoch
        if self.opts.pretrain_file:
            checkpoint = torch.load(self.opts.pretrain_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            last_epoch = checkpoint['last_epoch']
            print_log(f'Load model file {self.opts.pretrain_file} successfully!')

        loss_obj = MAELoss()

        for e in range(last_epoch, self.opts.end_epoch):
            t1 = time.time()
            # with torch.autograd.set_detect_anomaly(True):
            self.__train_epoch(model, loss_obj, train_loader, optimizer, e, train_num)
            t2 = time.time()
            scheduler.step()
            print_log(f"learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}", Fore.BLUE)
            print_log("Training consumes %.2f second\n" % (t2 - t1), Fore.RED)
            with open(os.path.join(self.opts.checkpoints_dir, 'log.txt'), 'a+') as log_f:
                log_f.write(f'Training one epoch consumes %.2f second\n' % (t2 - t1))

                if e % self.opts.save_frequency == 0 or e == self.opts.end_epoch:
                    self.__save_model(model, optimizer, e)


if __name__ == '__main__':
    train = MAETrain()
    train.main()
