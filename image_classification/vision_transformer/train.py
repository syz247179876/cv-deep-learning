import os.path
import time

import torch
from colorama import Fore
from torch.utils.data import DataLoader

from augment import args_train
from data import ViTDataset, FlowerTransform, ImageAugmentation, z_score_
from loss import ViTLoss
from utils import classify_collate, print_log, print_detail
from model import ModelFactory, VisionTransformer


class ViTTrain(object):
    def __init__(self):
        self.opts = args_train.opts

    def __save_model(
            self,
            model: VisionTransformer,
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

        learning rate decrease five times every 30 epoch
        """
        max_lr = self.opts.lr_max
        min_lr = self.opts.lr_min
        batch_size = self.opts.batch_size
        lr = min(max(batch_size / 64 * self.opts.lr_base, min_lr), max_lr)
        return lr

    def __train_epoch(
            self,
            model: VisionTransformer,
            loss_obj: ViTLoss,
            train_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            epoch: int,
            train_num: int,
    ) -> None:
        total_loss = 0.
        epoch_accu_num = 0
        with open(os.path.join(self.opts.checkpoints_dir, 'log.txt'), 'a+') as f:
            for batch, (x, labels) in enumerate(train_loader):
                if self.opts.use_gpu:
                    x = x.to(self.opts.gpu_id)
                    labels = labels.to(self.opts.gpu_id)
                pred = model(x)
                cur_loss, accu_num = loss_obj(pred, labels)
                epoch_accu_num += accu_num
                optimizer.zero_grad()
                cur_loss.backward()
                optimizer.step()
                total_loss += cur_loss.item()

                if batch % self.opts.print_frequency == 0:
                    print_detail(epoch, self.opts.end_epoch, batch, train_num // self.opts.batch_size,
                                 cur_loss.item(), avg_loss=total_loss / (batch + 1), log_f=f, write=True)
            accu_ratio = epoch_accu_num / train_num
            print(f'epoch-{epoch} accuracy: {accu_ratio}')

    @staticmethod
    def freeze_layers(model: torch.nn.Module):
        """
        use linear probing
        when training custom dataset, freeze some layers except head.
        """

        for name, params in model.named_parameters():
            if 'head' not in name and 'pre_logit' not in name:
                params.requires_grad = False

    def main(self):
        if not os.path.exists(self.opts.checkpoints_dir):
            os.mkdir(self.opts.checkpoints_dir)

        train_dataset = ViTDataset(mode='train', anno_transform=FlowerTransform,
                                   img_augmentation=ImageAugmentation, normalization=z_score_, distort=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.opts.batch_size,
            shuffle=self.opts.shuffle,
            num_workers=self.opts.num_workers,
            drop_last=self.opts.drop_last,
            collate_fn=classify_collate,
        )
        train_num = len(train_dataset)

        model = ModelFactory(self.opts.model).model
        print_log(f'Init model---vit-{self.opts.model} successfully!')
        if self.opts.use_gpu:
            model.to(self.opts.gpu_id)
        model.train()

        # freeze some layers
        if self.opts.freeze_lagers:
            self.freeze_layers(model)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.init_lr, momentum=0.9,
                                    weight_decay=self.opts.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.opts.decrease_interval, gamma=0.9)

        # load model, optimizer, last epoch
        last_epoch = self.opts.start_epoch
        if self.opts.pretrain_file:
            checkpoint = torch.load(self.opts.pretrain_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            last_epoch = checkpoint['last_epoch']
            print_log(f'Load model file {self.opts.pretrain_file} successfully!')

        loss_obj = ViTLoss()

        for e in range(last_epoch, self.opts.end_epoch):
            t1 = time.time()
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
    train = ViTTrain()
    train.main()
