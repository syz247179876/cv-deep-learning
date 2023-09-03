"""
训练模型
"""
import argparse
import os.path
import time

import torch.nn
from torch.utils.data import DataLoader
from data import YOLOV1Dataset
from model import YoloV1Net
from settings import GRID_NUM
from loss import YoloV1Loss
from argument import Args


class YoloV1Train(object):

    def __init__(self, opts: argparse.Namespace):
        """
        初始化命令行参数
        """
        self.opts = opts

    @staticmethod
    def __train_epoch(
            model: YoloV1Net,
            loss_obj: YoloV1Loss,
            train_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            epoch: int,
            train_num: int,
            opts: argparse.Namespace
    ) -> None:
        """
        完成一个epoch的训练
        """
        # 保证 BN 层能够用到每一批数据的均值和方差
        print(f'------starting training------epoch time: {epoch}')
        model.train()
        device = opts.gpu_id

        # 将每一epoch的avg_loss写入日志
        avg_loss = .0

        with open(os.path.join(opts.checkpoints_dir, 'log.txt'), 'a+') as log_f:
            for batch, (x, labels) in enumerate(train_loader):
                labels = labels.view(opts.batch_size, GRID_NUM, GRID_NUM, -1)
                if opts.use_gpu:
                    x = x.to(device)
                    labels = labels.to(device)
                # 训练时前向传播需要网络生成计算图, 而训练时则不需要
                pred = model(x)
                # 计算损失函数
                loss = loss_obj.calculate_loss(pred, labels)
                # 梯度初始化0
                optimizer.zero_grad()
                # 反向传播计算梯度
                loss.backward()
                # 更新网络参数
                optimizer.step()
                # 每一个batch, 计算当前平均损失
                avg_loss = (avg_loss * batch + loss.item()) / (batch + 1)

                if batch % opts.print_frequency == 0:
                    print("Epoch %d/%d | Iter %d/%d | training loss = %.3f, avg_loss = %.3f" %
                          (epoch, opts.epoch, batch, train_num // opts.batch_size, loss.item(), avg_loss))
                    log_f.write("Epoch %d/%d | Iter %d/%d | training loss = %.3f, avg_loss = %.3f\n" %
                                (epoch, opts.epoch, batch, train_num // opts.batch_size, loss.item(), avg_loss))
                    log_f.flush()

    @staticmethod
    def __save_model(
            model: YoloV1Net,
            epoch: int,
            opts: argparse.Namespace
    ) -> None:
        """
        存储模型
        """
        model_name = f'epoch{epoch}.pkl'
        torch.save(model, os.path.join(opts.checkpoints_dir, model_name))

    def main(self) -> None:
        """
        训练入口
        """

        opts: argparse.Namespace = self.opts

        # 创建checkpoint目录
        if not os.path.exists(opts.checkpoints_dir):
            os.mkdir(opts.checkpoints_dir)
        train_dataset = YOLOV1Dataset(dataset_dir=opts.dataset_dir, mode='train')
        train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=opts.shuffle,
                                  num_workers=opts.num_workers, drop_last=opts.drop_last)
        # validate_dataset = YOLOV1Dataset(dataset_dir=opts.dataset_dir, mode='validate')
        # val_dataloader = DataLoader(validate_dataset, batch_size=opts.batch_size, shuffle=opts.shuffle,
        #                             num_workers=opts.num_workers)
        num_train = len(train_dataset)
        # 是否中途开始训练
        if opts.pretrain is None:
            model = YoloV1Net()
        else:
            model = torch.load(opts.pretrain)
        loss_obj = YoloV1Loss()
        # 指定使用的GPU
        if opts.use_gpu:
            model.to(opts.gpu_id)

        # 随机梯度下降算法
        optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
        # 迭代训练, 每一epoch训练完后, 保存训练的参数, 以防训练中断
        for e in range(opts.start_epoch, opts.epoch + 1):
            t1 = time.time()
            self.__train_epoch(model, loss_obj, train_loader, optimizer, e, num_train, opts)
            t2 = time.time()
            print("Training consumes %.2f second\n" % (t2 - t1))

            with open(os.path.join(opts.checkpoints_dir, 'log.txt'), 'a+') as log_f:
                log_f.write(f'Training one epoch consumes %.2f second\n' % (t2 - t1))

            # 存储model
            if e % self.opts.save_frequency == 0 or e == opts.epoch:
                self.__save_model(model, e, opts)


if __name__ == '__main__':
    # 训练网络代码
    args = Args()
    args.set_train_args()
    yolo_v1_train = YoloV1Train(args.opts)
    yolo_v1_train.main()
