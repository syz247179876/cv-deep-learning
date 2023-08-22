import argparse

import torch.cuda


class Args(object):
    """
    设置命令行参数接口
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opts = None

    def set_train_args(self):
        """
        设置训练参数
        """
        # 构建dataloader时是否打乱顺序
        self.parser.add_argument('--shuffle', action='store_true', default=True)
        self.parser.add_argument('--drop_last', action='store_true', default=True)
        self.parser.add_argument('--batch_size', type=int, default=4)
        self.parser.add_argument('--epoch', type=int, default=66)
        self.parser.add_argument('--start_epoch', type=int, default=59)
        # action用于存储True、False、list等特殊值
        self.parser.add_argument('--use_gpu', action='store_true')
        self.parser.add_argument('--gpu_id', type=int, default=None)
        # 数据集加载的工作进程个数
        self.parser.add_argument('--num_workers', type=int, default=4)

        self.parser.add_argument('--dataset_dir', type=str,
                                 default=r'C:\Users\24717\Projects\pascal voc2012\VOCdevkit\VOC2012\YOLO_V1_TrainTest')
        self.parser.add_argument('--checkpoints_dir', type=str, default=r'./checkpoints_dir')

        # 每隔固定epoch打印一次训练信息, 如loss
        self.parser.add_argument('--print_frequency', type=int, default=60)
        # 每隔固定epoch保存一次模型参数
        self.parser.add_argument('--save_frequency', type=int, default=2)
        # optimizer的learning rate和正则化系数
        self.parser.add_argument('-lr', type=float, default=.001)
        self.parser.add_argument("--weight_decay", type=float, default=1e-4)
        # 用于断点训练
        self.parser.add_argument('--pretrain', type=str, default=r'./checkpoints_dir/epoch58.pkl')

        # 随机数种子
        self.parser.add_argument('--random_seed', type=int, default=42)

        self.opts = self.parser.parse_args()

        # 本机是否支持gpu
        if torch.cuda.is_available():
            self.opts.use_gpu = True
            self.opts.gpu_id = torch.cuda.current_device()

    def set_test_args(self):
        """
        设置测试参数
        """
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--use_gpu', action='store_true', default=True)
        self.parser.add_argument('--gpu_id', type=int, default=None)
        self.parser.add_argument('--dataset_dir', type=str,
                                 default=r'C:\Users\24717\Projects\pascal voc2012\VOCdevkit\VOC2012\YOLO_V1_TrainTest')
        self.parser.add_argument('--model_dir', type=str,
                                 default=r'./checkpoints_dir/epoch58.pkl')
        self.parser.add_argument('--nms_threshold', type=float, default=0.5)
        self.opts = self.parser.parse_args()

        # 本机是否支持gpu
        if torch.cuda.is_available():
            self.opts.use_gpu = True
            self.opts.gpu_id = torch.cuda.current_device()
