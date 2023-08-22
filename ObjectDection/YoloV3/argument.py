"""
hyper-parameters
"""
import argparse

import torch.cuda


class Args(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opts = None

    def set_process_args(self):
        """
        set params of pretreatment
        """
        self.parser.add_argument('--random_seed', type=int, default=333, help='use to randomly initialize anchor boxes')
        self.parser.add_argument('--anchors_num', type=int, default=5, help='the number of anchor boxes')
        self.parser.add_argument('--max_iter', type=int, default=300,
                                 help='the max iteration to find best anchor boxes')
        self.parser.add_argument('--base_dir',
                                 type=str,
                                 default=r'C:\Users\24717\Projects\pascal voc2012\VOCdevkit\VOC2012',
                                 help='the base dir of dataset'
                                 )

        self.opts = self.parser.parse_args()

    def set_train_args(self):
        """
        set params of train
        """
        self.parser.add_argument('--anchors_num', type=int, default=5)
        self.parser.add_argument('--base_dir',
                                 type=str,
                                 default=r'C:\Users\24717\Projects\pascal voc2012\VOCdevkit\VOC2012',
                                 help='the base dir of dataset'
                                 )
        self.parser.add_argument('--anchors_positive_thresh', type=float, default=0.7,
                                 help='threshold for selecting positive sample anchors'
                                 )
        self.parser.add_argument('--anchors_ignore_thresh', type=float, default=0.5,
                                 help='threshold for selecting ignore sample anchors')
        self.parser.add_argument('--anchors_negative_thresh', type=float, default=0.2,
                                 help='threshold for selecting negative sample anchors')

        self.parser.add_argument('--epoch', type=int, default=0, help='the epoch of current training')
        self.parser.add_argument('--batch_size', type=int, default=4)
        self.parser.add_argument('--use_gpu', action='store_true')
        self.parser.add_argument('--gpu_id', type=int, default=None)
        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--coord_loss_mode', type=str, default='giou',
                                 help='mode for calculating coordinate loss')
        # learning rate interval
        self.parser.add_argument('--lr_base', type=float, default=5e-2)
        self.parser.add_argument('--lr_max', type=float, default=5e-2, help='maximum of learning rate')
        self.parser.add_argument('--lr_min', type=float, default=5e-5, help='minimum of learning rate')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='regularization coefficient')
        self.parser.add_argument('--pretrain_file', type=str, default=r'./checkpoints_dir/epoch90.pkl',
                                 help='store the latest model file')
        self.parser.add_argument('--random_seed', type=int, default=42)
        self.parser.add_argument('--print_frequency', type=int, default=60, help='print interval')
        self.parser.add_argument('--save_frequency', type=int, default=2, help='store model interval')
        self.parser.add_argument('--checkpoints_dir', type=str, default=r'./checkpoints_dir', help='store the model')
        self.parser.add_argument('--shuffle', action='store_true', default=True)
        self.parser.add_argument('--drop_last', action='store_true', default=True)

        self.parser.add_argument('--start_epoch', type=int, default=81)
        self.parser.add_argument('--end_epoch', type=int, default=100)

        # loss weight
        self.parser.add_argument('--coord_weight', type=float, default=0.5)
        self.parser.add_argument('--no_obj_weight', type=float, default=1.)

        self.parser.add_argument('--decrease_interval', type=int, default=30,
                                 help='decrease learning rate every 30 epoch')
        self.opts = self.parser.parse_args()

        if torch.cuda.is_available():
            self.opts.use_gpu = True
            self.opts.gpu_id = torch.cuda.current_device()

    def set_test_args(self):
        """
        set params of test
        """
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--use_gpu', action='store_true')
        self.parser.add_argument('--gpu_id', type=int, default=None)
        self.parser.add_argument('--num_workers', type=int, default=4)

        self.parser.add_argument('--checkpoints_dir', type=str, default=r'./checkpoints_dir', help='store the model')
        self.parser.add_argument('--shuffle', action='store_true', default=True)
        self.parser.add_argument('--drop_last', action='store_true', default=True)
        self.parser.add_argument('--pretrain_file', type=str, default='./checkpoints_dir/epoch90.pkl',
                                 help='store the latest model file')
        self.parser.add_argument('--conf_thresh', type=float, default=0.5,
                                 help='filter the bbox which confidence less than confidence threshold')
        self.parser.add_argument('--iou_thresh', type=float, default=0.5,
                                 help='apply to NMS, filter the bbox which iou greater than iou threshold')

        self.opts = self.parser.parse_args()

        if torch.cuda.is_available():
            self.opts.use_gpu = True
            self.opts.gpu_id = torch.cuda.current_device()


args_process = Args()
args_process.set_process_args()
args_train = Args()
args_train.set_train_args()
args_test = Args()
args_test.set_test_args()
