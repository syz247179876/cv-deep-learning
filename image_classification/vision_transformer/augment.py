import argparse

import torch.cuda


class Args(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opts = None

    def set_train_args(self):
        self.parser.add_argument('--epoch', type=int, default=0, help='the epoch of current training')
        self.parser.add_argument('--batch_size', type=int, default=4)
        self.parser.add_argument('--use_gpu', action='store_true')
        self.parser.add_argument('--gpu_id', type=int, default=None)
        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--base_dir',
                                 type=str,
                                 default=r'C:\Users\24717\Projects\dataset',
                                 help='the base dir of dataset'
                                 )
        self.parser.add_argument('--shuffle', action='store_true', default=True)
        self.parser.add_argument('--drop_last', action='store_true', default=True)
        self.parser.add_argument('--pretrain_file', type=str,
                                 help='store the latest model file')
        self.parser.add_argument('--print_frequency', type=int, default=60, help='print interval')
        self.parser.add_argument('--save_frequency', type=int, default=4, help='store model interval')
        self.parser.add_argument('--checkpoints_dir', type=str, default=r'./checkpoints_dir', help='store the model')

        self.parser.add_argument('--start_epoch', type=int, default=0)
        self.parser.add_argument('--end_epoch', type=int, default=100)
        self.opts = self.parser.parse_args()
        if torch.cuda.is_available():
            self.opts.use_gpu = True
            self.opts.gpu_id = torch.cuda.current_device()

    def set_test_args(self):
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--use_gpu', action='store_true')
        self.parser.add_argument('--gpu_id', type=int, default=None)
        self.parser.add_argument('--base_dir',
                                 type=str,
                                 default=r'',
                                 help='the base dir of dataset'
                                 )
        self.parser.add_argument('--shuffle', action='store_true', default=True)
        self.parser.add_argument('--drop_last', action='store_true', default=True)
        self.parser.add_argument('--pretrain_file', type=str,
                                 help='store the latest model file')
        self.parser.add_argument('--random_seed', type=int, default=42)
        self.opts = self.parser.parse_args()
        if torch.cuda.is_available():
            self.opts.use_gpu = True
            self.opts.gpu_id = torch.cuda.current_device()


args_train = Args()
args_train.set_train_args()
args_test = Args()
args_test.set_test_args()
