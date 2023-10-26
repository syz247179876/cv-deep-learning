"""
Basic train and validate Class
"""
import argparse
import math
import os.path
import random
import time

import numpy as np
import torch.cuda
import typing as t
import torch.nn as nn
import torch
import datetime
from pathlib import Path

from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from colorama import Fore
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
from alchemy.settings import *
from alchemy.utils.weight import pretrained_weight

FILE = r'C:\dataset\flower_dataset'
ROOT = Path(FILE).resolve()

M = t.TypeVar('M', bound=nn.Module)


def print_log(txt: str, color: t.Any = Fore.GREEN):
    print(color, txt)


def base_collate(batch: t.Iterable[t.Tuple]):
    labels = []
    images = []
    for img, label in batch:
        images.append(img)
        labels.append(label)
    return torch.stack(images, dim=0), torch.tensor(labels)


def shuffle(arr: t.List[t.Union[t.Tuple, str]], arr_len: int) -> None:
    """
    基于Fisher-Yates的洗牌算法打乱arr
    时间复杂度为O(N)
    """
    for i in range(arr_len - 1):
        idx = random.randint(i, arr_len - 1)
        arr[i], arr[idx] = arr[idx], arr[i]


class Args(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opts = None

    def set_train_args(self):
        self.parser.add_argument('--model_name', type=str, help='Models that need to be loaded')
        self.parser.add_argument('--start_epoch', type=int, default=0, help='The epoch of current training')
        self.parser.add_argument('--end_epoch', type=int, default=101, help='Maximum number of iterations')
        self.parser.add_argument('--batch_size', type=int, default=16, help='Number of data processed per batch')

        self.parser.add_argument('--use_gpu', action='store_true', default=True,
                                 help='whether use GPU to training or test')
        self.parser.add_argument('--gpu_id', type=int, default=0, help='GPU number used')

        self.parser.add_argument('--num_workers', type=int, default=6,
                                 help='Number of processes used to load batch into RAM, '
                                      'default is number of CPU Kernel')

        self.parser.add_argument('--dataset_name', type=str, default='flower',
                                 help='Method prefixes related to data names')

        self.parser.add_argument('--dataset_path', type=str, default=ROOT, help='dataset path')
        self.parser.add_argument('--shuffle', action='store_true', default=True,
                                 help='whether shuffle the whole dataset for each epoch')

        self.parser.add_argument('--drop_last', action='store_true', default=False,
                                 help='Discard the last insufficient batch_ Size')

        self.parser.add_argument('--pretrained', action='store_true', default=False,
                                 help='whether use pretrain file')
        self.parser.add_argument('--checkpoint_file', type=str, default='',
                                 help='latest best model file path')

        self.parser.add_argument('--checkpoints_dir', type=str, default=r'./checkpoints_dir',
                                 help='model file path')

        self.parser.add_argument('--print_frequency', type=int, default=1,
                                 help='Frequency of checking and printing models information')
        self.parser.add_argument('--validate_frequency', type=int, default=5,
                                 help='Frequency of validating model')

        self.parser.add_argument('--lr_base', type=float, default=1.5e-3, help='base learning rate')
        self.parser.add_argument('--lr_max', type=float, default=5e-2, help='maximum of learning rate')
        self.parser.add_argument('--lr_min', type=float, default=1e-5, help='minimum of learning rate')
        self.parser.add_argument('--weight_decay', type=float, default=0.05, help='regularization coefficient')

        self.parser.add_argument('--model', type=str, default='large',
                                 help='different kind of models with scale'
                                      'value can be "base", "huge", "large", default is "base"')

        self.parser.add_argument('--freeze_lagers', action='store_false', default=False,
                                 help='if set true, freeze some layers except head layer')

        self.parser.add_argument('--train_ratio', type=float, default=0.7,
                                 help='Ratio of dividing training sets')
        self.parser.add_argument('--validate_ratio', type=float, default=0.1,
                                 help='Ratio of dividing validation sets')
        self.parser.add_argument('--test_ratio', type=float, default=0.2,
                                 help='Ratio of dividing test sets')

        self.parser.add_argument('--decrease_interval', type=int, default=6,
                                 help='LR func')
        self.parser.add_argument('--use_amp', action='store_true', default=False,
                                 help='whether open amp')

        self.opts = self.parser.parse_args()
        if torch.cuda.is_available():
            self.opts.use_gpu = True
            self.opts.gpu_id = torch.cuda.current_device()


class FLOWERExtract(object):
    """
    Flower Dataset preprocess
    """

    @classmethod
    def extract(cls, img_dir: str, use_shuffle: bool = False) -> t.Tuple[t.List, t.Dict]:
        files_path: t.List = []
        cls_map: t.Dict = {}
        for root, dirs, files in os.walk(img_dir):
            if files:
                cls_name = root.split('\\')[-1]
                for file in files:
                    cls_map[file] = cls.cls_id_map()[cls_name]
                    files_path.append(os.path.join(root, file))
        if use_shuffle:
            shuffle(files_path, len(files_path))
        return files_path, cls_map

    @classmethod
    def cls_id_map(cls):
        return {
            'daisy': 0,
            'dandelion': 1,
            'roses': 2,
            'sunflowers': 3,
            'tulips': 4,
        }

    @classmethod
    def id_cls_map(cls):
        return {
            0: 'daisy',
            1: 'dandelion',
            2: 'roses',
            3: 'sunflowers',
            4: 'tulips',
        }


class ImageAugmentationBase(object):

    def __init__(self):
        super(ImageAugmentationBase, self).__init__()

    def __call__(
            self,
            img: JpegImageFile,
            mean: t.List,
            std: t.List,
            input_shape: t.Tuple = (224, 224),
    ):
        img = img.convert('RGB')
        transform = transforms.Compose([
            # Random cropping
            transforms.RandomResizedCrop(input_shape, scale=(0.2, 1.0)),  # 3 is bicubic
            # flip horizontal
            transforms.RandomHorizontalFlip(),
            # divide 255.
            transforms.ToTensor(),
            # using z-score normalization can improve the latent representation learned by the model during training
            transforms.Normalize(mean, std),
        ])
        return transform(img)


class TrainBase(object):

    def __init__(
            self,
            args_obj: Args,
            model: M,
            model_name: str,
            optimizer: t.Optional[torch.optim.Optimizer] = None,
            scheduler: t.Optional[torch.optim.lr_scheduler.LRScheduler] = None,
            loss_obj: t.Optional[M] = None,
            train_dataset: t.Optional[Dataset] = None,
            validate_dataset: t.Optional[Dataset] = None,
            train_dataloader: t.Optional[DataLoader] = None,
            validate_dataloader: t.Optional[DataLoader] = None,
            use_amp: bool = False
    ):
        self.opts = args_obj.opts
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_obj = loss_obj
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        self.train_dataset = train_dataset or DatasetBase(self.opts, mode='train',
                                                          img_augmentation=ImageAugmentationBase)
        self.validate_dataset = validate_dataset or DatasetBase(self.opts, mode='validate',
                                                                img_augmentation=ImageAugmentationBase)
        self.train_loader = train_dataloader or DataLoader(
            self.train_dataset,
            batch_size=self.opts.batch_size,
            shuffle=self.opts.shuffle,
            num_workers=self.opts.num_workers,
            drop_last=self.opts.drop_last,
            collate_fn=base_collate,
        )
        self.validate_loader = validate_dataloader or DataLoader(
            self.validate_dataset,
            batch_size=self.opts.batch_size,
            shuffle=self.opts.shuffle,
            num_workers=self.opts.num_workers,
            drop_last=self.opts.drop_last,
            collate_fn=base_collate,
        )
        self.train_num = len(self.train_dataset)
        self.validate_num = len(self.validate_dataset)
        self.last_epoch = 0
        self.last_acc = 0.

    @property
    def optimizer_(self) -> torch.optim.Optimizer:
        return self.optimizer

    @optimizer_.setter
    def optimizer_(self, optimizer: M) -> None:
        self.optimizer = optimizer

    @property
    def scheduler_(self) -> torch.optim.lr_scheduler.LRScheduler:
        return self.scheduler

    @scheduler_.setter
    def scheduler_(self, scheduler: M) -> None:
        self.scheduler = scheduler

    @property
    def loss_(self) -> M:
        return self.loss_obj

    @loss_.setter
    def loss_(self, loss: M) -> None:
        self.loss_obj = loss

    def __save_model(
            self,
            epoch: int,
            accuracy: float
    ) -> None:
        """
        save model, last_accuracy means the best last time
        """
        model_name = f'{self.model_name}-epoch{epoch}-{round(accuracy, 4)}-{str(datetime.date.today())}.pth'
        torch.save({
            'last_epoch': epoch,
            'last_accuracy': accuracy,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, os.path.join(self.opts.checkpoints_dir, model_name))

    def load_model(self):
        """
        Load weights saved through transfer training
        """
        if self.opts.checkpoint_file:
            checkpoint = torch.load(self.opts.checkpoint_file)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.last_epoch = checkpoint.get('last_epoch', 0)
            self.last_acc = checkpoint.get('last_accuracy', 0.)
            print_log(f'Load model file {self.opts.checkpoint_file} to {self.model_name}successfully!')

    def init_lr(self, max_batch: int = 64) -> float:
        """
        Initial learning curvature
        """
        return min(max(self.opts.batch_size / max_batch * self.opts.lr_base, self.opts.lr_min), self.opts.lr_max)

    def print_detail(
            self,
            loop: tqdm,
            cur_epoch: int,
            end_epoch: int,
            batch: int,
            iter_total: int,
            cur_batch_loss: float,
            avg_loss: float,
            accuracy: float,
            num: int,
            log_f: t.TextIO,
            write: bool = False,
    ) -> None:
        info = f'''Epoch: {cur_epoch} / {end_epoch}\t Iter: {batch} / {iter_total}\t
                   Cur Batch Loss: {round(cur_batch_loss, 5)}\t Avg Loss: {round(avg_loss, 5)}\n
                   Accuracy: {round(accuracy, 3)}
                   '''

        loop.set_description(f'Epoch [{cur_epoch + 1}/{end_epoch}]  Iter [{batch}/{num // self.opts.batch_size}]')
        loop.set_postfix(cur_batch_loss=round(cur_batch_loss, 5), avg_loss=round(avg_loss, 5),
                         accuracy=round(accuracy, 3))

        if write:
            log_f.write(info)
            log_f.flush()

    def load_pretrained(self):
        """
        load partial pretrained weights, such as resnet18, resnet50,  resnet101 and others,
        then make adjustments and freezes.
        """
        model_name = self.opts.model_name
        weight_file = pretrained_weight.get_weight(model_name)
        weights = torch.load(weight_file)
        model_dict = self.model.state_dict()
        for key_p, key_w in zip(list(model_dict.keys())[:-2], list(weights.keys())[: -2]):
            value_p, value_w = model_dict[key_p], weights[key_w]
            if value_p.shape != value_w.shape:
                raise ValueError(f'{key_p}: {value_p.shape} and {key_w}: {value_w.shape} shape not match')
            model_dict[key_p] = value_w
        self.model.load_state_dict(model_dict)

    def freeze_layers(self):
        pass

    def __train_epoch(
            self,
            model: M,
            loss_obj: M,
            train_loader: DataLoader,
            validate_loader: DataLoader,
            optimizer: Optimizer,
            scaler: GradScaler,
            epoch: int,
    ) -> float:

        total_loss = 0.
        total_acc = 0.
        log_name = f'{self.model_name}_log.txt'
        batch_num = 0

        # train
        print_log('start train...')
        self.model.train()
        with open(os.path.join(self.opts.checkpoints_dir, log_name), 'a+') as f:
            loop = tqdm(train_loader, desc='training...', colour=BAR_TRAIN_COLOR)
            for batch, (x, label) in enumerate(loop):
                batch_num += 1
                if self.opts.use_gpu:
                    x = x.to(self.opts.gpu_id)
                    label = label.to(self.opts.gpu_id)
                if self.use_amp:
                    with autocast():
                        pred = model(x)
                        cur_loss, acc_num = loss_obj(pred, label)
                else:
                    pred = model(x)
                    cur_loss, acc_num = loss_obj(pred, label)
                optimizer.zero_grad()
                if self.use_amp:
                    scaler.scale(cur_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    cur_loss.backward()
                    optimizer.step()
                total_loss += cur_loss.item()
                total_acc += acc_num.item()

                if batch % self.opts.print_frequency == 0:
                    self.print_detail(loop, epoch, self.opts.end_epoch, batch, self.train_num // self.opts.batch_size,
                                      cur_batch_loss=cur_loss.item(), avg_loss=total_loss / (batch + 1),
                                      accuracy=total_acc / self.train_num, num=self.train_num, log_f=f, write=True)

        total_loss = 0.
        total_acc = 0.
        batch_num = 0

        # validate
        if (epoch + 1) % self.opts.validate_frequency == 0 or epoch > int(0.6 * self.opts.end_epoch):
            print_log('\nstart validate...')
            self.model.eval()
            with open(os.path.join(self.opts.checkpoints_dir, log_name), 'a+') as f:
                with torch.no_grad():
                    loop = tqdm(validate_loader, desc='validating...', colour=BAR_VALIDATE_COLOR)
                    for batch, (x, label) in enumerate(loop):
                        batch_num += 1
                        if self.opts.use_gpu:
                            x = x.to(self.opts.gpu_id)
                            label = label.to(self.opts.gpu_id)
                        pred = model(x)
                        cur_loss, acc_num = loss_obj(pred, label)
                        total_loss += cur_loss.item()
                        total_acc += acc_num.item()

                        if batch % self.opts.print_frequency == 0:
                            self.print_detail(loop, epoch, self.opts.end_epoch, batch,
                                              self.validate_num // self.opts.batch_size,
                                              cur_batch_loss=cur_loss.item(), avg_loss=total_loss / (batch + 1),
                                              accuracy=total_acc / self.validate_num, num=self.validate_num,
                                              log_f=f, write=True)

        return total_acc / self.validate_num

    def main(self):
        """
        train entrance
        """
        if not os.path.exists(self.opts.checkpoints_dir):
            os.mkdir(self.opts.checkpoints_dir)
        print_log(f'Init model --- {self.model_name} successfully!')
        if self.opts.use_gpu:
            self.model = self.model.to(self.opts.gpu_id)
        if self.opts.pretrained:
            self.load_pretrained()
        else:
            self.load_model()
        self.freeze_layers()

        for e in range(self.last_epoch, self.opts.end_epoch):
            t1 = time.time()
            avg_acc = self.__train_epoch(self.model, self.loss_obj, self.train_loader, self.validate_loader,
                                         self.optimizer, self.scaler, e)
            t2 = time.time()
            self.scheduler.step()
            print_log(f"learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}", Fore.BLUE)
            print_log("Training consumes %.2f second\n" % (t2 - t1), Fore.RED)

            if avg_acc > self.last_acc:
                self.__save_model(e, avg_acc)
                self.last_acc = avg_acc


class ClassifyLoss(nn.Module):

    def __init__(self):
        super(ClassifyLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        pred_classes_idx = torch.max(pred, dim=1)[1]
        acc_num = torch.eq(pred_classes_idx, labels).sum()
        return self.loss_func(pred, labels), acc_num


class DatasetBase(Dataset):
    """
    Base Dataset
    """

    def __init__(
            self,
            opts: argparse.Namespace,
            mode: str = 'train',
            input_shape: t.Tuple[int, int] = (224, 224),
            img_augmentation: t.Optional[t.Callable] = None,
            img_augmentation_kwargs: t.Optional[t.Dict] = None
    ):
        super(DatasetBase, self).__init__()
        self.opts = opts
        self.input_shape = input_shape
        dataset_name = self.opts.dataset_name.upper()
        self.img_dir = self.opts.dataset_path
        self.mean = globals().get(f'{dataset_name}_MEAN', IMAGENET_MEAN)
        self.std = globals().get(f'{dataset_name}_STD', IMAGENET_STD)
        img_augmentation_kwargs = img_augmentation_kwargs or {}
        self.img_augmentation = img_augmentation and img_augmentation(**img_augmentation_kwargs)
        self.anno_transform = globals().get(f'{dataset_name}Extract', None)
        assert self.anno_transform is not None and getattr(self.anno_transform, 'extract'), \
            f'Class {dataset_name}Extract should be define and extract function should be implement!'
        self.total_images, self.cls_map = self.anno_transform.extract(self.img_dir)
        self.use_images = []
        func = getattr(self, f'{mode}_data')
        func()

    def __len__(self):
        return len(self.use_images)

    def __getitem__(self, item):
        image_path: str = self.use_images[item]
        img = Image.open(image_path)
        if self.img_augmentation:
            img = self.img_augmentation(img, self.mean, self.std, self.input_shape)
        else:
            img = torch.tensor(np.array(img)).float()
        return img, self.cls_map[image_path.split('\\')[-1]]

    def train_data(self):
        end_idx = int(len(self.total_images) * self.opts.train_ratio)
        self.use_images = self.total_images[: end_idx]

    def validate_data(self):
        start_idx = int(len(self.total_images) * self.opts.train_ratio)
        num = int(len(self.total_images) * self.opts.validate_ratio)
        self.use_images = self.total_images[start_idx: start_idx + num]

    def test_data(self):
        start_idx = int(len(self.total_images) * self.opts.train_ratio) + \
                    int(len(self.total_images) * self.opts.validate_ratio)
        self.use_images = self.total_images[start_idx:]


def basic_run(model: M, model_name: str, args: Args, loss_obj: ClassifyLoss, train_class: t.Callable = TrainBase):
    """
    base-model family training
    """
    params = [p for p in model.parameters() if p.requires_grad]
    train = train_class(args, model, model_name, loss_obj=loss_obj, use_amp=args.opts.use_amp)
    print(args.opts)
    optimizer = torch.optim.SGD(params, lr=train.init_lr(), momentum=0.9,
                                weight_decay=args.opts.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.opts.decrease_interval + 1e-8),
                                0.5 * (math.cos(epoch / args.opts.end_epoch * math.pi) + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

    train.scheduler_ = scheduler
    train.optimizer_ = optimizer
    train.main()
