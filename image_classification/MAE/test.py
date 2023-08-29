import os

import torch
import typing as t
from torch.utils.data import DataLoader
from augment import args_test
from data import MAEDataset, ImageAugmentation
from model.mae import model_factory
from loss import MAELoss
from util import mae_collate, print_log, reconstruction
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt


class MAETest(object):

    def __init__(self):
        self.opts = args_test.opts

    def main(
            self
    ):
        test_dataset = MAEDataset(mode='train', img_augmentation=ImageAugmentation)
        print(self.opts)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.opts.batch_size,
            shuffle=self.opts.shuffle,
            num_workers=self.opts.num_workers,
            drop_last=self.opts.drop_last,
            collate_fn=mae_collate
        )
        model = model_factory(model_name=self.opts.model, load_file=True)
        assert self.opts.pretrain_file, 'no specified weight file to load!'
        checkpoint = torch.load(self.opts.pretrain_file)
        model.load_state_dict(checkpoint['model'])
        print_log(f'Load model file {self.opts.pretrain_file} successfully!')
        if self.opts.use_gpu:
            model.to(self.opts.gpu_id)
        model.eval()
        loss_obj = MAELoss()
        for batch, (x, img_path) in enumerate(test_loader):
            img_path: t.List
            with torch.no_grad():
                if self.opts.use_gpu:
                    x = x.to(self.opts.gpu_id)
                pred, mask, target = model(x)
                cur_loss = loss_obj(pred, target, mask)
                print(f'avg_loss: {cur_loss.item()}')
                recons_img_t, patches_img_t = reconstruction(pred, target, test_dataset.mean, test_dataset.std)
                recons_img_t = recons_img_t.cpu().squeeze(0)
                patches_img_t = patches_img_t.cpu().squeeze(0)

                recons_img = ToPILImage(mode='RGB')(recons_img_t)
                patches_img = ToPILImage(mode='RGB')(patches_img_t)
                # save_file = os.path.join(os.path.dirname(__file__), img_path[0].split('\\')[-1])
                # recons_img.save(os.path.join(f'{os.path.dirname(__file__)}', img_path[0].split(r"\\")[-1]))

                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(recons_img)
                plt.subplot(1, 2, 2)
                plt.imshow(patches_img)
                plt.show()


if __name__ == '__main__':
    test = MAETest()
    test.main()
