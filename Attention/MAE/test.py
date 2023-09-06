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
        test_dataset = MAEDataset(mode='test', img_augmentation=ImageAugmentation)
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
        loss_obj = MAELoss(model.patches_tuple, model.patch_size)
        for batch, (x, img_path) in enumerate(test_loader):
            img_path: t.List
            with torch.no_grad():
                if self.opts.use_gpu:
                    x = x.to(self.opts.gpu_id)
                pred, mask, target = model(x)
                cur_loss, mask_targets = loss_obj(pred, x, mask, test=True)
                print(f'avg_loss: {cur_loss.item()}')
                recons_img_t, original_img_t, mask_img_t = reconstruction(pred, x, mask_targets, test_dataset.mean,
                                                                          test_dataset.std)
                recons_img_t = recons_img_t.cpu().squeeze(0)
                original_img_t = original_img_t.cpu().squeeze(0)
                mask_img_t = mask_img_t.cpu().squeeze(0)

                recons_img = ToPILImage(mode='RGB')(recons_img_t)
                original_img = ToPILImage(mode='RGB')(original_img_t)
                mask_img = ToPILImage(mode='RGB')(mask_img_t)
                # save_file = os.path.join(os.path.dirname(__file__), img_path[0].split('\\')[-1])
                # recons_img.save(os.path.join(f'{os.path.dirname(__file__)}', img_path[0].split(r"\\")[-1]))

                plt.figure()
                plt.subplot(1, 3, 1)
                plt.title('Restored image')
                plt.imshow(recons_img)
                plt.subplot(1, 3, 2)
                plt.title('Mask image')
                plt.imshow(mask_img)
                plt.subplot(1, 3, 3)
                plt.title('Original image')
                plt.imshow(original_img)
                plt.show()


if __name__ == '__main__':
    test = MAETest()
    test.main()
