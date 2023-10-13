import torch
from colorama import Fore
from torch.utils.data import DataLoader

from augment import args_test
from data import ViTDataset, FlowerTransform, ImageAugmentation, z_score_
from loss import ViTLoss
from model import ModelFactory
from utils import print_log, draw_image, classify_collate_test


class ViTTest(object):

    def __init__(self):
        self.opts = args_test().opts

    def main(
            self,
            count: int = 10
    ):
        avg_accu = 0.
        for i in range(1, count + 1):
            test_dataset = ViTDataset(mode='test', anno_transform=FlowerTransform,
                                      img_augmentation=ImageAugmentation, normalization=z_score_, distort=True)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.opts.batch_size,
                shuffle=self.opts.shuffle,
                num_workers=1,
                drop_last=self.opts.drop_last,
                collate_fn=classify_collate_test,
            )
            model = ModelFactory(self.opts.model).model
            checkpoints = torch.load(self.opts.pretrain_file)
            model.load_state_dict(checkpoints['model_state_dict'])
            print_log(f'load model {self.opts.pretrain_file} successfully!')
            if self.opts.use_gpu:
                model = model.to(self.opts.gpu_id)
            model.eval()
            loss_obj = ViTLoss()
            test_num = len(test_dataset)
            accu_num = 0
            for batch, (x, labels, img_path) in enumerate(test_loader):
                with torch.no_grad():
                    if self.opts.use_gpu:
                        x = x.to(self.opts.gpu_id)
                        labels = labels.to(self.opts.gpu_id)
                    pred = model(x)
                    cur_loss, _ = loss_obj(pred, labels)
                    if draw_image(img_path[0], cur_loss.item(), pred, test_dataset.classes, labels[0].item(), False):
                        accu_num += 1
            print_log(f'test-{i} accuracy: {accu_num / test_num}', Fore.YELLOW)
            avg_accu += accu_num / test_num
        print_log(f'total test {count} epoch, accuracy: {avg_accu / count}', Fore.BLUE)


if __name__ == '__main__':
    test = ViTTest()
    test.main()
