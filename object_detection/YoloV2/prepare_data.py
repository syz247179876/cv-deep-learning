"""
预处理数据
"""
import os
import xml.etree.ElementTree as ET
import typing as t

from util import normalization
from argument import Args
from settings import *


class VOCProcess(object):
    """
    处理VOC数据集
    """

    def __init__(self):
        args = Args()
        args.set_process_args()
        self.opts = args.opts
        self.base_dir = self.opts.base_dir

    # def remove(self):
    #     annotations = os.path.join(self.base_dir, 'Annotations')
    #     filenames = os.listdir(annotations)
    #     for filename in filenames:
    #         file_path = os.path.join(annotations, filename)
    #         if os.path.isfile(file_path) and file_path.endswith('.txt'):
    #             os.remove(file_path)

    def retrieve_boxes_info(self):
        """
        抽取VOC数据集中所有bbox的具体信息
        """
        annotations = os.path.join(self.base_dir, 'Annotations')
        labels_dir = os.path.join(self.base_dir, f'{MODEL_NAME}_Labels')
        if not os.path.exists(labels_dir):
            os.mkdir(labels_dir)
        filenames = os.listdir(annotations)
        for filename in filenames:
            file_path = os.path.join(annotations, filename)
            img_id = filename.split('.')[0]
            tree = ET.parse(file_path)
            root = tree.getroot()

            size = root.find('size')
            pic_w, pic_h = int(size.find('width').text), int(size.find('height').text)

            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in VOC_CLASSES:
                    continue
                cls_id = VOC_CLASSES.index(cls)
                bbox = obj.find('bndbox')
                x_min, y_min, x_max, y_max = float(bbox.find('xmin').text), float(bbox.find('ymin').text), float(
                    bbox.find('xmax').text), float(bbox.find('ymax').text)

                bbox_info = normalization(x_min, y_min, x_max, y_max, pic_w, pic_h=pic_h, pic_w=pic_w)
                with open(os.path.join(labels_dir, f'{img_id}.txt'), 'a') as f:
                    f.write(f'{cls_id} {" ".join([str(b) for b in bbox_info])}\n')


if __name__ == "__main__":
    voc_p = VOCProcess()
    print('---start processing---')
    voc_p.retrieve_boxes_info()
    print('---process finished---')
