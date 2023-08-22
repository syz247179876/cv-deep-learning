"""
数据预处理
1.根据Annotations中的xml提取出所有bbox信息, 以txt格式保存到Labels目录下
2.对每一图像做padding, 再resize到448 x 448, 将新img保存到ProcessImg, 同时更新Labels目录下所有图像的bbox信息
3.随机划分训练集和测试集，并分别另存它们的图像路径和转换为7x7x30张量格式的数据到TrainTest目录下
"""
import torch

from debug import debug_yolo_v1, DEBUG_OPEN
from settings import *
import typing as t
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os
import random


def normalization(pic_w, pic_h, *bbox_coordinate) -> t.Tuple[float, float, float, float]:
    """
    对pic_w, pic_h归一化, 将bbox左上角，右下角点坐标转为bbox中心点坐标，并归一化
    """
    n_w = 1.0 / pic_w
    n_h = 1.0 / pic_h
    mid_x = (bbox_coordinate[0] + bbox_coordinate[2]) / 2.0
    mid_y = (bbox_coordinate[1] + bbox_coordinate[3]) / 2.0
    bbox_w = bbox_coordinate[2] - bbox_coordinate[0]
    bbox_y = bbox_coordinate[3] - bbox_coordinate[1]
    return mid_x * n_w, mid_y * n_h, bbox_w * n_w, bbox_y * n_h


def retrieve_bbox_info(annotation_dir: str, image_id: str, labels_dir: str):
    """
    根据image_id提取对应image_id.xml中的bbox信息, 将信息写入labels文件
    信息包含: bbox的width, height, left-top coordinate, class
    """
    xml_file = os.path.join(annotation_dir, image_id)
    image_id = image_id.split('.')[0]
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    pic_w, pic_h = int(size.find('width').text), int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # 对不在希望训练的class中或识别难度等于1的bbox过滤掉
        if cls not in CLASSES or int(difficult) == 1:
            continue
        cls_id = CLASSES.index(cls)
        bbox = obj.find('bndbox')
        x_min, y_min, x_max, y_max = float(bbox.find('xmin').text), float(bbox.find('ymin').text), float(
            bbox.find('xmax').text), float(bbox.find('ymax').text)

        # 归一化
        bbox_info = normalization(pic_w, pic_h, x_min, y_min, x_max, y_max)
        with open(os.path.join(labels_dir, f'{image_id}.txt'), 'a') as f:
            f.write(f'{cls_id} {" ".join([str(co) for co in bbox_info])}\n')
    #         img = cv2.imread(rf'C:\Users\24717\Projects\pascal voc2012\VOCdevkit\VOC2012\JPEGImages\{image_id}.jpg')
    #         h, w, dim = img.shape
    #         if DEBUG_OPEN:
    #             top_left = (
    #                 int(bbox_info[0] * w - bbox_info[2] * w / 2),
    #                 int(bbox_info[1] * h - bbox_info[3] * h / 2)
    #             )
    #             bottom_right = (
    #                 int(bbox_info[0] * w + bbox_info[2] * w / 2),
    #                 int(bbox_info[1] * h + bbox_info[3] * h / 2)
    #             )
    #
    #             debug_yolo_v1.draw_box(img, top_left, bottom_right, (255, 0, 0))
    # if DEBUG_OPEN:
    #     cv2.imshow(f'bbox-class6666666666', img)
    #     cv2.waitKey(0)


def img_padding(img_dir: str, save_img_dir: str, labels_dir: str, bbox_img_dir: str):
    """
    部分图片像素不足448 x 448
    1.同时对每张图像进行padding, 弥补图像边缘像素点信息在多层conv中发挥作用小的缺点以及多层conv导致最终的 feature map过小的问题
    对padding后的图像在scale到448 x 448, 保存到新的文件中
    2.确保所有输入的图像尺寸一致
    Annotation: 向下resize会导致cnn识别准确率降低
    """
    img_list = [f'{filename.split(".")[0]}.jpg' for filename in os.listdir(labels_dir)]
    print(f'img dir: {img_dir}')
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        bbox_img_path = os.path.join(bbox_img_dir, img_name)
        # print(f'start process img: {img_name}')
        img = cv2.imread(img_path)
        h, w, dim = img.shape

        # 先统一padding图像为正方形
        pad_w, pad_h = 0, 0
        if h > w:
            # padding width
            pad_w = (h - w) // 2
            img = np.pad(img, ((0, 0), (pad_w, pad_w), (0, 0)), 'constant', constant_values=0)
        elif h < w:
            # padding height
            pad_h = (w - h) // 2
            img = np.pad(img, ((pad_h, pad_h), (0, 0), (0, 0)), 'constant', constant_values=0)

        # 缩放到yolo v1规定的448 x 448
        img = cv2.resize(img, (INPUT_IMG_W, INPUT_IMG_H))
        cv2.imwrite(os.path.join(save_img_dir, img_name), img)

        # 修改labels中bbox的信息
        file_path = os.path.join(labels_dir, f'{img_name.split(".")[0]}.txt')
        with open(file_path, 'r') as f:
            bboxes_info: t.List = f.read().split('\n')[:-1]
        bboxes_info = [bbox.split() for bbox in bboxes_info]
        bboxes_info: t.List[float] = [float(x) for y in bboxes_info for x in y]

        assert len(bboxes_info) % 5 == 0, f'file：{file_path} parse error'
        top_left, bottom_right = None, None
        # 根据padding、resize调整bbox的长、宽
        len_bboxes = len(bboxes_info)
        if h > w:
            for i in range(len_bboxes // 5):
                # 更新相对于size=h来说, bbox中心w和其width的归一化值
                bboxes_info[i * 5 + 1] = (bboxes_info[i * 5 + 1] * w + pad_w) / h
                bboxes_info[i * 5 + 3] = (bboxes_info[i * 5 + 3] * w) / h
        if w > h:
            # 调整bbox的height
            for i in range(len_bboxes // 5):
                # 更新相对于size=w来说, bbox中心h和其height的归一化值
                bboxes_info[i * 5 + 2] = (bboxes_info[i * 5 + 2] * h + pad_h) / w
                bboxes_info[i * 5 + 4] = (bboxes_info[i * 5 + 4] * h) / w

        # 覆盖重写label(bbox info)
        with open(file_path, 'w') as f:
            for i in range(len_bboxes // 5):
                # content: class x_mid y_mid width height
                info = f'{int(bboxes_info[i * 5])} {" ".join([str(co) for co in bboxes_info[i * 5 + 1: i * 5 + 5]])}\n'
                # print(f'bbox information: {info}')
                f.write(info)
                # if DEBUG_OPEN:
                #     top_left = (
                #         int(bboxes_info[i * 5 + 1] * INPUT_IMG_W - bboxes_info[i * 5 + 3] * INPUT_IMG_W / 2),
                #         int(bboxes_info[i * 5 + 2] * INPUT_IMG_H - bboxes_info[i * 5 + 4] * INPUT_IMG_H / 2)
                #     )
                #     bottom_right = (
                #         int(bboxes_info[i * 5 + 1] * INPUT_IMG_W + bboxes_info[i * 5 + 3] * INPUT_IMG_W / 2),
                #         int(bboxes_info[i * 5 + 2] * INPUT_IMG_H + bboxes_info[i * 5 + 4] * INPUT_IMG_H / 2)
                #     )
                #     debug_yolo_v1.draw_box(img, top_left, bottom_right, (255, 125, 125))
        # if DEBUG_OPEN:
        #     cv2.imshow(f'bbox-class-{CLASSES[int(bboxes_info[0])]}', img)
        #     cv2.waitKey(0)
        # 保存带有目标bbox的图像
        cv2.imwrite(bbox_img_path, img)


def generate_label(annotation_dir: str, labels_dir: str) -> None:
    """
    遍历annotation下所有xml, 生成labels信息,写入数据集下的labels目录
    """
    filenames = os.listdir(annotation_dir)
    for filename in filenames:
        retrieve_bbox_info(annotation_dir, filename, labels_dir)


def compute_actual_pos(bbox_info: t.List[float]) -> np.ndarray:
    """
    计算中心点相对其所在的grid的位置(确定正样本及正样本位置信息)
    定义计算bbox的中心点相对所在grid的top-left边界的偏移cx, cy的计算方式,如下：
        1.将bbox中x,y反归一后, 除以stride降采样倍数64 = 448 / 7, 记作(grid_x_o, grid_y_o) 得到grid中的cx_1, cy_1, 相对整个pic的pos
        2.对步骤1中除以降采样后的结果取整, 得到其所在的grid的top-left的pos，计算(grid_x, grid_y)
        3.计算偏移量, cx = grid_x_o - grid_x; cy = grid_y_o - grid_y

    Annotations: 根据7x7x30的张量结构来看，可以发现YOLO V1算法的一个缺陷, 当在一个grid中存在多个gt的中心点, 则后一个gt的label会覆盖前者
                ，那么该grid在学习过程中将遗漏部分标签。
    """
    lens = len(bbox_info)
    temp_label = np.zeros((7, 7, 5 * BBOX_NUM + len(CLASSES)))
    for i in range(lens // 5):
        grid_x_o = bbox_info[i * 5 + 1] * GRID_NUM
        grid_y_o = bbox_info[i * 5 + 2] * GRID_NUM
        grid_x, grid_y = int(grid_x_o), int(grid_y_o)
        cx, cy = grid_x_o - grid_x, grid_y_o - grid_y

        # 将第grid_y行，grid_x列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
        temp_label[grid_y, grid_x, 0: 5] = np.array([cx, cy, bbox_info[i * 5 + 3], bbox_info[i * 5 + 4], 1])
        temp_label[grid_y, grid_x, 5: 10] = np.array([cx, cy, bbox_info[i * 5 + 3], bbox_info[i * 5 + 4], 1])
        temp_label[grid_y, grid_x, 10 + int(bbox_info[i * 5])] = 1

    # 扁平化为维度为7x7x30特征map
    temp_label = temp_label.reshape(1, -1)
    return temp_label


def split_train_test(
        arr: t.List[str],
        train_test_dir: str,
        save_img_dir: str,
        labels_dir: str
) -> None:
    """
    分割训练集和测试集
    生成train_num/test_num x (7x7x30)的二维csv格式数据
    其中7x7表示将448x448的图像划分为7x7个grid,
    每个grid预测两个bbox, 每个bbox预测cx, cy, width, height, 置信度, 每个class的概率
    因此最终对每个bbox, 需预测7x7x30的 tensor
    """
    lens = len(arr)
    train_num = int(lens * SPLIT_RATIO)
    test_num = lens - train_num
    train_img = arr[:train_num]
    test_img = arr[train_num:]
    train_csv = np.zeros((train_num, GRID_NUM * GRID_NUM * (5 * BBOX_NUM + len(CLASSES))), dtype=np.float32)
    test_csv = np.zeros((test_num, GRID_NUM * GRID_NUM * (5 * BBOX_NUM + len(CLASSES))), dtype=np.float32)
    # 存储需要train和test的img_path到train.txt中
    with open(os.path.join(train_test_dir, 'train.txt'), 'w') as img_f:
        for idx, img_name in enumerate(train_img):
            img_path = os.path.join(save_img_dir, img_name)
            img_f.write(f'{img_path}\n')
            # 取出需要train的img的label信息转换为7x7x30张量格式
            with open(os.path.join(labels_dir, f'{img_name.split(".")[0]}.txt')) as label_f:
                bbox_info = [float(co) for co in label_f.read().split()]
                cur_label: np.ndarray = compute_actual_pos(bbox_info)
            train_csv[idx, :] = cur_label
    np.savetxt(os.path.join(train_test_dir, 'train.csv'), train_csv)

    with open(os.path.join(train_test_dir, 'test.txt'), 'w') as img_f:
        for idx, img_name in enumerate(test_img):
            img_path = os.path.join(save_img_dir, img_name)
            img_f.write(f'{img_path}\n')
            with open(os.path.join(labels_dir, f'{img_name.split(".")[0]}.txt')) as label_f:
                bbox_info = [float(co) for co in label_f.read().split()]
                cur_label: np.ndarray = compute_actual_pos(bbox_info)
            test_csv[idx, :] = cur_label
    np.savetxt(os.path.join(train_test_dir, 'test.csv'), test_csv)


def shuffle(arr: t.List[str], arr_len: int) -> None:
    """
    基于Fisher-Yates的洗牌算法打乱arr
    时间复杂度为O(N)
    """
    for i in range(arr_len - 1):
        idx = random.randint(i, arr_len - 1)
        arr[i], arr[idx] = arr[idx], arr[i]


def create_csv_txt(annotation_dir: str, train_test_dir: str, img_dir: str) -> None:
    # 1.根据Annotations中的xml提取出所有bbox信息, 以txt格式保存到Labels目录下
    labels_dir = os.path.join(STATIC_DATA_PATH, f'{MODEL_NAME}_Labels')
    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)
        generate_label(annotation_dir, labels_dir)
        print('retrieve info successfully!')
    save_img_dir = os.path.join(os.path.join(STATIC_DATA_PATH, f'{MODEL_NAME}_ProcessImg'))
    bbox_img_dir = os.path.join(os.path.join(STATIC_DATA_PATH, f'{MODEL_NAME}_BboxImg'))
    # 2.对每一图像做padding, 再resize到448 x 448, 将新img保存到ProcessImg, 同时更新Labels目录下所有图像的bbox信息
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)
        os.mkdir(bbox_img_dir)
        img_padding(img_dir, save_img_dir, labels_dir, bbox_img_dir)

    # 3.使用Fisher-Yates洗牌算法打乱整个img_list, 分割测试集和训练集
    img_list = os.listdir(save_img_dir)
    img_len = len(img_list)
    shuffle(img_list, img_len)
    split_train_test(img_list, train_test_dir, save_img_dir, labels_dir)

def back():
    """
    根据构建的train.txt和train.csv, 要学习的label, 反向还原其原始图像的bbox的位置, 用于检测是否正确预处理labels
    :return:
    """
    dir_ = r'C:\Users\24717\Projects\pascal voc2012\VOCdevkit\VOC2012\YOLO_V1_TrainTest\train.txt'
    label_path = r'C:\Users\24717\Projects\pascal voc2012\VOCdevkit\VOC2012\YOLO_V1_TrainTest\train.csv'
    with open(dir_, 'r') as f:
        img_paths = [co for co in f.read().split('\n')]
    labels = np.loadtxt(label_path, dtype=np.float32)
    labels = torch.tensor(labels)
    for idx, path in enumerate(img_paths):
        img = cv2.imread(path)
        label = labels[idx].view(1, GRID_NUM, GRID_NUM, -1)
        for i in range(1):
            for c in range(GRID_NUM):  # x轴
                for r in range(GRID_NUM):  # y轴
                    # 该grid中包含obj
                    if label[i, r, c, 4] == 1:
                        gt_bbox_xy = ((label[i, r, c, 0] + c) * 64 - label[i, r, c, 2] * 224,
                                      (label[i, r, c, 1] + r) * 64 - label[i, r, c, 3] * 224,
                                      (label[i, r, c, 0] + c) * 64 + label[i, r, c, 2] * 224,
                                      (label[i, r, c, 1] + r) * 64 + label[i, r, c, 3] * 224)
                        debug_yolo_v1.draw_box(img, (int(gt_bbox_xy[0]), int(gt_bbox_xy[1])), (int(gt_bbox_xy[2]), int(gt_bbox_xy[3])), (0, 255, 0))
        cv2.imshow('666', img)
        cv2.waitKey(0)



if __name__ == '__main__':
    # 当数组元素比较多的时候，如果输出该数组，那么会出现省略号
    # np.set_printoptions(threshold=np.inf)
    # img_dir = os.path.join(STATIC_DATA_PATH, 'JPEGImages')
    # save_dir = os.path.join(STATIC_DATA_PATH, f'{MODEL_NAME}_TrainTest')
    # anno_dirs = [os.path.join(STATIC_DATA_PATH, 'Annotations')]
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # for anno_dir in anno_dirs:
    #     create_csv_txt(anno_dir, save_dir, img_dir)
    # print('process finish!')
    ttt()
