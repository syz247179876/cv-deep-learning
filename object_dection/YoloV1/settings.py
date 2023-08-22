"""
定义全局配置项
"""

# 数据集目录
DATASET_DIR = r'C:\Users\24717\Projects\pascal voc2012\VOCdevkit\VOC2012\YOLO_V1_TrainTest'
# 定义总类别
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
# 定义每个网格检测的bbox数
BBOX_NUM = 2
# 定义网格数
GRID_NUM = 7
# 静态资源路径
STATIC_DATA_PATH = r'C:\Users\24717\Projects\pascal voc2012\VOCdevkit\VOC2012'
# 网络中图像输入的尺寸, YOLO v1规定为448 x 448
INPUT_IMG_W = 448
INPUT_IMG_H = 448
# 降采样总倍数
STRIDE = 64
# 划分训练集和测试集的比例
SPLIT_RATIO = 0.7
# 模型名
MODEL_NAME = 'YOLO_V1'
# 是否开启调试
DEBUG_OPEN = True
# 绘制框类别颜色
COLOR = [(255, 0, 0), (255, 125, 0), (255, 255, 0), (255, 0, 125), (255, 0, 250),
         (255, 125, 125), (255, 125, 250), (125, 125, 0), (0, 255, 125), (255, 0, 0),
         (0, 0, 255), (125, 0, 255), (0, 125, 255), (0, 255, 255), (125, 125, 255),
         (0, 255, 0), (125, 255, 125), (255, 255, 255), (100, 100, 100), (0, 0, 0), ]  # 用来标识20个类别的bbox颜色，可自行设定
