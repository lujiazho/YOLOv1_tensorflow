import os

###################################### 文件和数据集路径 ######################################
# 数据根目录
DATA_PATH = 'data'
# pascal目录
PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')
# cache目录,存放保存为可用格式的voc数据
CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')
# 输出目录
OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')
# 预训练模型
WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')


###################################### 模型参数 ######################################
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
# 是否翻转
FLIPPED = True


###################################### 超参数 ########################################
# 图片大小
IMAGE_SIZE = 448
# n * n 网格的n数
CELL_SIZE = 7
# 一个网格的bounding box数
BOXES_PER_CELL = 2
# 激活函数leaky_relu的alpha值
ALPHA = 0.1


# 损失函数参数
OBJECT_SCALE = 1.0
# 原文中这个是 λnoobj = 0.5
NOOBJECT_SCALE = 0.5
# 原文中这里是 1.0
CLASS_SCALE = 1.0
# 对应原文的λcoord = 5
COORD_SCALE = 5.0


###################################### 训练参数 ######################################
GPU = ''
# 学习率
LEARNING_RATE = 0.001
# 衰减学习率中的参数
# 衰减步数
DECAY_STEPS = 500
# 衰减率
DECAY_RATE = 0.1
# 表示衰变的方式，具体公式是decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
# 当STAIRCASE为True，（global_step/decay_steps）则被转化为整数, 因此衰减是突变的，整个衰变过程成阶梯状
STAIRCASE = True
# batch size
BATCH_SIZE = 45
# 训练迭代次数，一个epoch包含 [labels_num / batch_size] 个iters
MAX_ITER = 1500
# 多少迭代总结一次训练日志信息，这个信息是记录到了tensorboard的文件里
SUMMARY_ITER = 100
# 多少迭代保存一次
SAVE_ITER = 100


###################################### 测试参数 ######################################
# 条件类别概率的阈值
THRESHOLD = 0.2
# NMS用的阈值，IOU超过多少就删掉小的那个
IOU_THRESHOLD = 0.5
# 这里没有包含评估模型效果mAP的IOU阈值
