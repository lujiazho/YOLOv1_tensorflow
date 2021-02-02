import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import yolo.config as cfg


class pascal_voc(object):
    def __init__(self, pattern, re_export=False):
        # 根据VOC的目录组织方式，得到image所在path
        self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')
        self.data_path = os.path.join(self.devkil_path, 'VOC2007')
        # 暂存目录
        self.cache_path = cfg.CACHE_PATH
        # 训练参数
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
        # 创建根据类别得到id的字典，id 为0-19
        self.class_to_id = dict(zip(self.classes, range(len(self.classes))))
        # 是否翻转
        self.flipped = cfg.FLIPPED
        # phase为 train就是训练，其他都当作test
        self.pattern = pattern
        # 重新从数据集建立labels
        self.re_export = re_export
        # get函数里提取self.gt_labels时计数作用，提取完全部ground truth labels后，打乱从0继续提取
        self.cursor = 0
        self.epoch = 1 # 记录这次是第几个epoch，一个epoch就是提取全部数据集一次（包括数据增强部分）
        # 最终的labels，不仅包括原始的，可能还包括翻转的
        self.gt_labels = None
        # 准备labels
        self.prepare()

    '''
    该函数在该py文件下没被使用，训练时直接调用，用于根据label获取数据
    '''
    def get(self):
        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros(
            (self.batch_size, self.cell_size, self.cell_size, 25))
        count = 0
        # 一次性提取一个batch
        while count < self.batch_size:
            img_name = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            # 图像根据gt_labels里记录的数据增强信息进行修改
            images[count, :, :, :] = self.image_read(img_name, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            # 表示一个epoch结束了
            if self.cursor >= len(self.gt_labels):
                # 重新打乱
                np.random.shuffle(self.gt_labels)
                # 下一个batch又从0开始提取
                self.cursor = 0
                # epoch 记录+1
                self.epoch += 1
        return images, labels

    def image_read(self, img_name, flipped=False):
        image = cv2.imread(img_name)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # 对图像进行归一化，范围为[-1, 1]
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            # 水平翻转
            image = image[:, ::-1, :]
        return image

    def prepare(self):
        # 得到所有图片的ground truth labels
        gt_labels = self.load_labels()
        # 如果翻转
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for id_ in range(len(gt_labels_cp)):
                gt_labels_cp[id_]['flipped'] = True
                # 每个网格在水平方向翻转（第一个维度控制竖直方向，第二个维度才是水平方向）
                gt_labels_cp[id_]['label'] =\
                    gt_labels_cp[id_]['label'][:, ::-1, :]
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        # 每个包含label 的网格，变换其label的box的x坐标（也就是[i, j, 1]，[i, j, 2]是y坐标）
                        if gt_labels_cp[id_]['label'][i, j, 0] == 1:
                            gt_labels_cp[id_]['label'][i, j, 1] = \
                                self.image_size - 1 -\
                                gt_labels_cp[id_]['label'][i, j, 1]
            # 不会替代原来的label，而是加在后面
            gt_labels += gt_labels_cp
        # 随机打乱
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    # 返回所有图片的labels
    def load_labels(self):
        # cache目录的path，第一次处理ground truth 的labels后会将处理后的文件存储于该目录，下次就能直接使用
        cache_file = os.path.join(self.cache_path, 'pascal_' + self.pattern + '_gt_labels.pkl')

        # 如果不需要reexport，则直接从cache目录下读取ground truth 的label(原始的，没有翻转的label)
        if os.path.isfile(cache_file) and not self.re_export:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)

        # 从注释文件建立labels，创建cache用于存放输出文件
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if self.pattern == 'train':
            # 这里的data_path已经在VOC2007，其下就有ImageSets、annotations等文件夹
            txt_name = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        else:
            # 原始的VOC2007下没有test.txt文件，这里只是为了完整性添加的代码
            txt_name = os.path.join(self.data_path, 'ImageSets', 'Main', 'test.txt')

        # self.image_index则是所有train的image的文件名，也是label的文件名（不包含后缀）
        with open(txt_name, 'r') as f:
            self.image_names = [x.strip() for x in f.readlines()]

        gt_labels = []
        for img_name in self.image_names:
            # 得到该图片的ground truth label
            label, num = self.load_pascal_annotation(img_name)
            # 若物体为0，则直接忽略
            if num == 0:
                continue
            img_name = os.path.join(self.data_path, 'JPEGImages', img_name + '.jpg')
            # 默认不翻转
            gt_labels.append({'imname': img_name,
                              'label': label,
                              'flipped': False})

        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            # 写入原始的ground truth labels 不会包括翻转等label
            pickle.dump(gt_labels, f)
        return gt_labels

    # 读取单张图片的label，从xml文件加载image和bounding boxes的信息
    def load_pascal_annotation(self, index):
        img_name = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        im = cv2.imread(img_name)
        # 计算缩放比例
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]
        # im = cv2.resize(im, [self.image_size, self.image_size])

        # ground truth的label
        label = np.zeros((self.cell_size, self.cell_size, 25))
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        # 输入文件名，并解析xml文件
        tree = ET.parse(filename)
        # 获取该图片的所有object信息
        objs = tree.findall('object')
        ''' 单个object如下
        <object>
            <name>chair</name>
            <pose>Rear</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>263</xmin>
                <ymin>211</ymin>
                <xmax>324</xmax>
                <ymax>339</ymax>
            </bndbox>
        </object>
        '''
        for obj in objs:
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based 并变换到目标尺寸
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            # 得到对应id
            cls_id = self.class_to_id[obj.find('name').text.lower().strip()]
            # 得到center w h的格式
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            # x_id、y_id表示该物体在哪一个格子
            x_id = int(boxes[0] * self.cell_size / self.image_size)
            y_id = int(boxes[1] * self.cell_size / self.image_size)
            # 如果该格子已经被标注过一个物体的label了，则忽略它，这也是YOLOv1的局限
            if label[y_id, x_id, 0] == 1:
                continue
            # confidence标记为1
            label[y_id, x_id, 0] = 1
            label[y_id, x_id, 1:5] = boxes
            # 属于哪个类，哪个类就为1，其他默认为0
            label[y_id, x_id, 5 + cls_id] = 1

        # 实际上这里的label中标记的物体数是 ≤ len(objs) 的
        return label, len(objs)
