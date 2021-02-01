import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer

class Detector(object):
    def __init__(self, model, weight_file):
        self.model = model
        self.weights_file = weight_file

        # 模型参数
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL

        # 条件类别概率阈值
        self.threshold = cfg.THRESHOLD
        # NMS用的阈值，IOU超过多少就删掉小的那个
        self.iou_threshold = cfg.IOU_THRESHOLD
        # 输出分割
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        # 创建会话并初始化参数，其实也不用初始化
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # 从训练好的模型文件restore权重等参数
        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def draw_result(self, img, result):
        # 遍历每个要画的bounding box
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            # cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), 矩形框的RGB颜色, 所画线的宽度)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            cv2.putText(
                img, result[i][0] + ' : %.2f' % result[i][5],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType)

    def detect(self, img):
        # 得到长宽高
        img_h, img_w, _ = img.shape
        # resize成输入所需的大小
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        # 归一化
        inputs = (inputs / 255.0) * 2.0 - 1.0
        # reshape成batch size为1 的shape
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        # 从cv格式的图片进行检测，这里返回第一个图片的检测结果，因为我们每次检测（不论是从camera还是图片），都只有一帧(一张图片)，而没有整个batch
        result = self.detect_from_cvmat(inputs)[0]

        # 遍历这张图片的每个检测目标
        for i in range(len(result)):
            # 全部转换成原始图片的尺寸，以便画在原始图像上
            result[i][1] *= (1.0 * img_w / self.image_size) # x
            result[i][2] *= (1.0 * img_h / self.image_size) # y
            result[i][3] *= (1.0 * img_w / self.image_size) # w
            result[i][4] *= (1.0 * img_h / self.image_size) # h
        return result

    def detect_from_cvmat(self, inputs):
        # 将图片输入模型，得到输出，net_output的shape为batch_size个7 * 7 * 30
        net_output = self.sess.run(self.model.outputs, feed_dict={self.model.images: inputs})
        # 保存每张图片的检测结果
        results = []
        # net_output.shape[0]为batch size大小，即遍历batch中每个被检测的图片
        for i in range(net_output.shape[0]):
            # 进行output解析
            results.append(self.interpret_output(net_output[i]))
        # 返回整个batch的results
        return results

    def interpret_output(self, output):
        probs = np.zeros((self.cell_size, self.cell_size, self.boxes_per_cell, self.num_class))

        # 得到类别概率
        class_probs = np.reshape(output[0:self.boundary1], (self.cell_size, self.cell_size, self.num_class))
        # 置信度
        confidences = np.reshape(output[self.boundary1:self.boundary2], (self.cell_size, self.cell_size, self.boxes_per_cell))
        # bounding box坐标
        boxes = np.reshape(output[self.boundary2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        # 得到偏移，形如[[0,0],[1,1]...[6,6]]，这是x方向的偏移
        offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                [self.boxes_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))

        # 每个x加上各自的偏移量
        boxes[:, :, :, 0] += offset
        # 每个y加上各自的偏移量
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        # x和y都得到基于整个image的坐标（仍处于[0-1]）
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        # 宽高进行平方
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        # x、y、w、h都乘上image的尺寸，变成真实的像素值，便于画在图片上
        boxes *= self.image_size

        # 对每个网格的每个bounding box
        for i in range(self.boxes_per_cell):
            # 遍历每个类别概率
            for j in range(self.num_class):
                # 类别概率×置信度 = 条件类别概率，每个网格的每个box的每个class都对应有该值
                # 同一个网格不同box的class_probs[:, :, j]相同，不同类别同一个box的置信度相同
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], confidences[:, :, i])

        # 大于阈值的类别为True，filter_mat_probs的shape还是为（7，7，2，20）
        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')

        '''
        np.nonzero是用于得到数组array中非零元素的位置（数组索引）的函数，filter_mat_probs有多少维，就会得到几个array，每个array的长度即非0元素的个数
        这里的filter_mat_boxes可能为(array([3, 3, 5]), array([2, 2, 1]), array([0, 1, 0]), array([14, 14, 11]))
        表示（3，2，0，14）等3个bounding box的那个类别是类别概率大于了阈值的
        '''
        filter_mat_boxes = np.nonzero(filter_mat_probs)

        ''' boxes_filtered则提取这3个可能包含物体的bounding box的坐标
        把每个包含非0元素的bounding box先提取出来，boxes_filtered就是二维数组了
        [[162.31715  247.20142  101.67404  277.54727 ]
         [155.00407  242.30052   78.952705 209.73154 ]
         [102.4614   325.6552    87.022316 124.1645  ]]
        '''
        boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        ''' probs_filtered则把这三个大于了阈值的条件类别概率直接取出来
        [0.602121   0.24487814 0.4548721 ]
        '''
        probs_filtered = probs[filter_mat_probs]

        ''' classes_num_filtered 则把这三个概率的index取出来(从0开始记)
        [14 14 11]
        '''
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        ''' argsort()是将元素从小到大排序后，提取对应的索引index，这里得到的argsort就是数值上从大到小的index了
        就是给概率[0.602121   0.24487814 0.4548721 ]排序，如下所示，第0个最大，第2个第二，第1个最小
        [0 2 1]
        '''
        argsort = np.array(np.argsort(probs_filtered))[::-1]

        ''' boxes_filtered[argsort]的操作就是把bounding box的坐标按照概率大小重新排了一遍
        [[162.31715  247.20142  101.67404  277.54727 ]
         [102.4614   325.6552    87.022316 124.1645  ]
         [155.00407  242.30052   78.952705 209.73154 ]]
        '''
        boxes_filtered = boxes_filtered[argsort]

        ''' 把概率也重新排一遍
        [0.602121   0.4548721  0.24487814]
        '''
        probs_filtered = probs_filtered[argsort]
        # 类别的index也重新排一遍：[14 11 14]
        classes_num_filtered = classes_num_filtered[argsort]

        # NMS非极大值抑制，遍历每个bounding box
        for i in range(len(boxes_filtered)):
            # 原始肯定是不会等于0的，不然也过不了阈值那关；不过后面NMS可能会修改这个值，过滤一些重复检测
            if probs_filtered[i] == 0:
                continue
            # 遍历每个比自己概率小的
            for j in range(i + 1, len(boxes_filtered)):
                # 如果大于了IOU阈值
                if self.cal_iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    # 过滤重复检测
                    probs_filtered[j] = 0.0

        # filter_nms为[ True  True False]
        filter_nms = np.array(probs_filtered > 0.0, dtype='bool')
        ''' boxes_filtered
        [[162.31715  247.20142  101.67404  277.54727 ]
         [102.4614   325.6552    87.022316 124.1645  ]]
        '''
        boxes_filtered = boxes_filtered[filter_nms]
        # probs_filtered为[0.602121  0.4548721]
        probs_filtered = probs_filtered[filter_nms]
        # classes_num_filtered：[14 11]
        classes_num_filtered = classes_num_filtered[filter_nms]

        results = []
        # 遍历每个检测出来的bounding box
        for i in range(len(boxes_filtered)):
            # 返回类别，坐标以及概率
            results.append([self.classes[classes_num_filtered[i]],
                 boxes_filtered[i][0], boxes_filtered[i][1],
                 boxes_filtered[i][2], boxes_filtered[i][3],
                 probs_filtered[i]])
        return results

    # 计算IOU
    def cal_iou(self, box1, box2):
        # 最小的右边界 - 最大的左边界，如果大于0才可能有交集
        inner_w = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        # 最小的上边界 - 最大的下边界，大于0才可能有交集
        inner_h = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        # 如果其中一个小于0都没有交集
        inner_area = 0 if inner_w < 0 or inner_h < 0 else inner_w * inner_h
        return inner_area / (box1[2] * box1[3] + box2[2] * box2[3] - inner_area)

    # # 从camera检测
    # def camera_detector(self, cap, wait=10):
    #     detect_timer = Timer()
    #     ret, _ = cap.read()

    #     while ret:
    #         ret, frame = cap.read()
    #         detect_timer.tic()
    #         result = self.detect(frame)
    #         detect_timer.toc()
    #         print('Average detecting time: {:.3f}s'.format(
    #             detect_timer.average_time))

    #         self.draw_result(frame, result)
    #         cv2.imshow('Camera', frame)
    #         cv2.waitKey(wait)

    #         ret, frame = cap.read()

    # 从图像进行检测
    def image_detector(self, img_name, wait=0):
        # 计时
        detect_timer = Timer()
        image = cv2.imread(img_name)

        detect_timer.tic()
        # 得到检测结果
        result = self.detect(image)
        detect_timer.toc()
        # 输出检测时间
        print('Average detecting time: {:.3f}s'.format(
            detect_timer.average_time))
        # 把结果画在图像上
        self.draw_result(image, result)
        # cv2.imshow('Image', image)
        cv2.imwrite('detected.jpg', image)
        cv2.waitKey(wait)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="./data/weights/YOLO_small.ckpt", type=str)
    parser.add_argument('--img', default='./images/person.jpg', type=str)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 测试没有label、反向传播等等
    yolo = YOLONet(False)
    # 初始化检测器
    detector = Detector(yolo, args.weights)

    # detect from camera
    # cap = cv2.VideoCapture(-1)
    # detector.camera_detector(cap)

    # 从img文件检测对象
    detector.image_detector(args.img)
