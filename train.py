import os
import argparse
import datetime
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc import pascal_voc
'''
对于tensorflow.contrib这个库，tensorflow官方对它的描述是：此目录中的任何代码未经官方支持，可能会随时更改或删除。每个目录下都有指定的所有者
slim是一个使构建，训练，评估神经网络变得简单的库。它可以消除原生tensorflow里面很多重复的模板性的代码，让代码更紧凑，更具备可读性
另外slim提供了很多计算机视觉方面的著名模型（VGG, AlexNet等），我们不仅可以直接使用，甚至能以各种方式进行扩展
'''
slim = tf.contrib.slim

class Trainer(object):
    def __init__(self, model, data):
        # 模型和数据
        self.model = model
        self.data = data
        # 默认就是None
        self.weights_file_path = cfg.WEIGHTS_FILE
        # 训练轮数
        self.iters = cfg.MAX_ITER
        self.initial_learning_rate = cfg.LEARNING_RATE
        # 衰减学习率用到的参数
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        # 为True表示衰减方式为突变型
        self.staircase = cfg.STAIRCASE
        # 每多少轮，输出log信息
        self.summary_iter = cfg.SUMMARY_ITER
        # 每多少轮，保存模型文件
        self.save_iter = cfg.SAVE_ITER
        # 输出目录，文件夹名为时间戳
        self.output_dir = os.path.join(
            cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        # 不存在就创建
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # 保存配置信息
        self.save_cfg()

        # tf.global_variables()查看全部变量，这里就是模型的全部变量，在yolo_net.py里那些
        self.variable_to_restore = tf.global_variables()
        # tf.train.Saver 第一个参数用于选择要保存和恢复的变量，可以是列表或字典，变量与变量名一一对应
        # max_to_keep表示最多保存的个数，超过了就会覆盖之前的
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        # 存放checkpoint文件的目录
        self.ckpt_file_path = os.path.join(self.output_dir, 'weights')
        # 不存在就创建
        if not os.path.exists(self.ckpt_file_path):
            os.makedirs(self.ckpt_file_path)
        # 自动将所有summary信息整合到一起
        self.summary_op = tf.summary.merge_all()
        # tf.summary.FileWriter 第一个参数指定输出路径，可以通过add_summary将self.summary_op写入文件，便于tensorboard查看
        # 文件名类似这种：events.out.tfevents.1612084986.1195155b8bd2
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        # tensorflow中用于创建一个全局的步数的方法，初始化为0，这里用于衰减学习率的当前迭代次数的计数，训练的轮数是自己计数
        self.global_step = tf.train.create_global_step()
        # 衰减学习率，其中的self.global_step能够在optimizor反向传播时自动进行计数, STAIRCASE默认是False
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        # 当STAIRCASE为True，（global_step/decay_steps）则被转化为整数, 因此衰减是突变的，整个衰变过程成阶梯状
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)
        # 一旦我们确定了模型、损失函数和优化器，我们就可以调用slim.learning.create_train_op() 和 slim.learning.train()来执行优化操作
        # 这个train算子有两个作用，第一个是计算损失，第二个是计算梯度值
        self.train_op = slim.learning.create_train_op(
            self.model.total_loss, self.optimizer, global_step=self.global_step)

        # tf.GPUOptions()有参数per_process_gpu_memory_fraction，默认为1，意味完全使用所有gpu
        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        # 初始化所有变量
        self.sess.run(tf.global_variables_initializer())

        # 如果有pretrain模型，则可以直接提取其变量来覆盖初始化的变量
        if self.weights_file_path is not None:
            print('Restoring weights from: ' + self.weights_file_path)
            self.saver.restore(self.sess, self.weights_file_path)
        # 当前参数的graph写入到tensorboard中，也就是网络结构图
        self.writer.add_graph(self.sess.graph)

    def train(self):
        # 计时
        train_timer = Timer()
        load_timer = Timer()

        for step in range(1, self.iters + 1):
            load_timer.tic()
            # 根据labels获取images
            images, labels = self.data.get()
            load_timer.toc()
            # 输入数据
            feed_dict = {self.model.images: images,
                         self.model.labels: labels}

            if step % self.summary_iter == 0:
                train_timer.tic()
                summary_str, loss, _ = self.sess.run(
                    [self.summary_op, self.model.total_loss, self.train_op],
                    # feed_dict就是用于给placeholder创建的images和labels赋值，也会自动将数据类型转化为tf的对应类型
                    feed_dict=feed_dict)
                train_timer.toc()

                # 在命令行输出部分日志信息
                print('''{} Epoch: {}, Step: {}, Learning rate: {},'''\
                ''' Loss: {:5.3f}\nSpeed: {:.3f}s/iter,'''\
                '''' Load: {:.3f}s/iter, Remain: {}'''.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),     # 当前时间
                    self.data.epoch,                                        # 当前epoch
                    int(step),                                              # 当前iter
                    round(self.learning_rate.eval(session=self.sess), 6),   # learning_rate的值
                    loss,                                                   # 当前总loss
                    train_timer.average_time,                               # 训练一个iter平均时间
                    load_timer.average_time,                                # load一个batch平均时间
                    train_timer.remain(step, self.iters))                # 预估剩余时间
                )

                # 写入tensorboard的文件
                self.writer.add_summary(summary_str, step)

            else:
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_timer.toc()

            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                # 这个self.global_step和step的值是一样的，只是数据类型不同，一个int，一个tf.variable的int64_ref
                # self.global_step是在self.train_op被run之后+1的
                self.saver.save(self.sess, self.ckpt_file_path+'/yolo.ckpt', global_step=self.global_step)

    def save_cfg(self):
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            # 将其字典属性提取出来，其中就包含了每个属性，以及其他各种信息
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                # 其中大写开头的就是我们自己定的配置信息
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)

# python train.py --weights YOLO_small.ckpt --gpu 0
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default='', type=str)
    opt = parser.parse_args()

    cfg.GPU = opt.gpu
    # 设置环境变量中的指定GPU设备，使之对tf可见，才能调用
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    # 模型class实例化
    yolo = YOLONet()
    # 数据集class实例化
    pascal = pascal_voc('train')

    trainer = Trainer(yolo, pascal)

    print('Start training ...')
    trainer.train()
    print('Done training.')
