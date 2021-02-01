import numpy as np
import tensorflow as tf
import yolo.config as cfg

slim = tf.contrib.slim

class YOLONet(object):
    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        # 网格在一个边上的个数，即论文中所说的 S = 7
        self.cell_size = cfg.CELL_SIZE
        # 一个网格的bounding box数，即 B = 2
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        # 这个人工计算即可，论文给的是 7 * 7 * 30
        self.output_size = (self.cell_size * self.cell_size) * (self.num_class + self.boxes_per_cell * 5)
        # 一个网格的size
        self.scale = 1.0 * self.image_size / self.cell_size
        # 输出的分割，这个表示，20个类在前
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        # 而中间是两个box的置信度，最后是两个box的坐标
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        # 损失函数参数
        self.object_lambda = cfg.OBJECT_SCALE
        self.noobject_lambda = cfg.NOOBJECT_SCALE
        self.class_lambda = cfg.CLASS_SCALE
        self.coord_lambda = cfg.COORD_SCALE

        # 训练参数
        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        # 激活函数leaky_relu的alpha值
        self.alpha = cfg.ALPHA

        # np.array 得到14个一维数组，reshape后得到分成两拨的一维数组，即shape =（2，7，7）
        # 再transpose一下，按后面的（1，2，0）参数进行转换，即第0维到最后，1和2向前移，所以得到shape =（7，7，2）
        # 其形如[[0,0],[1,1]...[6,6]]，对应图中每个网格的x值的偏移量，偏移量和x相加组成中心点坐标，这是对predicts用的
        # 让predicts变成和ground truth一样的scale，才能比较计算loss
        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        # 第一个None是batch size，不用提前指定
        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')
        self.outputs = self.build_network(
            self.images, num_outputs=self.output_size, alpha=self.alpha,
            is_training=is_training)

        if is_training:
            # 这个label是对应于每张图片的每个bounding box的，得和output保持一致的shape
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.cell_size, self.cell_size, 5 + self.num_class])
            # 最后来加loss计算层
            self.loss_layer(self.outputs, self.labels)
            # 提前在loss计算层用tf.losses.add_loss把几大块loss张量加起来了，这里就直接返回总的loss，当然也是个张量
            self.total_loss = tf.losses.get_total_loss()
            # tf.summary.scalar 用来显示标量信息，其格式为 tf.summary.scalar(tags, values, collections=None, name=None)
            tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self, images, num_outputs, alpha, keep_prob=0.5, is_training=True, scope='yolo'):
        # 传进来的参数keep_prob就是dropout，准确叫做1 - dropout_rate，按字面看就是保留的概率
        with tf.variable_scope(scope):
            # 这个函数的作用是给list_ops_or_scope中的内容设置默认值，在这里就是给函数slim.conv2d和slim.fully_connected设置默认值
            # 这两个函数在tensorflow中被定义的时候就已经添加了@add_arg_scope，这里就能直接使用slim.arg_scope进行设置默认值了
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                activation_fn=self.leaky_relu(alpha),
                weights_regularizer=slim.l2_regularizer(0.0005), # 正则化防止过拟合，系数为0.0005，很小的惩罚
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
            ):
                # pad第二个变量表示填充方式，第一个[0,0]表示在batch维度上不填充，如果填充就成了增加batch了
                # 第一个[3,3]表示在图像y轴方向，即上下填充3排；第二个[3,3]表示在x轴方向左右填充3排
                # 第二个[0,0]表示在图像通道的维度不填充，不然就增加图像通道了
                yolonet = tf.pad(
                    images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
                    name='pad_1')
                # 卷积公式：W = (W + 2*padding - kernel_size)/stride + 1
                # 第二个变量是卷积数即num_outputs，第三个是kernel_size，第四个是stride
                # padding默认SAME，自动帮助pad以至于输出的image size不变（前提是stride为1，如果不为，则还是变小:换算公式n_out = [n_in/stride] 向上取整）
                # padding VALID 则公式为（n_out = [(n_in - kernel_size + 1) / stride]），当stride为1时，即不会自动加padding，padding为0
                # 最后的scope是可选项，用于共享参数，比如让别人可以使用自己scope的参数
                yolonet = slim.conv2d(
                    yolonet, 64, 7, 2, padding='VALID', scope='conv_2')
                yolonet = slim.max_pool2d(yolonet, 2, padding='SAME', scope='pool_3') # 这个2表示size，stride默认就是2
                # 到这里 112 * 112 * 64
                yolonet = slim.conv2d(yolonet, 192, 3, scope='conv_4') # stride 默认为 1，这里padding默认SAME，则应该会周围叠加一层
                yolonet = slim.max_pool2d(yolonet, 2, padding='SAME', scope='pool_5')
                # 到这里就是56 * 56 * 192
                yolonet = slim.conv2d(yolonet, 128, 1, scope='conv_6')
                yolonet = slim.conv2d(yolonet, 256, 3, scope='conv_7')
                yolonet = slim.conv2d(yolonet, 256, 1, scope='conv_8')
                yolonet = slim.conv2d(yolonet, 512, 3, scope='conv_9')
                # 这里是56 * 56 * 512
                yolonet = slim.max_pool2d(yolonet, 2, padding='SAME', scope='pool_10')
                # 到这 28 * 28 * 512
                # 接下来四个连环，都不改变size
                yolonet = slim.conv2d(yolonet, 256, 1, scope='conv_11')
                yolonet = slim.conv2d(yolonet, 512, 3, scope='conv_12')
                yolonet = slim.conv2d(yolonet, 256, 1, scope='conv_13')
                yolonet = slim.conv2d(yolonet, 512, 3, scope='conv_14')
                yolonet = slim.conv2d(yolonet, 256, 1, scope='conv_15')
                yolonet = slim.conv2d(yolonet, 512, 3, scope='conv_16')
                yolonet = slim.conv2d(yolonet, 256, 1, scope='conv_17')
                yolonet = slim.conv2d(yolonet, 512, 3, scope='conv_18')
                # 这儿还是 28 * 28 * 512
                yolonet = slim.conv2d(yolonet, 512, 1, scope='conv_19')
                yolonet = slim.conv2d(yolonet, 1024, 3, scope='conv_20')
                # 这里 28 * 28 * 1024
                yolonet = slim.max_pool2d(yolonet, 2, padding='SAME', scope='pool_21')
                # 这儿 14 * 14 * 1024
                yolonet = slim.conv2d(yolonet, 512, 1, scope='conv_22')
                yolonet = slim.conv2d(yolonet, 1024, 3, scope='conv_23')
                yolonet = slim.conv2d(yolonet, 512, 1, scope='conv_24')
                yolonet = slim.conv2d(yolonet, 1024, 3, scope='conv_25')
                yolonet = slim.conv2d(yolonet, 1024, 3, scope='conv_26')
                # 到这儿还是 14 * 14 * 1024
                # 要人工包一层，其实从size的角度来看，没必要（这些只能去官方证实看是不是这么用的）
                yolonet = tf.pad(
                    yolonet, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    name='pad_27')
                # 16 * 16 * 1024
                yolonet = slim.conv2d(
                    yolonet, 1024, 3, 2, padding='VALID', scope='conv_28')
                # 7 * 7 * 1024
                yolonet = slim.conv2d(yolonet, 1024, 3, scope='conv_29')
                yolonet = slim.conv2d(yolonet, 1024, 3, scope='conv_30')
                # 7 * 7 * 1024
                yolonet = tf.transpose(yolonet, [0, 3, 1, 2], name='trans_31')
                # 到这儿，batch size还在前面，但通道所在维度到了第二个位置，这里的shape就是（batch，1024，7，7）
                # flatten：将把维度压缩成一个具有形状[batch_size, k]的平坦张量
                yolonet = slim.flatten(yolonet, scope='flat_32')
                # 在这儿，shape = (batch, 1024*7*7)，每个平化后的7*7图像一个接一个排列，一共1024个
                yolonet = slim.fully_connected(yolonet, 512, scope='fc_33')
                yolonet = slim.fully_connected(yolonet, 4096, scope='fc_34')
                # 这里就是（batch, 4096）
                yolonet = slim.dropout(
                    yolonet, keep_prob=keep_prob, is_training=is_training,
                    scope='dropout_35')
                # 这里的最后一层没有给激活函数，其实原文说的是线性激活函数
                yolonet = slim.fully_connected(
                    yolonet, num_outputs, activation_fn=None, scope='fc_36')
                # 这里就是变成了（batch，7*7*30）
        return yolonet

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        ''' 计算IOU
        参数:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        返回:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        '''
        with tf.variable_scope(scope):
            # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)，一个左上角，一个右下角
            boxes1_ = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)

            boxes2_ = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # calculate the left up point & right down point of the inner square
            lu = tf.maximum(boxes1_[..., :2], boxes2_[..., :2])
            rd = tf.minimum(boxes1_[..., 2:], boxes2_[..., 2:])

            # intersection
            # 这里的tf.maximum很高级，如果当真没有交际，不会返回一个0.0，而是以rd - lu相同的形式（点）进行返回，即（0.0，0.0）
            # 准确来说，应该是<tf.Tensor: shape=(2,), dtype=float32, numpy=array([0., 0.], dtype=float32)>
            intersection = tf.maximum(0.0, rd - lu)
            # 这样在这一步就不会出现报错了，如果只有一维，inter[0]和inter[...,0]就是同样的返回值
            inter_square = intersection[..., 0] * intersection[..., 1]

            # calculate the boxs1 square and boxs2 square 这个地方原作者写错了，如注释所示
            # square1 = boxes1[..., 2] * boxes1[..., 3]
            # square2 = boxes2[..., 2] * boxes2[..., 3]
            square1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
            square2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)
        ''' tf.clip_by_value将一个张量中的数值限制在一个范围之内
        当inter_square / union_square小于0.0时，输出0.0；
        当inter_square / union_square大于0.0小于1.0时，输出原值；
        当inter_square / union_square大于1.0时，输出1.0；
        '''
        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        # predicts 的 shape 是（batch，7*7*30）
        # self.boundary1 = 7*7*20
        with tf.variable_scope(scope):
            # predicts的归类
            # 前 7*7*20 全是集中存放每个网格的类别概率
            predict_classes = tf.reshape(
                predicts[:, :self.boundary1],
                [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            # 接着是 7*7*2 个置信度，每个网格的每个bounding box都有一个
            predict_confidence = tf.reshape(
                predicts[:, self.boundary1:self.boundary2],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            # 最后是 7*7*8 的bounding box的坐标，得到shape = （batch_size, 7, 7, 2, 4）
            predict_boxes = tf.reshape(
                predicts[:, self.boundary2:],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            # 这个三个点...比:范围更广，除了最后一个维度，其代表前面全部维度
            # label 的shape 为[None, self.cell_size, self.cell_size, 5 + self.num_class]
            # 因此这个labels[..., 0]（这里等价于labels[:,:,:,0]）就代表了每个batch每个网格里的第一个，即置信度，当然，不是0就是1
            response = tf.reshape(
                labels[..., 0],
                [self.batch_size, self.cell_size, self.cell_size, 1])
            # 这代表box的x、y、w、h，reshape成5个维度是为了和predicts里保持一致
            boxes = tf.reshape(
                labels[..., 1:5],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            # tf.tile用来对张量(Tensor)进行复制扩展，只改变size，但最终维度不变
            # 这里是对boxes进行复制扩展，因为box有5个维度，因此第二个变量得是1*5的一维向量，分别表示哪个维度扩展多少倍
            # 这里即表示只对第4个维度进行扩展，扩展成原来的两倍，新增的一倍的值和被复制的是一样的，即两个bounding box的x、y、w、h是一样的
            # 这样一来ground truth和predicts的boxes的shape就完全一样了
            # 最后还除去self.image_size，因为ground truth的size没有标准化成[0-1]范围，这里就把x、y、w和h都标准化了
            boxes = tf.tile(
                boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            # 提取classes的概率，当然只有一个class是1，其他都是0（这是ground truth）
            classes = labels[..., 5:]

            # 计算offset，tf.constant就是转化类型；offset 的shape是（7，7，2），这里就变成了（1，7，7，2），最前面加了一个维度而已
            offset = tf.reshape(tf.constant(self.offset, dtype=tf.float32), [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            # 在第一个维度扩展batchsize倍，变成（batch_size，7，7，2），每个batch的offset的值都是一模一样的
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            # 这里第一个维度还是batch size，只是把第3和第2个维度换了
            # 本来是形如[[0,0],[1,1]...[6,6]]再接6个[[0,0],[1,1]...[6,6]]，现在是[[0,0],[0,0]...[0,0]]接[[1,1],[1,1]...[1,1]]...
            offset_tran = tf.transpose(offset, (0, 2, 1, 3))

            '''
            tf.stack实现拼接，相比于tf.concat会在原来的基础上增加一个维度，axis=-1表示在最后一个维度上拼接，即会把每一对x、y、w、h拼在一起
            最后predict_boxes_tran的维度应该是（batch_size, 7, 7, 2, 4）
            predict_boxes[..., 0]得到的维度是（batch_size, 7, 7, 2）,每个数都是代表每个bounding box的x，这里offset的shape也是(batch_size，7，7，2)
            所以offset第一排的[0,0]对应第一排第一个网格的两个bounding box的x，第一排的[1,1]对应第一排第二个网格两个bounding box的x
            offset第二排的[0,0]对应第二排第一个网格两个bounding box的x，以此类推；所以，offset只是负责每个bounding box的x值的offset
            类比offset_tran也是一样，这个负责每个bounding box的y值，所以越往下，offset值越大，因此第一排是[[0,0],[0,0]...[0,0]]
            '''
            predict_boxes_tran = tf.stack(
                # 把[0-1]的范围从image尺度转化为网格的尺度
                [(predict_boxes[..., 0] + offset) / self.cell_size,
                 (predict_boxes[..., 1] + offset_tran) / self.cell_size,
                 # 把[0-1]的根号w和根号h变成正常的[0-1]范围
                 tf.square(predict_boxes[..., 2]),
                 tf.square(predict_boxes[..., 3])], axis=-1)
            # 最后predict_boxes_tran得到的x和y就是物体中心在整个image的比例[0-1]，而w和h也是在image中的比列，范围也是[0-1]
            # 这里之所以要tf.square平方，是因为原文说了：我们对物体的w和h的平方根进行predict，而不是直接预测宽高，这里要平方一下才是真实宽高
            # 在计算loss确实是用的qurt，所以这次转换不是为了计算loss，而是计算IOU，需要转化为以image的比例且不能开根
            # 当然这里平方后的范围也是在[0-1]，不会是真实的像素值，那样计算值太大了，而且对不同size的image也没法很好处理

            # 得到IOU
            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # IOU 的shape [BATCH_SIZE, 7, 7, 2]
            # tf.reduce_max计算一个张量的各个维度上元素的最大值，这里是计算第3个维度，即两个bounding box谁的IOU大，就留下谁
            # object_mask的shape应该是（BATCH_SIZE, 7, 7, 1）
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            # tf.cast实现张量数据类型转换，这里是将True换成1.0，False换成0.0
            # iou_predict_truth >= object_mask 其实是本质是拿一个一维数组和一个数比较（在tf数据类型下），返回值也是一个数组，大的就是True，小则False
            # 当然这里的（iou_predict_truth >= object_mask）返回的shape是 [BATCH_SIZE, 7, 7, 2]
            # 这个response是ground truth的confidence = Pr(object) * IOU，可见只有包含物体的网格的confidence才是1，其他都是0
            # 最后这个object_mask的shape是[BATCH_SIZE, 7, 7, 2]，即所有的bounding box，没物体的就是0.0，有物体且IOU最大的那个就是1.0
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            # tf.ones_like创建一个和输入tensor维度一样且元素都为1的张量
            # 这个noobject_mask其实就是1 - object_mask，里面的值都取了一个反（1->0, 0->1）
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask

            # ground truth的boxes的x和y已经归一化到[0-1]之间，但是按image的比例，现转化成按单个网格的比例
            # 而w和h也sqrt一下，因为loss计算是用的sqrt
            boxes_tran = tf.stack(
                [boxes[..., 0] * self.cell_size - offset,
                 boxes[..., 1] * self.cell_size - offset_tran,
                 tf.sqrt(boxes[..., 2]),
                 tf.sqrt(boxes[..., 3])], axis=-1)

            # class_loss
            class_delta = response * (predict_classes - classes)
            # 使用tf.reduce_mean对batch中每个图片的loss和取平均，这样才保证不会因为batch size大，loss也跟着变大batch倍
            class_loss = tf.reduce_mean(
                # class_delta有4维，axis=[1, 2, 3]表示把后面的维度的数都加起来，即每个网格的每个类的prob，剩第0维即batch，最后shape为(batch,)
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                name='class_loss') * self.class_lambda # class_scale在原文为1

            # object_loss
            # predict_scales是预测置信度，这里的ground truth之所以不是1而是IOU，是因为原文说了：当有物体，我们希望的是置信度恰好等于IOU的值
            object_delta = object_mask * (predict_confidence - iou_predict_truth)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_lambda # object_scale在原文为1

            # noobject_loss
            # 这里似乎没有ground truth的置信度，其实是有的，对没物体的loss，ground truth的置信度本就是0
            noobject_delta = noobject_mask * predict_confidence
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_lambda # noobject_scale原文为0.5

            # coord_loss
            # tf.expand_dims 给object_mask增加了第5维，相当于把每个数都单独加了一个[]框起来，即增加了1维；因为bounding box都是5维的
            # 不过经实验发现，似乎不增加这一维，最后的结果也是一样的
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_lambda # 原文这个coord_scale是5

            # loss作为损耗张量加入到tf.losses，便于总和计算
            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            tf.summary.histogram('iou', iou_predict_truth)

    def leaky_relu(self, alpha):
        def op(inputs):
            return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
        return op
