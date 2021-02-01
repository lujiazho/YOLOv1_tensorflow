# YOLO_tensorflow
2016年[YOLO](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)实现，包含所有源码注释并解决一些原作者编码中的bug

## Requirements
未完待续

## Installation
1. Clone yolo_tensorflow repository
```Shell
$ git clone https://github.com/leaving-voider/YOLOv1_tensorflow.git
$ cd YOLOv1_tensorflow
```

2. Download Pascal VOC dataset and YOLO_small.ckpt to correct catalogue automatically
```Shell
$ ./download_data.sh
```
where YOLO_small.ckpt file will be pt in `data/weights`

3. Modify configuration in `yolo/config.py` such as batch size

## Run
1. Training
- with GPU
```Shell
$ python train.py --weights YOLO_small.ckpt --gpu 0
```
- without GPU
```Shell
$ python train.py
```

2. Test
```Shell
$ python test.py --weights ./data/weights/YOLO_small.ckpt --img ./images/emotion.jpg
```
