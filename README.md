# YOLO_tensorflow
2016年[YOLO](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)实现，包含所有源码注释并解决一些原作者编码中的bug

## Demo Show
<img src="https://github.com/leaving-voider/YOLOv1_tensorflow/blob/main/test/detected.jfif" width = "640" height = "423" alt="" align=center />

## Requirements
```Shell
$ pip install -r requirements.txt
```

## Installation
1. Clone yolo_tensorflow repository
```Shell
$ git clone https://github.com/leaving-voider/YOLOv1_tensorflow.git
$ cd YOLOv1_tensorflow
```

2. Download Pascal VOC dataset and YOLO_small.ckpt to correct catalogue automatically
```Shell
# (if needed)
# !chmod +x download_data.sh
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
$ python test.py --weights ./data/pascal_voc/output/2021_02_02_07_43/weights/yolo.ckpt-15 --img ./images/emotion.jpg
```
