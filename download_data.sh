echo "Creating data directory..."
mkdir -p data && cd data
mkdir weights
mkdir pascal_voc

echo "Downloading Pascal VOC 2007 data..."
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

echo "Extracting VOC data..."
tar xf VOCtrainval_06-Nov-2007.tar

mv VOCdevkit pascal_voc/.

FILE_NAME='./data/weights'
FILE_ID='0B5aC8pI-akZUNVFZMmhmcVRpbTA'

echo "Downloading YOLO_small.ckpt..."
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=$FILE_ID" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=$FILE_ID" -o $FILE_NAME

echo "Done."
