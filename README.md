# Official YOLOv7



## Installation

``` shell
conda create -n env python=3.8
conda activate env
pip install -r requirements.txt
```

## Testing

``` shell
python custom_detect.py --classes 0 --source 'DATASET_PATH' --name "OUTPUT_NAME.txt" --img 640 --conf 0.5 --weights yolov7.pt
```
