# Official YOLOv7

## Data Preparation
- 테스트 할 데이터 (영상, GT 레이블)들을 모두 단일 폴더에 이동 하여 저장 e.g. /home/user/DATASET_PATH

## Installation
- Dependency 세팅 

``` shell
conda create -n env python=3.8
conda activate env
pip install -r requirements.txt
```

## Testing
- 테스트 데이터셋의 경로를 --source의 인자로, 저장될 파일의 이름을 --name의 인자로 전달하여 실행
- 정상 실행 된다면 ./f1_res 에 txt 파일로 결과물 저장됨
- 사전 진행한 실험의 결과 또한 ./f1_res/final49.txt에 기록 돼있음

``` shell
python custom_detect.py --classes 0 --source 'DATASET_PATH' --name "OUTPUT_NAME.txt" --img 640 --conf 0.5 --weights yolov7.pt
```
