# Data Augmentation OpenCV for YOLO
본 코드는 OpenCV를 이용하여 [YOLO](https://docs.ultralytics.com/)를 위한 학습 데이터 셋을 만들기 위해 제작되었습니다.

## Dependencies
```
opencv Python
    python -m pip install --upgrade pip
    python -m pip install opencv-python
```
```
pyyaml Python
    pip install pyyaml
```

## License
Example Data for [BG-20K](https://www.kaggle.com/datasets/nguyenquocdungk16hl/bg-20o) : MIT

## How to Use
사용자는 data_augmentation.py의 최상단, Hyper Parameters만 수정하면 됩니다.
현재 배경 이미지 폴더로 "BG-20k"를, 증강 데이터 폴더로 GreenLights, RedLights가 예시로 제공됩니다.

이후 코드를 실행하면, YOLOv8 학습에 필요한 datasets.yaml이 생성됩니다. 사용자는 위 Datasets를 이용하여 YOLOv8 학습을 수행하면 됩니다. 각 Hyper Parameter에 대한 설명은 아래와 같습니다.

1. back_ground_images_folder(String): 배경 이미지 폴더의 이름
2. augment_images_foler(List for String): 증강 임지 폴더의 이름 목록
3. test_data_ratio(float): test data 비율
4. labels(Dictionary): key: label, value: name
5. mean_width(Int): 증강 데이터의 평균 폭, original data의 폭, 높이 비율은 유지됩니다.
6. max_num_object(Int): 한 사진에 등장하는 최대 오브젝트 수
7. scale_bias(float): 이미지 스케일링 bias
8. scale_fator(float): 이미지 스케일링 배수
**ImageScale = random.random() * scale_factor + scale_bias**