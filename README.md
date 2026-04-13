# trustworthy-ai-assignment2
DeepXplore on CIFAR-10 ResNet50

## 개요
DeepXplore를 활용하여 CIFAR-10으로 학습된 두 개의 ResNet50 모델에 대해 차분 테스트를 수행하였습니다.

원본 DeepXplore (Keras / TensorFlow 기반)의 핵심 알고리즘 (뉴런 커버리지 최대화 + gradient 기반 차분 테스트)을 PyTorch로 재구현하여 CIFAR-10/ResNet 50에 적용하였습니다.

## 환경 설정
```bash
conda create -n deepxplore310 python=3.10
conda activate deepxplore310
pip install -r requirements.txt
```

## 프로젝트 구조
```
trustworthy-ai-assignment2/
├── deepxplore/          # 원본 DeepXplore 레포지토리 (참고용)
├── models/              # 학습된 ResNet50 모델 저장 (train_models.py 실행 시 생성)
├── results/             # 생성된 disagreement 시각화 결과
├── gen_diff_cifar.py    # CIFAR-10용으로 재구현된 DeepXplore 메인
├── utils_cifar.py       # CIFAR-10용 유틸리티 함수
├── train_models.py      # ResNet50 학습 스크립트
├── test.py              # 데모 실행 파일
└── requirements.txt     # 의존성 목록
```

## DeepXplore 수정 사항
원본 DeepXplore는 Keras / TensorFlow 기반으로 VGG16, VGG19, ResNet50 (ImageNet)을 대상으로 설계되었습니다. 제 프로젝트에서는 다음과 같이 수정하였습니다.

1. **프레임워크 변경** : Keras/TensorFlow -> PyTorch (CUDA 12.5 호환)
2. **모델 변경** : VGG16/VGG19/ResNet50(ImageNet) -> ResNet50 x 2개 (CIFAR-10)
3. **입력 크기 변경** : 224x224 -> 32x32
4. **클래스 수 변경** : 1000개 (ImageNet) -> 10개 (CIFAR-10)
5. **이미지 저장** : 'scipy.misc.imsave' -> 'matplotlib'
6. **Python 호환성** : 'xrange' -> 'range' (Python 3)
7. **이미지 클리핑** : gradient ascent 중 정규화 범위 초과 방지 로직 추가
8. **포화 이미지 필터링** : 픽셀값이 포화된 이미지는 시각화에서 제외하였습니다.

## 실행 방법
```bash
python test.py
```
모델 파일이 없으면 자동으로 학습 후 DeepXplore를 실행합니다.
결과는 'results/' 폴더에 PNG 파일로 저장됩니다.

## 실험 결과
- **Disagreement 유발 입력 수** : 94개 (100개 seed 중)
- **모델1 뉴런 커버리지**: 65.6%
- **모델2 뉴런 커버리지**: 71.7%
- **시각화 저장**: 'results/' 폴더에 7개 PNG 저장