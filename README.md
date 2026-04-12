# trustworthy-ai-assignment2
DeepXplore on CIFAR-10 ResNet50

## 개요
DeepXplore를 활용하여 CIFAR-10으로 학습된 두 개의 ResNet50 모델에 대해 차분 테스트를 수행함

## 환경 설정
'''bash
conda create -n deepxplore python=3.8
conda activate deepxplore
pip install -r requirements.txt
'''

## 프로젝트 구조
├── deepxplore/          # 원본 DeepXplore 레포지토리 (수정 없음)
├── models/              # 학습된 ResNet50 모델 저장
├── results/             # 생성된 disagreement 시각화 결과
├── gen_diff_cifar.py    # CIFAR-10용으로 수정된 DeepXplore 메인
├── utils_cifar.py       # CIFAR-10용 유틸리티 함수
├── train_models.py      # ResNet50 학습 스크립트
├── test.py              # 데모 실행 파일
└── requirements.txt     # 의존성 목록

## DeepXplore 수정 사항
(추후 작성 예정 - 수정 완료 후 채울 것)

## 실행 방법
(추후 작성 예정)