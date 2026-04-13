import os
import torch

def check_and_train():
    model1_path = 'models/resnet50_cifar10_model1.pth'
    model2_path = 'models/resnet50_cifar10_model2.pth'

    if not os.path.exists(model1_path) or not os.path.exists(model2_path):
        print("=== 모델 파일이 없습니다. 학습을 시작합니다... ===", flush=True)
        from train_models import train_and_save
        os.makedirs('models', exist_ok=True)
        train_and_save(model_id=1, seed=42, lr=0.001, epochs=30)
        train_and_save(model_id=2, seed=123, lr=0.0005, epochs=30)
        print("=== 모델 학습 완료 ===", flush=True)
    else:
        print("=== 모델 파일 확인 완료 ===", flush=True)

def run_test():
    from gen_diff_cifar import run_deepxplore

    print("\n=== DeepXplore 차분 테스팅 시작 ===", flush=True)
    print("설정: transformation=light, seeds=100, grad_iterations=50", flush=True)

    disagreement_count, pct1, pct2 = run_deepxplore(
        transformation='light',
        weight_diff=1.0,
        weight_nc=0.1,
        step=0.01,
        num_seeds=100,
        grad_iterations=50,
        threshold=0.5
    )

    print("\n========== 최종 결과 ==========", flush=True)
    print(f"Disagreement 유발 입력 수: {disagreement_count}", flush=True)
    print(f"모델1 뉴런 커버리지: {pct1:.3f} ({pct1*100:.1f}%)", flush=True)
    print(f"모델2 뉴런 커버리지: {pct2:.3f} ({pct2*100:.1f}%)", flush=True)
    print(f"시각화 결과 저장 위치: results/", flush=True)
    print(f"저장된 이미지 수: {len(os.listdir('results'))}", flush=True)
    print("================================", flush=True)

if __name__ == '__main__':
    check_and_train()
    run_test()