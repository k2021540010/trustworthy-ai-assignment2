import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import os
import random

from utils_cifar import (load_model, init_coverage_tables, neuron_to_cover,
                         neuron_covered, update_coverage, normalize,
                         constraint_light, constraint_black, constraint_occl,
                         CLASS_NAMES)

def get_seed_inputs(num_seeds=100):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    indices = random.sample(range(len(testset)), num_seeds)
    seeds = [testset[i] for i in indices]
    return seeds

def deprocess_image(tensor):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img

def run_deepxplore(transformation='light', weight_diff=1.0, weight_nc=0.1,
                   step=0.01, num_seeds=100, grad_iterations=50, threshold=0.5):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}", flush=True)

    model1 = load_model('models/resnet50_cifar10_model1.pth', device)
    model2 = load_model('models/resnet50_cifar10_model2.pth', device)
    print("모델 로드 완료", flush=True)

    model_layer_dict1, model_layer_dict2 = init_coverage_tables(model1, model2)

    os.makedirs('results', exist_ok=True)
    seeds = get_seed_inputs(num_seeds)

    disagreement_count = 0
    disagreement_inputs = []

    for seed_idx, (seed_img, true_label) in enumerate(seeds):
        print(f"[{seed_idx+1}/{num_seeds}] 처리 중...", flush=True)

        gen_img = seed_img.unsqueeze(0).clone().requires_grad_(True)
        gen_img = gen_img.to(device)

        with torch.no_grad():
            pred1 = model1(gen_img)
            pred2 = model2(gen_img)
        label1 = pred1.argmax().item()
        label2 = pred2.argmax().item()

        if label1 != label2:
            print(f"  → 이미 disagreement: 모델1={CLASS_NAMES[label1]}, 모델2={CLASS_NAMES[label2]}", flush=True)
            update_coverage(gen_img, model1, model_layer_dict1, threshold, device)
            update_coverage(gen_img, model2, model_layer_dict2, threshold, device)
            disagreement_count += 1
            disagreement_inputs.append((gen_img.detach().cpu(), label1, label2, true_label))
            continue

        orig_label = label1

        for i in range(grad_iterations):
            gen_img_var = gen_img.detach().requires_grad_(True)

            pred1 = model1(gen_img_var)
            pred2 = model2(gen_img_var)

            loss1 = -weight_diff * pred1[0, orig_label]
            loss2 = weight_diff * pred2[0, orig_label]

            layer_name1, idx1 = neuron_to_cover(model_layer_dict1)
            layer_name2, idx2 = neuron_to_cover(model_layer_dict2)

            total_loss = loss1 + loss2
            total_loss.backward()

            grads = gen_img_var.grad.data
            grads = normalize(grads)

            if transformation == 'light':
                grads = constraint_light(grads)
            elif transformation == 'blackout':
                grads = constraint_black(grads)
            elif transformation == 'occl':
                grads = constraint_occl(grads)

            gen_img = (gen_img.detach() + grads * step).requires_grad_(True)

            with torch.no_grad():
                pred1 = model1(gen_img)
                pred2 = model2(gen_img)
            label1 = pred1.argmax().item()
            label2 = pred2.argmax().item()

            if label1 != label2:
                update_coverage(gen_img, model1, model_layer_dict1, threshold, device)
                update_coverage(gen_img, model2, model_layer_dict2, threshold, device)
                disagreement_count += 1
                disagreement_inputs.append((gen_img.detach().cpu(), label1, label2, true_label))
                print(f"  → disagreement 발견! 모델1={CLASS_NAMES[label1]}, 모델2={CLASS_NAMES[label2]}", flush=True)
                break

    covered1, total1, pct1 = neuron_covered(model_layer_dict1)
    covered2, total2, pct2 = neuron_covered(model_layer_dict2)
    print(f"\n=== 결과 ===", flush=True)
    print(f"Disagreement 유발 입력 수: {disagreement_count}", flush=True)
    print(f"모델1 뉴런 커버리지: {covered1}/{total1} ({pct1:.3f})", flush=True)
    print(f"모델2 뉴런 커버리지: {covered2}/{total2} ({pct2:.3f})", flush=True)

    visualize_disagreements(disagreement_inputs[:10])

    return disagreement_count, pct1, pct2

def visualize_disagreements(disagreement_inputs):
    if not disagreement_inputs:
        print("시각화할 disagreement가 없습니다.", flush=True)
        return

    for i, (img_tensor, label1, label2, true_label) in enumerate(disagreement_inputs):
        img = deprocess_image(img_tensor)
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(f"True: {CLASS_NAMES[true_label]}\n"
                  f"Model1: {CLASS_NAMES[label1]} | Model2: {CLASS_NAMES[label2]}")
        plt.axis('off')
        save_path = f'results/disagreement_{i+1}_m1_{CLASS_NAMES[label1]}_m2_{CLASS_NAMES[label2]}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"저장: {save_path}", flush=True)

if __name__ == '__main__':
    run_deepxplore(transformation='light', weight_diff=1.0, weight_nc=0.1,
                   step=0.01, num_seeds=100, grad_iterations=50, threshold=0.5)