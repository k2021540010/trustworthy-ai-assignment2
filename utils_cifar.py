import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_model(model_path, device):
    model = resnet50(weights=None, num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def init_coverage_tables(model1, model2):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    return model_layer_dict1, model_layer_dict2

def init_dict(model, model_layer_dict):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            output_size = get_output_size(module)
            if output_size:
                for index in range(output_size):
                    model_layer_dict[(name, index)] = False

def get_output_size(module):
    if isinstance(module, nn.Conv2d):
        return module.out_channels
    elif isinstance(module, nn.Linear):
        return module.out_features
    elif isinstance(module, nn.BatchNorm2d):
        return module.num_features
    return None

def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(list(model_layer_dict.keys()))
    return layer_name, index

def neuron_covered(model_layer_dict):
    covered = len([v for v in model_layer_dict.values() if v])
    total = len(model_layer_dict)
    return covered, total, covered / float(total)

def update_coverage(input_data, model, model_layer_dict, threshold=0.5, device='cuda'):
    activation = {}

    def hook_fn(name):
        def hook(module, input, output):
            activation[name] = output.detach()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        model(input_data.to(device))

    for name, output in activation.items():
        if output.dim() == 4:  
            mean_activation = output.mean(dim=[0, 2, 3])
        elif output.dim() == 2:  
            mean_activation = output.mean(dim=0)
        else:
            mean_activation = output.mean(dim=0)

        for idx in range(mean_activation.size(0)):
            if mean_activation[idx].item() > threshold:
                model_layer_dict[(name, idx)] = True

    for hook in hooks:
        hook.remove()

def normalize(x):
    return x / (torch.sqrt(torch.mean(x ** 2)) + 1e-5)

def constraint_light(grads):
    return torch.ones_like(grads) * grads.mean() * 1e4

def constraint_black(grads, rect_shape=(4, 4)):
    start_h = random.randint(0, grads.shape[2] - rect_shape[0])
    start_w = random.randint(0, grads.shape[3] - rect_shape[1])
    new_grads = torch.zeros_like(grads)
    patch = grads[:, :, start_h:start_h+rect_shape[0], start_w:start_w+rect_shape[1]]
    if patch.mean() < 0:
        new_grads[:, :, start_h:start_h+rect_shape[0], start_w:start_w+rect_shape[1]] = -1
    return new_grads

def constraint_occl(grads, start_point=(0, 0), rect_shape=(8, 8)):
    new_grads = torch.zeros_like(grads)
    new_grads[:, :, start_point[0]:start_point[0]+rect_shape[0],
              start_point[1]:start_point[1]+rect_shape[1]] = \
        grads[:, :, start_point[0]:start_point[0]+rect_shape[0],
              start_point[1]:start_point[1]+rect_shape[1]]
    return new_grads