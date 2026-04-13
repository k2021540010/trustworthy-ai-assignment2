import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import os

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def get_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def build_model(seed):
    torch.manual_seed(seed)
    model = resnet50(weights=None, num_classes=10)
    return model

def train_and_save(model_id, seed, lr, epochs=30):
    print(f"\n=== 모델 {model_id} 학습 시작 (seed={seed}, lr={lr}) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}", flush = True)

    trainloader, testloader = get_dataloader()
    model = build_model(seed).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(trainloader):.4f} | Test Acc: {acc:.4f}", flush = True)

    save_path = f'models/resnet50_cifar10_model{model_id}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"모델 {model_id} 저장 완료: {save_path}")

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    train_and_save(model_id=1, seed=42,  lr=0.001, epochs=30)
    train_and_save(model_id=2, seed=123, lr=0.0005, epochs=30)
    print("\n=== 두 모델 학습 완료 ===")