import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:0')

print("="*70)
print("拓展研究1: 网络结构与参数量对非线性误差影响")
print("="*70)

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=False)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

nl_alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class NonLinearWrapper(nn.Module):
    def __init__(self, base_model, alpha):
        super().__init__()
        self.base_model = base_model
        self.alpha = alpha

    def forward(self, x):
        if self.alpha > 0:
            x = x + self.alpha * (x ** 3 - x)
        return self.base_model(x)

def train_model(model, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            acc = 100. * correct / total
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%")

    return model

def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def test_nonlinear_sensitivity(model, model_name):
    results = {}
    print(f"\n--- {model_name} 非线性敏感度测试 ---")

    baseline_acc = evaluate(model, test_loader)
    results['baseline'] = baseline_acc
    print(f"α=0.0 (基准): {baseline_acc:.2f}%")

    for alpha in nl_alphas[1:]:
        nl_model = NonLinearWrapper(model, alpha)
        acc = evaluate(nl_model, test_loader)
        results[alpha] = acc
        print(f"α={alpha:.1f}: {acc:.2f}% (衰减: {acc - baseline_acc:+.2f}%)")

    return results

results = {}

models_config = [
    ('ResNet18', models.resnet18(weights=None)),
    ('ResNet34', models.resnet34(weights=None)),
    ('MobileNetV2', models.mobilenet_v2(weights=None)),
]

for model_name, base_model in models_config:
    print(f"\n{'='*70}")
    print(f"训练 {model_name}...")
    print(f"{'='*70}")

    if 'ResNet' in model_name:
        base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    else:
        base_model.classifier[1] = nn.Linear(base_model.classifier[1].in_features, 10)

    model_path = f'../结果/extended_research/{model_name}_cifar10.pth'

    if os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        base_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        base_model = base_model.to(device)
        base_model = train_model(base_model, epochs=50)
        os.makedirs('../结果/extended_research', exist_ok=True)
        torch.save(base_model.state_dict(), model_path)
        print(f"模型已保存: {model_path}")

    base_model = base_model.to(device)
    params = count_parameters(base_model)
    print(f"{model_name} 参数量: {params/1e6:.2f}M")

    results[model_name] = {
        'params': params,
        'sensitivity': test_nonlinear_sensitivity(base_model, model_name)
    }

print("\n" + "="*70)
print("结果汇总")
print("="*70)

print("\n| 模型 | 参数量 | α=0.0 | α=0.1 | α=0.2 | α=0.3 | α=0.4 | α=0.5 | 平均 | 衰减 |")
print("|------|--------|-------|-------|-------|-------|-------|-------|------|------|")

for model_name, data in results.items():
    sens = data['sensitivity']
    avg = sum(sens.values()) / len(sens)
    decay = sens[0.0] - sens[0.5]
    params = data['params'] / 1e6
    print(f"| {model_name} | {params:.1f}M | {sens[0.0]:.2f}% | {sens[0.1]:.2f}% | {sens[0.2]:.2f}% | {sens[0.3]:.2f}% | {sens[0.4]:.2f}% | {sens[0.5]:.2f}% | {avg:.2f}% | {decay:.2f}% |")

save_dir = '../结果/extended_research'
os.makedirs(save_dir, exist_ok=True)

with open(f'{save_dir}/network_structure_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=float)

print(f"\n结果已保存到: {save_dir}/network_structure_results.json")
print("\n实验完成!")