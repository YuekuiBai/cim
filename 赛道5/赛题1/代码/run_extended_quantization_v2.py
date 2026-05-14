import os
import json
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:0')

print("="*70)
print("拓展研究3: 量化误差 + 非线性误差联合分析 (简化版)")
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

def load_baseline_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    checkpoint = torch.load('../结果/baseline_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint)
    return model.to(device)

class NonLinearWrapper(nn.Module):
    def __init__(self, base_model, alpha):
        super().__init__()
        self.base_model = base_model
        self.alpha = alpha

    def forward(self, x):
        if self.alpha > 0:
            x = x + self.alpha * (x ** 3 - x)
        return self.base_model(x)

class QuantNoiseWrapper(nn.Module):
    def __init__(self, base_model, n_bits=8):
        super().__init__()
        self.base_model = base_model
        self.n_bits = n_bits
        self.scale = 2 ** n_bits

    def quantize(self, x):
        return torch.round(x * self.scale) / self.scale

    def forward(self, x):
        x_quant = self.quantize(x)
        return self.base_model(x_quant)

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

print("\n加载基准模型...")
baseline_model = load_baseline_model()
baseline_acc = evaluate(baseline_model, test_loader)
print(f"FP32基准精度: {baseline_acc:.2f}%")

nl_alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
n_bits_list = [8, 4, 2]

print("\n" + "="*70)
print("实验A: 不同量化位数对非线性误差的影响")
print("="*70)

results = {'baseline': {}, 'quant_8bit': {}, 'quant_4bit': {}, 'quant_2bit': {}}

print("\n--- 基准 (无量化) ---")
for alpha in nl_alphas:
    model = NonLinearWrapper(baseline_model, alpha)
    acc = evaluate(model, test_loader)
    results['baseline'][alpha] = acc
    print(f"α={alpha:.1f}: {acc:.2f}%")

print("\n--- INT8量化 (8-bit) ---")
for alpha in nl_alphas:
    model = NonLinearWrapper(baseline_model, alpha)
    model = QuantNoiseWrapper(model, n_bits=8)
    acc = evaluate(model, test_loader)
    results['quant_8bit'][alpha] = acc
    print(f"α={alpha:.1f}: {acc:.2f}%")

print("\n--- 4-bit量化 ---")
for alpha in nl_alphas:
    model = NonLinearWrapper(baseline_model, alpha)
    model = QuantNoiseWrapper(model, n_bits=4)
    acc = evaluate(model, test_loader)
    results['quant_4bit'][alpha] = acc
    print(f"α={alpha:.1f}: {acc:.2f}%")

print("\n--- 2-bit量化 ---")
for alpha in nl_alphas:
    model = NonLinearWrapper(baseline_model, alpha)
    model = QuantNoiseWrapper(model, n_bits=2)
    acc = evaluate(model, test_loader)
    results['quant_2bit'][alpha] = acc
    print(f"α={alpha:.1f}: {acc:.2f}%")

print("\n" + "="*70)
print("结果汇总表")
print("="*70)

print("\n| 量化方式 | α=0.0 | α=0.1 | α=0.2 | α=0.3 | α=0.4 | α=0.5 |")
print("|----------|-------|-------|-------|-------|-------|-------|")
print(f"| FP32 | {results['baseline'][0.0]:.2f}% | {results['baseline'][0.1]:.2f}% | {results['baseline'][0.2]:.2f}% | {results['baseline'][0.3]:.2f}% | {results['baseline'][0.4]:.2f}% | {results['baseline'][0.5]:.2f}% |")
print(f"| INT8 | {results['quant_8bit'][0.0]:.2f}% | {results['quant_8bit'][0.1]:.2f}% | {results['quant_8bit'][0.2]:.2f}% | {results['quant_8bit'][0.3]:.2f}% | {results['quant_8bit'][0.4]:.2f}% | {results['quant_8bit'][0.5]:.2f}% |")
print(f"| 4-bit | {results['quant_4bit'][0.0]:.2f}% | {results['quant_4bit'][0.1]:.2f}% | {results['quant_4bit'][0.2]:.2f}% | {results['quant_4bit'][0.3]:.2f}% | {results['quant_4bit'][0.4]:.2f}% | {results['quant_4bit'][0.5]:.2f}% |")
print(f"| 2-bit | {results['quant_2bit'][0.0]:.2f}% | {results['quant_2bit'][0.1]:.2f}% | {results['quant_2bit'][0.2]:.2f}% | {results['quant_2bit'][0.3]:.2f}% | {results['quant_2bit'][0.4]:.2f}% | {results['quant_2bit'][0.5]:.2f}% |")

print("\n" + "="*70)
print("量化对非线性误差敏感度的影响分析")
print("="*70)

print("\n| 量化方式 | 基准衰减(α=0→0.5) | 敏感度变化 |")
print("|----------|------------------|------------|")
baseline_decay = results['baseline'][0.0] - results['baseline'][0.5]
print(f"| FP32 | {baseline_decay:.2f}% | 基准 |")
quant8_decay = results['quant_8bit'][0.0] - results['quant_8bit'][0.5]
print(f"| INT8 | {quant8_decay:.2f}% | {quant8_decay - baseline_decay:+.2f}% |")
quant4_decay = results['quant_4bit'][0.0] - results['quant_4bit'][0.5]
print(f"| 4-bit | {quant4_decay:.2f}% | {quant4_decay - baseline_decay:+.2f}% |")
quant2_decay = results['quant_2bit'][0.0] - results['quant_2bit'][0.5]
print(f"| 2-bit | {quant2_decay:.2f}% | {quant2_decay - baseline_decay:+.2f}% |")

save_dir = '../结果/extended_research'
os.makedirs(save_dir, exist_ok=True)

with open(f'{save_dir}/quantization_vs_nonlinear.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n结果已保存到: {save_dir}/quantization_vs_nonlinear.json")
print("\n实验完成!")