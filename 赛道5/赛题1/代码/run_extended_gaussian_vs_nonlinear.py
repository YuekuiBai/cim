import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:0')

print("="*70)
print("拓展研究2: 高斯噪声 vs 非线性失真对比实验")
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
    model = model.to(device)
    return model

def gaussian_noise_injection(model, sigma):
    class GaussianNoiseWrapper(nn.Module):
        def __init__(self, base_model, sigma):
            super().__init__()
            self.base_model = base_model
            self.sigma = sigma

        def forward(self, x):
            if self.sigma > 0:
                noise = torch.randn_like(x) * sigma
                x = x + noise
            return self.base_model(x)
    return GaussianNoiseWrapper(model, sigma)

def nonlinear_injection(model, alpha):
    class NonLinearWrapper(nn.Module):
        def __init__(self, base_model, alpha):
            super().__init__()
            self.base_model = base_model
            self.alpha = alpha

        def forward(self, x):
            if self.alpha > 0:
                x = x + self.alpha * (x ** 3 - x)
            return self.base_model(x)
    return NonLinearWrapper(model, alpha)

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
print(f"基准精度 (无噪声): {baseline_acc:.2f}%")

print("\n" + "="*70)
print("实验A: 高斯噪声实验")
print("="*70)

gaussian_sigmas = [0.05, 0.10, 0.15, 0.20]
gaussian_results = {}

for sigma in gaussian_sigmas:
    model = gaussian_noise_injection(baseline_model, sigma)
    acc = evaluate(model, test_loader)
    gaussian_results[sigma] = acc
    print(f"σ={sigma:.2f}: {acc:.2f}% (衰减: {acc - baseline_acc:+.2f}%)")

print("\n" + "="*70)
print("实验B: 非线性失真实验 (基准)")
print("="*70)

nonlinear_alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
nonlinear_results = {}

for alpha in nonlinear_alphas:
    model = nonlinear_injection(baseline_model, alpha)
    acc = evaluate(model, test_loader)
    nonlinear_results[alpha] = acc
    print(f"α={alpha:.1f}: {acc:.2f}% (衰减: {acc - baseline_acc:+.2f}%)")

print("\n" + "="*70)
print("对比分析")
print("="*70)

print("\n高斯噪声实验结果:")
print(f"{'σ值':<10} {'精度':<12} {'相对基准衰减':<15}")
print("-"*40)
for sigma, acc in gaussian_results.items():
    print(f"{sigma:<10.2f} {acc:<12.2f} {acc - baseline_acc:+.2f}%")

print("\n非线性失真实验结果:")
print(f"{'α值':<10} {'精度':<12} {'相对基准衰减':<15}")
print("-"*40)
for alpha, acc in nonlinear_results.items():
    print(f"{alpha:<10.1f} {acc:<12.2f} {acc - baseline_acc:+.2f}%")

print("\n等效性分析:")
print("="*50)
gaussian_losses = {k: baseline_acc - v for k, v in gaussian_results.items()}
nonlinear_losses = {k: baseline_acc - v for k, v in nonlinear_results.items() if k > 0}

print("\n寻找等效衰减条件:")
for g_sigma, g_loss in gaussian_losses.items():
    for n_alpha, n_loss in nonlinear_losses.items():
        if abs(g_loss - n_loss) < 2.0:
            print(f"高斯噪声 σ={g_sigma:.2f} (衰减={g_loss:+.2f}%) ≈ 非线性 α={n_alpha:.1f} (衰减={n_loss:+.2f}%)")

save_dir = '../结果/extended_research'
os.makedirs(save_dir, exist_ok=True)

all_results = {
    'baseline_acc': baseline_acc,
    'gaussian_results': gaussian_results,
    'nonlinear_results': nonlinear_results
}

with open(f'{save_dir}/gaussian_vs_nonlinear.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n结果已保存到: {save_dir}/gaussian_vs_nonlinear.json")
print("\n实验完成!")