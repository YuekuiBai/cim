import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')

print("="*70)
print("从头训练严谨对比实验 (From Scratch)")
print("="*70)

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

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

train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=False)
test_dataset = CIFAR10(root='./data', train=False, transform=test_transform, download=False)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

class NonLinearWrapper(nn.Module):
    def __init__(self, model, alpha):
        super().__init__()
        self.model = model
        self.alpha = alpha

    def forward(self, x):
        if self.alpha > 0:
            x = x + self.alpha * (x ** 3 - x)
        return self.model(x)

def get_model(alpha):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    if alpha > 0:
        model = NonLinearWrapper(model, alpha)
    return model.to(device)

def train_from_scratch(alpha, epochs=50, lr=0.01):
    model = get_model(alpha)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 10 == 0:
            print(f'    Epoch {epoch+1}/{epochs}: Acc={acc:.2f}%')

    return best_acc

alpha_values = [0.1, 0.2, 0.3]
num_runs = 3
save_dir = './results/task2_training/scratch_rigorous'
os.makedirs(save_dir, exist_ok=True)

scratch_results = {alpha: [] for alpha in alpha_values}

for alpha in alpha_values:
    print(f'\n--- Alpha={alpha} ---')
    for run in range(num_runs):
        print(f'  Run {run+1}/{num_runs}...', end=' ')
        acc = train_from_scratch(alpha, epochs=50, lr=0.01)
        scratch_results[alpha].append(acc)
        print(f'Acc={acc:.2f}%')

        temp_results = {'scratch_results': scratch_results}
        with open(f'{save_dir}/scratch_results_temp.json', 'w') as f:
            json.dump(temp_results, f, indent=2)

print('\n从头训练结果:')
for alpha in alpha_values:
    results = scratch_results[alpha]
    print(f'  Alpha={alpha}: Mean={np.mean(results):.2f}%, Std={np.std(results):.2f}%')

all_results = {'scratch_results': scratch_results}
with open(f'{save_dir}/scratch_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

finetune_results = {
    "0.1": [85.34, 85.11, 85.30],
    "0.2": [85.34, 85.57, 85.30],
    "0.3": [84.96, 84.93, 85.06]
}

print('\n' + '='*70)
print('生成对比图表')
print('='*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

alphas = [0.1, 0.2, 0.3]
ft_means = [np.mean(finetune_results[str(a)]) for a in alphas]
ft_stds = [np.std(finetune_results[str(a)]) for a in alphas]
sc_means = [np.mean(scratch_results[a]) for a in alphas]
sc_stds = [np.std(scratch_results[a]) for a in alphas]

x = np.arange(len(alphas))
width = 0.35

axes[0].bar(x - width/2, ft_means, width, yerr=ft_stds, label='Fine-tune', color='steelblue', alpha=0.8)
axes[0].bar(x + width/2, sc_means, width, yerr=sc_stds, label='From Scratch', color='coral', alpha=0.8)
axes[0].set_xlabel('Alpha')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Fine-tune vs From Scratch (3 runs avg)')
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'α={a}' for a in alphas])
axes[0].legend()
axes[0].set_ylim([75, 90])

for i, alpha in enumerate(alphas):
    axes[0].errorbar(i - width/2, ft_means[i], yerr=ft_stds[i], color='black', capsize=5)
    axes[0].errorbar(i + width/2, sc_means[i], yerr=sc_stds[i], color='black', capsize=5)

improvements = [(ft_means[i] - sc_means[i]) for i in range(len(alphas))]
colors = ['green' if imp > 0 else 'red' for imp in improvements]
bars = axes[1].bar([f'α={a}' for a in alphas], improvements, color=colors, alpha=0.7)
axes[1].set_xlabel('Alpha')
axes[1].set_ylabel('Improvement (%)')
axes[1].set_title('Fine-tune Improvement over From Scratch')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
for bar, imp in zip(bars, improvements):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{imp:.2f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f'{save_dir}/comparison_chart.png', dpi=150, bbox_inches='tight')
print(f'图表已保存: {save_dir}/comparison_chart.png')

print('\n' + '='*70)
print('统计分析')
print('='*70)
from scipy import stats
print(f'{"策略":<15} {"Alpha":<8} {"Mean":<12} {"Std":<12} {"p-value"}')
print('-'*60)
for alpha in alphas:
    ft = finetune_results[str(alpha)]
    sc = scratch_results[alpha]
    t_stat, p_value = stats.ttest_ind(ft, sc)
    print(f'{"微调":<15} {alpha:<8.1f} {np.mean(ft):>8.2f}%  {np.std(ft):>8.2f}%  {p_value:.4f}')
    print(f'{"从头训练":<15} {alpha:<8.1f} {np.mean(sc):>8.2f}%  {np.std(sc):>8.2f}%')
    print()

print(f'\n结果已保存到: {save_dir}/')