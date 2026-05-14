import os
import json
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.intrinsic as nni
import torch.nn.quantized as nnq

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:0')

print("="*70)
print("拓展研究3: 量化误差 + 非线性误差联合分析")
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

class QuantizedWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.quant = torch.quantization.quantize_dynamic(
            self.base_model, {nn.Linear}, dtype=torch.qint8
        )

    def forward(self, x):
        return self.quant(x)

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

print("\n" + "="*70)
print("实验A: FP32 + 非线性失真 (已有数据)")
print("="*70)

fp32_nl_results = {}
nl_alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

for alpha in nl_alphas:
    model = NonLinearWrapper(baseline_model, alpha)
    acc = evaluate(model, test_loader)
    fp32_nl_results[alpha] = acc
    print(f"FP32 + α={alpha:.1f}: {acc:.2f}%")

print("\n" + "="*70)
print("实验B: FP16 + 非线性失真")
print("="*70)

fp16_nl_results = {}

for alpha in nl_alphas:
    model = NonLinearWrapper(baseline_model, alpha)
    model = model.half()
    acc = evaluate(model, test_loader)
    fp16_nl_results[alpha] = acc
    print(f"FP16 + α={alpha:.1f}: {acc:.2f}%")

print("\n" + "="*70)
print("实验C: INT8动态量化 + 非线性失真")
print("="*70)

int8_nl_results = {}

for alpha in nl_alphas:
    model = NonLinearWrapper(baseline_model, alpha)
    model_cpu = model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu, {nn.Linear}, dtype=torch.qint8
    )
    quantized_model = quantized_model.to(device)
    acc = evaluate(quantized_model, test_loader)
    int8_nl_results[alpha] = acc
    print(f"INT8 + α={alpha:.1f}: {acc:.2f}%")

print("\n" + "="*70)
print("结果汇总")
print("="*70)

print("\n| 配置 | α=0.0 | α=0.3 | α=0.5 | 备注 |")
print("|------|-------|-------|-------|------|")
print(f"| FP32 | {fp32_nl_results[0.0]:.2f}% | {fp32_nl_results[0.3]:.2f}% | {fp32_nl_results[0.5]:.2f}% | 基准 |")
print(f"| FP16 | {fp16_nl_results[0.0]:.2f}% | {fp16_nl_results[0.3]:.2f}% | {fp16_nl_results[0.5]:.2f}% | 半精度 |")
print(f"| INT8 | {int8_nl_results[0.0]:.2f}% | {int8_nl_results[0.3]:.2f}% | {int8_nl_results[0.5]:.2f}% | 动态量化 |")

print("\n量化对非线性误差的影响分析:")
for alpha in [0.0, 0.3, 0.5]:
    fp32_loss = fp32_nl_results[0.0] - fp32_nl_results[alpha]
    fp16_loss = fp16_nl_results[0.0] - fp16_nl_results[alpha]
    int8_loss = int8_nl_results[0.0] - int8_nl_results[alpha]
    print(f"α={alpha:.1f}: FP32衰减={fp32_loss:.2f}%, FP16衰减={fp16_loss:.2f}%, INT8衰减={int8_loss:.2f}%")

save_dir = '../结果/extended_research'
os.makedirs(save_dir, exist_ok=True)

all_results = {
    'fp32_nl_results': fp32_nl_results,
    'fp16_nl_results': fp16_nl_results,
    'int8_nl_results': int8_nl_results
}

with open(f'{save_dir}/quantization_vs_nonlinear.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n结果已保存到: {save_dir}/quantization_vs_nonlinear.json")
print("\n实验完成!")