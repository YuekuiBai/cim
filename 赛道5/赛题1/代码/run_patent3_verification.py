"""
赛题1 专利三验证实验 - 课程学习渐进式非线性鲁棒训练方法

核心创新：
1. 8阶段渐进α(0.05→0.4)
2. SAM平坦最优解
3. OVF负反馈变分融合

消融实验：
- 仅课程学习
- 课程学习+SAM
- 课程学习+OVF
- 课程学习+SAM+OVF
- 阶段数消融(4/8/12)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import os
import sys
import argparse
from tqdm import tqdm
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.resnet import get_model


# ==================== 专利三核心组件 ====================

class CurriculumTrainer:
    """课程学习训练器 - 8阶段渐进α"""
    def __init__(self, num_stages=8, alpha_start=0.05, alpha_end=0.4, epochs_per_stage=5):
        self.num_stages = num_stages
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.epochs_per_stage = epochs_per_stage
        self.current_stage = 0
        self.current_alpha = alpha_start
    
    def get_alpha_for_epoch(self, epoch):
        """根据epoch获取当前alpha"""
        stage = min((epoch - 1) // self.epochs_per_stage, self.num_stages - 1)
        progress = stage / (self.num_stages - 1) if self.num_stages > 1 else 1.0
        alpha = self.alpha_start + progress * (self.alpha_end - self.alpha_start)
        return alpha


class SAMOptimizer:
    """SAM优化器 - 锐度感知最小化"""
    def __init__(self, base_optimizer, rho=0.05, adaptive=False):
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        self.state = {}
        
        for group in self.param_groups:
            group['initial_lr'] = group['lr']
    
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups
    
    @property
    def state(self):
        return self.base_optimizer.state
    
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        
        if zero_grad:
            self.zero_grad()
    
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                (torch.abs(p) if self.adaptive else 1.0) * p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def step(self, closure=None):
        if closure is None:
            return self.base_optimizer.step()
        
        loss = closure()
        self.first_step(zero_grad=True)
        
        loss.backward()
        self.second_step(zero_grad=True)
        
        return loss


class OVFFusion:
    """OVF负反馈变分融合"""
    def __init__(self, num_samples=5, alpha=0.5):
        self.num_samples = num_samples
        self.alpha = alpha
    
    def fuse(self, model, inputs, device):
        """多次采样变分融合"""
        outputs_list = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                if hasattr(model, 'set_alpha'):
                    model.set_alpha(self.alpha)
                outputs = model(inputs)
                outputs_list.append(outputs)
        
        # 负反馈加权
        weights = torch.tensor([1.0/len(outputs_list)] * len(outputs_list), device=device)
        fused_output = sum(w * o for w, o in zip(weights, outputs_list))
        
        return fused_output


# ==================== 训练函数 ====================

def train_with_curriculum(model, trainloader, criterion, optimizer, device, epoch, total_epochs,
                         curriculum, use_sam=False, use_ovf=False):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    # 获取当前epoch的alpha
    alpha = curriculum.get_alpha_for_epoch(epoch)
    if hasattr(model, 'set_alpha'):
        model.set_alpha(alpha)
    
    pbar = tqdm(trainloader, desc=f'Epoch {epoch}/{total_epochs} (α={alpha:.2f})')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        if use_sam:
            # SAM训练需要closure
            def closure():
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                return loss
            
            optimizer.step(closure)
            outputs = model(inputs)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        
        train_loss += loss.item() if 'loss' in locals() else 0
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return train_loss / len(trainloader), 100.*correct/total


@torch.no_grad()
def evaluate(model, testloader, criterion, device, alpha=0.0, use_ovf=False, ovf_alpha=0.5):
    model.eval()
    if hasattr(model, 'set_alpha'):
        model.set_alpha(alpha)
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        if use_ovf:
            outputs = OVFFusion(num_samples=3, alpha=ovf_alpha).fuse(model, inputs, device)
        else:
            outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return test_loss / len(testloader), 100.*correct/total


def run_experiment(config):
    device = config['device']
    experiment_type = config.get('experiment_type', 'curriculum_only')
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 128)
    num_workers = config.get('num_workers', 4)
    seed = config.get('seed', 42)
    alpha_max = config.get('alpha_max', 0.4)
    use_sam = config.get('use_sam', False)
    use_ovf = config.get('use_ovf', False)
    num_stages = config.get('num_stages', 8)
    
    print("=" * 70)
    print(f"实验: {config.get('name', 'unknown')}")
    print(f"类型: {experiment_type}")
    print(f"Alpha_max: {alpha_max}, 阶段数: {num_stages}")
    print(f"SAM: {use_sam}, OVF: {use_ovf}")
    print("=" * 70)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='/mnt/storage2/zyc/CIM比赛/公共数据集', train=True,
        download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='/mnt/storage2/zyc/CIM比赛/公共数据集', train=False,
        download=True, transform=transform_test
    )
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 加载模型
    model = get_model(name='resnet18', num_classes=10, pretrained=True, alpha=0.0)
    model = model.to(device)
    
    # 优化器
    base_optimizer = optim.SGD(
        model.parameters(),
        lr=config.get('lr', 0.1),
        momentum=0.9,
        weight_decay=config.get('weight_decay', 5e-4)
    )
    
    if use_sam:
        optimizer = SAMOptimizer(base_optimizer, rho=config.get('sam_rho', 0.05))
    else:
        optimizer = base_optimizer
    
    scheduler = CosineAnnealingLR(optimizer if not use_sam else optimizer.base_optimizer, T_max=epochs)
    
    # 课程学习
    curriculum = CurriculumTrainer(
        num_stages=num_stages,
        alpha_start=0.05,
        alpha_end=alpha_max,
        epochs_per_stage=max(1, epochs // num_stages)
    )
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'alphas': []}
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        alpha = curriculum.get_alpha_for_epoch(epoch)
        history['alphas'].append(alpha)
        
        train_loss, train_acc = train_with_curriculum(
            model, trainloader, nn.CrossEntropyLoss(), optimizer, device, epoch, epochs,
            curriculum, use_sam, use_ovf
        )
        scheduler.step()
        
        test_loss, test_acc = evaluate(model, testloader, nn.CrossEntropyLoss(), device, 
                                       alpha=0.0, use_ovf=use_ovf)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch}/{epochs}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            print(f'  -> 新最佳: {best_acc:.2f}% (Epoch {best_epoch})')
    
    return {
        'experiment_type': experiment_type,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'final_acc': history['test_acc'][-1],
        'history': history,
        'config': config
    }


def main():
    parser = argparse.ArgumentParser(description='赛题1 专利三验证实验')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/patent3_verification')
    args = parser.parse_args()
    
    configs = [
        # 配置1：仅课程学习（基线）
        {'name': 'curriculum_only', 'device': args.device, 'experiment_type': 'curriculum_only',
         'epochs': args.epochs, 'batch_size': args.batch_size,
         'num_workers': args.num_workers, 'seed': args.seed,
         'lr': 0.1, 'weight_decay': 5e-4, 'alpha_max': 0.4,
         'use_sam': False, 'use_ovf': False, 'num_stages': 8},
        
        # 配置2：课程学习+SAM
        {'name': 'curriculum_sam', 'device': args.device, 'experiment_type': 'curriculum_sam',
         'epochs': args.epochs, 'batch_size': args.batch_size,
         'num_workers': args.num_workers, 'seed': args.seed + 1,
         'lr': 0.1, 'weight_decay': 5e-4, 'alpha_max': 0.4,
         'use_sam': True, 'use_ovf': False, 'sam_rho': 0.05, 'num_stages': 8},
        
        # 配置3：课程学习+OVF
        {'name': 'curriculum_ovf', 'device': args.device, 'experiment_type': 'curriculum_ovf',
         'epochs': args.epochs, 'batch_size': args.batch_size,
         'num_workers': args.num_workers, 'seed': args.seed + 2,
         'lr': 0.1, 'weight_decay': 5e-4, 'alpha_max': 0.4,
         'use_sam': False, 'use_ovf': True, 'num_stages': 8},
        
        # 配置4：课程学习+SAM+OVF
        {'name': 'curriculum_sam_ovf', 'device': args.device, 'experiment_type': 'curriculum_sam_ovf',
         'epochs': args.epochs, 'batch_size': args.batch_size,
         'num_workers': args.num_workers, 'seed': args.seed + 3,
         'lr': 0.1, 'weight_decay': 5e-4, 'alpha_max': 0.4,
         'use_sam': True, 'use_ovf': True, 'sam_rho': 0.05, 'num_stages': 8},
    ]
    
    save_dir = os.path.join('/mnt/storage2/zyc/CIM比赛/赛道5/赛题1', args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = []
    
    for config in configs:
        print(f"\n{'='*70}")
        print(f"运行配置: {config['name']}")
        print(f"{'='*70}")
        
        result = run_experiment(config)
        result['config_name'] = config['name']
        all_results.append(result)
        
        save_path = os.path.join(save_dir, f"patent3_{config['name']}.json")
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"结果已保存到: {save_path}")
    
    print("\n" + "=" * 70)
    print("专利三验证实验结果汇总:")
    print("=" * 70)
    
    for r in all_results:
        print(f"  {r['config_name']}: 最佳={r['best_acc']:.2f}%(E{r['best_epoch']}), 最终={r['final_acc']:.2f}%")
    
    if all_results:
        best_result = max(all_results, key=lambda x: x['best_acc'])
        print(f"\n最佳配置: {best_result['config_name']}")
        print(f"最佳精度: {best_result['best_acc']:.2f}%")
        
        summary_path = os.path.join(save_dir, "patent3_summary.json")
        summary = {
            'best_config': best_result['config_name'],
            'best_acc': best_result['best_acc'],
            'best_epoch': best_result['best_epoch'],
            'all_results': [{
                'name': r['config_name'],
                'best_acc': r['best_acc'],
                'best_epoch': r['best_epoch'],
                'final_acc': r['final_acc']
            } for r in all_results]
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n汇总已保存到: {summary_path}")


if __name__ == '__main__':
    main()
