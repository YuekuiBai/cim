"""
赛道五赛题一主入口脚本

完整实验流程：
1. 敏感性分析（任务一）
2. 非线性感知训练（任务二）
3. 鲁棒性增强方法（任务三）
4. 拓展研究
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys
import os
import argparse
import yaml

# Add project root path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet import get_model, NonLinearWrapper
from noise.nonlinearity import set_model_alpha
from evaluation.sensitivity import comprehensive_sensitivity_analysis
from evaluation.robustness import comprehensive_robustness_analysis
from evaluation.robustness_v2 import (
    comprehensive_robustness_analysis as comprehensive_robustness_analysis_v2,
    train_improved_calibration,
    mixed_alpha_training
)
from evaluation.extended import comprehensive_extended_analysis
from training.train import NonlinearityAwareTrainer, TrainingConfig, compare_training_strategies


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_data_loaders(config: dict, batch_size=None, num_workers=None):
    """获取数据加载器"""
    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(config['dataset']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(config['dataset']['image_size']),
        transforms.CenterCrop(config['dataset']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    # 加载数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root=config['dataset']['root'], train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=config['dataset']['root'], train=False, download=True, transform=test_transform
    )
    
    bs = batch_size if batch_size is not None else config['training']['batch_size']
    nw = num_workers if num_workers is not None else config['dataset'].get('num_workers', 4)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=nw
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, num_workers=nw
    )
    
    return train_loader, test_loader


def run_task1(model, test_loader, alpha_values, device, save_dir):
    """
    任务一：敏感性分析
    - 整网精度衰减趋势
    - 单层输出分布偏移
    - 误差逐层累积行为
    """
    print("\n" + "=" * 70)
    print("任务一：非线性误差敏感性分析")
    print("=" * 70)
    
    return comprehensive_sensitivity_analysis(
        model, test_loader, alpha_values, device,
        save_dir=os.path.join(save_dir, 'task1_sensitivity')
    )


def run_task2(model, train_loader, test_loader, alpha_values, device, save_dir, config,
              training_mode='fixed', alpha_max=0.5, num_stages=8, total_epochs=40,
              batch_size=None, num_workers=None, seed=42):
    """
    任务二：非线性感知训练
    - fine-tuning vs from-scratch 对比
    - 收敛性分析
    - 泛化性分析
    - 混合Alpha随机采样训练
    - 课程学习渐进式训练
    """
    print("\n" + "=" * 70)
    print("任务二：非线性感知训练")
    print("=" * 70)
    
    import numpy as np
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    results = {}
    
    if training_mode == 'mixed':
        print(f"\n混合Alpha训练: alpha ~ U(0, {alpha_max})")
        train_model = get_model(
            name=config['model']['name'],
            num_classes=config['model']['num_classes'],
            pretrained=True,
            alpha=0.0
        )
        train_model = NonLinearWrapper(train_model, alpha=0.0)
        train_model = train_model.to(device)
        
        optimizer = torch.optim.SGD(train_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=20)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
        
        for epoch in range(20):
            alpha_t = np.random.uniform(0, alpha_max)
            train_model.set_alpha(alpha_t)
            train_model.train()
            
            total_loss = 0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = train_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            scheduler.step()
            train_acc = 100. * correct / total
            
            train_model.set_alpha(0.0)
            train_model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = train_model(inputs)
                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()
            test_acc = 100. * test_correct / test_total
            
            history['train_loss'].append(total_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            
            if test_acc > best_acc:
                best_acc = test_acc
                os.makedirs(os.path.join(save_dir, 'task2_training/mixed_alpha'), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': train_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': test_acc,
                    'alpha_max': alpha_max,
                    'seed': seed
                }, os.path.join(save_dir, 'task2_training/mixed_alpha/model_best.pth'))
            
            print(f'Epoch {epoch+1}/20: alpha={alpha_t:.3f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%')
        
        results['mixed_alpha'] = {'best_acc': best_acc, 'history': history}
    
    elif training_mode == 'curriculum':
        print(f"\n课程学习训练: {num_stages}阶段, 总{total_epochs}epoch")
        train_model = get_model(
            name=config['model']['name'],
            num_classes=config['model']['num_classes'],
            pretrained=True,
            alpha=0.0
        )
        train_model = NonLinearWrapper(train_model, alpha=0.0)
        train_model = train_model.to(device)
        
        optimizer = torch.optim.SGD(train_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)
        criterion = nn.CrossEntropyLoss()
        
        alpha_start, alpha_end = 0.05, 0.4
        epochs_per_stage = total_epochs // num_stages
        
        best_acc = 0
        history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'alpha_values': []}
        
        for stage in range(num_stages):
            alpha_stage = alpha_start + (alpha_end - alpha_start) * stage / (num_stages - 1)
            print(f"\n阶段 {stage+1}/{num_stages}: alpha={alpha_stage:.3f}")
            
            for epoch in range(epochs_per_stage):
                train_model.set_alpha(alpha_stage)
                train_model.train()
                
                total_loss = 0
                correct = 0
                total = 0
                
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = train_model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                
                scheduler.step()
                train_acc = 100. * correct / total
                
                train_model.set_alpha(0.0)
                train_model.eval()
                test_correct = 0
                test_total = 0
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = train_model(inputs)
                        _, predicted = outputs.max(1)
                        test_total += targets.size(0)
                        test_correct += predicted.eq(targets).sum().item()
                test_acc = 100. * test_correct / test_total
                
                history['train_loss'].append(total_loss / len(train_loader))
                history['train_acc'].append(train_acc)
                history['test_acc'].append(test_acc)
                history['alpha_values'].append(alpha_stage)
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    os.makedirs(os.path.join(save_dir, f'task2_training/curriculum_K{num_stages}'), exist_ok=True)
                    torch.save({
                        'epoch': stage * epochs_per_stage + epoch,
                        'model_state_dict': train_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': test_acc,
                        'num_stages': num_stages,
                        'seed': seed
                    }, os.path.join(save_dir, f'task2_training/curriculum_K{num_stages}/model_best.pth'))
                
                print(f'  Epoch {epoch+1}/{epochs_per_stage}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%')
        
        results['curriculum'] = {'best_acc': best_acc, 'history': history}
    
    else:
        for alpha in [0.1, 0.2, 0.3]:
            print(f"\n训练 alpha={alpha} 的模型...")
            train_model = get_model(
                name=config['model']['name'],
                num_classes=config['model']['num_classes'],
                pretrained=True,
                alpha=alpha
            )
            
            train_config = TrainingConfig(
                epochs=20,
                lr=0.01,
                alpha=alpha,
                alpha_schedule='fixed',
                save_dir=os.path.join(save_dir, f'task2_training/alpha_{alpha}')
            )
            
            trainer = NonlinearityAwareTrainer(train_model, train_config, device)
            history = trainer.train(train_loader, test_loader)
            results[f'alpha_{alpha}'] = history
    
    return results


def run_task3(model, train_loader, test_loader, alpha_values, device, save_dir):
    """
    任务三：鲁棒性增强方法
    - 改进的预失真补偿
    - 增强版校准层
    - 多Alpha混合训练
    """
    print("\n" + "=" * 70)
    print("任务三：鲁棒性增强方法（改进版）")
    print("=" * 70)

    nat_models = {}

    for alpha in [0.1, 0.2, 0.3]:
        model_path = os.path.join(save_dir, f'task2_training/alpha_{alpha}/best_model.pth')
        if os.path.exists(model_path):
            from models.resnet import get_model
            nat_model = get_model('resnet18', num_classes=10, pretrained=False)
            nat_model = NonLinearWrapper(nat_model, alpha=alpha)

            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            nat_model.load_state_dict(state_dict)
            nat_model = nat_model.to(device)
            nat_models[alpha] = nat_model
            print(f'已加载NAT模型 (alpha={alpha})')

    print("\n开始综合鲁棒性分析...")
    return comprehensive_robustness_analysis_v2(
        model, train_loader, test_loader, nat_models, alpha_values, device,
        save_dir=os.path.join(save_dir, 'task3_robustness')
    )


def run_extended(model, test_loader, alpha_values, device, save_dir):
    """
    拓展研究
    - 噪声类型对比
    - 量化+非线性组合
    """
    print("\n" + "=" * 70)
    print("拓展研究")
    print("=" * 70)
    
    return comprehensive_extended_analysis(
        model, test_loader, alpha_values, device,
        save_dir=os.path.join(save_dir, 'extended')
    )


def main():
    parser = argparse.ArgumentParser(description='赛道五赛题一：非线性误差研究')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--task', type=str, default='all', 
                        choices=['all', 'task1', 'task2', 'task3', 'extended', 'eval'],
                        help='要运行的任务')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--training_mode', type=str, default='fixed',
                        choices=['fixed', 'mixed', 'curriculum'],
                        help='训练模式: fixed/mixed/curriculum')
    parser.add_argument('--alpha_max', type=float, default=0.5, help='混合Alpha最大上界')
    parser.add_argument('--num_stages', type=int, default=8, help='课程学习阶段数')
    parser.add_argument('--total_epochs', type=int, default=40, help='课程学习总epoch数')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小(覆盖配置)')
    parser.add_argument('--num_workers', type=int, default=None, help='数据加载线程数(覆盖配置)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--model_path', type=str, default=None, help='模型路径(用于eval任务)')
    parser.add_argument('--alpha', type=float, default=None, help='评估时的alpha值')
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f'使用设备: {torch.cuda.get_device_name(0)}')
    else:
        print(f'使用设备: {device}')
    
    # 数据
    print('\n加载数据...')
    train_loader, test_loader = get_data_loaders(config, args.batch_size, args.num_workers)

    # 模型
    print('\n加载模型...')
    base_model = get_model(
        name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        alpha=0.0
    )

    model_path = os.path.join('results', 'baseline_model.pth')
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        base_model.load_state_dict(state_dict)
        print(f'已加载训练权重: {model_path}')
    else:
        print(f'警告: 未找到训练权重 {model_path}，使用随机初始化')

    model = NonLinearWrapper(base_model, alpha=0.0)
    model = model.to(device)

    alpha_values = config['noise']['alpha_values']
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    
    if args.task in ['all', 'task1']:
        run_task1(model, test_loader, alpha_values, device, save_dir)
    
    if args.task in ['all', 'task2']:
        run_task2(model, train_loader, test_loader, alpha_values, device, save_dir, config,
                  training_mode=args.training_mode, alpha_max=args.alpha_max,
                  num_stages=args.num_stages, total_epochs=args.total_epochs,
                  batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed)
    
    if args.task in ['all', 'task3']:
        run_task3(model, train_loader, test_loader, alpha_values, device, save_dir)
    
    if args.task in ['all', 'extended']:
        run_extended(model, test_loader, alpha_values, device, save_dir)
    
    if args.task == 'eval':
        if args.model_path is None:
            print("错误: eval任务需要指定--model_path")
            return
        if args.alpha is None:
            print("错误: eval任务需要指定--alpha")
            return
        
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        eval_model = get_model(
            name=config['model']['name'],
            num_classes=config['model']['num_classes'],
            pretrained=False,
            alpha=0.0
        )
        eval_model = NonLinearWrapper(eval_model, alpha=0.0)
        eval_model.load_state_dict(checkpoint['model_state_dict'])
        eval_model = eval_model.to(device)
        
        eval_model.set_alpha(args.alpha)
        eval_model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = eval_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        print(f'评估结果: alpha={args.alpha}, accuracy={acc:.2f}%')
    
    print("\n" + "=" * 70)
    print("所有任务完成！")
    print(f"结果保存在: {save_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
