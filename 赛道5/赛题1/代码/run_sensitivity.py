"""
赛道五赛题一主入口脚本

完整实验流程：
1. 敏感性分析（任务一）
2. 非线性感知训练（任务二）
3. 鲁棒性增强方法（任务三）
4. 拓展研究
"""

import torch
import torchvision
import torchvision.transforms as transforms
import sys
import os
import argparse
import yaml

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def get_data_loaders(config: dict):
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
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, num_workers=4
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


def run_task2(model, train_loader, test_loader, alpha_values, device, save_dir, config):
    """
    任务二：非线性感知训练
    - fine-tuning vs from-scratch 对比
    - 收敛性分析
    - 泛化性分析
    """
    print("\n" + "=" * 70)
    print("任务二：非线性感知训练")
    print("=" * 70)
    
    results = {}
    
    # 1. 非线性感知训练（固定alpha）
    for alpha in [0.1, 0.2, 0.3]:
        print(f"\n训练 alpha={alpha} 的模型...")
        
        # 创建新模型
        train_model = get_model(
            name=config['model']['name'],
            num_classes=config['model']['num_classes'],
            pretrained=True,
            alpha=alpha
        )
        
        train_config = TrainingConfig(
            epochs=20,  # 简化实验
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
                        choices=['all', 'task1', 'task2', 'task3', 'extended'],
                        help='要运行的任务')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # 使用 GPU 0
        print(f'使用设备: {torch.cuda.get_device_name(0)}')
    else:
        print(f'使用设备: {device}')
    
    # 数据
    print('\n加载数据...')
    train_loader, test_loader = get_data_loaders(config)

    # 模型：先创建普通模型加载权重，再包装非线性
    print('\n加载模型...')
    base_model = get_model(
        name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        alpha=0.0  # get_model(alpha=0)不会包装NonLinearWrapper
    )

    # 加载训练好的权重
    model_path = os.path.join('results', 'baseline_model.pth')
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        base_model.load_state_dict(state_dict)
        print(f'已加载训练权重: {model_path}')
    else:
        print(f'警告: 未找到训练权重 {model_path}，使用随机初始化')

    # 用NonLinearWrapper包装，以便动态控制alpha
    model = NonLinearWrapper(base_model, alpha=0.0)
    model = model.to(device)

    # Alpha值范围
    alpha_values = config['noise']['alpha_values']
    
    # 结果保存目录
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    
    # 执行任务
    if args.task in ['all', 'task1']:
        run_task1(model, test_loader, alpha_values, device, save_dir)
    
    if args.task in ['all', 'task2']:
        run_task2(model, train_loader, test_loader, alpha_values, device, save_dir, config)
    
    if args.task in ['all', 'task3']:
        run_task3(model, train_loader, test_loader, alpha_values, device, save_dir)
    
    if args.task in ['all', 'extended']:
        run_extended(model, test_loader, alpha_values, device, save_dir)
    
    print("\n" + "=" * 70)
    print("所有任务完成！")
    print(f"结果保存在: {save_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
