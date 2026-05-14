"""
鲁棒性增强方法模块

任务三：针对非线性误差的优化方法
- 逆非线性预失真补偿
- 可学习校准层
- 非线性感知微调
- 多种增强方法对比
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple
from noise.nonlinearity import (
    NonLinearWrapper, 
    InverseNonLinearity, 
    CalibrationLayer,
    set_model_alpha
)


class RobustModelWrapper(nn.Module):
    """
    鲁棒模型包装器
    
    在模型输出后添加校准层或逆非线性补偿
    """
    def __init__(self, model: nn.Module, method: str = 'calibration', 
                 alpha: float = 0.0, num_features: int = None):
        """
        Args:
            model: 基础模型
            method: 增强方法 ('calibration', 'inverse', 'none')
            alpha: 非线性强度
            num_features: 校准层的特征数（对于分类任务为类别数）
        """
        super().__init__()
        self.model = model
        self.method = method
        self.alpha = alpha
        
        if method == 'calibration' and num_features is not None:
            self.compensation = CalibrationLayer(num_features, learn_cubic=True)
        elif method == 'inverse':
            self.compensation = InverseNonLinearity(alpha=alpha)
        else:
            self.compensation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return self.compensation(output)
    
    def set_alpha(self, alpha: float):
        """设置非线性强度"""
        self.alpha = alpha
        if isinstance(self.model, NonLinearWrapper):
            self.model.set_alpha(alpha)
        else:
            set_model_alpha(self.model, alpha)
        
        if isinstance(self.compensation, InverseNonLinearity):
            self.compensation.alpha = alpha


class PreDistortionWrapper(nn.Module):
    """
    预失真补偿包装器
    
    在输入模型前应用逆非线性变换
    """
    def __init__(self, model: nn.Module, alpha: float = 0.0):
        super().__init__()
        self.model = model
        self.predistortion = InverseNonLinearity(alpha=alpha)
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 预失真补偿
        x_corrected = self.predistortion(x)
        # 正常前向传播
        return self.model(x_corrected)
    
    def set_alpha(self, alpha: float):
        self.alpha = alpha
        self.predistortion.alpha = alpha
        if isinstance(self.model, NonLinearWrapper):
            self.model.set_alpha(alpha)


def train_calibration_layer(model: nn.Module, train_loader, test_loader,
                            alpha: float, device: torch.device,
                            epochs: int = 10, lr: float = 0.01) -> Dict:
    """
    训练校准层
    
    固定模型权重，仅训练校准层参数
    
    Args:
        model: 带非线性注入的模型
        train_loader: 训练数据
        test_loader: 测试数据
        alpha: 非线性强度
        device: 设备
        epochs: 训练轮数
        lr: 学习率
    
    Returns:
        训练结果
    """
    # 设置非线性
    if isinstance(model, NonLinearWrapper):
        model.set_alpha(alpha)
    else:
        set_model_alpha(model, alpha)
    
    model = model.to(device)
    model.eval()  # 冻结模型
    
    # 获取输出特征数（假设是分类任务）
    num_classes = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            num_classes = module.out_features
            break
    
    if num_classes is None:
        num_classes = 10  # 默认值
    
    # 创建校准层
    calibration = CalibrationLayer(num_classes, learn_cubic=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(calibration.parameters(), lr=lr)
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    print(f"\n训练校准层 (alpha={alpha})...")
    
    for epoch in range(epochs):
        # 训练
        calibration.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            with torch.no_grad():
                features = model(inputs)
            outputs = calibration(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        history['train_loss'].append(total_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        
        # 测试
        calibration.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                features = model(inputs)
                outputs = calibration(features)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%')
    
    return {
        'history': history,
        'final_test_acc': test_acc,
        'calibration_state': calibration.state_dict()
    }


def evaluate_robustness_methods(model: nn.Module, test_loader, 
                                 alpha_values: List[float], device: torch.device,
                                 methods: List[str] = None) -> Dict:
    """
    评估不同鲁棒性增强方法的效果
    
    Args:
        model: 基础模型
        test_loader: 测试数据
        alpha_values: 非线性强度列表
        device: 设备
        methods: 要评估的方法列表
    
    Returns:
        各方法在不同alpha下的精度
    """
    if methods is None:
        methods = ['none', 'inverse']
    
    results = {method: {} for method in methods}
    
    # 获取输出特征数
    num_classes = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            num_classes = module.out_features
            break
    if num_classes is None:
        num_classes = 10
    
    for method in methods:
        print(f"\n评估方法: {method}")
        
        for alpha in tqdm(alpha_values, desc=f'{method}'):
            # 创建对应方法的包装器
            if method == 'none':
                wrapped_model = model
                if isinstance(wrapped_model, NonLinearWrapper):
                    wrapped_model.set_alpha(alpha)
                else:
                    set_model_alpha(wrapped_model, alpha)
            elif method == 'inverse':
                wrapped_model = RobustModelWrapper(
                    model, method='inverse', alpha=alpha, num_features=num_classes
                )
            elif method == 'calibration':
                wrapped_model = RobustModelWrapper(
                    model, method='calibration', alpha=alpha, num_features=num_classes
                )
            else:
                continue
            
            wrapped_model = wrapped_model.to(device)
            wrapped_model.eval()
            
            # 评估
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = wrapped_model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            acc = 100. * correct / total
            results[method][alpha] = acc
            print(f'  alpha={alpha:.2f}: accuracy={acc:.2f}%')
    
    return results


def compare_all_methods(baseline_model: nn.Module, nat_model: nn.Module,
                        test_loader, alpha_values: List[float], 
                        device: torch.device) -> Dict:
    """
    对比所有鲁棒性增强方法
    
    Args:
        baseline_model: 基线模型（无增强）
        nat_model: 非线性感知训练后的模型
        test_loader: 测试数据
        alpha_values: 非线性强度列表
        device: 设备
    
    Returns:
        完整对比结果
    """
    results = {
        'baseline': {},
        'nat': {},
        'inverse': {},
        'calibration': {}
    }
    
    # 获取输出特征数
    num_classes = None
    for module in baseline_model.modules():
        if isinstance(module, nn.Linear):
            num_classes = module.out_features
            break
    if num_classes is None:
        num_classes = 10
    
    for alpha in tqdm(alpha_values, desc='评估所有方法'):
        # 1. 基线模型
        if isinstance(baseline_model, NonLinearWrapper):
            baseline_model.set_alpha(alpha)
        else:
            set_model_alpha(baseline_model, alpha)
        baseline_model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = baseline_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        results['baseline'][alpha] = 100. * correct / total
        
        # 2. NAT模型
        if isinstance(nat_model, NonLinearWrapper):
            nat_model.set_alpha(alpha)
        else:
            set_model_alpha(nat_model, alpha)
        nat_model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = nat_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        results['nat'][alpha] = 100. * correct / total
        
        # 3. 逆非线性补偿
        inverse_wrapper = RobustModelWrapper(
            baseline_model, method='inverse', alpha=alpha, num_features=num_classes
        ).to(device)
        inverse_wrapper.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = inverse_wrapper(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        results['inverse'][alpha] = 100. * correct / total
    
    return results


def plot_robustness_comparison(results: Dict, save_path: str):
    """绘制鲁棒性方法对比图"""
    import matplotlib.pyplot as plt
    
    # 获取alpha值
    alphas = sorted(list(results['baseline'].keys()))
    
    plt.figure(figsize=(12, 6))
    
    # 绘制各方法曲线
    for method, data in results.items():
        if not data:
            continue
        accs = []
        for a in alphas:
            accs.append(data.get(a, None))
        valid_pairs = [(a, acc) for a, acc in zip(alphas, accs) if acc is not None]
        if valid_pairs:
            plot_alphas, plot_accs = zip(*valid_pairs)
            plt.plot(plot_alphas, plot_accs, '-o', label=method.upper(), linewidth=2, markersize=8)
    
    plt.xlabel('Nonlinearity Alpha', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Robustness Enhancement Methods Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(alphas)
    plt.ylim(0, 100)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"对比图已保存至: {save_path}")


def comprehensive_robustness_analysis(model: nn.Module, train_loader, test_loader,
                                       alpha_values: List[float], device: torch.device,
                                       save_dir: str = '../结果/robustness') -> Dict:
    """
    综合鲁棒性分析
    
    执行完整的鲁棒性增强方法评估
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("开始综合鲁棒性分析")
    print("=" * 60)
    
    # 1. 评估基线模型
    print("\n[1/3] 评估基线模型...")
    baseline_results = {}
    for alpha in tqdm(alpha_values, desc='基线评估'):
        if isinstance(model, NonLinearWrapper):
            model.set_alpha(alpha)
        else:
            set_model_alpha(model, alpha)
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        baseline_results[alpha] = 100. * correct / total
    
    # 2. 评估逆非线性补偿
    print("\n[2/3] 评估逆非线性补偿...")
    inverse_results = evaluate_robustness_methods(
        model, test_loader, alpha_values, device, methods=['inverse']
    )
    
    # 3. 训练并评估校准层
    print("\n[3/3] 训练并评估校准层...")
    calibration_results = {}
    for alpha in [0.1, 0.2, 0.3]:  # 选择几个典型alpha值
        if alpha in alpha_values:
            calib_result = train_calibration_layer(
                model, train_loader, test_loader, alpha, device, epochs=10
            )
            calibration_results[alpha] = calib_result['final_test_acc']
    
    # 汇总结果
    all_results = {
        'baseline': baseline_results,
        'inverse': inverse_results.get('inverse', {}),
        'calibration': calibration_results
    }
    
    # 绘制对比图
    plot_robustness_comparison(all_results, os.path.join(save_dir, 'robustness_comparison.png'))
    
    # 保存结果
    with open(os.path.join(save_dir, 'robustness_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("鲁棒性分析完成！")
    print(f"结果保存在: {save_dir}")
    print("=" * 60)
    
    return all_results
