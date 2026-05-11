import os
import sys
import subprocess

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config.yaml')
    output_dir = os.path.join(base_dir, '..', '结果')
    chart_dir = os.path.join(base_dir, '..', '图表')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(chart_dir, exist_ok=True)

    print("=" * 60)
    print("赛道四 - 赛题一：神经网络在模拟存内计算系统的映射")
    print("=" * 60)
    print(f"\n配置路径: {config_path}")
    print(f"输出目录: {output_dir}")
    print(f"图表目录: {chart_dir}")

    scripts = [
        ("存算阵列仿真", "array_simulation.py"),
        ("网络映射分析", "network_mapping.py"),
        ("性能评估", "performance_evaluation.py"),
    ]

    for name, script in scripts:
        print("\n" + "=" * 60)
        print(f"运行: {name}")
        print("=" * 60)
        cmd = [sys.executable, script, config_path, output_dir]
        result = subprocess.run(cmd, cwd=base_dir)
        if result.returncode != 0:
            print(f"错误: {name} 执行失败")
            return 1

    print("\n" + "=" * 60)
    print("生成图表")
    print("=" * 60)
    cmd = [sys.executable, "visualization.py", output_dir]
    result = subprocess.run(cmd, cwd=base_dir)
    if result.returncode != 0:
        print("警告: 图表生成失败")
        return 1

    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)
    print(f"\n结果文件: {output_dir}")
    print(f"图表文件: {chart_dir}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
