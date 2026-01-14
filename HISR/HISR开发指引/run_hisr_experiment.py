#!/usr/bin/env python3
"""
HISR实验运行脚本 - 依次运行各个模块的简化版本

这个脚本提供了一个更方便的命令行界面来运行HISR实验，
避免Jupyter Notebook的交互复杂性。
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def setup_paths():
    """设置项目路径"""
    current_dir = Path.cwd()
    hisr_scripts_dir = current_dir / "关键脚本实现示例（不一定完美复刻报告中的模型，仅参考）"
    ucl_repo_dir = current_dir / "参考代码仓库" / "uclpe-sketch-master"
    
    # 添加到Python路径
    sys.path.insert(0, str(hisr_scripts_dir))
    sys.path.insert(0, str(ucl_repo_dir))
    
    return hisr_scripts_dir, ucl_repo_dir

def check_dependencies():
    """检查依赖包"""
    required_packages = ['torch', 'numpy', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请使用以下命令安装: pip install " + " ".join(missing_packages))
        return False
    return True

def find_data_files(ucl_repo_dir):
    """查找数据文件"""
    data_dir = ucl_repo_dir / "data"
    if data_dir.exists():
        dat_files = list(data_dir.glob("*.dat"))
        return dat_files
    return []

def run_module_test(hisr_scripts_dir):
    """测试模块导入"""
    modules_to_test = [
        "bucketize", "prefix", "local_operator", 
        "encoder_bipartite", "decoder_prefix_tree"
    ]
    
    print("\n📋 测试模块导入:")
    all_ok = True
    for module_name in modules_to_test:
        module_path = hisr_scripts_dir / f"{module_name}.py"
        if module_path.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"  ✅ {module_name}")
            except Exception as e:
                print(f"  ❌ {module_name}: {e}")
                all_ok = False
        else:
            print(f"  ❌ {module_name}.py 文件不存在")
            all_ok = False
    
    return all_ok

def run_hisr_training(hisr_scripts_dir, data_file, break_number=100000, train_steps=100):
    """运行HISR训练脚本"""
    script_path = hisr_scripts_dir / "train_eval_hisr.py"
    
    if not script_path.exists():
        print(f"❌ 主脚本不存在: {script_path}")
        return False
    
    cmd = [
        "python", str(script_path),
        "--data", "network" if data_file else "synthetic",
        "--break_number", str(break_number),
        "--train_steps", str(train_steps)
    ]
    
    if data_file:
        cmd.extend(["--data_path", str(data_file)])
    else:
        cmd.extend(["--skewness", "1.5"])
    
    print(f"\n🚀 执行命令: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, 
                              cwd=str(hisr_scripts_dir),
                              capture_output=True, 
                              text=True,
                              timeout=600)  # 10分钟超时
        
        print("📋 标准输出:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  标准错误:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ HISR训练执行成功！")
            return True
        else:
            print(f"❌ HISR训练失败，返回码: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 训练超时（超过10分钟）")
        return False
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='HISR实验运行脚本')
    parser.add_argument('--break_number', type=int, default=100000, 
                       help='处理的数据包数量')
    parser.add_argument('--train_steps', type=int, default=100, 
                       help='训练步数')
    parser.add_argument('--check_only', action='store_true',
                       help='只检查环境，不运行训练')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 HISR实验运行脚本")
    print("=" * 60)
    
    # 1. 设置路径
    hisr_scripts_dir, ucl_repo_dir = setup_paths()
    print(f"📁 HISR脚本目录: {hisr_scripts_dir}")
    print(f"📁 UCL仓库目录: {ucl_repo_dir}")
    
    # 2. 检查依赖
    print("\n1. 检查依赖包:")
    if not check_dependencies():
        sys.exit(1)
    
    # 3. 检查数据文件
    print("\n2. 检查数据文件:")
    data_files = find_data_files(ucl_repo_dir)
    if data_files:
        print("✅ 找到数据文件:")
        for f in data_files:
            print(f"  - {f.name}")
        data_file = data_files[0]
    else:
        print("⚠️  未找到数据文件，将使用合成数据")
        data_file = None
    
    # 4. 测试模块导入
    print("\n3. 测试模块导入:")
    if not run_module_test(hisr_scripts_dir):
        print("❌ 模块导入测试失败，请检查代码")
        sys.exit(1)
    
    if args.check_only:
        print("\n✅ 环境检查完成！")
        return
    
    # 5. 运行训练
    print("\n4. 开始HISR训练...")
    success = run_hisr_training(hisr_scripts_dir, data_file, args.break_number, args.train_steps)
    
    # 6. 检查结果
    if success:
        print("\n5. 检查训练结果:")
        check_files = ["checkpoints", "training_log.txt", "model.pth"]
        for f in check_files:
            if (hisr_scripts_dir / f).exists():
                print(f"  ✅ {f}")
            else:
                print(f"  ❌ {f}")
        
        print("\n🎉 实验完成！")
    else:
        print("\n❌ 实验失败")

if __name__ == "__main__":
    main()