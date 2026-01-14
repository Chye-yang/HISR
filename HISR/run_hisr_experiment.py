#!/usr/bin/env python3
"""
HISR项目主入口脚本

这个脚本提供了HISR项目的统一入口点，方便运行各种实验和测试。
"""

import os
import sys
from pathlib import Path

def main():
    print("🚀 HISR - 高性能不变性网络测量框架")
    print("=" * 50)
    
    # 添加项目路径
    project_root = Path(__file__).parent
    scripts_dir = project_root / "HISR开发指引" / "关键脚本实现示例（不一定完美复刻报告中的模型，仅参考）"
    
    if scripts_dir.exists():
        sys.path.insert(0, str(scripts_dir))
        print(f"✅ 已添加脚本目录: {scripts_dir}")
    else:
        print("⚠️  脚本目录不存在，部分功能可能无法使用")
    
    # 检查核心文件
    core_files = [
        "train_eval_hisr.py",
        "encoder_bipartite.py", 
        "decoder_prefix_tree.py",
        "bucketize.py"
    ]
    
    print("\n📁 检查核心文件:")
    for file in core_files:
        file_path = scripts_dir / file
        if file_path.exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (未找到)")
    
    print("\n🎯 可用功能:")
    print("1. 数据编码 (encoder_bipartite.py)")
    print("2. 前缀树解码 (decoder_prefix_tree.py)") 
    print("3. 分桶处理 (bucketize.py)")
    print("4. 训练和评估 (train_eval_hisr.py)")
    print("\n💡 请直接运行对应的Python脚本进行实验")
    print("   例如: python HISR开发指引/关键脚本实现示例（不一定完美复刻报告中的模型，仅参考）/train_eval_hisr.py")

if __name__ == "__main__":
    main()