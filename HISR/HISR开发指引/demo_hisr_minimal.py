#!/usr/bin/env python3
"""
HISR最小演示脚本 - 专注于核心功能测试
"""

import sys
from pathlib import Path

def main():
    print("🎯 HISR核心功能演示")
    print("=" * 50)
    
    # 设置路径
    hisr_dir = Path("关键脚本实现示例（不一定完美复刻报告中的模型，仅参考）")
    print(f"📁 HISR目录: {hisr_dir.absolute()}")
    
    # 列出所有.py文件
    py_files = list(hisr_dir.glob("*.py"))
    print(f"\n📋 找到 {len(py_files)} 个Python文件:")
    for py_file in py_files:
        print(f"  - {py_file.name}")
    
    # 测试核心模块导入
    print(f"\n🔧 测试模块导入:")
    modules_to_test = ["bucketize", "prefix", "local_operator"]
    
    for module_name in modules_to_test:
        module_path = hisr_dir / f"{module_name}.py"
        if module_path.exists():
            try:
                # 简化导入测试 - 只检查语法
                with open(module_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # 简单的语法检查
                compile(content, module_path.name, 'exec')
                print(f"  ✅ {module_name}.py (语法正确)")
                
                # 尝试导入模块基本信息
                if module_name == "bucketize":
                    print("    - 功能: 逻辑分桶策略")
                elif module_name == "prefix":
                    print("    - 功能: 前缀层次处理")
                elif module_name == "local_operator":
                    print("    - 功能: 桶本地二分图算子")
                    
            except SyntaxError as e:
                print(f"  ❌ {module_name}.py (语法错误: {e})")
            except Exception as e:
                print(f"  ⚠️  {module_name}.py (导入警告: {e})")
        else:
            print(f"  ❌ {module_name}.py (文件不存在)")
    
    # 检查主训练脚本
    print(f"\n🚀 主训练脚本检查:")
    main_script = hisr_dir / "train_eval_hisr.py"
    if main_script.exists():
        print(f"  ✅ train_eval_hisr.py 存在")
        
        with open(main_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键组件
        components = {
            "BucketGraph": "二分图数据结构",
            "EncoderOutput": "编码器输出",
            "PrefixTreeDecoder": "前缀树解码器", 
            "HISRConfig": "配置类",
            "HISRPipeline": "主流程"
        }
        
        print("    关键组件:")
        for comp, desc in components.items():
            if comp in content:
                print(f"      ✅ {comp} - {desc}")
            else:
                print(f"      ❌ {comp} - 未找到")
                
    else:
        print(f"  ❌ train_eval_hisr.py 不存在")
    
    print(f"\n📋 使用说明:")
    print("  1. 运行单个模块测试: python demo_hisr_minimal.py")
    print("  2. 运行完整实验: python run_hisr_experiment.py")
    print("  3. 交互式实验: jupyter notebook HISR实验流程.ipynb")
    
    print(f"\n✅ 演示完成!")

if __name__ == "__main__":
    main()