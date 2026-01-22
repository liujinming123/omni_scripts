#!/usr/bin/env python3
"""vLLM-Omni环境测试脚本"""

import sys
import os

def test_environment():
    """测试环境安装"""
    print("=" * 50)
    print("vLLM-Omni 环境测试")
    print("=" * 50)
    
    tests = []
    
    # 1. 测试vLLM
    try:
        import vllm
        tests.append(("vLLM", True, vllm.__version__))
    except ImportError as e:
        tests.append(("vLLM", False, str(e)))
    
    # 2. 测试vLLM-Omni
    try:
        import vllm_omni
        tests.append(("vLLM-Omni", True, vllm_omni.__version__))
    except ImportError as e:
        tests.append(("vLLM-Omni", False, str(e)))
    
    # 3. 测试Transformers
    try:
        import transformers
        tests.append(("Transformers", True, transformers.__version__))
    except ImportError as e:
        tests.append(("Transformers", False, str(e)))
    
    # 4. 测试PyTorch
    try:
        import torch
        tests.append(("PyTorch", True, torch.__version__))
    except ImportError as e:
        tests.append(("PyTorch", False, str(e)))
    
    # 5. 测试模型路径
    model_path = "/vepfs/public/model-public/Qwen3-Omni-30B-A3B-Instruct"
    model_exists = os.path.exists(model_path)
    tests.append(("模型路径", model_exists, model_path if model_exists else "不存在"))
    
    # 6. 测试CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        tests.append(("CUDA", cuda_available, "可用" if cuda_available else "不可用"))
    except:
        tests.append(("CUDA", False, "测试失败"))
    
    # 打印结果
    print(f"{'组件':<15} {'状态':<10} {'版本/信息'}")
    print("-" * 50)
    for name, success, info in tests:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name:<15} {status:<10} {info}")
    
    print("-" * 50)
    
    # 返回结果
    all_passed = all(test[1] for test in tests)
    if all_passed:
        print("✅ 所有测试通过！环境搭建成功。")
        return 0
    else:
        print("❌ 部分测试失败，请检查错误信息。")
        return 1

if __name__ == "__main__":
    sys.exit(test_environment())
