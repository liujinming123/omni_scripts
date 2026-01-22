#!/bin/bash
# vLLM-Omni服务启动脚本

# 激活conda环境
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate base

# 模型路径
# MODEL_PATH="/vepfs/group03/user/ljm/base_model/Qwen/Qwen3-Omni-30B-A3B-Thinking"
MODEL_PATH="/vepfs/public/model-public/Qwen3-Omni-30B-A3B-Instruct"

# 端口配置
PORT=${PORT:-8091}

# GPU内存使用率
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}

echo "=========================================="
echo "vLLM-Omni服务启动"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "端口: $PORT"
echo "GPU内存利用率: $GPU_MEM_UTIL"
echo "=========================================="

# 启动vLLM-Omni服务
vllm serve "$MODEL_PATH" \
  --omni \
  --port "$PORT" \
  --host 0.0.0.0 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  -tp 4 \
  "$@"

  # --enable-prefix-caching \

echo "服务已启动: http://0.0.0.0:$PORT"