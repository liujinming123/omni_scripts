# Qwen3-Omni 音频交互测试

## 简介

本测试脚本用于验证 Qwen3-Omni 模型的音频交互能力，支持：
- 文本 + 音频输入 → 音频输出
- 交互式多轮对话
- 实时音频播放和保存

## 环境要求

### Python 依赖
```bash
pip install soundfile openai numpy
```

### 系统要求
- 音频播放器（可选，用于实时播放）：
  - macOS: `afplay`（系统自带）
  - Linux (PulseAudio): `paplay`
  - Linux (ALSA): `aplay`
  - SoX: `play`

安装示例（Ubuntu）:
```bash
sudo apt-get install libasound2-plugins paplay sox
```

## 使用方法

### 1. 基础测试

测试纯文本输入和文本+音频输入的能力：
```bash
python test_omni_audio_chat.py --audio-path /path/to/your/audio.wav
```

参数说明：
- `--audio-path`: 测试音频文件路径（可选，如果不提供则只测试纯文本）
- `--server-url`: API 服务器地址（默认: http://localhost:8091/v1）
- `--model`: 模型名称（默认: Qwen/Qwen3-Omni-30B-A3B-Instruct）
- `--output-dir`: 输出目录（默认: ./outputs）

示例：
```bash
python test_omni_audio_chat.py \
    --audio-path /vepfs/group03/user/ljm/data/test_audio.wav \
    --server-url http://localhost:8091/v1 \
    --output-dir /vepfs/group03/user/ljm/scripts/outputs
```

### 2. 交互式对话

启动交互式多轮对话：
```bash
python test_omni_audio_chat.py --interactive
```

交互命令：
- 直接输入文本进行对话
- 输入 `audio:/path/to/audio.wav` 添加音频输入
- 输入 `quit` 退出对话

示例对话：
```
👤 你: 你好，请介绍一下自己
🤖 Qwen3-Omni: [文本回复]
🔊 正在播放音频...
✅ 音频已保存: outputs/turn_1_response.wav

👤 你: audio:/path/to/test.wav
🤖 Qwen3-Omni: [根据音频内容的回复]
🔊 正在播放音频...
✅ 音频已保存: outputs/turn_2_response.wav

👤 你: quit
👋 对话结束
```

### 3. 自定义系统提示词

修改对话的系统行为：
```bash
python test_omni_audio_chat.py \
    --interactive \
    --system-prompt "你是一个专业的客服助手，请用礼貌的语气回答。"
```

### 4. 自定义对话轮数

限制最大对话轮数：
```bash
python test_omni_audio_chat.py --interactive --max-turns 10
```

## 输出文件

脚本会在指定的输出目录生成以下文件：
- `{turn}_response.wav`: 每轮对话的音频回复
- `{turn}_text.txt`: 每轮对话的文本内容

## 常见问题

### Q1: 没有音频播放器怎么办？

脚本会检测可用的播放器，如果没有找到会提示手动播放：
```
⚠️ 未找到音频播放器
📁 音频文件已保存: outputs/turn_1_response.wav
💡 可使用以下命令播放:
   - macOS: afplay outputs/turn_1_response.wav
   - Linux (PulseAudio): paplay outputs/turn_1_response.wav
```

### Q2: 服务器连接失败

确保 vLLM-Omni 服务已启动：
```bash
bash start_vllm_omni.sh
```

检查服务状态：
```bash
curl http://localhost:8091/v1/models
```

### Q3: 音频格式要求

推荐使用以下规格的 WAV 音频：
- 采样率: 16kHz 或 24kHz
- 声道: 单声道（Mono）
- 位深度: 16-bit

### Q4: 内存不足错误

如果遇到内存错误，可以：
1. 减少 `--max-turns` 参数
2. 使用较短的音频文件
3. 检查 GPU 内存使用情况

### Q5: 如何查看完整的帮助信息

```bash
python test_omni_audio_chat.py --help
```

## 与 vLLM-Omni 服务配合使用

### 步骤 1: 启动服务

```bash
bash start_vllm_omni.sh
```

### 步骤 2: 运行测试

在新终端中：
```bash
python test_omni_audio_chat.py --interactive
```

## 高级配置

### 4×A100 GPU 配置建议

如果使用 4 张 A100 GPU，可以调整以下参数：
```bash
python test_omni_audio_chat.py \
    --audio-path test.wav \
    --max-turns 10 \
    --system-prompt "你是一个专业的语音助手..."
```

### 批量测试

可以通过循环脚本进行批量测试：
```bash
#!/bin/bash
# batch_test.sh

for audio in /path/to/audio_files/*.wav; do
    echo "Testing: $audio"
    python test_omni_audio_chat.py --audio-path "$audio"
    echo "---"
done
```

## 日志和调试

### 启用详细输出

脚本默认会输出详细的执行信息。如需更详细的调试信息，可以修改脚本中的日志级别。

### 检查输出文件

所有输出文件都会保存在指定的输出目录中，可以使用以下命令查看：
```bash
ls -la outputs/
```

## 注意事项

1. **首次运行较慢**: 首次调用可能需要较长时间来加载模型
2. **音频文件大小**: 较大的音频文件会延长处理时间
3. **对话历史**: 过多的对话轮数会占用更多内存
4. **并发请求**: 不建议同时运行多个测试实例

## 技术细节

### 音频处理流程

1. **输入处理**:
   - 加载音频文件（支持 WAV, FLAC 等格式）
   - 转换为 base64 编码的 data URL
   - 构建 multimodal 消息

2. **API 调用**:
   - 使用 OpenAI 兼容 API
   - 发送 multimodal 消息
   - 设置采样参数以获得高质量音频

3. **输出处理**:
   - 解码 base64 音频数据
   - 保存为 WAV 文件
   - 播放音频（如有播放器）

### 消息格式

```python
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": system_prompt}]
    },
    {
        "role": "user",
        "content": [
            {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}},
            {"type": "text", "text": user_text}
        ]
    }
]
```

## 故障排除

### 错误: "ModuleNotFoundError: No module named 'soundfile'"

安装依赖：
```bash
pip install soundfile
```

### 错误: "ModuleNotFoundError: No module named 'openai'"

安装依赖：
```bash
pip install openai
```

### 错误: "CUDA out of memory"

降低 GPU 内存使用：
- 减少对话轮数
- 使用较小的音频文件
- 重启服务释放内存

### 错误: "Connection refused"

检查服务是否运行：
```bash
curl http://localhost:8091/v1/models
```

如未运行，启动服务：
```bash
bash start_vllm_omni.sh
```

## 参考资料

- [vLLM-Omni 文档](https://docs.vllm.ai/projects/vllm-omni/)
- [Qwen3-Omni GitHub](https://github.com/QwenLM/Qwen3-Omni)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [soundfile 文档](https://python-soundfile.readthedocs.io/)
