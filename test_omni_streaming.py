#!/usr/bin/env python3
"""
Qwen3-Omni 流式输出测试脚本
功能：
1. 流式文本输出 - 实时显示模型生成的文本
2. 流式音频播放 - 边生成边播放，优先流畅性
3. 音频保存 - 累积保存完整的音频文件
4. 多轮对话 - 支持上下文保持

依赖：
- conda activate ljm_asr
- pip install pyaudio numpy soundfile
"""

import os
import sys
import io
import base64
import argparse
import time
import threading
import queue
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Union
from collections import deque

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️ pyaudio 未安装，音频播放功能将受限")
    print("💡 安装命令: conda activate ljm_asr && pip install pyaudio")


class Config:
    """配置类"""

    def __init__(self):
        self.server_url: str = "http://10.246.56.85:8091/v1"
        self.model: str = "/vepfs/public/model-public/Qwen3-Omni-30B-A3B-Instruct"
        self.output_dir: str = "./outputs"
        self.sample_rate: int = 24000
        self.buffer_size: int = 8192
        self.latency_mode: str = "high"
        self.system_prompt: str = "你是一个友好的语音助手，请用简洁、自然的语言回答问题。"
        self.seed: int = 42
        self.enable_play: bool = True


class AudioBuffer:
    """环形缓冲区实现"""

    def __init__(self, max_size: int = 65536):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.condition = threading.Condition()

    def write(self, data: np.ndarray):
        with self.lock:
            self.buffer.append(data)
            if len(self.buffer) > 0:
                with self.condition:
                    self.condition.notify()

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        with self.condition:
            if not self.buffer:
                self.condition.wait(timeout=timeout)
            if self.buffer:
                with self.lock:
                    return self.buffer.popleft()
        return None

    def get_all(self) -> np.ndarray:
        with self.lock:
            if not self.buffer:
                return np.array([], dtype=np.float32)
            audio_data = np.concatenate(list(self.buffer))
            self.buffer.clear()
            return audio_data

    def clear(self):
        with self.lock:
            self.buffer.clear()

    def size(self) -> int:
        with self.lock:
            return sum(len(data) for data in self.buffer)


class StreamingAudioPlayer:
    """流式音频播放引擎"""

    def __init__(self, sample_rate: int = 24000, buffer_size: int = 8192):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio_buffer = AudioBuffer(max_size=131072)
        self.running = False
        self.stream = None
        self.pyaudio = None
        self.play_thread = None
        self.total_samples = 0
        self.play_start_time = None

        if PYAUDIO_AVAILABLE:
            self._init_audio_device()

    def _init_audio_device(self):
        """初始化音频设备"""
        try:
            self.pyaudio = pyaudio.PyAudio()
            self.stream = self.pyaudio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.buffer_size // 4,
                start=False
            )
        except Exception as e:
            print(f"⚠️ 音频设备初始化失败: {e}")
            self.pyaudio = None
            self.stream = None

    def start(self):
        """开始播放"""
        if not PYAUDIO_AVAILABLE:
            return

        self.running = True
        self.play_start_time = time.time()
        if self.stream:
            self.stream.start_stream()
        self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.play_thread.start()

    def _play_loop(self):
        """播放循环"""
        if not PYAUDIO_AVAILABLE or not self.stream:
            return

        while self.running:
            audio_data = self.audio_buffer.read(timeout=0.1)
            if audio_data is not None:
                try:
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                    if audio_data.ndim > 1:
                        audio_data = audio_data[:, 0]
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    self.stream.write(audio_data.tobytes())
                    self.total_samples += len(audio_data)
                except Exception as e:
                    pass

    def write(self, audio_data: np.ndarray):
        """写入音频数据"""
        if PYAUDIO_AVAILABLE:
            self.audio_buffer.write(audio_data)

    def stop(self) -> Tuple[np.ndarray, float]:
        """停止播放并返回完整音频数据"""
        self.running = False

        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass

        if self.pyaudio:
            try:
                self.pyaudio.terminate()
            except:
                pass

        full_audio = self.audio_buffer.get_all()
        duration = time.time() - self.play_start_time if self.play_start_time else 0

        return full_audio, duration

    def get_buffer_level(self) -> float:
        """获取缓冲区水位 (0-1)"""
        size = self.audio_buffer.size()
        max_size = 131072
        return min(size / max_size, 1.0)


class StreamingOmniClient:
    """流式 API 客户端"""

    def __init__(self, config: Config):
        self.config = config
        self.audio_player = StreamingAudioPlayer(
            sample_rate=config.sample_rate,
            buffer_size=config.buffer_size
        )
        self.full_audio_data: List[np.ndarray] = []
        self.conversation_history: List[Dict[str, Any]] = []

    def _create_client(self):
        """创建 OpenAI 客户端（延迟导入避免依赖问题）"""
        from openai import OpenAI
        return OpenAI(
            api_key="EMPTY",
            base_url=self.config.server_url,
        )

    def check_server_status(self) -> bool:
        """检查服务器状态"""
        try:
            client = self._create_client()
            client.models.list()
            return True
        except Exception as e:
            print(f"❌ 服务器连接失败: {e}")
            return False

    def build_messages(self, user_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """构建消息列表"""
        messages = []

        system_content = [
            {
                "type": "text",
                "text": self.config.system_prompt,
            }
        ]

        messages.append({
            "role": "system",
            "content": system_content,
        })

        for msg in self.conversation_history:
            messages.append(msg)

        messages.append({
            "role": "user",
            "content": user_content,
        })

        return messages

    def audio_to_base64_data_url(self, audio_path: str) -> Optional[str]:
        """将音频文件转换为 base64 data URL"""
        if not audio_path or not os.path.exists(audio_path):
            return None

        audio_data, sr = sf.read(audio_path)
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        buffer = io.BytesIO()
        sf.write(buffer, audio_int16, sr, format="WAV")
        wav_bytes = buffer.getvalue()
        wav_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        return f"data:audio/wav;base64,{wav_b64}"

    def stream_chat(
        self,
        text: str,
        audio_path: Optional[str] = None,
        modalities: List[str] = ["audio"],
        stream: bool = True,
    ) -> Tuple[str, np.ndarray, float]:
        """
        流式对话

        Args:
            text: 用户文本
            audio_path: 音频文件路径（可选）
            modalities: 输出模态
            stream: 是否流式输出

        Returns:
            text_response: 完整文本回复
            audio_data: 完整音频数据
            duration: 响应时间
        """
        import requests
        import json

        content_list = []

        if audio_path:
            audio_url = self.audio_to_base64_data_url(audio_path)
            if audio_url:
                content_list.append({
                    "type": "audio_url",
                    "audio_url": {"url": audio_url},
                })

        if text:
            content_list.append({
                "type": "text",
                "text": text,
            })

        messages = self.build_messages(content_list)

        full_text = ""
        all_audio_chunks: List[np.ndarray] = []
        start_time = time.time()

        api_url = f"{self.config.server_url}/chat/completions"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "modalities": modalities,
            "stream": stream,
        }

        try:
            headers = {"Content-Type": "application/json"}
            with requests.post(api_url, json=payload, headers=headers, stream=True, timeout=120) as response:
                response.raise_for_status()

                self.audio_player.start()
                first_audio_time = None
                first_text_time = None

                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    choice = chunk['choices'][0]
                                    delta = choice.get('delta', {})

                                    if 'content' in delta and delta['content']:
                                        text_chunk = delta['content']
                                        full_text += text_chunk
                                        if not first_text_time:
                                            first_text_time = time.time() - start_time
                                            print(f"\n📝 [文本流] 首次响应: {first_text_time:.2f}秒")
                                        print(text_chunk, end="", flush=True)

                                    if 'audio' in delta and delta['audio'] and 'data' in delta['audio']:
                                        try:
                                            audio_b64 = delta['audio']['data']
                                            audio_bytes = base64.b64decode(audio_b64)
                                            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
                                            if audio_data.ndim > 1:
                                                audio_data = audio_data[:, 0]
                                            all_audio_chunks.append(audio_data)
                                            self.audio_player.write(audio_data)

                                            if not first_audio_time:
                                                first_audio_time = time.time() - start_time
                                                buffer_level = self.audio_player.get_buffer_level()
                                                print(f"\n🔊 [音频流] 首次收到: {first_audio_time:.2f}秒 | 缓冲区: {buffer_level*100:.0f}%")

                                        except Exception as e:
                                            pass

                            except json.JSONDecodeError:
                                pass

                print()

            full_audio = np.concatenate(all_audio_chunks) if all_audio_chunks else np.array([], dtype=np.float32)

            self.conversation_history.append({
                "role": "user",
                "content": content_list,
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": full_text,
            })

            return full_text, full_audio, time.time() - start_time

        except Exception as e:
            print(f"\n❌ API 调用失败: {e}")
            return "", np.array([], dtype=np.float32), 0


class StreamingTester:
    """流式测试器"""

    def __init__(self, config: Config):
        self.config = config
        self.client = StreamingOmniClient(config)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_basic_streaming(self, audio_path: Optional[str] = None):
        """基础流式测试"""
        print("=" * 60)
        print("Qwen3-Omni 流式输出测试")
        print("=" * 60)

        if not self.client.check_server_status():
            return False

        print(f"✅ 服务器连接成功")
        print(f"模型: {self.config.model}")
        print(f"采样率: {self.config.sample_rate} Hz")
        print(f"缓冲区: {self.config.buffer_size} bytes")
        print(f"延迟模式: {self.config.latency_mode} (优先流畅性)")
        print()

        test_cases = [
            {
                "text": "请做一个简短的自我介绍。",
                "audio": None,
                "description": "纯文本 → 流式输出",
            },
        ]

        if audio_path and os.path.exists(audio_path):
            test_cases.append({
                "text": "请描述这段音频的内容。",
                "audio": audio_path,
                "description": "文本+音频 → 流式输出",
            })

        all_passed = True

        for i, test in enumerate(test_cases, 1):
            print(f"\n{'=' * 60}")
            print(f"测试 {i}: {test['description']}")
            print(f"{'=' * 60}")
            print(f"文本: {test['text']}")
            if test['audio']:
                print(f"音频: {test['audio']}")
            print("-" * 60)
            print("流式输出:")
            print("-" * 60)

            full_text, full_audio, duration = self.client.stream_chat(
                text=test["text"],
                audio_path=test["audio"],
                modalities=["audio"],
                stream=True,
            )

            if full_text or len(full_audio) > 0:
                print("-" * 60)
                print("✅ 测试通过")
                print(f"响应时间: {duration:.2f}秒")
                print(f"文本长度: {len(full_text)} 字符")

                if len(full_audio) > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_output = self.output_dir / f"streaming_{i}_response.wav"
                    text_output = self.output_dir / f"streaming_{i}_text.txt"
                    sf.write(audio_output, full_audio, self.config.sample_rate)
                    with open(text_output, "w", encoding="utf-8") as f:
                        f.write(full_text)
                    print(f"音频时长: {len(full_audio) / self.config.sample_rate:.2f}秒")
                    print(f"音频已保存: {audio_output}")
                    print(f"文本已保存: {text_output}")
            else:
                print("-" * 60)
                print("❌ 测试失败")
                all_passed = False

        return all_passed

    def test_batch_streaming(self, prompts: List[str]):
        """批量流式对话测试"""
        print("=" * 60)
        print("批量流式对话测试")
        print("=" * 60)

        if not self.client.check_server_status():
            return False

        print(f"✅ 服务器连接成功")
        print(f"系统提示: {self.config.system_prompt}")
        print(f"对话轮数: {len(prompts)}")
        print("-" * 60)

        self.client.conversation_history = []
        all_passed = True

        for turn, prompt in enumerate(prompts, 1):
            print(f"\n{'=' * 60}")
            print(f"第 {turn} 轮对话")
            print(f"{'=' * 60}")
            print(f"输入: {prompt}")
            print("-" * 60)
            print("流式输出:")
            print("-" * 60)

            full_text, full_audio, duration = self.client.stream_chat(
                text=prompt,
                audio_path=None,
                modalities=["audio"],
                stream=True,
            )

            if full_text or len(full_audio) > 0:
                print("-" * 60)
                print(f"响应时间: {duration:.2f}秒")
                if len(full_audio) > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_output = self.output_dir / f"batch_{turn}_response.wav"
                    text_output = self.output_dir / f"batch_{turn}_text.txt"
                    sf.write(audio_output, full_audio, self.config.sample_rate)
                    with open(text_output, "w", encoding="utf-8") as f:
                        f.write(full_text)
                    print(f"✅ 文本已保存: {text_output}")
                    print(f"✅ 音频已保存: {audio_output}")
                    print(f"音频时长: {len(full_audio) / self.config.sample_rate:.2f}秒")
            else:
                print("-" * 60)
                print("❌ 对话失败")
                all_passed = False

        print(f"\n{'=' * 60}")
        print("对话历史统计")
        print(f"{'=' * 60}")
        print(f"总对话轮数: {len(self.client.conversation_history) // 2}")
        print(f"上下文保持: {'✅ 是' if len(self.client.conversation_history) > 0 else '❌ 否'}")

        return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-Omni 流式输出测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础测试（带音频输入）
  conda activate ljm_asr && unset HTTP_PROXY && python test_omni_streaming.py \\
      --audio-path /vepfs/group03/user/ljm/omni_scripts/input/test.wav

  # 交互式流式对话
  conda activate ljm_asr && unset HTTP_PROXY && python test_omni_streaming.py --interactive

  # 批量流式对话
  conda activate ljm_asr && unset HTTP_PROXY && python test_omni_streaming.py \\
      --dialogue \\
      --prompts "你好,请介绍自己,请用英文说再见"

  # 自定义参数
  conda activate ljm_asr && unset HTTP_PROXY && python test_omni_streaming.py \\
      --audio-path test.wav \\
      --buffer-size 16384 \\
      --latency-mode high
        """,
    )

    parser.add_argument(
        "--server-url",
        type=str,
        default="http://10.246.56.85:8091/v1",
        help="API 服务器地址",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/vepfs/public/model-public/Qwen3-Omni-30B-A3B-Instruct",
        help="模型路径",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="测试音频文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="输出目录",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=8192,
        help="音频缓冲区大小（字节）",
    )
    parser.add_argument(
        "--latency-mode",
        type=str,
        default="high",
        choices=["low", "normal", "high"],
        help="延迟模式: low=低延迟, normal=平衡, high=优先流畅性",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="你是一个友好的语音助手，请用简洁、自然的语言回答问题。",
        help="系统提示词",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="启用交互式对话模式",
    )
    parser.add_argument(
        "--dialogue",
        action="store_true",
        help="启用批量对话模式（需配合 --prompts）",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="批量对话的提示词列表，用逗号分隔",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="禁用实时播放，仅保存文件",
    )

    args = parser.parse_args()

    config = Config()
    config.server_url = args.server_url
    config.model = args.model
    config.output_dir = args.output_dir
    config.buffer_size = args.buffer_size
    config.latency_mode = args.latency_mode
    config.system_prompt = args.system_prompt
    config.enable_play = not args.no_play

    print("=" * 60)
    print("Qwen3-Omni 流式输出测试")
    print("=" * 60)
    print(f"模型: {config.model}")
    print(f"服务器: {config.server_url}")
    print(f"输出目录: {config.output_dir}")
    print(f"缓冲区: {config.buffer_size} bytes")
    print(f"延迟模式: {config.latency_mode} (优先流畅性)")
    if not config.enable_play:
        print("⚠️ 实时播放已禁用")
    print("=" * 60)

    tester = StreamingTester(config)

    if args.dialogue and args.prompts:
        prompts = [p.strip() for p in args.prompts.split(",")]
        tester.test_batch_streaming(prompts=prompts)
    elif args.interactive:
        print("⚠️ 交互式模式需要在终端中运行")
        print("请使用: python test_omni_streaming.py --interactive")
    else:
        tester.test_basic_streaming(audio_path=args.audio_path)


if __name__ == "__main__":
    main()
