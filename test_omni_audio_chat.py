#!/usr/bin/env python3
"""
Qwen3-Omni 音频交互测试脚本
功能：
1. 测试文本 + 音频输入 → 音频输出的能力
2. 支持交互式多轮对话
3. 实时播放音频并保存到文件
"""

import os
import sys
import io
import base64
import argparse
import time
import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from typing import Optional, Tuple, List, Dict, Any


class Config:
    """配置类"""

    def __init__(self):
        self.server_url: str = "http://localhost:8091/v1"
        self.model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        self.output_dir: str = "./outputs"
        self.max_turns: int = 5
        self.system_prompt: str = "你是一个友好的语音助手，请用简洁、自然的语言回答问题。"
        self.seed: int = 42


class AudioProcessor:
    """音频处理器"""

    @staticmethod
    def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
        """加载音频文件"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        audio_data, sample_rate = sf.read(audio_path)
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]
        return audio_data, sample_rate

    @staticmethod
    def audio_to_base64_data_url(audio_data: np.ndarray, sample_rate: int) -> str:
        """将音频转换为 base64 data URL"""
        audio_int16 = AudioProcessor._normalize_to_int16(audio_data)
        buffer = io.BytesIO()
        sf.write(buffer, audio_int16, sample_rate, format="WAV")
        wav_bytes = buffer.getvalue()
        wav_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        return f"data:audio/wav;base64,{wav_b64}"

    @staticmethod
    def _normalize_to_int16(audio_data: np.ndarray) -> np.ndarray:
        """归一化音频数据到 int16 范围"""
        if audio_data.dtype == np.int16:
            return audio_data
        audio_float = audio_data.astype(np.float32)
        audio_float = np.clip(audio_float, -1.0, 1.0)
        return (audio_float * 32767).astype(np.int16)

    @staticmethod
    def decode_audio_from_base64(audio_b64: str) -> Tuple[np.ndarray, int]:
        """从 base64 数据解码音频（不保存文件）"""
        audio_bytes = base64.b64decode(audio_b64)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]
        return audio_data, sample_rate

    @staticmethod
    def save_audio_from_base64(audio_b64: str, output_path: str) -> Tuple[np.ndarray, int]:
        """从 base64 数据保存音频文件"""
        audio_data, sample_rate = AudioProcessor.decode_audio_from_base64(audio_b64)
        sf.write(output_path, audio_data, sample_rate)
        return audio_data, sample_rate
        sf.write(output_path, audio_data, sample_rate)
        return audio_data, sample_rate

    @staticmethod
    def play_audio(audio_data: np.ndarray, sample_rate: int, output_path: str = None):
        """播放音频（使用系统播放器）"""
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output_path = f"/tmp/omni_audio_{timestamp}.wav"

            sf.write(output_path, audio_data, sample_rate)

            player = None
            for cmd in ["afplay", "aplay", "paplay", "mplayer", "play"]:
                if subprocess.call(["which", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                    player = cmd
                    break

            if player:
                subprocess.run([player, output_path], check=True)
            else:
                print(f"⚠️ 未找到音频播放器")
                print(f"📁 音频文件已保存: {output_path}")
                print(f"💡 可使用以下命令播放:")
                print(f"   - macOS: afplay {output_path}")
                print(f"   - Linux (PulseAudio): paplay {output_path}")
                print(f"   - Linux (ALSA): aplay {output_path}")
                print(f"   - SoX: play {output_path}")

        except subprocess.CalledProcessError as e:
            print(f"⚠️ 音频播放失败: {e}")
            print(f"📁 音频文件已保存: {output_path}")
        except Exception as e:
            print(f"⚠️ 音频播放失败: {e}")
            if output_path:
                print(f"📁 音频文件已保存: {output_path}")


class Qwen3OmniClient:
    """Qwen3-Omni API 客户端"""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=config.server_url,
        )
        self.conversation_history: List[Dict[str, Any]] = []

    def check_server_status(self) -> bool:
        """检查服务器状态"""
        try:
            self.client.models.list()
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

        if self.conversation_history:
            for msg in self.conversation_history:
                messages.append(msg)

        messages.append({
            "role": "user",
            "content": user_content,
        })

        return messages

    def chat(
        self,
        text: str,
        audio_path: Optional[str] = None,
        modalities: List[str] = ["audio"],
        stream: bool = False,
    ) -> Tuple[str, Optional[np.ndarray], Optional[int]]:
        """
        发送对话请求

        Args:
            text: 用户文本
            audio_path: 音频文件路径（可选）
            modalities: 输出模态 ["text"] 或 ["audio"]
            stream: 是否使用流式输出

        Returns:
            text_response: 文本回复
            audio_data: 音频数据（如果请求了音频输出）
            sample_rate: 采样率
        """
        content_list = []

        if audio_path:
            audio_data, sample_rate = AudioProcessor.load_audio(audio_path)
            audio_url = AudioProcessor.audio_to_base64_data_url(audio_data, sample_rate)
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

        extra_body = {
            "sampling_params_list": [
                {
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "top_k": 1,
                    "max_tokens": 16384,
                    "detokenize": True,
                    "repetition_penalty": 1.05,
                    "stop_token_ids": [151645],
                    "seed": self.config.seed,
                },
                {
                    "temperature": 0.9,
                    "top_k": 50,
                    "max_tokens": 4096,
                    "seed": self.config.seed,
                    "detokenize": False,
                    "repetition_penalty": 1.05,
                    "stop_token_ids": [2150],
                },
                {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": 4096 * 16,
                    "seed": self.config.seed,
                    "detokenize": True,
                    "repetition_penalty": 1.1,
                },
            ],
        }

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.config.model,
                modalities=modalities,
                extra_body=extra_body,
                stream=stream,
            )

            text_response = ""
            audio_data = None
            sample_rate = None

            if not stream:
                for choice in response.choices:
                    if choice.message.content:
                        text_response = choice.message.content
                    if choice.message.audio:
                        audio_b64 = choice.message.audio.data
                        audio_data, sample_rate = AudioProcessor.decode_audio_from_base64(
                            audio_b64
                        )

                self.conversation_history.append({
                    "role": "user",
                    "content": content_list,
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": text_response,
                })

            return text_response, audio_data, sample_rate

        except Exception as e:
            print(f"❌ API 调用失败: {e}")
            return "", None, None


class AudioChatTester:
    """音频交互测试器"""

    def __init__(self, config: Config):
        self.config = config
        self.client = Qwen3OmniClient(config)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_basic_chat(self, audio_path: Optional[str] = None):
        """测试基础对话功能"""
        print("=" * 50)
        print("基础对话功能测试")
        print("=" * 50)

        if not self.client.check_server_status():
            return False

        print(f"✅ 服务器连接成功")
        print(f"模型: {self.config.model}")
        print(f"服务器: {self.config.server_url}")
        print()

        test_cases = [
            {
                "text": "请做一个简短的自我介绍。",
                "audio": None,
                "description": "纯文本输入 → 音频输出",
            },
        ]

        if audio_path and os.path.exists(audio_path):
            test_cases.append({
                "text": "请描述这段音频的内容。",
                "audio": audio_path,
                "description": "文本 + 音频输入 → 音频输出",
            })

        all_passed = True

        for i, test in enumerate(test_cases, 1):
            print(f"\n测试 {i}: {test['description']}")
            print("-" * 40)
            print(f"文本: {test['text']}")
            if test['audio']:
                print(f"音频: {test['audio']}")
            print()

            start_time = time.time()
            text_response, audio_data, sample_rate = self.client.chat(
                text=test["text"],
                audio_path=test["audio"],
                modalities=["audio"],
            )
            elapsed = time.time() - start_time

            if text_response or audio_data is not None:
                print("✅ 测试通过")
                print(f"响应时间: {elapsed:.2f}秒")
                print(f"文本回复长度: {len(text_response)} 字符")

                if audio_data is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_output = self.output_dir / f"test_{i}_response.wav"
                    sf.write(audio_output, audio_data, sample_rate)
                    print(f"音频已保存: {audio_output}")
                    print(f"音频时长: {len(audio_data) / sample_rate:.2f}秒")
                    print("🔊 正在播放音频...")
                    AudioProcessor.play_audio(audio_data, sample_rate, str(audio_output))
            else:
                print("❌ 测试失败")
                all_passed = False

        return all_passed

    def test_interactive_chat(self, max_turns: int = 5):
        """交互式对话测试"""
        print("=" * 50)
        print("交互式对话测试")
        print("=" * 50)

        if not self.client.check_server_status():
            return False

        print(f"✅ 服务器连接成功")
        print(f"系统提示: {self.config.system_prompt}")
        print(f"最大对话轮数: {max_turns}")
        print("-" * 50)
        print("提示: 输入 'quit' 退出对话，输入 'audio:<path>' 添加音频输入")
        print("-" * 50)

        self.client.conversation_history = []

        for turn in range(1, max_turns + 1):
            print(f"\n{'=' * 50}")
            print(f"第 {turn} 轮对话")
            print(f"{'=' * 50}")

            user_input = input("👤 你: ").strip()

            if user_input.lower() == "quit":
                print("\n👋 对话结束")
                break

            audio_path = None
            text_prompt = user_input

            if user_input.lower().startswith("audio:"):
                audio_path = user_input[6:].strip()
                text_prompt = "请分析这段音频的内容并回答。"
                print(f"🎵 音频文件: {audio_path}")

            if not user_input and not audio_path:
                print("⚠️ 请输入内容或提供音频文件")
                continue

            start_time = time.time()
            text_response, audio_data, sample_rate = self.client.chat(
                text=text_prompt if user_input else None,
                audio_path=audio_path if audio_path else None,
                modalities=["audio"],
            )
            elapsed = time.time() - start_time

            if text_response or audio_data is not None:
                print(f"\n🤖 Qwen3-Omni:")
                print("-" * 40)
                if text_response:
                    print(text_response)
                else:
                    print("(仅音频回复)")

                print(f"\n响应时间: {elapsed:.2f}秒")

                if audio_data is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_output = self.output_dir / f"turn_{turn}_response.wav"
                    text_output = self.output_dir / f"turn_{turn}_text.txt"
                    sf.write(audio_output, audio_data, sample_rate)
                    with open(text_output, "w", encoding="utf-8") as f:
                        f.write(text_response)
                    print(f"\n✅ 文本已保存: {text_output}")
                    print(f"✅ 音频已保存: {audio_output}")
                    print(f"音频时长: {len(audio_data) / sample_rate:.2f}秒")
                    print("🔊 正在播放音频...")
                    AudioProcessor.play_audio(audio_data, sample_rate, str(audio_output))
            else:
                print("❌ 对话失败，请重试")

        print(f"\n{'=' * 50}")
        print("对话历史统计")
        print(f"{'=' * 50}")
        print(f"总对话轮数: {len(self.client.conversation_history) // 2}")

        return True

    def test_batch_dialogue(self, prompts: List[str]):
        """批量对话测试（多轮对话，上下文保持）"""
        print("=" * 50)
        print("批量对话测试")
        print("=" * 50)

        if not self.client.check_server_status():
            return False

        print(f"✅ 服务器连接成功")
        print(f"系统提示: {self.config.system_prompt}")
        print(f"对话轮数: {len(prompts)}")
        print("-" * 50)

        self.client.conversation_history = []

        for turn, prompt in enumerate(prompts, 1):
            print(f"\n{'=' * 50}")
            print(f"第 {turn} 轮对话")
            print(f"{'=' * 50}")
            print(f"输入: {prompt}")
            print()

            start_time = time.time()
            text_response, audio_data, sample_rate = self.client.chat(
                text=prompt,
                audio_path=None,
                modalities=["audio"],
            )
            elapsed = time.time() - start_time

            if text_response or audio_data is not None:
                print(f"🤖 回复:")
                print("-" * 40)
                if text_response:
                    print(text_response)
                else:
                    print("(仅音频回复)")

                print(f"\n响应时间: {elapsed:.2f}秒")

                if audio_data is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_output = self.output_dir / f"dialogue_{turn}_response.wav"
                    text_output = self.output_dir / f"dialogue_{turn}_text.txt"
                    sf.write(audio_output, audio_data, sample_rate)
                    with open(text_output, "w", encoding="utf-8") as f:
                        f.write(text_response)
                    print(f"\n✅ 文本已保存: {text_output}")
                    print(f"✅ 音频已保存: {audio_output}")
                    if sample_rate:
                        print(f"音频时长: {len(audio_data) / sample_rate:.2f}秒")
                    print("🔊 正在播放音频...")
                    AudioProcessor.play_audio(audio_data, sample_rate, str(audio_output))
            else:
                print("❌ 对话失败，请重试")

        print(f"\n{'=' * 50}")
        print("对话历史统计")
        print(f"{'=' * 50}")
        print(f"总对话轮数: {len(self.client.conversation_history) // 2}")
        print(f"对话历史保持: {'✅ 是' if len(self.client.conversation_history) > 0 else '❌ 否'}")

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-Omni 音频交互测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础测试
  python test_omni_audio_chat.py --audio-path test.wav

  # 交互式对话
  python test_omni_audio_chat.py --interactive

  # 自定义配置
  python test_omni_audio_chat.py \
      --server-url http://localhost:8091/v1 \
      --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
      --output-dir ./results \
      --interactive
        """,
    )

    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8091/v1",
        help="API 服务器地址 (默认: http://localhost:8091/v1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="模型名称 (默认: Qwen/Qwen3-Omni-30B-A3B-Instruct)",
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
        help="输出目录 (默认: ./outputs)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="交互模式最大对话轮数 (默认: 5)",
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
        help="运行批量对话测试模式（需配合 --prompts 使用）",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="批量对话的提示词列表，用逗号分隔，例如：'你好,请介绍自己,请总结'",
    )

    args = parser.parse_args()

    config = Config()
    config.server_url = args.server_url
    config.model = args.model
    config.output_dir = args.output_dir
    config.max_turns = args.max_turns
    config.system_prompt = args.system_prompt

    tester = AudioChatTester(config)

    print("=" * 50)
    print("Qwen3-Omni 音频交互测试")
    print("=" * 50)
    print(f"模型: {config.model}")
    print(f"服务器: {config.server_url}")
    print(f"输出目录: {config.output_dir}")
    print("=" * 50)

    if args.dialogue and args.prompts:
        prompts = [p.strip() for p in args.prompts.split(",")]
        tester.test_batch_dialogue(prompts=prompts)
    elif args.interactive:
        tester.test_interactive_chat(max_turns=config.max_turns)
    else:
        tester.test_basic_chat(audio_path=args.audio_path)


if __name__ == "__main__":
    main()
