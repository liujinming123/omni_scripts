#!/usr/bin/env python3
"""
Qwen3-Omni Non-streaming Audio Chat Test Script
Features:
1. Non-streaming text -> audio output
2. Interactive multi-turn dialogue
3. Audio playback and saving

Dependencies:
- conda activate asr
- pip install pyaudio soundfile openai
"""

import os
import sys
import io
import base64
import argparse
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from openai import OpenAI

PYAUDIO_AVAILABLE = True
try:
    import pyaudio
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("[WARNING] pyaudio not installed, audio playback disabled")
    print("[INFO] Install command: conda activate asr && pip install pyaudio")


class Config:
    """Configuration class"""

    def __init__(self):
        self.server_url: str = "http://10.246.56.85:8091/v1"
        self.model: str = "/vepfs/public/model-public/Qwen3-Omni-30B-A3B-Instruct"
        self.output_dir: str = "./outputs"
        self.sample_rate: int = 24000
        self.system_prompt: str = "You are a friendly voice assistant. Please respond concisely and naturally."
        self.seed: int = 42
        self.enable_play: bool = True


class SimpleAudioPlayer:
    """Simple audio player (non-streaming)"""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.pyaudio = None
        self.stream = None

        if PYAUDIO_AVAILABLE:
            self._init_audio_device()

    def _init_audio_device(self):
        """Initialize audio device"""
        try:
            self.pyaudio = pyaudio.PyAudio()
            self.stream = self.pyaudio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=4096,
            )
        except Exception as e:
            print("[WARNING] Audio device initialization failed: " + str(e))
            self.pyaudio = None
            self.stream = None

    def play(self, audio_data: np.ndarray):
        """Play complete audio"""
        if not PYAUDIO_AVAILABLE or not self.stream:
            return

        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]
        audio_data = np.clip(audio_data, -1.0, 1.0)

        try:
            self.stream.write(audio_data.tobytes())
        except Exception as e:
            print("[WARNING] Audio playback failed: " + str(e))

    def close(self):
        """Close audio device"""
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


class OmniClient:
    """Non-streaming API client"""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=config.server_url,
        )
        self.audio_player = SimpleAudioPlayer(sample_rate=config.sample_rate)
        self.conversation_history: List[Dict[str, Any]] = []

    def check_server_status(self) -> bool:
        """Check server status"""
        try:
            self.client.models.list()
            return True
        except Exception as e:
            print("[ERROR] Server connection failed: " + str(e))
            return False

    def build_messages(self, user_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build message list"""
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

    def chat(
        self,
        text: str,
        audio_path: Optional[str] = None,
        modalities: List[str] = ["audio"],
    ) -> Tuple[str, np.ndarray, int]:
        """
        Non-streaming chat

        Args:
            text: User text
            audio_path: Audio file path (optional)
            modalities: Output modalities

        Returns:
            text_response: Complete text response
            audio_data: Complete audio data
            sample_rate: Sample rate
        """
        content_list = []

        if audio_path:
            audio_data, sr = sf.read(audio_path)
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]
            audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
            buffer = io.BytesIO()
            sf.write(buffer, audio_int16, sr, format="WAV")
            wav_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            audio_url = f"data:audio/wav;base64,{wav_b64}"
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
                stream=False,
            )

            text_response = ""
            audio_data = None
            sample_rate = None

            for choice in response.choices:
                if choice.message.content:
                    text_response = choice.message.content
                if choice.message.audio:
                    audio_b64 = choice.message.audio.data
                    audio_bytes = base64.b64decode(audio_b64)
                    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
                    if audio_data.ndim > 1:
                        audio_data = audio_data[:, 0]

            self.conversation_history.append({
                "role": "user",
                "content": content_list,
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": text_response,
            })

            return text_response, audio_data, sample_rate or self.config.sample_rate

        except Exception as e:
            print("[ERROR] API call failed: " + str(e))
            return "", np.array([], dtype=np.float32), self.config.sample_rate

    def close(self):
        """Close client"""
        self.audio_player.close()


class AudioChatTester:
    """Audio chat tester"""

    def __init__(self, config: Config):
        self.config = config
        self.client = OmniClient(config)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_basic_chat(self):
        """Basic chat test"""
        print("=" * 60)
        print("Qwen3-Omni Non-streaming Chat Test")
        print("=" * 60)

        if not self.client.check_server_status():
            return False

        print("[OK] Server connected")
        print("Model: " + self.config.model)
        print("Sample rate: " + str(self.config.sample_rate) + " Hz")
        print()

        test_text = "Please give a brief self-introduction."
        print("-" * 60)
        print("Input: " + test_text)
        print("-" * 60)
        print("Waiting for response...")
        print()

        start_time = time.time()
        text_response, audio_data, sample_rate = self.client.chat(
            text=test_text,
            audio_path=None,
            modalities=["audio"],
        )
        elapsed = time.time() - start_time

        if text_response or len(audio_data) > 0:
            print("[OK] Test passed")
            print("Response time: {:.2f} seconds".format(elapsed))

            if text_response:
                print()
                print("Response:")
                print("-" * 40)
                print(text_response)

            if len(audio_data) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_output = self.output_dir / "response_{}.wav".format(timestamp)
                text_output = self.output_dir / "text_{}.txt".format(timestamp)
                sf.write(audio_output, audio_data, sample_rate)
                with open(text_output, "w", encoding="utf-8") as f:
                    f.write(text_response)
                print()
                print("Audio saved: " + str(audio_output))
                print("Text saved: " + str(text_output))
                print("Audio duration: {:.2f} seconds".format(len(audio_data) / sample_rate))

                if self.config.enable_play:
                    print()
                    print("Playing audio...")
                    self.client.audio_player.play(audio_data)
                    print("Playback complete")
        else:
            print("[FAIL] Test failed")
            return False

        return True

    def test_interactive_chat(self, max_turns: int = 10):
        """Interactive multi-turn dialogue"""
        print("=" * 60)
        print("Interactive Multi-turn Dialogue")
        print("=" * 60)

        if not self.client.check_server_status():
            return False

        print("[OK] Server connected")
        print("System prompt: " + self.config.system_prompt)
        print("Max turns: " + str(max_turns))
        print("-" * 60)
        print("Tip: Enter 'quit' to exit")
        print("-" * 60)

        self.client.conversation_history = []

        for turn in range(1, max_turns + 1):
            print()
            print("=" * 60)
            print("Turn " + str(turn))
            print("=" * 60)

            user_input = input("You: ").strip()

            if user_input.lower() == "quit":
                print()
                print("Goodbye!")
                break

            if not user_input:
                print("[WARNING] Please enter some text")
                continue

            print("Waiting for response...")
            start_time = time.time()
            text_response, audio_data, sample_rate = self.client.chat(
                text=user_input,
                audio_path=None,
                modalities=["audio"],
            )
            elapsed = time.time() - start_time

            if text_response or len(audio_data) > 0:
                print()
                print("Qwen3-Omni:")
                print("-" * 40)
                if text_response:
                    print(text_response)
                else:
                    print("(Audio only response)")

                print()
                print("Response time: {:.2f} seconds".format(elapsed))

                if len(audio_data) > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_output = self.output_dir / "turn_{}_{}.wav".format(turn, timestamp)
                    text_output = self.output_dir / "turn_{}_{}.txt".format(turn, timestamp)
                    sf.write(audio_output, audio_data, sample_rate)
                    with open(text_output, "w", encoding="utf-8") as f:
                        f.write(text_response)
                    print()
                    print("Audio saved: " + str(audio_output))
                    print("Text saved: " + str(text_output))
                    print("Audio duration: {:.2f} seconds".format(len(audio_data) / sample_rate))

                    if self.config.enable_play:
                        print()
                        print("Playing audio...")
                        self.client.audio_player.play(audio_data)
                        print("Playback complete")
            else:
                print("[FAIL] Dialogue failed, please try again")

        print()
        print("=" * 60)
        print("Dialogue History Summary")
        print("=" * 60)
        print("Total dialogue turns: " + str(len(self.client.conversation_history) // 2))

        return True

    def test_batch_dialogue(self, prompts: List[str]):
        """Batch dialogue test"""
        print("=" * 60)
        print("Batch Dialogue Test")
        print("=" * 60)

        if not self.client.check_server_status():
            return False

        print("[OK] Server connected")
        print("System prompt: " + self.config.system_prompt)
        print("Dialogue turns: " + str(len(prompts)))
        print("-" * 60)

        self.client.conversation_history = []

        for turn, prompt in enumerate(prompts, 1):
            print()
            print("=" * 60)
            print("Turn " + str(turn))
            print("=" * 60)
            print("Input: " + prompt)
            print("Waiting for response...")

            start_time = time.time()
            text_response, audio_data, sample_rate = self.client.chat(
                text=prompt,
                audio_path=None,
                modalities=["audio"],
            )
            elapsed = time.time() - start_time

            if text_response or len(audio_data) > 0:
                print()
                print("Response:")
                print("-" * 40)
                if text_response:
                    print(text_response)
                else:
                    print("(Audio only response)")

                print()
                print("Response time: {:.2f} seconds".format(elapsed))

                if len(audio_data) > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_output = self.output_dir / "batch_{}_{}.wav".format(turn, timestamp)
                    text_output = self.output_dir / "batch_{}_{}.txt".format(turn, timestamp)
                    sf.write(audio_output, audio_data, sample_rate)
                    with open(text_output, "w", encoding="utf-8") as f:
                        f.write(text_response)
                    print()
                    print("Audio saved: " + str(audio_output))
                    print("Text saved: " + str(text_output))
                    print("Audio duration: {:.2f} seconds".format(len(audio_data) / sample_rate))

                    if self.config.enable_play:
                        print()
                        print("Playing audio...")
                        self.client.audio_player.play(audio_data)
                        print("Playback complete")
            else:
                print("[FAIL] Dialogue failed, please try again")

        print()
        print("=" * 60)
        print("Dialogue History Summary")
        print("=" * 60)
        print("Total dialogue turns: " + str(len(self.client.conversation_history) // 2))
        has_context = "Yes" if len(self.client.conversation_history) > 0 else "No"
        print("Context preserved: " + has_context)

        return True

    def close(self):
        """Close tester"""
        self.client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-Omni Non-streaming Audio Chat Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test
  conda activate asr && python test_omni_streaming.py

  # Interactive dialogue
  conda activate asr && python test_omni_streaming.py --interactive

  # Batch dialogue
  conda activate asr && python test_omni_streaming.py --dialogue --prompts "Hello,Introduce yourself,Say goodbye in English"

  # Disable playback
  conda activate asr && python test_omni_streaming.py --no-play --interactive
        """,
    )

    parser.add_argument(
        "--server-url",
        type=str,
        default="http://10.246.56.85:8091/v1",
        help="API server URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/vepfs/public/model-public/Qwen3-Omni-30B-A3B-Instruct",
        help="Model path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Audio sample rate",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a friendly voice assistant. Please respond concisely and naturally.",
        help="System prompt",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive dialogue mode",
    )
    parser.add_argument(
        "--dialogue",
        action="store_true",
        help="Enable batch dialogue mode (use with --prompts)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Batch dialogue prompts, comma-separated",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Max dialogue turns in interactive mode",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Disable audio playback",
    )

    args = parser.parse_args()

    config = Config()
    config.server_url = args.server_url
    config.model = args.model
    config.output_dir = args.output_dir
    config.sample_rate = args.sample_rate
    config.system_prompt = args.system_prompt
    config.enable_play = not args.no_play

    print("=" * 60)
    print("Qwen3-Omni Non-streaming Audio Chat Test")
    print("=" * 60)
    print("Model: " + config.model)
    print("Server: " + config.server_url)
    print("Output dir: " + config.output_dir)
    print("Sample rate: " + str(config.sample_rate) + " Hz")
    if not config.enable_play:
        print("[WARNING] Audio playback disabled")
    print("=" * 60)

    tester = AudioChatTester(config)

    try:
        if args.dialogue and args.prompts:
            prompts = [p.strip() for p in args.prompts.split(",")]
            tester.test_batch_dialogue(prompts=prompts)
        elif args.interactive:
            tester.test_interactive_chat(max_turns=args.max_turns)
        else:
            tester.test_basic_chat()
    finally:
        tester.close()


if __name__ == "__main__":
    main()
