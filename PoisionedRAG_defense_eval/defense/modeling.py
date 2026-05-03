from __future__ import annotations

import base64
import os
from pathlib import Path
import re

from defense.registry import ModelSpec
from defense.model_types import ModelInput, ModelOutput, VictimModel


def load_model_config(model_spec: ModelSpec) -> dict:
    provider = model_spec.provider if model_spec.provider != "openai_compatible" else "openai"
    return {
        "models": {
            "victim_provider": provider,
            "victim_model": model_spec.model_name,
            "temperature": 0.0,
            "max_tokens": 256,
            "request_timeout_seconds": 120.0,
            "request_max_retries": 2,
            "openai_api_key_env": "OPENAI_API_KEY",
            "gemini_api_key_env": "GEMINI_API_KEY",
            "qwen_api_base_env": model_spec.api_base_env or "QWEN_API_BASE",
            "qwen_api_key_env": model_spec.api_key_env or "QWEN_API_KEY",
        }
    }


def build_generation_model(model_spec: ModelSpec, backend_override: str = "native") -> VictimModel:
    if backend_override == "local":
        return LocalHeuristicAudioModel(model_name=f"local-{model_spec.key}")
    if backend_override != "native":
        raise ValueError(f"Unsupported generation backend override: {backend_override}")
    if model_spec.provider == "openai_compatible":
        return OpenAICompatibleAudioModel(
            model_name=model_spec.model_name,
            api_base_env=model_spec.api_base_env or "ANYGPT_API_BASE",
            api_key_env=model_spec.api_key_env or "ANYGPT_API_KEY",
        )
    config = load_model_config(model_spec)
    return build_native_generation_model(config["models"])


def _audio_as_base64(audio_path: str) -> tuple[str, str]:
    suffix = Path(audio_path).suffix.lower()
    audio_format = "mp3" if suffix == ".mp3" else "wav"
    encoded = base64.b64encode(Path(audio_path).read_bytes()).decode("utf-8")
    return encoded, audio_format


class OpenAICompatibleAudioModel(VictimModel):
    def __init__(
        self,
        model_name: str,
        api_base_env: str,
        api_key_env: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        from openai import OpenAI

        api_base = os.getenv(api_base_env)
        api_key = os.getenv(api_key_env)
        if not api_base or not api_key:
            raise EnvironmentError(f"Missing connection env vars {api_base_env} or {api_key_env}")
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, model_input: ModelInput) -> ModelOutput:
        encoded_audio, audio_format = _audio_as_base64(model_input.audio_path)
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": encoded_audio, "format": audio_format}},
                        {"type": "text", "text": model_input.prompt},
                    ],
                }
            ],
        )
        text = response.choices[0].message.content or ""
        return ModelOutput(model_name=self.model_name, text=text, raw_response=response)


class OpenAIAudioModel(VictimModel):
    def __init__(
        self,
        model_name: str,
        api_key_env: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        request_timeout_seconds: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        from openai import OpenAI

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing OpenAI API key in env var {api_key_env}")
        self.client = OpenAI(api_key=api_key, timeout=request_timeout_seconds, max_retries=max_retries)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, model_input: ModelInput) -> ModelOutput:
        encoded_audio, audio_format = _audio_as_base64(model_input.audio_path)
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": encoded_audio, "format": audio_format}},
                        {"type": "text", "text": model_input.prompt},
                    ],
                }
            ],
        )
        text = response.choices[0].message.content or ""
        return ModelOutput(model_name=self.model_name, text=text, raw_response=response)


class GeminiAudioModel(VictimModel):
    def __init__(
        self,
        model_name: str,
        api_key_env: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        import google.generativeai as genai

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing Gemini API key in env var {api_key_env}")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model_name)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, model_input: ModelInput) -> ModelOutput:
        response = self.model.generate_content(
            [
                {"mime_type": "audio/wav", "data": Path(model_input.audio_path).read_bytes()},
                model_input.prompt,
            ],
            generation_config={"temperature": self.temperature, "max_output_tokens": self.max_tokens},
        )
        text = ""
        try:
            text = response.text or ""
        except ValueError:
            candidates = getattr(response, "candidates", None) or []
            parts = []
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                for part in getattr(content, "parts", []) or []:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        parts.append(part_text)
            text = "\n".join(parts)
        return ModelOutput(model_name=self.model_name, text=text, raw_response=response)


class LocalHeuristicAudioModel(VictimModel):
    def __init__(self, model_name: str = "local-heuristic") -> None:
        self.model_name = model_name

    def generate(self, model_input: ModelInput) -> ModelOutput:
        lines = [line.strip() for line in model_input.retrieved_context.splitlines() if line.strip()]
        candidates = []
        for line in lines:
            candidate = re.sub(r"^\d+\.\s*", "", line).strip()
            if candidate.lower().startswith("context:"):
                continue
            if candidate:
                candidates.append(candidate)
        if candidates:
            response = (
                f"Based on the retrieved audio context, the sound is most likely: {candidates[0]}. "
                f"Supporting retrieved evidence also includes {len(candidates)} context item(s)."
            )
        else:
            response = "No retrieved context was available, so I cannot confidently identify the sound."
        return ModelOutput(model_name=self.model_name, text=response, raw_response=None)


def build_native_generation_model(config: dict) -> VictimModel:
    provider = config["victim_provider"].lower()
    model_name = config["victim_model"]
    temperature = float(config.get("temperature", 0.0))
    max_tokens = int(config.get("max_tokens", 256))
    request_timeout_seconds = float(config.get("request_timeout_seconds", 120.0))
    max_retries = int(config.get("request_max_retries", 2))
    if provider == "openai":
        return OpenAIAudioModel(
            model_name=model_name,
            api_key_env=config["openai_api_key_env"],
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout_seconds=request_timeout_seconds,
            max_retries=max_retries,
        )
    if provider == "gemini":
        return GeminiAudioModel(
            model_name=model_name,
            api_key_env=config["gemini_api_key_env"],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "qwen":
        return OpenAICompatibleAudioModel(
            model_name=model_name,
            api_base_env=config["qwen_api_base_env"],
            api_key_env=config["qwen_api_key_env"],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "local":
        return LocalHeuristicAudioModel(model_name=model_name)
    raise ValueError(f"Unsupported victim provider: {provider}")
