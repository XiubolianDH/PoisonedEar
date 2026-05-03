from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass
class CaptionPerplexityResult:
    text: str
    perplexity: float


class BasePerplexityScorer:
    backend = "base"

    def score_text(self, text: str) -> float:
        raise NotImplementedError

    def score_many(self, texts: list[str]) -> list[CaptionPerplexityResult]:
        return [CaptionPerplexityResult(text=text, perplexity=self.score_text(text)) for text in texts]


class HuggingFacePerplexityScorer(BasePerplexityScorer):
    backend = "huggingface"

    def __init__(self, model_name: str = "gpt2", device: str | None = None, max_length: int = 512) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def score_text(self, text: str) -> float:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with self.torch.inference_mode():
            outputs = self.model(**encoded, labels=encoded["input_ids"])
        return float(math.exp(float(outputs.loss)))


class HeuristicPerplexityScorer(BasePerplexityScorer):
    backend = "heuristic"

    def score_text(self, text: str) -> float:
        tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
        if not tokens:
            return 0.0
        unique_ratio = len(set(tokens)) / float(len(tokens))
        avg_token_length = sum(len(token) for token in tokens) / float(len(tokens))
        punctuation_count = len(re.findall(r"[,:;()\[\]{}]", text))
        uppercase_ratio = sum(1 for ch in text if ch.isupper()) / float(max(len(text), 1))
        return float((unique_ratio * 25.0) + avg_token_length + (punctuation_count * 1.5) + (uppercase_ratio * 10.0))


def build_perplexity_scorer(backend: str, model_name: str) -> BasePerplexityScorer:
    normalized = backend.lower()
    if normalized == "heuristic":
        return HeuristicPerplexityScorer()
    if normalized == "auto":
        try:
            return HuggingFacePerplexityScorer(model_name=model_name)
        except Exception:
            return HeuristicPerplexityScorer()
    if normalized == "huggingface":
        return HuggingFacePerplexityScorer(model_name=model_name)
    raise ValueError(f"Unsupported perplexity backend: {backend}")
