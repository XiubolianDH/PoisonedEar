from __future__ import annotations

import os
from dataclasses import dataclass

from defense.types import DefenseResult, RetrievedCaption


@dataclass
class DefenseRuntimeConfig:
    paraphrase_backend: str = "rule_based"
    paraphrase_model: str = "gpt-4o-mini"
    paraphrase_api_env: str = "OPENAI_API_KEY"
    ppl_backend: str = "huggingface"
    ppl_model: str = "gpt2"
    ppl_threshold: float = 80.0


class BaseDefense:
    key = "base"

    def apply(self, user_query: str, retrieved_list: list[RetrievedCaption]) -> DefenseResult:
        raise NotImplementedError


class NoDefense(BaseDefense):
    key = "no_defense"

    def apply(self, user_query: str, retrieved_list: list[RetrievedCaption]) -> DefenseResult:
        return DefenseResult(
            defense_name=self.key,
            rewritten_query=user_query,
            kept_contexts=list(retrieved_list),
            removed_contexts=[],
            metadata={},
        )


class RuleBasedParaphraser:
    def paraphrase(self, text: str) -> str:
        stripped = text.strip()
        if stripped.lower() == "what is this sound?":
            return "Please identify the sound in this audio clip."
        if stripped.endswith("?"):
            return f"Please answer this question about the audio: {stripped[:-1]}."
        return f"Please answer this audio question in other words: {stripped}"


class OpenAIParaphraser:
    def __init__(self, model_name: str, api_key_env: str) -> None:
        from openai import OpenAI

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing OpenAI API key in env var {api_key_env}")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def paraphrase(self, text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": "Rewrite the user's query in natural English without changing intent. Return only the rewritten query.",
                },
                {"role": "user", "content": text},
            ],
        )
        return (response.choices[0].message.content or text).strip()


class ParaphrasingDefense(BaseDefense):
    key = "paraphrasing"

    def __init__(self, config: DefenseRuntimeConfig) -> None:
        self.config = config
        if config.paraphrase_backend == "openai":
            self.paraphraser = OpenAIParaphraser(config.paraphrase_model, config.paraphrase_api_env)
        else:
            self.paraphraser = RuleBasedParaphraser()

    def apply(self, user_query: str, retrieved_list: list[RetrievedCaption]) -> DefenseResult:
        rewritten_query = self.paraphraser.paraphrase(user_query)
        return DefenseResult(
            defense_name=self.key,
            rewritten_query=rewritten_query,
            kept_contexts=list(retrieved_list),
            removed_contexts=[],
            metadata={"paraphrase_backend": self.config.paraphrase_backend},
        )


class PerplexityFilteringDefense(BaseDefense):
    key = "perplexity_filtering"

    def __init__(self, config: DefenseRuntimeConfig) -> None:
        self.config = config
        from defense.perplexity import build_perplexity_scorer

        self.scorer = build_perplexity_scorer(backend=config.ppl_backend, model_name=config.ppl_model)

    def apply(self, user_query: str, retrieved_list: list[RetrievedCaption]) -> DefenseResult:
        scored = self.scorer.score_many([item.text for item in retrieved_list])
        kept_contexts: list[RetrievedCaption] = []
        removed_contexts: list[RetrievedCaption] = []
        for item, scored_item in zip(retrieved_list, scored):
            item.ppl = scored_item.perplexity
            if scored_item.perplexity > self.config.ppl_threshold:
                removed_contexts.append(item)
            else:
                kept_contexts.append(item)
        return DefenseResult(
            defense_name=self.key,
            rewritten_query=user_query,
            kept_contexts=kept_contexts,
            removed_contexts=removed_contexts,
            metadata={
                "ppl_backend": getattr(self.scorer, "backend", self.config.ppl_backend),
                "ppl_model": self.config.ppl_model,
                "ppl_threshold": self.config.ppl_threshold,
            },
        )


def build_defense(defense_key: str, config: DefenseRuntimeConfig) -> BaseDefense:
    if defense_key == "no_defense":
        return NoDefense()
    if defense_key == "paraphrasing":
        return ParaphrasingDefense(config)
    if defense_key == "perplexity_filtering":
        return PerplexityFilteringDefense(config)
    raise KeyError(f"Unknown defense key: {defense_key}")
