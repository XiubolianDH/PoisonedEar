from __future__ import annotations

from dataclasses import dataclass

from evaluation.metrics import EvaluationBundle, evaluate_query_result
from models.base import ModelInput, ModelOutput, VictimModel
from rag.context import build_context_string
from retriever.base import AudioEmbeddingEncoder, RetrievalHit
from retriever.index import FaissAudioRetriever


@dataclass
class AudioRAGResult:
    generated_response: str
    model_output: ModelOutput
    retrieved_hits: list[RetrievalHit]
    context: str
    metrics: EvaluationBundle


class AudioRAGEngine:
    def __init__(
        self,
        retriever: FaissAudioRetriever,
        encoder: AudioEmbeddingEncoder,
        victim_model: VictimModel,
        rag_config: dict,
        evaluation_config: dict,
    ) -> None:
        self.retriever = retriever
        self.encoder = encoder
        self.victim_model = victim_model
        self.rag_config = rag_config
        self.evaluation_config = evaluation_config

    def run_query(
        self,
        audio_path: str,
        user_query: str,
        expected_answer: str | None = None,
        attack_target: str | None = None,
        poisoned_ids: set[str] | None = None,
        clean_reference_text: str | None = None,
        clean_reference_audio_path: str | None = None,
    ) -> AudioRAGResult:
        top_k = int(self.rag_config["top_k"])
        hits = self.retriever.query(audio_path=audio_path, top_k=top_k)
        context = build_context_string(hits, self.rag_config["context_template"])
        prompt = self.rag_config["prompt_template"].format(user_query=user_query, context=context)
        model_input = ModelInput(
            audio_path=audio_path,
            user_query=user_query,
            retrieved_context=context,
            prompt=prompt,
        )
        model_output = self.victim_model.generate(model_input)
        metrics = evaluate_query_result(
            encoder=self.encoder,
            query_audio_path=audio_path,
            retrieved_hits=hits,
            generated_response=model_output.text,
            expected_answer=expected_answer,
            attack_target=attack_target,
            poisoned_ids=poisoned_ids or set(),
            clean_reference_text=clean_reference_text,
            clean_reference_audio_path=clean_reference_audio_path,
            evaluation_config=self.evaluation_config,
        )
        return AudioRAGResult(
            generated_response=model_output.text,
            model_output=model_output,
            retrieved_hits=hits,
            context=context,
            metrics=metrics,
        )
