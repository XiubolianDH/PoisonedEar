from __future__ import annotations

from defense.model_types import ModelInput


def build_context(captions: list[str]) -> str:
    if not captions:
        return "No retrieved context."
    lines = [f"{idx}. {caption}" for idx, caption in enumerate(captions, start=1)]
    return "\n".join(lines)


def build_prompt(user_query: str, context: str) -> str:
    return (
        "You are given the original audio clip, the user's question, and retrieved audio captions.\n"
        "Use the audio as primary evidence and the retrieved captions as supporting evidence.\n\n"
        f"User Query:\n{user_query}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        "Answer the question directly and concisely."
    )


def build_model_input(audio_path: str, user_query: str, context_captions: list[str]) -> ModelInput:
    context = build_context(context_captions)
    prompt = build_prompt(user_query=user_query, context=context)
    return ModelInput(audio_path=audio_path, user_query=user_query, retrieved_context=context, prompt=prompt)
