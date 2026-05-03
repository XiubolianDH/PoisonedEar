from __future__ import annotations

from retriever.base import RetrievalHit


def build_context_string(hits: list[RetrievalHit], template: str) -> str:
    item_lines = [f"{index}. {hit.sample.text}" for index, hit in enumerate(hits, start=1)]
    return template.format(items="\n".join(item_lines))

