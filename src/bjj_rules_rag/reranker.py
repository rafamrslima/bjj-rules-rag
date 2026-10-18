from functools import lru_cache

from sentence_transformers import CrossEncoder

from bjj_rules_rag.models import RuleMatch

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    # Loading takes a few seconds, so do it once per process, on first use.
    return CrossEncoder(RERANK_MODEL)


def rerank(query: str, matches: list[RuleMatch], top_n: int) -> list[RuleMatch]:
    """Reorder `matches` by cross-encoder relevance to `query` and keep the best `top_n`."""
    if not matches:
        return []
    # The section label carries headings like "4.4 Mount and Back Mount (4 points)"
    # that aren't in the chunk text, so the reranker needs to see it too.
    pairs = [(query, f"{m.source_section}\n{m.chunk_text}") for m in matches]
    scores = _model().predict(pairs)
    ranked = sorted(zip(scores, matches), key=lambda pair: pair[0], reverse=True)
    return [match for _, match in ranked[:top_n]]
