from pathlib import Path

from bjj_rules_rag.chunking import chunk_rules, format_for_embedding
from bjj_rules_rag.db import get_connection
from bjj_rules_rag.models import Rule, RuleChunk
from bjj_rules_rag.ollama_embed import embed_text
from bjj_rules_rag.repositories.rules_repo import insert_rule

RULES_PATH = Path(__file__).resolve().parents[2] / "IBJJF_RULES.txt"


def main() -> None:
    print("Hello from bjj-rules-rag!")
    insert_chunks()


def insert_chunks() -> None:
    content = RULES_PATH.read_text(encoding="utf-8")
    chunks: list[RuleChunk] = chunk_rules(content)
    print(chunks)

    with get_connection() as conn:
        for idx, chunk in enumerate(chunks):
            embedding = embed_text(format_for_embedding(chunk))
            rule = Rule(
                chunk_text=chunk.text,
                source_section=chunk.source_section,
                chunk_index=idx,
                embedding=embedding,
            )
            insert_rule(conn, rule)
            print(f"[{idx + 1}/{len(chunks)}] {chunk.source_section}")

main()
