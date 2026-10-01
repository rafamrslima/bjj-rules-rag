from pathlib import Path

from bjj_rules_rag.chunking import chunk_rules, format_for_embedding
from bjj_rules_rag.db import get_connection
from bjj_rules_rag.models import Rule, RuleChunk
from bjj_rules_rag.ollama_embed import embed_text
from bjj_rules_rag.repositories.rules_repo import insert_rule, search_rule
from openai import OpenAI


RULES_PATH = Path(__file__).resolve().parents[2] / "IBJJF_RULES.txt"
client = OpenAI()

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

def user_query(prompt: str) -> str:
    embedding = embed_text(prompt)

    with get_connection() as conn:
        results = search_rule(conn, embedding)

    context = "\n".join([chunk.source_section + ":" + chunk.chunk_text for chunk in results])

    system_prompt = (
    "You are a helpful assistant. Use the provided context to answer the user's question. "
    "If the answer cannot be found in the context, say 'I do not know based on the context provided.'\n\n"
    f"Context:\n{context}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0
    )
    print(response.choices[0].message.content)

if __name__ == "__main__": 
    # main()
    user_query("Heel hook is allowed in white belts?")
