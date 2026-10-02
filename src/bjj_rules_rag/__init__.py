from pathlib import Path

from bjj_rules_rag.chunking import chunk_rules, format_for_embedding
from bjj_rules_rag.db import get_connection
from bjj_rules_rag.models import Rule, RuleChunk
from bjj_rules_rag.ollama_embed import embed_text
from bjj_rules_rag.repositories.rules_repo import insert_rule, search_rule
from openai import OpenAI
import sys


RULES_PATH = Path(__file__).resolve().parents[2] / "IBJJF_RULES.txt"
client = OpenAI()

def ask() -> None:
    if len(sys.argv) < 2:
        print('Usage: bjj-ask "your question"')
        sys.exit(1)
    print(user_query(" ".join(sys.argv[1:])))

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

def user_query(prompt: str) -> str | None:
    embedding = embed_text(f"search_query: {prompt}")

    with get_connection() as conn:
        results = search_rule(conn, embedding)

    context = "\n\n---\n\n".join(
        f"[{i}] {chunk.source_section}\n{chunk.chunk_text}"
        for i, chunk in enumerate(results, start=1)
    )
    # print(context)

    system_prompt = (
        "You answer questions about the IBJJF Jiu-Jitsu rules using only the numbered "
        "rule excerpts in the context below.\n\n"
        "How to answer:\n"
        "- Use the excerpts as your only source of facts, but you may combine excerpts "
        "and draw reasonable conclusions from them.\n"
        "- After each claim, cite the excerpt(s) it comes from, like [2] or [1][4].\n"
        "- End with a 'Sources:' line listing the section of each excerpt you cited.\n"
        "- If the excerpts cover the topic only partially, answer what they support and "
        "say what is missing.\n"
        "- Only if no excerpt is relevant to the question, reply exactly: "
        "'I do not know based on the context provided.'\n\n"
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

    answer = response.choices[0].message.content
    if answer is None:
        raise RuntimeError("Model returned no content")

    print(answer)
    return answer
