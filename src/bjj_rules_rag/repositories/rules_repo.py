import psycopg

from bjj_rules_rag.models import Rule, RuleMatch


def insert_rule(conn: psycopg.Connection, rule: Rule) -> int:
    """Insert a rule chunk and return the id assigned by the database."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO rules (chunk_text, source_section, chunk_index, embedding)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (rule.chunk_text, rule.source_section, rule.chunk_index, rule.embedding),
        )
        row = cur.fetchone()
        assert row is not None
        return row[0]


def search_rule(conn: psycopg.Connection, vector: list[float], top_k: int = 5) -> list[RuleMatch]:
    """Return the `top_k` chunks closest to `vector` by cosine distance."""
    with conn.cursor() as cur:
        # A plain list is sent as double precision[]; the cast makes `<=>` resolve to
        # pgvector's cosine-distance operator, which the HNSW index accelerates.
        cur.execute(
            """
            SELECT id, chunk_text, source_section, chunk_index,
                   embedding <=> %(vector)s::vector AS distance
            FROM rules
            ORDER BY embedding <=> %(vector)s::vector
            LIMIT %(top_k)s
            """,
            {"vector": vector, "top_k": top_k},
        )
        return [RuleMatch(*row) for row in cur.fetchall()]
