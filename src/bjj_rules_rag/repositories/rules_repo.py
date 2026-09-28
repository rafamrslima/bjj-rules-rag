import psycopg

from bjj_rules_rag.models import Rule


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
