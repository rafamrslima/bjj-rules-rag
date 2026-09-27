CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rules (
    id             BIGSERIAL PRIMARY KEY,
    chunk_text     TEXT        NOT NULL,
    source_section TEXT        NOT NULL,
    chunk_index    INTEGER     NOT NULL,
    embedding      VECTOR(768) NOT NULL
);

-- Approximate nearest-neighbour index for cosine-distance search (<=>).
CREATE INDEX IF NOT EXISTS rules_embedding_idx
    ON rules USING hnsw (embedding vector_cosine_ops);
