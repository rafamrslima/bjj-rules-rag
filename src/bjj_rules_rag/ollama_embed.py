import os

import httpx
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_URL = os.environ["OLLAMA_API_EMBEDDING"]


def embed_text(text: str, model: str = "nomic-embed-text") -> list[float]:
    response = httpx.post(
        EMBEDDING_URL,
        json={"model": model, "prompt": text},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["embedding"]

def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = start + chunk_size - overlap   # step forward, minus overlap
    return chunks