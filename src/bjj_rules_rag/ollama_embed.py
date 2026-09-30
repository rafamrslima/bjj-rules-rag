import os

import httpx
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_URL = os.environ["OLLAMA_API_EMBEDDING"]
OLLAMA_MODEL = "nomic-embed-text"

def embed_text(text: str) -> list[float]:
    response = httpx.post(
        EMBEDDING_URL,
        json={"model": OLLAMA_MODEL, "prompt": text},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["embedding"]