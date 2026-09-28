import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """Open a connection to the rules database.

    Registers the pgvector type adapter so `list[float]` values can be
    sent and read back as `vector` columns directly.
    """
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        yield conn
