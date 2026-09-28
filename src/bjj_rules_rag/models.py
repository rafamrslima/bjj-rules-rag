from dataclasses import dataclass


@dataclass
class Rule:
    """A single chunk of the rulebook, ready to be embedded and stored.

    `id` is left unset (None) for a rule that has not been inserted yet;
    the database assigns it on insert.
    """

    chunk_text: str
    source_section: str
    chunk_index: int
    embedding: list[float]
    id: int | None = None
