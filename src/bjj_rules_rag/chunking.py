import math
import re
from dataclasses import dataclass

from bjj_rules_rag.models import RuleChunk

BOOKLETS = frozenset(
    {"RULES BOOK", "GENERAL COMPETITION GUIDELINES", "COMPETITION FORMAT MANUAL"}
)
END_MARKER = "2014-2024©INTERNATIONAL BRAZILIAN"

ARTICLE_RE = re.compile(r"^ARTICLE \d+\b")
NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+)+) ")
LETTERED_RE = re.compile(r"^([A-Z])\) ")
LABEL_ONLY_RE = re.compile(r"^[\d\s]+$")

HEADING_MAX_CHARS = 80
MAX_UNIT_CHARS = 1500
SECTION_SEP = " > "

# Lines that must stay in the same part as the line before them when a unit is split.
KEEP_WITH_PREVIOUS = ("GESTURE:", "VERBAL COMMAND:", "Note", "Obs:", "OBS:", "Ex:", "*")


@dataclass
class _Line:
    kind: str  # "booklet" | "article" | "numbered" | "lettered" | "text"
    text: str
    number: str | None = None


def chunk_rules(content: str) -> list[RuleChunk]:
    units = _build_units(_classify(content))
    return [part for unit in units for part in _split_oversized(unit)]


def format_for_embedding(chunk: RuleChunk) -> str:
    return f"search_document: [{chunk.source_section}]\n{chunk.text}"


def _classify(content: str) -> list[_Line]:
    lines: list[_Line] = []
    for raw in content.splitlines():
        text = raw.strip()
        if text.startswith(END_MARKER):
            break
        # Blank lines and bare figure labels ("1 2", "4 5 6") carry no meaning.
        if not text or LABEL_ONLY_RE.match(text):
            continue
        if text in BOOKLETS:
            lines.append(_Line("booklet", text))
        elif ARTICLE_RE.match(text):
            lines.append(_Line("article", text))
        elif m := NUMBERED_RE.match(text):
            lines.append(_Line("numbered", text, m.group(1)))
        elif m := LETTERED_RE.match(text):
            lines.append(_Line("lettered", text, m.group(1)))
        else:
            lines.append(_Line("text", text))
    return lines


def _is_heading(lines: list[_Line], i: int) -> bool:
    """A short numbered line followed by its own children: '1.1 Authority of Referee' -> 1.1.1,
    or '6.2.2 Serious Fouls' -> A), B), ..."""
    line = lines[i]
    if len(line.text) > HEADING_MAX_CHARS:
        return False
    for nxt in lines[i + 1 :]:
        if nxt.kind in ("booklet", "article"):
            return False
        if nxt.kind == "lettered":
            return True
        if nxt.kind == "numbered":
            return nxt.number.startswith(line.number + ".")
    return False


def _build_units(lines: list[_Line]) -> list[RuleChunk]:
    units: list[RuleChunk] = []
    booklet: str | None = None
    article: str | None = None
    # Ancestors of the current rule: (number, label). Headings are labelled with
    # their title ("6.2.2 Serious Fouls"), plain rules with their number ("3.1").
    stack: list[tuple[str, str]] = []
    section: str | None = None
    body: list[str] = []

    def flush() -> None:
        if section and body:
            units.append(RuleChunk(text="\n".join(body), source_section=section))

    def path(*extra: str) -> str:
        parts = [booklet, article, *(label for _, label in stack), *extra]
        return SECTION_SEP.join(p for p in parts if p)

    for i, line in enumerate(lines):
        if line.kind == "booklet":
            flush()
            booklet, article, stack, section, body = line.text, None, [], None, []
        elif line.kind == "article":
            flush()
            article, stack, body = line.text, [], []
            section = path()
        elif line.kind == "numbered":
            flush()
            while stack and not line.number.startswith(stack[-1][0] + "."):
                stack.pop()
            heading = _is_heading(lines, i)
            stack.append((line.number, line.text if heading else line.number))
            section = path()
            body = [] if heading else [line.text]
        elif line.kind == "lettered":
            flush()
            section = path(f"{line.number})")
            body = [line.text]
        elif section:
            body.append(line.text)
    flush()
    return units


def _split_oversized(unit: RuleChunk) -> list[RuleChunk]:
    if len(unit.text) <= MAX_UNIT_CHARS:
        return [unit]

    groups: list[list[str]] = []
    for line in unit.text.split("\n"):
        if groups and line.startswith(KEEP_WITH_PREVIOUS):
            groups[-1].append(line)
        else:
            groups.append([line])

    # Aim for evenly sized parts so the last one isn't a tiny leftover.
    target = len(unit.text) / math.ceil(len(unit.text) / MAX_UNIT_CHARS)
    parts: list[list[str]] = []
    current: list[str] = []
    for group in groups:
        current.extend(group)
        if len("\n".join(current)) >= target:
            parts.append(current)
            current = []
    if current:
        parts.append(current)

    total = len(parts)
    return [
        RuleChunk(
            text="\n".join(part),
            source_section=f"{unit.source_section} (part {n}/{total})",
        )
        for n, part in enumerate(parts, start=1)
    ]
