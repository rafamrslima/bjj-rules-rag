# bjj-rules-rag

A retrieval-augmented question-answering system for Brazilian Jiu-Jitsu competition rules. Every answer is grounded in, and cited from, a specific published ruleset ([IBJJF Rules Book, version / date]).

```
Q: Is a heel hook legal in the adult brown belt gi division?
A: [example answer from your system, with its citations, e.g. "No. ... [IBJJF Rules, §X.Y, p. N]"]
```

## Why RAG

BJJ rules are public, and a capable base LLM will answer many rules questions from memory. That is the problem this project is built around, not a reason to skip retrieval.

Answers from memory fail in ways that matter here. They are often wrong on specifics: point values, time limits, which submissions are legal at which belt and age division. They come without sources, so a referee, coach or athlete cannot check them. And they mix rulesets: IBJJF, ADCC, UAEJJF and various submission-only formats differ on exactly the questions people ask (heel hooks, slams, stalling penalties, advantages), and a model recalling "BJJ rules" in general will blend them without saying so.

So the value of RAG here is not access to hidden information. It is that the answer is:

- **Scoped** to one named ruleset and version, not to whatever the model remembers.
- **Cited** to the section and page it came from, so it can be checked in seconds.
- **Measured**, with faithfulness to the retrieved text scored on a fixed evaluation set, so the claim that answers are grounded rests on numbers, not anecdotes.

The goal is an answer you can verify rather than one you have to trust. A system that answers from the rulebook and says when the rulebook does not cover a question is more useful than a more fluent system that is sometimes confidently wrong.

## Architecture

```
PDF ruleset ──► ingestion ──► chunking ──► embeddings ──► vector store
                                                              │
question ──► query embedding ──► hybrid retrieval (dense + BM25) ──► reranker ──► top-k chunks
                                                                                     │
                                                              LLM generation with citations ◄┘
```

**Ingestion.** The source is the official [IBJJF rules PDF, version / date]. Text is extracted with [PDF parser, e.g. PyMuPDF / unstructured / docling], keeping section numbers, headings and page numbers as metadata, because the citations depend on them. Tables (for example, the legal-techniques-by-division table) are [how tables are handled: extracted as structured rows / serialized to text / kept as separate chunks].

**Chunking.** Chunks follow the document's own structure: [strategy, e.g. one chunk per numbered article, split further at ~N tokens with M-token overlap]. Rulebooks are written as numbered articles that are meant to be read on their own, so structure-aware chunking keeps a rule and its exceptions together better than fixed-size windows do. Each chunk carries `section`, `title`, `page` and `ruleset_version` metadata.

**Embeddings.** [Embedding model, e.g. `nomic-embed-text` via Ollama / `bge-m3` / hosted model], chosen for [reason: runs locally, retrieval benchmark performance, context length, cost].

**Vector store.** [pgvector / Qdrant]. [Reason, e.g. "pgvector keeps vectors, metadata and full-text search in one Postgres instance, which is enough at the scale of one rulebook and simplifies hybrid search" or "Qdrant for native hybrid/sparse vectors and payload filtering"].

**Hybrid retrieval.** Dense retrieval alone misses exact terms that matter in this domain: technique names ("heel hook", "reaping", "slicer"), division names and article numbers. Dense results are combined with keyword search ([BM25 / Postgres full-text / sparse vectors]) using [fusion method, e.g. reciprocal rank fusion], retrieving [N] candidates.

**Reranking.** Candidates are reranked with [reranker, e.g. `bge-reranker-v2-m3` cross-encoder] and the top [k] go to the generator. The cross-encoder reads the query and passage together, which fixes the most common failure of the first stage: passages that share vocabulary with the question but concern a different division or situation.

**Generation.** [LLM, e.g. a local model via Ollama or a hosted model] is told to answer only from the supplied passages, to cite each claim as [citation format, e.g. `[§X.Y, p. N]`], and to say the ruleset does not address the question when the passages do not support an answer. Citations are [checked after generation against the retrieved chunk IDs / not yet validated automatically].

## Evaluation

Evaluation is the core of this project. Without it, "grounded" is a claim; with it, it is a measurement that can go up or down when the pipeline changes.

### Test set

The golden set has [N] question–answer pairs written by hand against the rulebook, each with the reference answer and the section(s) that support it. It is designed to cover the kinds of questions where retrieval and memory both fail:

- **Direct lookups**: point values, match durations, and similar single-article facts.
- **Division-conditional rules**: legality of techniques by belt, age and gi/no-gi, where the answer depends on a table or a cross-reference.
- **Multi-hop questions**: answers that need two or more articles, such as a penalty plus its escalation rule.
- **Ruleset confusion traps**: questions whose answer differs between IBJJF and other rulesets, to catch answers drawn from memory.
- **Out-of-scope questions**: questions the rulebook does not answer, where the correct response is to say so.

The set lives in [`path/to/golden_set.jsonl`].

### Metrics

Scored with [RAGAS, version]:

| Metric | What it measures |
|---|---|
| Faithfulness | Share of claims in the answer that are supported by the retrieved context. The main guard against hallucination. |
| Answer relevancy | Whether the answer addresses the question that was asked. |
| Context precision | Whether the relevant chunks are ranked near the top of what was retrieved. |
| Context recall | Whether the retrieved context contains everything needed for the reference answer. |

The judge model is [judge LLM]. LLM-judged metrics are noisy, so [number of runs / seed / temperature settings] and [whether a sample of judgments was checked by hand].

### Results

[Describe the specific change being measured, e.g. "Adding hybrid BM25 retrieval and cross-encoder reranking on top of dense-only retrieval."]

| Configuration | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| Baseline: [e.g. dense-only, fixed-size chunks] | [x.xx] | [x.xx] | [x.xx] | [x.xx] |
| [Change, e.g. + hybrid + rerank] | [x.xx] | [x.xx] | [x.xx] | [x.xx] |

[One or two sentences interpreting the result: which metric moved, why that change plausibly caused it, and any metric that got worse.]

For reference, the same golden set answered by [base LLM] with no retrieval: [accuracy / notes on how many answers were wrong or mixed rulesets]. [Include only if you actually ran this comparison.]

To reproduce:

```bash
[command to run the evaluation, e.g. uv run bjj-rules-rag eval --set data/golden_set.jsonl]
```

## Tech stack

- **Language:** Python 3.14, managed with [uv](https://docs.astral.sh/uv/)
- **PDF parsing:** [library]
- **Embeddings:** [model]
- **Vector store:** [pgvector on PostgreSQL N / Qdrant]
- **Keyword search:** [BM25 implementation / Postgres full-text]
- **Reranker:** [model]
- **LLM:** [model(s)]; local inference via [Ollama](https://ollama.com)
- **Orchestration:** [framework, or "plain Python, no framework"]
- **Evaluation:** [RAGAS]
- **Interface:** [CLI / FastAPI / Streamlit]
- **Infrastructure:** Docker Compose

## Setup

### Prerequisites

- Docker and Docker Compose
- [uv](https://docs.astral.sh/uv/) (for running outside Docker)
- [Ollama](https://ollama.com), for fully local inference

### Running locally with Ollama

The whole pipeline can run on your machine with no API keys and no cost, using Ollama for embeddings and generation.

```bash
git clone [repo URL]
cd bjj-rules-rag

# Pull local models
ollama pull [embedding model]
ollama pull [generation model]

# Start the vector store (and app, if containerized)
docker compose up -d

# Configure
cp .env.example .env   # [set LLM_PROVIDER=ollama, model names, DB URL]

# Ingest the ruleset
[ingest command, e.g. uv run bjj-rules-rag ingest data/ibjjf_rules.pdf]

# Ask a question
[query command, e.g. uv run bjj-rules-rag ask "Is slamming allowed?"]
```

### Using a hosted model

[Optional: which environment variables switch generation or embeddings to a hosted provider.]

## Failure modes and limitations

**Retrieval misses.** If the article that answers a question is not in the top-k after reranking, the model either declines or answers from a partial context. Context recall on the golden set is the direct measure of this. Questions that depend on tables and footnotes are the weakest area [confirm against your results].

**Cross-references.** Rules often say "see Article X". The pipeline does not follow these references, so multi-hop answers depend on both articles being retrieved independently.

**Faithful is not the same as correct.** Faithfulness measures whether the answer matches the retrieved text, not whether it is right. A faithful answer built on the wrong passage (for example, the juvenile division's rule when the adult rule was asked about) scores well on faithfulness and is still wrong. Context precision and the division-conditional questions in the golden set exist to catch this, but not perfectly.

**Ruleset versioning.** The index holds one version of one ruleset. IBJJF revises its rules periodically; a production system would need versioned ingestion, a version filter at query time, and answers that state which version they cite.

**Evaluation coverage.** [N] hand-written questions do not cover the long tail of edge cases that come up at real events. The metrics are a regression signal for this set, not a guarantee.

**Security is out of scope.** The system ingests one trusted PDF. It does not defend against prompt injection in source documents or user queries, SSRF or untrusted document ingestion, or abuse of a public endpoint. Accepting user-supplied documents or exposing a public API would require input sanitization, isolation of retrieved content from instructions, rate limiting and authentication.

**Not an official source.** Answers are a reading aid. Refereeing decisions follow the official rulebook and the referees on the day.

## Roadmap

- Follow in-document cross-references during retrieval.
- Index multiple rulesets (ADCC, UAEJJF) with explicit ruleset filtering, and extend the golden set with cross-ruleset questions.
- Versioned ingestion with change detection between rulebook releases.
- Automatic citation validation: reject or flag answers whose citations do not match retrieved chunks.
- Run evaluation in CI to catch regressions from prompt, model or chunking changes.
- [Your own items.]

## License

[License]
