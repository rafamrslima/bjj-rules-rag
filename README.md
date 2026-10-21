# bjj-rules-rag

A retrieval-augmented question-answering system for Brazilian Jiu-Jitsu competition rules. Every answer is grounded in, and cited from, a specific published ruleset (IBJJF Rules Book, June 2024).

```
Q: How close to the crease of the wrist must an athlete's Gi sleeve reach when their arms are extended forward?
A: An athlete's Gi sleeve must come to no more than 2 cm from the athlete's wrist when the arm is extended
   straight parallel to the ground (8.1.7).

   Sources: 8.1.7
```

## Why RAG

BJJ rules are public, and a capable base LLM will answer many rules questions from memory. That is the problem this project is built around, not a reason to skip retrieval.

Answers from memory fail in ways that matter here. They are often wrong on specifics: point values, time limits, which submissions are legal at which belt and age division. They come without sources, so a referee, coach or athlete cannot check them. And they mix rulesets: IBJJF, ADCC, UAEJJF and various submission-only formats differ on exactly the questions people ask (heel hooks, slams, stalling penalties, advantages), and a model recalling "BJJ rules" in general will blend them without saying so.

So the value of RAG here is not access to hidden information. It is that the answer is:

- **Scoped** to one named ruleset and version, not to whatever the model remembers.
- **Cited** to the numbered rule it came from, so it can be checked in seconds.
- **Measured**, with faithfulness to the retrieved text scored on a fixed evaluation set, so the claim that answers are grounded rests on numbers, not anecdotes.

The goal is an answer you can verify rather than one you have to trust. A system that answers from the rulebook and says when the rulebook does not cover a question is more useful than a more fluent system that is sometimes confidently wrong.

## Architecture

```
PDF ruleset ──► ingestion ──► chunking ──► embeddings ──► vector store
                                                              │
question ──► query embedding ──► dense retrieval (top 20) ──► reranker ──► top-10 chunks
                                                                                │
                                                         LLM generation with citations ◄┘
```

**Ingestion.** The source is the official IBJJF Rules Book PDF (`2024JUN_IBJJF_Rules_EN.pdf`, June 2024). Text is extracted with `pdftotext -layout` (Poppler) and normalized into `IBJJF_RULES.txt`: repeated page headers and footers are removed, wrapped lines are rejoined, and every rule number is placed directly before its text, because the citations depend on them. Page numbers are not kept. Tables (for example, the division table and the illegal-moves table) are not extracted as structured data: they are kept as serialized text, one row per line, and the per-division legality marks in the illegal-moves table are graphical, so they are lost in extraction.

**Chunking.** Chunks follow the document's own structure: the parser tracks the ruleset's three booklets (Rules Book, General Competition Guidelines, Competition Format Manual), their articles and their numbered rules, and makes one chunk per numbered rule, with lettered sub-items (`A)`, `B)`, …) as chunks of their own. Short heading lines such as `6.2.2 Serious Fouls` are folded into the section label of the chunks under them instead of becoming chunks. Rules longer than 1,500 characters are split at line boundaries into evenly sized parts (labelled `part n/m`), with no overlap. The result is 268 chunks. Rulebooks are written as numbered articles that are meant to be read on their own, so structure-aware chunking keeps a rule and its exceptions together better than fixed-size windows do. Each chunk carries a `source_section` label with its fully qualified position, for example `RULES BOOK > ARTICLE 4 – POINT SCORING POSITIONS > 4.4 Mount and Back Mount (4 points) > 4.4.1`, which is what the citations are built from.

**Embeddings.** `nomic-embed-text` via Ollama (768 dimensions), chosen because it runs locally at no cost. Documents are embedded with the `search_document:` prefix (followed by the section label and the chunk text) and questions with `search_query:`, as the model expects.

**Vector store.** pgvector 0.8.6 on PostgreSQL 17. pgvector keeps vectors and metadata in one Postgres instance, which is enough at the scale of one rulebook; an HNSW index serves cosine-distance search.

**Retrieval.** Retrieval is dense-only: the question is embedded and the 20 nearest chunks by cosine distance are retrieved as candidates for the reranker. Dense retrieval alone can miss exact terms that matter in this domain: technique names ("heel hook", "reaping", "slicer"), division names and article numbers. Hybrid retrieval with keyword search is not implemented yet (see the roadmap).

**Reranking.** Candidates are reranked with the `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder (via sentence-transformers) and the top 10 go to the generator. The cross-encoder reads the query and passage together, which fixes the most common failure of the first stage: passages that share vocabulary with the question but concern a different division or situation.

**Generation.** `gpt-4o-mini` (OpenAI API, temperature 0) is told to answer only from the supplied passages, to cite each claim by its rule number (for example `(4.1.1)`) and to end with a `Sources:` line, and to say the ruleset does not address the question when the passages do not support an answer. Citations are not yet validated automatically.

## Evaluation

Evaluation is the core of this project. Without it, "grounded" is a claim; with it, it is a measurement that can go up or down when the pipeline changes.

### Test set

The golden set has 20 question–answer pairs written by hand against the rulebook, each with a reference answer. It is designed to cover the kinds of questions where retrieval and memory both fail:

- **Direct lookups**: point values, match durations, and similar single-article facts.
- **Division-conditional rules**: legality of techniques by belt, age and gi/no-gi, where the answer depends on a table or a cross-reference.
- **Multi-hop questions**: answers that need two or more articles, such as a penalty plus its escalation rule.
- **Ruleset confusion traps**: questions whose answer differs between IBJJF and other rulesets, to catch answers drawn from memory.
- **Out-of-scope questions**: questions the rulebook does not answer, where the correct response is to say so.

The set lives in [`eval/eval_questions.json`](eval/eval_questions.json).

### Metrics

Scored with RAGAS 0.3.1:

| Metric | What it measures |
|---|---|
| Faithfulness | Share of claims in the answer that are supported by the retrieved context. The main guard against hallucination. |
| Answer relevancy | Whether the answer addresses the question that was asked. |
| Context precision | Whether the relevant chunks are ranked near the top of what was retrieved. |
| Context recall | Whether the retrieved context contains everything needed for the reference answer. |

The judge model is `llama3` (8B) via Ollama for the local runs and `gpt-4o` via the OpenAI API for the final run. LLM-judged metrics are noisy, so both judges run at temperature 0 with RAGAS's default seed, each configuration was scored once, and [whether a sample of judgments was checked by hand].

### Results

Adding cross-encoder reranking on top of dense-only retrieval, and tuning how many reranked chunks go to the generator.

| Configuration | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| Baseline: dense retrieval, top 10, no reranking (`llama3` judge) | 0.86 | 0.89 | 0.72 | 0.82 |
| + rerank, 20 candidates → top 5 (`llama3` judge) | 0.66 | 0.77 | 0.75 | 0.89 |
| + rerank, 20 candidates → top 10 (`llama3` judge) | 0.78 | 0.82 | 0.72 | 0.93 |
| Rerank, 20 candidates → top 10 (`gpt-4o` judge) | 0.80 | 0.80 | 0.65 | 0.65 |

Reranking down to 5 chunks raised context precision and recall but cut faithfulness from 0.86 to 0.66: with fewer passages, the model filled gaps with claims the context did not support. Keeping 10 reranked chunks recovered most of that (faithfulness 0.78) and gave the best context recall (0.93), though faithfulness is still below the no-rerank baseline, and answer relevancy dropped in both reranked configurations. The `gpt-4o` row is the most trustworthy result: the frontier judge is stricter and more reliable than the local 8B judge, as its much lower context recall on the same retrieved chunks shows.

To reproduce:

```bash
cd eval
uv run evaluate_ollama.py   # local judge (llama3 via Ollama)
uv run evaluate_openAI.py   # gpt-4o judge (uses OpenAI credits)
```

Each run writes per-question scores to `eval/results/` as a CSV. The no-rerank baseline is `user_query(question, rerank=False)`.

## Tech stack

- **Language:** Python 3.13, managed with [uv](https://docs.astral.sh/uv/)
- **PDF parsing:** `pdftotext -layout` (Poppler), plus text normalization
- **Embeddings:** `nomic-embed-text` (768 dimensions)
- **Vector store:** pgvector 0.8.6 on PostgreSQL 17
- **Keyword search:** none yet (dense retrieval only)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM:** `gpt-4o-mini` via the OpenAI API for generation; local inference via [Ollama](https://ollama.com) for embeddings and the `llama3` evaluation judge
- **Orchestration:** plain Python, no framework
- **Evaluation:** RAGAS 0.3.1
- **Interface:** CLI (`bjj-ingest`, `bjj-ask`)
- **Infrastructure:** Docker (a single pgvector container)

## Setup

### Prerequisites

- Docker
- [uv](https://docs.astral.sh/uv/) (for running outside Docker)
- [Ollama](https://ollama.com), for local embeddings
- An OpenAI API key, for generation

### Running locally with Ollama

Embeddings run on your machine through Ollama; generation calls the OpenAI API, so an `OPENAI_API_KEY` is required.

```bash
git clone [repo URL]
cd bjj-rules-rag

# Pull local models
ollama pull nomic-embed-text
ollama pull llama3            # only needed for the local evaluation judge

# Configure
cp .env.example .env   # set POSTGRES_PASSWORD, DATABASE_URL and OPENAI_API_KEY

# Start the vector store (creates the rules table on first start)
docker build -t bjj-rules-db .
docker run -d --name bjj-rules-db \
  --env-file .env \
  -p 5433:5432 \
  -v bjj-rules-pgdata:/var/lib/postgresql/data \
  bjj-rules-db

# Ingest the ruleset (reads IBJJF_RULES.txt from the repository root; run once on an empty table)
uv run bjj-ingest

# Ask a question
uv run bjj-ask "Is slamming allowed?"
```

The rulebook is not included in this repository. Download the official PDF from IBJJF yourself ([official rulebook URL]), save it as `2024JUN_IBJJF_Rules_EN.pdf` in the repository root, and extract it to `IBJJF_RULES.txt` with `pdftotext -layout` followed by the normalization described under Ingestion. Both files are gitignored.

### Using a hosted model

Generation already uses a hosted model (`gpt-4o-mini`), configured through `OPENAI_API_KEY`; there is no setting to switch it to a local model yet.

## Failure modes and limitations

**Tables are read as flat text.** The division and illegal-moves tables are kept as serialized text rather than extracted as structured rows. Even when the right table is retrieved, the model struggles to read it: asked under which divisions heel hooks and knee reaping are allowed, it answered that they are not allowed in any division.

**Gaps in the extracted text.** Some rules are not fully present in the ingested text. The per-division legality marks of the illegal-moves table are graphical and are lost in extraction, so no amount of retrieval can answer which belt may use a given technique. Like the table problem, this is a corpus and parsing problem, not a retrieval-algorithm problem: the fix is better extraction, not a better retriever.

**Retrieval misses.** If the article that answers a question is not in the top-k after reranking, the model either declines or answers from a partial context. Context recall on the golden set is the direct measure of this.

**Cross-references.** Rules often say "see Article X". The pipeline does not follow these references, so multi-hop answers depend on both articles being retrieved independently.

**Faithful is not the same as correct.** Faithfulness measures whether the answer matches the retrieved text, not whether it is right. A faithful answer built on the wrong passage (for example, the juvenile division's rule when the adult rule was asked about) scores well on faithfulness and is still wrong. Context precision and the division-conditional questions in the golden set exist to catch this, but not perfectly.

**Ruleset versioning.** The index holds one version of one ruleset. IBJJF revises its rules periodically; a production system would need versioned ingestion, a version filter at query time, and answers that state which version they cite.

**Evaluation coverage.** 20 hand-written questions do not cover the long tail of edge cases that come up at real events. The metrics are a regression signal for this set, not a guarantee.

**Security is out of scope.** The system ingests one trusted PDF. It does not defend against prompt injection in source documents or user queries, SSRF or untrusted document ingestion, or abuse of a public endpoint. Accepting user-supplied documents or exposing a public API would require input sanitization, isolation of retrieved content from instructions, rate limiting and authentication.

**Not an official source.** Answers are a reading aid. Refereeing decisions follow the official rulebook and the referees on the day.

## Roadmap

- Follow in-document cross-references during retrieval.
- Index multiple rulesets (ADCC, UAEJJF) with explicit ruleset filtering, and extend the golden set with cross-ruleset questions.
- Versioned ingestion with change detection between rulebook releases.
- Automatic citation validation: reject or flag answers whose citations do not match retrieved chunks.
- Run evaluation in CI to catch regressions from prompt, model or chunking changes.
- Hybrid retrieval: combine dense results with keyword search (BM25 or Postgres full-text) for exact technique names and article numbers.
- Structured table extraction, so the division and illegal-moves tables can be read row by row.
- [Your own items.]

## License

[License]
