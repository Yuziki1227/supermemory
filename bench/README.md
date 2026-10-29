# Retrieval benchmark

`cases.jsonl` is a small, self-contained regression suite for recall quality.
Every run seeds a temporary SQLite database, replays the queries, and reports
Recall@k, MRR, nDCG@k, negative-case behaviour and token cost.

```bash
python -m supermemory bench bench/cases.jsonl
python -m supermemory bench bench/cases.jsonl --json
python -m supermemory bench bench/cases.jsonl --fail-under-recall 0.85 --fail-under-mrr 0.80
```

CI runs the suite with thresholds, so a ranking change that breaks retrieval
fails the build.

## File format

One JSON object per line (JSONL). Blank lines and `#` comments are ignored. A
single JSON object with `memories` and `queries` arrays is also accepted.

```json
{"type": "memory", "id": "gate", "title": "Relevance gate", "content": "...", "kind": "decision", "tags": ["retrieval"], "importance": 0.8}
{"type": "query", "query": "how do I avoid unrelated hits", "expect": ["gate"]}
```

Memory fields:

| Field | Required | Default | Notes |
|---|---|---|---|
| `id` | yes | — | label used by `expect`; must be unique |
| `content` | yes | — | the memory body |
| `title` | no | derived | |
| `kind` | no | `fact` | one of the nine kinds |
| `tags` | no | `[]` | string or list |
| `importance` | no | `0.5` | 0–1 |
| `confidence` | no | `0.8` | 0–1 |
| `pinned` | no | `false` | |
| `chunk` | no | `false` | auto-chunk long content (one label may map to several rows) |

Query fields:

| Field | Required | Default | Notes |
|---|---|---|---|
| `query` | yes | — | natural language |
| `expect` | no | `[]` | one label or a list; empty means a negative case |
| `note` | no | `""` | why the case exists |

Unknown labels in `expect` are rejected at load time, so typos fail loudly.

## Metrics

- **Recall@k** — share of expected memories present in the top-k, averaged over
  positive queries.
- **MRR** — mean reciprocal rank of the first expected memory.
- **nDCG@k** — binary-relevance nDCG; rewards putting expected items first.
- **Negative pass** — negative queries that return nothing at all.
- **Tokens** — approximate context cost of returned summaries, per query.

## Writing good cases

- One case per retrieval behaviour you care about: tokenizer quirks, relevance
  gate, version history, namespaces, mixed Chinese/English.
- Include negative queries. They catch the failure mode where recency or
  importance drags unrelated memories into the results.
- Keep the committed corpus synthetic. Real memories may contain private project
  details; keep a private cases file outside the repository and pass its path
  instead.
- Benchmarks use `DEFAULT_CONFIG` by default. To measure a tuned configuration:

```bash
python -m supermemory bench my-cases.jsonl --config ~/.supermemory/config.json
```

`--keep` leaves the temporary database on disk for inspection; the path is
printed in the report.
