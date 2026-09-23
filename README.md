# doc-extractor

Multi-agent pipeline that extracts **attributes**, **real-world entities** and
**attribute-to-entity mappings** from unstructured documents (PDF, DOCX, Markdown,
plain text). Nine specialised agents share one LangGraph state; every stage is
validated, retried, checkpointed and fully traceable back to the source text.

- Python 3.12, LangGraph 1.x, Pydantic v2
- Azure OpenAI as the LLM backend (configured through `.env`), with a **deterministic
  rule-based fallback** so the pipeline and the whole test-suite run with no API key
- Byte-for-byte deterministic JSON outputs, SQLite checkpoints, `--resume` recovery

```
Input -> 1 Document Processing -> 2 Chunking -> 3 Attribute Extraction -> 4 Attribute Storage
      -> 5 Entity Generation -> 6 Entity Normalization -> 7 Entity Reviewer
      -> 8 Attribute-Entity Mapping -> 9 Final Output (JSON / graph)
```

---

## 1. Prerequisites

| Requirement | Notes |
| --- | --- |
| Python 3.12 | `python3.12 --version`. On Debian/Ubuntu also install `python3.12-venv`. |
| git | to clone the repository |
| (optional) Tesseract OCR | only for scanned PDFs: `apt-get install tesseract-ocr`, then `pip install -e ".[ocr]"` |
| (optional) Azure OpenAI resource | endpoint, key and a chat-completions deployment. Not needed for the offline mode. |

No database, Docker or other service is required. Checkpoints use the SQLite module
that ships with Python.

## 2. Install

```bash
git clone https://github.com/ashish-019-hash/test-repo.git
cd test-repo

python3.12 -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -e ".[dev]"              # runtime + pytest/ruff/mypy/reportlab

doc-extractor --version              # -> doc-extractor 0.1.0
```

## 3. Configure Azure OpenAI (`.env`)

```bash
cp .env.example .env
```

Fill the five variables in `.env`:

```dotenv
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_MODEL=gpt-4o
AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
```

- `.env` is loaded automatically (python-dotenv) from the current directory or any
  parent. Use `--env-file path/to/file` to load a different file. Shell variables that
  are already exported win over `.env`.
- `.env` is git-ignored. Never commit it.
- Provider selection (`LLM_PROVIDER` in `.env`, or `--provider` on the command line):

| Value | Behaviour |
| --- | --- |
| `auto` (default) | Azure when all five variables are set, otherwise a logged fallback (`llm.fallback`) to `rules` |
| `azure` | Azure only. Exit code 3 with the list of missing variables if the `.env` is incomplete |
| `rules` | Deterministic rule-based provider. No network. Used by the test-suite |
| `mock` | Replays recorded responses from `llm.replay_dir` (regression tests) |

Azure calls use `temperature=0`, a fixed `seed` and an on-disk response cache
(`<out>/llm_cache/`), so re-runs with the same input are free and repeatable.

## 4. Run

```bash
# list the nine stages with the state keys they need and produce
doc-extractor stages

# run the bundled sample specification (offline fallback if .env is empty)
doc-extractor run tests/fixtures/telecom_spec.pdf --out out/

# same with the graph export and JSON logs
doc-extractor run tests/fixtures/telecom_spec.pdf --out out/ --graph json --log-format json

# force a provider, use the project config file
doc-extractor run my_spec.docx --out out/my_spec --provider rules --config config.yaml

# show the effective configuration and its hash
doc-extractor config --show
```

The command prints a one-line summary and writes the output directory:

| File | Content | Deterministic |
| --- | --- | --- |
| `final.json` | document summary, attributes, canonical entities, mappings, `review` section, unmapped/dangling attribute ids and the `trace` edges Document -> Chunk -> Attribute -> Entity -> Mapping | yes |
| `attributes.json` | accepted + review-band attributes (Agent 4 output) | yes |
| `discarded.json` | candidates below threshold or promoted to entities, with the reason | yes |
| `entities.json` | canonical entities after duplicate merge and quality validation | yes |
| `mappings.json` | attribute -> canonical entity mappings with evidence and confidence | yes |
| `review_queue.json` | human-review items (attribute review band, possible duplicates, entity review) | yes |
| `run.json` | run id, timestamps, provider/model, `system_fingerprint`, config hash, stage records, traces | no |
| `stages/NN_<stage>.json` | full state snapshot after each stage (used by `--resume-from`) | no (timestamps) |
| `checkpoints.sqlite` | LangGraph checkpoints (used by `--resume`) | no |
| `llm_cache/` | cached Azure responses (Azure mode only) | - |
| `graph.json` / `graph.graphml` | optional node-link graph (`--graph`) | yes |

Exit codes: `0` success, `2` a stage failed after its retries (the message names the
stage and the resume command), `3` invalid input, configuration or usage.

## 5. Recover from a failure

Every stage retries transient errors (`retry.max_attempts`, default 3, exponential
backoff). If a stage still fails, the run stops, `run.json` records the failed stage
and the checkpoint stays on disk.

```bash
# continue after the last successful stage (the failed stage runs again)
doc-extractor run my_spec.pdf --out out/my_spec --resume

# re-run from a chosen stage using the stored snapshot of its predecessor
doc-extractor run my_spec.pdf --out out/my_spec --resume-from entity_generation
```

`--resume-from` is also the way to re-run the entity stages after you change the
reviewer thresholds without paying for extraction again.

## 6. Tune the configuration

All thresholds live in `config.yaml` (a copy of the packaged defaults in
`src/doc_extractor/config/default.yaml`). Pass your copy with `--config`. Any value can
also be overridden with an environment variable
`DOC_EXTRACTOR__<section>__<key>=<yaml value>`, for example:

```bash
DOC_EXTRACTOR__scoring__routing__accept=0.9 doc-extractor run spec.pdf --out out/
```

Key settings:

| Section | Setting | Default | Meaning |
| --- | --- | --- | --- |
| `scoring.weights` | per-signal weights | see spec (+0.45 spec table row ... -1.0 furniture) | attribute scoring ensemble |
| `scoring.routing` | `accept` / `review` | 0.85 / 0.50 | >= accept auto-accept, >= review human queue, below discard |
| `scoring.short_circuit` | clean spec-table row | on, confidence 1.0 | worked example CIR -> 1.0 |
| `binding.scope_multipliers` | table_subject ... document_default | 1.0 ... 0.5 | entity binding confidence by scope |
| `entity_generation` | `min_mentions`, `min_confidence` | 2, 0.5 | entity candidate filters |
| `reviewer.duplicate` | `duplicate_threshold` / `possible_threshold` | 0.90 / 0.70 | DUPLICATE / POSSIBLE_DUPLICATE / DISTINCT |
| `reviewer.quality` | `accept` / `review` | 0.75 / 0.50 | ACCEPT / REVIEW / REJECT |
| `mapping` | `min_confidence` | 0.5 | minimum mapping confidence |
| `chunking` | `strategy`, `max_tokens`, `overlap_tokens` | heading_aware, 800, 100 | `by_page`, `fixed_tokens` or `heading_aware` |
| `lexicons.*` | YAML lexicon files | packaged telecom set | property nouns, gazetteer, known entities, SID ABEs, aliases, furniture, lifecycle verbs |

The config hash (printed by `config --show`, stored in `run.json`) covers the merged
configuration and the lexicon file contents, so a changed threshold always starts a
new checkpoint thread.

## 7. Test

```bash
pytest                    # offline: unit + integration, ~200 tests, no network
pytest --cov=doc_extractor --cov-report=term
pytest -m live            # Azure smoke test; skips cleanly when .env is empty
pytest --update-golden    # rewrite tests/fixtures/expected/final.<fmt>.json after an intended change
ruff check src tests && ruff format --check src tests
mypy src
python scripts/make_fixtures.py   # regenerate telecom_spec.pdf / .docx from telecom_spec.md
```

The integration tests run the compiled graph on the Markdown, PDF and DOCX fixtures,
compare `final.json` to the golden files, run twice and assert byte equality, and
inject a failure into the reviewer stage to prove `--resume` does not re-run earlier
stages.

## 8. Reset

```bash
rm -rf out/                 # all run outputs, checkpoints and the LLM cache
rm -rf .venv                # the virtual environment (then repeat section 2)
find . -name __pycache__ -type d -exec rm -rf {} +   # optional
```

To restart one document from scratch, delete its output directory or run without
`--resume`; a fresh run clears the thread for that document and config.

## 9. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `llm.fallback ... missing=[AZURE_OPENAI_...]` warning and `provider=rules` | `.env` is incomplete. Fill all five variables, or accept the offline mode. `--provider azure` makes this a hard error (exit 3). |
| `BadRequestError ... response_format / json_schema` | Your API version does not support structured outputs. The provider retries automatically with `json_object`; set `AZURE_OPENAI_API_VERSION=2024-10-21` or newer to use `json_schema`. |
| `DeploymentNotFound` / 404 | `AZURE_OPENAI_DEPLOYMENT` must be the deployment name in Azure AI Foundry, not the model name. |
| `document.needs_ocr: true`, zero attributes (exit 0) | The PDF is scanned; the run completes with empty results. Install Tesseract, `pip install -e ".[ocr]"`, set `ingest.ocr_enabled: true`. |
| Table rows extracted as prose | pdfplumber found no ruling lines. Try the DOCX/Markdown source, or lower `scoring.routing.review` to keep key-value candidates for review. |
| `database is locked` | Another process holds `checkpoints.sqlite`. Wait for it to finish or use a different `--out`. |
| `Nothing to resume` (exit 3) | No checkpoint for this document + config in `--out`. Run once without `--resume`. |
| `snapshot ... not found` with `--resume-from` | The predecessor stage never completed in this `--out`. Use `--resume` or run from scratch. |
| `snapshot ... belongs to document` with `--resume-from` | The `--out` directory holds another document's snapshots. Use one `--out` per document. The existing checkpoint is left untouched. |
| Windows: long paths / `\` in `.env` | Quote paths in `.env` and prefer forward slashes; run inside PowerShell with the venv activated. |
| `python3.12: command not found` | Install Python 3.12 (`pyenv install 3.12`, `uv python install 3.12`, or the OS package). |

## 10. Architecture

```
                 +----------------------------- LangGraph StateGraph (linear) -----------------------------+
 input file ---> | document_processing -> chunking -> attribute_extraction -> attribute_storage           |
                 |   -> entity_generation -> entity_normalization -> entity_reviewer -> attribute_mapping |
                 |   -> final_output                                                                      |
                 +--------------------------------------------------------------------------------------+
                        |                 |                          |                     |
                  SqliteSaver       stages/NN_*.json            LLMProvider          run.json + logs
                 (checkpoints)      (state snapshots)      azure | rules | mock      (structlog, traces)
```

| # | Agent | Reads | Writes | Responsibility |
| --- | --- | --- | --- | --- |
| 1 | DocumentProcessingAgent | `input_path` | `document` | PDF/DOCX/MD/TXT text with pages, blocks, tables, headings, OCR flag |
| 2 | ChunkingAgent | `document` | `chunks` | heading-aware / by-page / fixed-token chunks that never split sentences or table rows |
| 3 | AttributeExtractionAgent | `chunks`, `document` | `attributes`, `discarded_attributes`, review items | signal ensemble (structural, lexical, value-domain, syntactic, negative), clamp, routing, entity binding, normalised record |
| 4 | AttributeStorageAgent | `attributes` | `attributes_path` (+ `attributes.json`, `discarded.json`) | schema-only validation and deterministic JSON persistence |
| 5 | EntityGenerationAgent | `attributes`, `chunks` | `entities` | grounded entity candidates from bindings, promotions and lexicons |
| 6 | EntityNormalizationAgent | `entities` | `normalized_entities` | textual normalisation with step log; never merges |
| 7 | EntityReviewerAgent | entities, attributes, chunks | `duplicate_groups`, `canonical_entities`, `review_decisions` | DUPLICATE / POSSIBLE_DUPLICATE / DISTINCT and eight-criterion quality validation |
| 8 | AttributeMappingAgent | canonical entities, attributes | `mappings`, `unmapped_attribute_ids` | evidence-backed attribute -> canonical entity mappings |
| 9 | FinalOutputAgent | everything | `final_output` + files | final JSON, review section, trace edges, optional graph |

Every agent extends `BaseAgent`: `validate_input` -> `execute` -> `validate_output`,
with retries on `LLMTransientError` and output-validation errors, a `StageRecord` and
`TraceRecord`s in state. Agents are instantiated with a config and a provider and are
testable without the graph (`tests/unit/agents/`).

## 11. Determinism guarantees

- Identifiers are content hashes: `doc-<sha16(file bytes)>`, `chunk-<doc>-p<page>-<nnnn>`,
  `attr-<sha16(doc|chunk|name|source)>`, `ent-<sha16(doc|name|type)>`, `cent-<sha16(sorted members)>`.
- All deterministic files are written with canonical JSON: sorted keys, record lists
  sorted by id, floats rounded to `output.float_precision`, no timestamps, run ids,
  hostnames or absolute paths (`provenance.file` is the base name only).
- The rules provider is pure. Azure determinism is best-effort (`temperature=0`,
  fixed `seed`, response cache); the `system_fingerprint` is recorded in `run.json`.
- `tests/integration/test_determinism.py` asserts byte equality across two runs.

## 12. Observability

- structlog events: `run.start`, `stage.start`, `stage.retry`, `stage.done`, `stage.failed`,
  `llm.call`, `llm.cache_hit`, `llm.fallback`, `run.done`. Choose `--log-format console|json`
  and `--log-level`.
- LangSmith: set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` in `.env`;
  LangGraph picks them up without any code change.
