# STARK-QA: answering questions about stories

STARK-QA investigates how passage search and a lightweight knowledge graph can
work together on narrative comprehension. It loads FairytaleQA story sections,
retrieves evidence through both routes, and produces a short answer using
Mistral or a local extractive answerer. Its name expands to **Story-grounded
Text Answering with Retrieval & Knowledge**.

Basic-RAG and Basic-KAG provide separate passage-only and graph-only comparisons.
This is a command-line experiment pipeline, with no web service or model-training
component.

## How the pipeline works

1. The dataset loader reads a cached test sample or downloads FairytaleQA. It
   groups distinct sections by story and removes repeated sections.
2. `RobustRAG` builds overlapping passages, then combines normalized dense
   embedding scores with TF-IDF scores. Lexical retrieval remains available
   when the dense model cannot load.
3. `SimpleKnowledgeGraph` extracts entities, relationships, and supporting
   sentences. Entity matches and question cues select useful evidence.
4. The combined system joins passage and graph evidence, putting graph context
   first for questions beginning with "who". Mistral receives a few-shot prompt
   requesting a concise answer grounded in that context. Without API access,
   or after failed requests, an extractive answerer selects an answer span.
5. The evaluator scores all three systems and exports predictions, metrics,
   plots, and a comparison report.

## Local setup

Use Python 3.11 or 3.12, as required by `pyproject.toml`. With uv:

```sh
uv sync --python 3.11
uv run python -m spacy download en_core_web_sm
```

For pip, create and activate a virtual environment, then install the stack:

```sh
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The dependency files specify version ranges. This snapshot does not include a
tracked lockfile. uv is configured to use CPU PyTorch wheels on Windows and
Linux. Installation and uncached model loading can require network downloads.

## Answer generation and settings

The local extractive path does not require an API key. To enable Mistral, copy
`.env.example` to `.env` and fill in:

```dotenv
MISTRAL_API_KEY=your_key_here
MISTRAL_MODEL=mistral-small-latest
```

`main.py` loads `.env` if python-dotenv is installed; Git ignores that file.
`config.py` defines retrieval parameters, model defaults, answer length limits,
and request throttling and retries.

Several optional libraries have fallback paths, but NumPy, scikit-learn, and
pandas are still required by core modules. Offline dataset access needs an
appropriate cached sample, and offline dense retrieval needs cached model files.

## Running the comparison

```sh
uv run python validate.py
uv run python main.py --test_size 10
uv run python main.py --test_size 100
```

Inside an activated environment, use `python` in place of `uv run python`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--test_size` | `100` | Number of test questions |
| `--cache_dir` | `cache/` | Dataset and retrieval cache directory |
| `--output_dir` | `results/` | Evaluation output directory |

The repository includes samples of 3, 10, and 100 questions. Smaller requested
samples can be sliced from a larger cache. A run replaces outputs in its selected
directory; choose another output directory to preserve the included benchmark.

## Supplied benchmark snapshot

The existing comparison CSV contains these scores for 100 questions. These
artifacts were supplied with the project and were not regenerated during local
repository setup.

| Measure | STARK-QA | Basic-RAG | Basic-KAG |
| --- | ---: | ---: | ---: |
| Exact match | 0.350 | 0.000 | 0.010 |
| Token F1 | 0.611 | 0.176 | 0.041 |
| ROUGE-1 | 0.629 | 0.196 | 0.058 |
| ROUGE-2 | 0.467 | 0.128 | 0.010 |
| ROUGE-L | 0.627 | 0.186 | 0.055 |
| BLEU | 0.422 | 0.085 | 0.008 |
| Semantic similarity | 0.729 | 0.381 | 0.193 |
| BERTScore F1 | 0.893 | 0.729 | 0.657 |

STARK-QA leads in this saved sample. The baselines use extractive answers,
whereas the supplied combined-system results describe Mistral generation. The
comparison therefore changes both retrieval and answer generation. Different
samples, model versions, and fallback modes can produce different results.

The evaluator scores against the primary answer even though the loader keeps
the second reference. Questions search a shared corpus of selected stories
without an explicit restriction to their originating story. Both choices affect
how the reported performance should be interpreted.

## Where to find things

| Path | Role |
| --- | --- |
| `main.py` | Loads data, builds indices, and compares systems |
| `validate.py` | Checks imports and a tiny end-to-end example |
| `src/data_loader.py`, `src/utils.py` | Dataset preparation and text helpers |
| `src/rag/` | Hybrid retrieval and cached passage indices |
| `src/kag/` | Entity, relation, and supporting-sentence retrieval |
| `src/models/` | Evidence fusion, Mistral calls, and extractive answers |
| `src/baselines/` | Basic-RAG and Basic-KAG |
| `src/evaluation/` | Metrics, visualization, and reporting |
| `cache/` | Dataset samples and index snapshots |
| `results/` | Saved predictions and evaluation artifacts |
| `report/` | Academic report, bibliography, style files, and PDF |

`results/system_comparison.csv` summarizes the systems, while
`results/evaluation_report.md` breaks down question categories. Each system has
a predictions CSV and detailed results JSON. Metric definitions are explained in
[METRICS_EXPLANATION.md](METRICS_EXPLANATION.md).

Corpus fingerprints help invalidate indices, but the helper hashes only each
story's beginning, end, and length. A same-length change confined to the middle
can therefore leave a stale cache undetected.

## Original project credits

The supplied project credits **Satvik Shrivastava (2024102029)**. Its original
course context is Introduction to NLP at IIIT Hyderabad, taught by
Prof. Manish Shrivastava in Spring 2025. The inspiration for this project was 
taken from this course.


