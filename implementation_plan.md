# TakeMeter — Implementation Plan & Progress Tracker

> **What this is.** The live feature definition and milestone tracker for TakeMeter, a
> fine-tuned text classifier that evaluates discourse quality in an online community.
> This file is the single source of truth for *what we're building* and *where we are*.
> It is written first and kept current as work lands.

---

## Project summary

Build and evaluate a text classifier that sorts community posts into a small, mutually
exclusive taxonomy of discourse types. Two systems are compared on the same locked test set:

1. **Zero-shot baseline** — Groq `llama-3.3-70b-versatile`, prompted with the taxonomy only.
2. **Fine-tuned model** — `distilbert-base-uncased` fine-tuned on our annotated dataset.

The deliverable is a comparative evaluation (accuracy + per-class precision/recall/F1), a
systematic error-pattern analysis, and full documentation.

---

## Decisions locked for this build

| Area | Decision |
|------|----------|
| Baseline model | Groq `llama-3.3-70b-versatile`, zero-shot, taxonomy-only prompt |
| Fine-tune model | `distilbert-base-uncased` |
| Train/Val/Test split | 70 / 15 / 15, stratified by label, `random_state=42` |
| Training config | 3 epochs · batch size 16 · lr `2e-5` · weight decay 0.01 · 50 warmup steps · T4 GPU · select best by weighted F1 |
| Notebook | Official AI201 starter (`takemeter_starter_notebook.md`) as the foundation; TODOs filled + 2 documented enhancements |
| Metrics | Overall accuracy + per-class Precision / Recall / F1 + confusion matrix |
| Dataset | ≥ 200 rows · no single label > 70% · CSV columns: `text`, `label`, `annotation_notes` |
| Execution env | Google Colab (T4 GPU); `GROQ_API_KEY` stored as a Colab Secret |
| **Community** | ✅ **r/PremierLeague** — football/soccer discussion |
| **Taxonomy** | ✅ `analysis` / `hot_take` / `reaction` (3-label, approved) |

---

## Status legend

✅ done · 🟡 in progress · ⬜ not started

---

## Milestone status table

| # | Milestone | Feature / deliverable | File(s) | Status |
|---|-----------|-----------------------|---------|--------|
| 1 | Community & Taxonomy | Pick community; design 2–4 label schema; boundary examples + edge-case rule | (decided → `planning.md`) | ✅ |
| 2 | Planning doc | Full design spec answering all required questions | `planning.md` | ✅ |
| 3 | Dataset assembly | CSV schema ✅; PullPush scraper ✅; harvested + balanced ✅; **246 rows annotated** (passes verify) ✅; difficult-cases log ✅ | `data/takemeter_dataset.csv`, `scripts/scrape_reddit.py`, `scripts/apply_annotations.py`, `scripts/verify_dataset.py` | ✅ (pending your review) |
| 4 | Colab fine-tuning | Data prep/split, tokenization, DistilBERT training cells | `notebooks/takemeter_colab.ipynb` | ✅ (built; runs after data ready) |
| 5 | Baseline integration | Groq zero-shot baseline cells over the test set | `notebooks/takemeter_colab.ipynb` | ✅ (built; runs after data ready) |
| 6 | Stretch + docs | Error-pattern parser ✅; README results filled (baseline 0.676 > FT 0.568) ✅ | `src/error_analysis.py`, `README.md`, `outputs/` | ✅ |
| — | Presentation | 3:30 script (supplied in brief) | `README.md` (appendix) | ✅ |

---

## Adapted directory layout

`PROJECT_STRUCTURE.md` describes an agent app; this is an ML pipeline, so the layout is
adapted while keeping the spec-first / self-documenting / graceful-degradation principles.

```text
ai201-project3-takemeter/
├── README.md                     # User-facing: framework, results, samples, reflections
├── planning.md                   # Design spec — written before data/code
├── implementation_plan.md        # This tracker
├── PROJECT_STRUCTURE.md          # Conventions (reference)
├── requirements.txt              # Pinned deps for the local scripts
├── .gitignore                    # Secrets, caches, model outputs
├── .env.example                  # Credential template (copy to .env, gitignored)
├── data/
│   ├── raw_posts.csv             # Raw harvested comments (broad + analysis queries)
│   ├── raw_reactions.csv         # Raw harvested comments (reaction-targeted queries)
│   └── takemeter_dataset.csv     # Annotated dataset (246 rows, passes verify)
├── scripts/
│   ├── scrape_reddit.py          # PullPush harvester -> data/raw_*.csv
│   ├── apply_annotations.py      # Applies reviewed label decisions -> takemeter_dataset.csv
│   ├── verify_dataset.py         # Validates row count, balance, null/malformed fields
│   └── build_notebook.py         # Build tool: regenerates the Colab notebook
├── notebooks/
│   └── takemeter_colab.ipynb     # Colab: data prep, DistilBERT fine-tune, Groq baseline
└── src/
    └── error_analysis.py         # Stretch: systematic error-pattern parser
```

---

## Per-milestone notes

### Milestone 1 — Community & Taxonomy *(in progress)*
- Choosing an active, text-heavy community with varying discourse depth.
- Will propose a 2–4 label mutually-exclusive, exhaustive taxonomy (≥90% coverage).
- Deliver 2 positive examples per label + 1 ambiguous edge case with a strict decision rule.
- **Output feeds directly into `planning.md`.**

### Milestone 2 — `planning.md`
- Community description + why its discourse is interesting.
- Label definitions (full sentences) + 2 examples each.
- Hard edge cases with exact classification rules.
- Data plan (sourcing, per-label targets, imbalance mitigation < 70%).
- Evaluation metrics + rationale; success criteria threshold.
- AI tool plan (label stress-testing, annotation assist, error-pattern ID).

### Milestone 3 — Dataset
- CSV schema: `text`, `label`, `annotation_notes`.
- Verification: ≥200 rows · no label > 70% · no null/malformed fields.
- Log 3 ambiguous examples into `planning.md`.

### Milestones 4 & 5 — Colab notebook *(based on the official AI201 starter)*
- **Foundation:** `takemeter_starter_notebook.md` (provided). `notebooks/takemeter_colab.ipynb`
  is generated from it by `scripts/build_notebook.py` with our TODOs filled in.
- **Filled in:** `LABEL_MAP` (analysis/hot_take/reaction) and the Groq `SYSTEM_PROMPT`
  (taxonomy definitions + one example each, from `planning.md`).
- **Two documented enhancements** to the starter (both non-breaking):
  1. `compute_metrics` also returns weighted F1/precision/recall; best checkpoint selected
     on F1 (`metric_for_best_model="f1"`) per the locked decision.
  2. an export cell writes `test_predictions.csv` so the stretch parser runs locally.
- **Starter outputs:** `confusion_matrix.png`, `evaluation_results.json` (commit these);
  plus our `test_predictions.csv`.
- Everything else (tokenization, training loop, baseline) is the starter verbatim.

### Milestone 6 — Stretch + docs
- Error parser groups mistakes by structural attributes (length, punctuation/markers,
  analysis↔hot_take lexical confusion).
- README with comparison matrices, sample-classification table (with confidence %),
  3 notable failures, and spec-alignment reflections.

---

## Open inputs needed from you (running list)

| When | What I need from you |
|------|----------------------|
| ~~M1~~ | ~~Confirm community + taxonomy~~ ✅ done (r/PremierLeague; 3-label) |
| ~~M3~~ | ~~Annotate dataset~~ ✅ AI-drafted, 246 rows, passes verify — **review the labels** in `data/takemeter_dataset.csv` and correct any you disagree with |
| **Now (M4/M5)** | **Run the notebook in Colab** (T4 GPU); add `GROQ_API_KEY` as a Colab Secret; download `test_predictions.csv` |
| M6 | Fill README result tables + run `src/error_analysis.py` after the Colab run |
