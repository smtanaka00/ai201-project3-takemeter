# Project Structure & Conventions (Reusable)

> **What this is.** A portable description of my preferred project structure, distilled
> from a working agent project. Drop this file into any new/unstructured repo, or paste it
> into a prompt as context, to get a consistent layout and file style.
>
> **This is a default, not a law.** Every choice below is a starting point. When a specific
> project's requirements call for something different — and especially when I say so in a
> prompt — override it. See [§ Adapting this for a specific project](#adapting-this-for-a-specific-project)
> at the bottom. Treat the prompt's instructions as higher priority than this document.

---

## 1. Core philosophy

Five ideas drive the whole layout. If a decision is ever ambiguous, fall back to these:

1. **Layered separation.** UI → orchestration → tools → data. Each layer only knows about
   the one beneath it. The UI never contains business logic; tools never know about the UI.
2. **Spec before code.** Design docs (the "what" and "why") are written and locked *before*
   implementation. Code should match the doc; if it diverges, the divergence is recorded.
3. **Single source of truth for state.** One state object per interaction is passed between
   steps. Nothing is shared between components except through it.
4. **Graceful degradation.** Functions handle their own edge cases and return safe fallbacks
   instead of raising. The orchestration layer decides when to stop, not when to crash.
5. **Self-documenting files.** Every file opens with a docstring saying what it is, how to
   use it, and how to run it. Comments explain *why*, not *what*.

---

## 2. Directory layout

A typical instance of this structure (names are illustrative — adapt to the domain):

```text
project-root/
├── README.md               # User-facing: setup, how to run, what each part does
├── planning.md             # Design spec — written first, drives implementation
├── implementation_plan.md  # Feature definition + milestone status tracker
├── PROJECT_STRUCTURE.md     # This file
├── requirements.txt        # Pinned/constrained dependencies
├── .gitignore              # Secrets, venvs, caches, local-only files
├── .env                    # Secrets — NEVER committed (listed in .gitignore)
│
├── app.py                  # UI / entry layer — thin, delegates to orchestration
├── agent.py                # Orchestration layer — the control loop + state object
├── tools.py                # Standalone, independently-testable unit functions
│
├── utils/
│   └── data_loader.py      # Data-access layer — load/parse data, resolve paths
│
├── data/
│   ├── <dataset>.json      # Static / mock data
│   └── <schema>.json       # Schemas, examples, templates
│
└── tests/
    └── test_tools.py       # Boundary + graceful-failure tests
```

### Layer responsibilities

| Layer | File(s) | Owns | Must NOT do |
|-------|---------|------|-------------|
| **UI / entry** | `app.py` | Input cleaning, calling orchestration, mapping results to output | Business logic, data access, tool calls |
| **Orchestration** | `agent.py` | The control loop, the state object, deciding flow & stop conditions | Reimplement tool internals; talk to the UI |
| **Tools** | `tools.py` | Discrete units of work; each callable & testable alone; graceful fallbacks | Know about the loop or the UI |
| **Data access** | `utils/data_loader.py` | Resolving paths, loading & parsing files, convenience getters | Business logic |
| **Data** | `data/*.json` | Static content, schemas, examples | (n/a — data only) |
| **Tests** | `tests/*` | Verifying boundaries & failure modes don't crash | (n/a) |

---

## 3. The three document files (write these first)

This structure is **spec-first**. Three markdown files carry the design and travel with the
project. Write `planning.md` before any code; keep the tracker current; finish the README last.

### `planning.md` — the design contract
Written before implementation and used as the prompt context when generating code. Sections:
- **Tools / units** — for each: what it does (1–2 sentences), input parameters (name, type,
  meaning), return value, and **what happens if it fails or returns nothing**.
- **Control loop** — how the next step is chosen, what conditions change behavior, how it
  knows it's done.
- **State management** — how information passes from one step to the next; which fields are
  tracked and who writes each.
- **Error handling** — a table of `unit | failure mode | response`.
- **Architecture** — an ASCII or Mermaid diagram showing flow and where error paths branch.
- **AI tool plan** — which AI tool, what input it gets, expected output, how output is
  verified before being trusted. ("I'll use AI to code" is not a plan.)
- **A complete interaction, step by step** — one concrete walkthrough end to end.

### `implementation_plan.md` — the tracker
Feature definition + live status. Contains:
- **Decisions locked for this build** — the choices that resolve ambiguity (input model,
  test strategy, model/library versions, etc.).
- **Status legend** (✅ done / 🟡 in progress / ⬜ not started).
- **Milestone status table** — `milestone | feature | file(s) | status`, updated as work lands.
- The system architecture diagram and per-milestone notes.

### `README.md` — user-facing
Fixed section order:
1. One-paragraph description of what it does and how (the mental model).
2. **Setup & Running** — install, env/secret setup, run command, test command.
3. **Inventory** of the main units with their real signatures.
4. **How the control loop works** — numbered steps.
5. **State management** — a table of `field | written by | holds`.
6. **Error handling** — table + a concrete worked example from real testing.
7. **Spec reflection** — one way the spec helped, one way the implementation diverged.
8. **AI usage** — specific instances: what was given, what was produced, what was changed.

---

## 4. File-internal conventions

These apply to every source file and are what make the codebase feel consistent.

### Module docstring (top of every file)
State the filename, its purpose, a usage example, and the run command. Example:

```python
"""
agent.py

The <project> control loop. Orchestrates the tools in response to user input,
passing state between them via a single session object.

Usage (once implemented):
    from agent import run_agent
    result = run_agent(query="...", context=...)
    print(result["output"])

Run with:
    python app.py
"""
```

### Section dividers
Group related functions under a labeled rule so files scan top-to-bottom:

```python
# ── session state ─────────────────────────────────────────────────────────────
# ── query parsing ───────────────────────────────────────────────────────────--
# ── Tool 1: <name> ──────────────────────────────────────────────────────────--
```

### Function docstrings
Every public function documents `Args:`, `Returns:`, and its failure behavior. During a
spec-first build, leave the `TODO:` step list from the design in the docstring until the
function is implemented, then replace it with the real behavior description.

### Comments explain *why*, not *what*
Good: `# ignore very short tokens so filler words don't inflate every score`.
Not: `# loop over the listings`.

### Private helpers prefixed `_`
`_new_session`, `_parse_query`, `_get_client` — anything not part of the public surface.

### A runnable `__main__` smoke test
End orchestration/data files with an `if __name__ == "__main__":` block that exercises the
happy path and a key edge case, so the file can be sanity-checked in isolation.

---

## 5. State management pattern

One **state object** (a dict) is created at the start of each interaction and is the single
source of truth. Each step reads only the fields it needs and writes its own result back.
Data flow is explicit and one-directional.

```python
def _new_session(query, context) -> dict:
    return {
        "query": query,           # original input
        "parsed": {},             # structured form of the input
        "results": [],            # intermediate output of step 1
        "selected": None,         # chosen item passed downstream
        "output": None,           # final result
        "error": None,            # set if the run ended early (else None)
        "notes": [],              # non-fatal events (e.g. a fallback was used)
    }
```

Convention: callers check `session["error"]` first — if set, the run stopped early and the
other output fields are `None`. `notes` carries non-fatal events worth surfacing.

---

## 6. Error-handling pattern

- **Tools degrade, the loop decides.** Each tool handles its own edge cases and returns a
  safe value (a fallback string, an empty list) instead of raising.
- **Retry-with-fallback** where it makes sense: if a strict query returns nothing, loosen one
  constraint and retry once before giving up; record that a fallback was used in `notes`.
- **Early-exit on empty.** The loop never feeds empty/invalid output into the next step — it
  sets `session["error"]` with a user-facing message and returns.
- **Deterministic fallbacks for non-deterministic units.** Anything that calls an external
  service wraps the call and falls back to output built from raw inputs on any error.

---

## 7. Testing conventions

- Tests target **boundaries and failure modes** — does each unit stay graceful at its edges
  (empty input, impossible query, missing field) without crashing.
- **Skip, don't fail, on missing prerequisites.** Tests needing a live API key/network are
  guarded so the pure-logic tests still run in a keyless environment:
  ```python
  requires_api = pytest.mark.skipif(not os.environ.get("API_KEY"), reason="API_KEY not set")
  ```
- Make the project root importable from inside `tests/` and keep assertions light when a test
  hits a paid/external service.

---

## 8. Dependencies, secrets, config

- **`requirements.txt`** — pin or constrain versions (`pkg==1.2.3` for things that must not
  drift, `pkg>=x` where a floor is enough).
- **`.env`** for secrets, loaded at startup; **never committed** — first entry in `.gitignore`.
- **`.gitignore`** covers, at minimum: `.env`, `__pycache__/`, `*.py[cod]`, virtualenvs
  (`.venv/`, `venv/`, `env/`), `.DS_Store`, and any local-only files.

---

## Adapting this for a specific project

This document is a **default**. Override it whenever the project warrants — and when a prompt
gives specific structural instructions, **those win over this file.** Common adaptations:

- **Different stack/language.** The layering (UI → orchestration → logic → data), the
  spec-first docs, the single-state-object and graceful-degradation patterns are
  language-agnostic. File names and idioms change; the principles carry.
- **Bigger projects.** Flat files (`tools.py`, `agent.py`) become packages
  (`tools/`, `agent/`) with the same role boundaries. `data_loader.py` may grow into a
  `data/` or `repositories/` layer.
- **No UI / library or service.** Drop `app.py`; the orchestration layer becomes the public
  entry point. For an API service, the route handlers play the role `app.py` plays here.
- **Heavier testing.** Add `tests/test_agent.py`, integration tests, fixtures, and mocks for
  external services so the suite runs without network access.
- **Different doc needs.** For a tiny project, `planning.md` and `implementation_plan.md` may
  collapse into one file. For a team project, they may expand (ADRs, a CONTRIBUTING guide).

**When prompting against this structure:** paste this file (or the relevant section) as
context, then state the deltas explicitly — e.g. "use this structure, but it's a FastAPI
service with no Gradio UI, split tools into a package, and add integration tests with mocked
external calls." The model should follow the prompt's deltas over the defaults here.
