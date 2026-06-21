"""
build_notebook.py

Generates notebooks/takemeter_colab.ipynb from the **official AI201 Project 3 starter
notebook** (takemeter_starter_notebook.md), with our project-specific TODOs filled in:
  • LABEL_MAP   -> our 3-label taxonomy (analysis / hot_take / reaction)
  • SYSTEM_PROMPT -> the Groq baseline prompt, built from planning.md's definitions
  • two documented enhancements over the starter, both non-breaking:
      1. compute_metrics also returns weighted F1/precision/recall, and the best
         checkpoint is selected on F1 (our locked decision in planning.md § 5).
      2. an export cell writes test_predictions.csv so the stretch error-analysis
         parser (src/error_analysis.py) can run locally on per-row predictions.

Keeping the notebook in a builder (rather than hand-edited JSON) means the cells stay
valid and easy to regenerate. Run once; commit the resulting .ipynb. This file is a
build tool, not part of the runtime pipeline.

Run with:
    python scripts/build_notebook.py
"""

import json
from pathlib import Path

OUT = Path("notebooks/takemeter_colab.ipynb")


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": _src(lines)}


def _src(lines):
    joined = "\n".join(lines)
    parts = joined.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


cells = []

# ── intro ─────────────────────────────────────────────────────────────────────--
cells.append(md(
    "# TakeMeter — Fine-Tuning Notebook",
    "### AI201 · Project 3 — r/PremierLeague discourse classifier",
    "",
    "Based on the official course starter notebook. Fine-tunes a DistilBERT classifier on",
    "our annotated dataset and compares it to a Groq zero-shot baseline.",
    "",
    "**Infrastructure (provided):** tokenization, the DistilBERT training loop, metrics +",
    "confusion matrix, and the Groq baseline comparison.",
    "",
    "**Our work (filled in):** the label map, the Groq classification prompt (from",
    "`planning.md`), the dataset upload, and the evaluation write-up.",
    "",
    "**Before you start:** Runtime → Change runtime type → **T4 GPU** → Save.",
))

# ── deps + imports ──────────────────────────────────────────────────────────────
cells.append(code(
    "# Install any dependencies not pre-installed on Colab",
    "!pip install -q groq python-dotenv",
    'print("✅ Dependencies ready")',
))
cells.append(code(
    "import pandas as pd",
    "import numpy as np",
    "import json",
    "import time",
    "",
    "from sklearn.model_selection import train_test_split",
    "from sklearn.metrics import (",
    "    accuracy_score, classification_report,",
    "    confusion_matrix, ConfusionMatrixDisplay,",
    "    precision_recall_fscore_support,   # added: per-class + weighted metrics",
    ")",
    "import matplotlib.pyplot as plt",
    "",
    "import torch",
    "from transformers import (",
    "    AutoTokenizer,",
    "    AutoModelForSequenceClassification,",
    "    TrainingArguments,",
    "    Trainer,",
    "    DataCollatorWithPadding,",
    ")",
    "from datasets import Dataset",
    "import warnings",
    'warnings.filterwarnings("ignore")',
    "",
    'print("✅ Imports complete")',
    'print(f"PyTorch version: {torch.__version__}")',
    'print(f"GPU available: {torch.cuda.is_available()}")',
    "if torch.cuda.is_available():",
    '    print(f"GPU: {torch.cuda.get_device_name(0)}")',
))

# ── section 1: load dataset ─────────────────────────────────────────────────────
cells.append(md(
    "## Section 1 — Load the dataset",
    "Upload the labeled CSV and define the label map. The CSV has `text`, `label`, and",
    "`annotation_notes` columns; only `text` and `label` are used here.",
))
cells.append(code(
    "# Our finalized 3-label taxonomy for r/PremierLeague (see planning.md § 2).",
    "# These are this project's real labels — not the starter's illustrative example.",
    "LABEL_MAP = {",
    '    "analysis":  0,   # reasoned, cites specific checkable evidence',
    '    "hot_take":  1,   # bold claim/opinion without verifiable support',
    '    "reaction":  2,   # purely emotional / expressive, no claim',
    "}",
    "",
    "ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}",
    "NUM_LABELS = len(LABEL_MAP)",
    'print(f"Labels: {LABEL_MAP}")',
    'print(f"Number of labels: {NUM_LABELS}")',
))
cells.append(code(
    "# Upload your CSV from your computer (select data/takemeter_dataset.csv)",
    "from google.colab import files",
    'print("Select your labeled dataset CSV file...")',
    "uploaded = files.upload()",
    "CSV_PATH = list(uploaded.keys())[0]",
    'print(f"Uploaded: {CSV_PATH}")',
))
cells.append(code(
    "# Load and validate the dataset",
    "df = pd.read_csv(CSV_PATH)",
    "",
    'print(f"Columns: {df.columns.tolist()}")',
    'print(f"Total examples: {len(df)}")',
    "print()",
    'print("Label distribution:")',
    'print(df["label"].value_counts())',
    "",
    "# Validate all labels are in LABEL_MAP",
    'unknown = set(df["label"].unique()) - set(LABEL_MAP.keys())',
    "if unknown:",
    '    print(f"\\n⚠️  Labels in CSV not found in LABEL_MAP: {unknown}")',
    "else:",
    '    print("\\n✅ All labels match your LABEL_MAP")',
    "",
    "# Convert string labels to integers",
    'df["label_id"] = df["label"].map(LABEL_MAP)',
    'df = df.dropna(subset=["label_id"])',
    'df["label_id"] = df["label_id"].astype(int)',
))

# ── section 2: prepare data ─────────────────────────────────────────────────────
cells.append(md(
    "## Section 2 — Prepare data for training",
    "Stratified 70/15/15 train/val/test split, then tokenize.",
))
cells.append(code(
    "# Train / val / test split — 70% / 15% / 15%, stratified by label",
    "train_df, temp_df = train_test_split(",
    '    df, test_size=0.30, random_state=42, stratify=df["label_id"]',
    ")",
    "val_df, test_df = train_test_split(",
    '    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"]',
    ")",
    "",
    'print(f"Train: {len(train_df)} examples")',
    'print(f"Validation: {len(val_df)} examples")',
    'print(f"Test: {len(test_df)} examples")',
    "print()",
    'print("Train label distribution:")',
    'print(train_df["label"].value_counts())',
    "print()",
    'print("Test label distribution:")',
    'print(test_df["label"].value_counts())',
    "",
    "# Reset indices (needed for clean HuggingFace Dataset conversion)",
    "train_df = train_df.reset_index(drop=True)",
    "val_df   = val_df.reset_index(drop=True)",
    "test_df  = test_df.reset_index(drop=True)",
    "",
    "# Load tokenizer and tokenize all splits",
    'MODEL_NAME = "distilbert-base-uncased"',
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)",
    "",
    "def tokenize(examples):",
    '    return tokenizer(examples["text"], truncation=True, max_length=256)',
    "",
    "def make_dataset(df_split):",
    "    ds = Dataset.from_pandas(",
    '        df_split[["text", "label_id"]].rename(columns={"label_id": "labels"})',
    "    )",
    "    return ds.map(tokenize, batched=True)",
    "",
    "train_dataset = make_dataset(train_df)",
    "val_dataset   = make_dataset(val_df)",
    "test_dataset  = make_dataset(test_df)",
    "",
    "data_collator = DataCollatorWithPadding(tokenizer=tokenizer)",
    'print("✅ Tokenization complete")',
))

# ── section 3: fine-tune ────────────────────────────────────────────────────────
cells.append(md(
    "## Section 3 — Fine-tune the model",
    "Loads `distilbert-base-uncased` with a classification head and trains for 3 epochs",
    "(~5–15 min on a T4). We select the best checkpoint on **weighted F1** (planning.md § 5).",
))
cells.append(code(
    "# Load DistilBERT with a classification head",
    "model = AutoModelForSequenceClassification.from_pretrained(",
    "    MODEL_NAME,",
    "    num_labels=NUM_LABELS,",
    "    id2label=ID_TO_LABEL,",
    "    label2id=LABEL_MAP,",
    ")",
    'print(f"✅ Model loaded: {MODEL_NAME}")',
))
cells.append(code(
    "def compute_metrics(eval_pred):",
    "    # Enhanced over the starter: accuracy plus weighted P/R/F1 so the per-class",
    "    # picture is logged each epoch and we can select the best checkpoint on F1.",
    "    logits, labels = eval_pred",
    "    predictions = np.argmax(logits, axis=-1)",
    "    acc = accuracy_score(labels, predictions)",
    "    precision, recall, f1, _ = precision_recall_fscore_support(",
    '        labels, predictions, average="weighted", zero_division=0)',
    '    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}',
    "",
    "",
    "training_args = TrainingArguments(",
    '    output_dir="./takemeter-model",',
    "    num_train_epochs=3,",
    "    per_device_train_batch_size=16,",
    "    per_device_eval_batch_size=32,",
    "    learning_rate=2e-5,",
    "    weight_decay=0.01,",
    "    warmup_steps=50,",
    '    eval_strategy="epoch",',
    '    save_strategy="epoch",',
    "    save_total_limit=1,",
    "    load_best_model_at_end=True,",
    '    metric_for_best_model="f1",   # planning.md § 5: select on weighted F1',
    "    logging_steps=10,",
    '    report_to="none",',
    ")",
    "",
    "trainer = Trainer(",
    "    model=model,",
    "    args=training_args,",
    "    train_dataset=train_dataset,",
    "    eval_dataset=val_dataset,",
    "    data_collator=data_collator,",
    "    compute_metrics=compute_metrics,",
    ")",
    "",
    'print("Starting fine-tuning... (5–15 minutes on T4 GPU)")',
    "trainer.train()",
    'print("\\n✅ Fine-tuning complete")',
))

# ── section 4: evaluate fine-tuned ───────────────────────────────────────────────
cells.append(md(
    "## Section 4 — Evaluate the fine-tuned model on the locked test set",
    "Accuracy, per-class metrics, a confusion matrix, and the wrong predictions to review.",
))
cells.append(code(
    'print("Running inference on test set...")',
    "ft_output = trainer.predict(test_dataset)",
    "ft_pred_ids = np.argmax(ft_output.predictions, axis=-1)",
    "ft_true_ids = ft_output.label_ids",
    "",
    "ft_probs = torch.nn.functional.softmax(",
    "    torch.tensor(ft_output.predictions), dim=-1",
    ").numpy()",
    "",
    "ft_accuracy = accuracy_score(ft_true_ids, ft_pred_ids)",
    'print(f"\\n🎯 Fine-tuned model accuracy: {ft_accuracy:.3f}")',
    "",
    "label_names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]",
    'print("\\nPer-class metrics (fine-tuned model):")',
    "print(classification_report(ft_true_ids, ft_pred_ids, target_names=label_names, zero_division=0))",
))
cells.append(code(
    "# Confusion matrix",
    "cm = confusion_matrix(ft_true_ids, ft_pred_ids)",
    "disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)",
    "fig, ax = plt.subplots(figsize=(7, 5))",
    'disp.plot(ax=ax, cmap="Blues", colorbar=False)',
    'ax.set_title("Fine-Tuned Model — Confusion Matrix (Test Set)")',
    "plt.tight_layout()",
    'plt.savefig("confusion_matrix.png", dpi=150)',
    "plt.show()",
    'print("✅ Saved: confusion_matrix.png")',
))
cells.append(code(
    "# Print wrong predictions — review these for the README error analysis.",
    "wrong_idx = np.where(ft_pred_ids != ft_true_ids)[0]",
    'print(f"Wrong predictions: {len(wrong_idx)} / {len(ft_true_ids)}\\n")',
    "",
    "for i, idx in enumerate(wrong_idx[:15]):",
    '    text = test_df.iloc[idx]["text"]',
    "    true_label = ID_TO_LABEL[ft_true_ids[idx]]",
    "    pred_label = ID_TO_LABEL[ft_pred_ids[idx]]",
    "    confidence = ft_probs[idx][ft_pred_ids[idx]]",
    '    print(f"--- #{i+1} ---")',
    "    print(f\"Text:      {text[:200]}{'...' if len(text) > 200 else ''}\")",
    '    print(f"True:      {true_label}")',
    '    print(f"Predicted: {pred_label}  (confidence: {confidence:.2f})")',
    "    print()",
))

# ── section 5: groq baseline ─────────────────────────────────────────────────────
cells.append(md(
    "## Section 5 — Zero-shot baseline (Groq)",
    "Classifies the test set with `llama-3.3-70b-versatile` using only the taxonomy",
    "definitions. Add `GROQ_API_KEY` via the 🔑 Secrets panel (notebook access ON).",
))
cells.append(code(
    "from groq import Groq",
    "from google.colab import userdata",
    "",
    '# Colab Secrets (recommended): add GROQ_API_KEY in the 🔑 panel, notebook access ON.',
    'GROQ_API_KEY = userdata.get("GROQ_API_KEY")',
    "",
    "assert GROQ_API_KEY, (",
    '    "GROQ_API_KEY not set — add it in the Colab Secrets panel (🔑, left sidebar) "',
    '    "and enable notebook access for this notebook."',
    ")",
    "",
    "client = Groq(api_key=GROQ_API_KEY)",
    'print("✅ Groq client initialized")',
))
cells.append(code(
    "# Classification prompt built from our taxonomy (planning.md § 2): definitions +",
    "# one example per label. The model must output ONLY a label string.",
    'SYSTEM_PROMPT = """',
    "You are classifying comments from r/PremierLeague, the Premier League football subreddit.",
    "Assign each comment to exactly one of the following categories.",
    "",
    "analysis: a reasoned argument that cites specific, checkable evidence (a stat, xG, a named",
    "sequence of play, a formation, a defined player role, or an explicit cause-and-effect).",
    'Example: "Arsenal\'s high line held because Saliba\'s recovery pace covered the space behind',
    '— City\'s second-half xG dropped once they stopped pressing the half-spaces."',
    "",
    "hot_take: a bold, declarative opinion or prediction stated without verifiable support, even",
    "if it gives a bare reason. Unbacked assertions and sweeping generalizations belong here.",
    'Example: "Haaland is overrated, he\'d be useless in any other league."',
    "",
    "reaction: a purely emotional or expressive response (celebration, despair, shock, a joke)",
    "with no claim to defend.",
    'Example: "WHAT A GOAL!!! I\'m actually shaking 😭🔥"',
    "",
    "Respond with ONLY the label name.",
    "Do not explain your reasoning.",
    "",
    "Valid labels:",
    "analysis",
    "hot_take",
    "reaction",
    '"""',
    "",
    'print("Prompt length:", len(SYSTEM_PROMPT), "characters")',
))
cells.append(code(
    "def classify_with_groq(text):",
    '    """Classify a single post. Returns a label string or None if unparseable."""',
    "    try:",
    "        response = client.chat.completions.create(",
    '            model="llama-3.3-70b-versatile",',
    "            messages=[",
    '                {"role": "system", "content": SYSTEM_PROMPT},',
    '                {"role": "user", "content": f"Classify this post:\\n\\n{text}"},',
    "            ],",
    "            temperature=0,",
    "            max_tokens=20,",
    "        )",
    "        raw = response.choices[0].message.content.strip().lower()",
    "        # Longest labels first so a label that is a substring of another can't mis-match.",
    "        for label in sorted(LABEL_MAP, key=len, reverse=True):",
    "            if raw == label or label in raw:",
    "                return label",
    "        return None",
    "    except Exception as e:",
    '        print(f"API error: {e}")',
    "        return None",
    "",
    "",
    'print(f"Running baseline on {len(test_df)} examples...")',
    "baseline_preds = []",
    "for i, (_, row) in enumerate(test_df.iterrows()):",
    '    pred = classify_with_groq(row["text"])',
    "    baseline_preds.append(pred)",
    "    if (i + 1) % 10 == 0:",
    '        print(f"  {i+1}/{len(test_df)} complete...")',
    "    time.sleep(0.1)",
    "",
    "none_count = baseline_preds.count(None)",
    "if none_count > 0:",
    '    print(f"\\n⚠️  {none_count} responses could not be parsed.")',
))
cells.append(code(
    "# Baseline metrics (exclude unparseable responses)",
    'valid = [(p, t) for p, t in zip(baseline_preds, test_df["label_id"]) if p is not None]',
    "bl_pred_ids = [LABEL_MAP[p] for p, _ in valid]",
    "bl_true_ids = [t for _, t in valid]",
    "",
    "bl_accuracy = accuracy_score(bl_true_ids, bl_pred_ids)",
    'print(f"🎯 Baseline accuracy: {bl_accuracy:.3f}  "',
    '      f"(evaluated on {len(valid)}/{len(test_df)} parseable responses)")',
    "print()",
    'print("Per-class metrics (baseline):")',
    "print(classification_report(bl_true_ids, bl_pred_ids, target_names=label_names, zero_division=0))",
))

# ── section 6: compare + export ─────────────────────────────────────────────────
cells.append(md(
    "## Section 6 — Compare results and export",
    "Side-by-side comparison, plus the artifacts to commit: `evaluation_results.json`,",
    "`confusion_matrix.png`, and `test_predictions.csv` (feeds `src/error_analysis.py`).",
))
cells.append(code(
    'print("=" * 50)',
    'print("RESULTS COMPARISON")',
    'print("=" * 50)',
    'print(f"{\'Model\':<35} {\'Accuracy\':>8}")',
    'print("-" * 45)',
    'print(f"{\'Zero-shot baseline (Groq)\':<35} {bl_accuracy:>8.3f}")',
    'print(f"{\'Fine-tuned DistilBERT\':<35} {ft_accuracy:>8.3f}")',
    'print("-" * 45)',
    "delta = ft_accuracy - bl_accuracy",
    '_direction = "improvement" if delta >= 0 else "regression"',
    'print(f"\\nFine-tuning {_direction}: {abs(delta):.3f}")',
))
cells.append(code(
    "# Save metrics JSON",
    "results = {",
    '    "baseline_accuracy": round(bl_accuracy, 4),',
    '    "finetuned_accuracy": round(ft_accuracy, 4),',
    '    "improvement": round(ft_accuracy - bl_accuracy, 4),',
    '    "test_set_size": len(test_df),',
    '    "label_map": LABEL_MAP,',
    '    "model": MODEL_NAME,',
    "}",
    'with open("evaluation_results.json", "w") as f:',
    "    json.dump(results, f, indent=2)",
    "",
    "# Export per-row predictions for the stretch error-analysis parser",
    "# (src/error_analysis.py expects: text, true_label, ft_pred, baseline_pred).",
    "# We also export ft_conf — the fine-tuned model's softmax confidence in its",
    "# predicted label — so the README sample table and demo can show confidence.",
    'pred_df = test_df[["text"]].copy()',
    'pred_df["true_label"] = [ID_TO_LABEL[i] for i in ft_true_ids]',
    'pred_df["ft_pred"] = [ID_TO_LABEL[i] for i in ft_pred_ids]',
    "pred_df[\"ft_conf\"] = [round(float(ft_probs[i][ft_pred_ids[i]]), 4) for i in range(len(ft_pred_ids))]",
    'pred_df["baseline_pred"] = [p if p else "unparseable" for p in baseline_preds]',
    'pred_df.to_csv("test_predictions.csv", index=False)',
    "",
    'print("✅ Files ready to download (Files panel 📁 → right-click → Download):")',
    'print("   evaluation_results.json")',
    'print("   confusion_matrix.png")',
    'print("   test_predictions.csv   (run src/error_analysis.py on this locally)")',
))

# ── section 7: live demo (for the video) ────────────────────────────────────────
cells.append(md(
    "## Section 7 — Live demo (record this for the video)",
    "Classifies a handful of posts with the **fine-tuned model** and prints the predicted",
    "label and its confidence — this is the screen to record for the demo video. The",
    "`DEMO_POSTS` list is pre-filled with real test-set posts (one per class, plus a known",
    "hard case); swap in any text you like. Run this cell on camera so label + confidence",
    "are visible, and narrate one correct and one incorrect prediction.",
))
cells.append(code(
    "# A reusable single-post classifier using the fine-tuned model in memory.",
    "def classify_finetuned(text):",
    '    """Return (label, confidence) from the fine-tuned DistilBERT for one post."""',
    "    enc = tokenizer(",
    "        text, truncation=True, max_length=256, return_tensors=\"pt\"",
    "    ).to(model.device)",
    "    model.eval()",
    "    with torch.no_grad():",
    "        logits = model(**enc).logits",
    "    probs = torch.nn.functional.softmax(logits, dim=-1)[0]",
    "    pred_id = int(torch.argmax(probs))",
    "    return ID_TO_LABEL[pred_id], float(probs[pred_id])",
    "",
    "",
    "# Pre-filled with real posts spanning the three classes + a hard analysis/hot_take case.",
    "# Replace or extend with your own examples for the recording.",
    "DEMO_POSTS = [",
    '    "Created more big chances than PSG, had more shots on target, and a higher xG over the 90 — '
    'the performance was there even if the result wasn\'t.",',
    '    "Haaland is overrated, he\'d be useless in any other league.",',
    '    "WHAT A GOAL!!! I\'m actually shaking, what a time to be alive 😭🔥",',
    '    "5 xG? Who cares, xG over a single game is so flawed.",',
    '    "Unbelievable really. Can\'t wait for the final day, I say both teams just have a party!",',
    "]",
    "",
    'print("=" * 64)',
    'print("TAKEMETER — FINE-TUNED MODEL LIVE DEMO")',
    'print("=" * 64)',
    "for i, post in enumerate(DEMO_POSTS, 1):",
    "    label, conf = classify_finetuned(post)",
    "    snippet = post if len(post) <= 100 else post[:100] + \"...\"",
    '    print(f"\\nPost {i}: {snippet}")',
    '    print(f"   → Predicted label: {label.upper():<9}  confidence: {conf:.1%}")',
    'print("\\n" + "=" * 64)',
))

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1))
print(f"Wrote {OUT} with {len(cells)} cells.")
