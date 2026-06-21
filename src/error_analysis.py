"""
error_analysis.py

Stretch feature: systematic error-pattern analysis. Takes the per-row test predictions
exported by the Colab notebook (`test_predictions.csv`) and groups the model's mistakes by
structural attributes — text length, presence of explicit markers (punctuation, emoji,
digits/stats), and which label pairs get confused. The goal is to move past a single
accuracy number toward *why* the model fails.

Each grouping is deterministic code; the prose summary it prints is what feeds the README's
failure analysis. Degrades gracefully: a missing file is reported (not raised), and with no
input it runs a small built-in demo so the file can be sanity-checked in isolation.

Usage:
    python src/error_analysis.py                          # uses test_predictions.csv
    python src/error_analysis.py --csv path/to/preds.csv
    python src/error_analysis.py --model baseline_pred    # analyze the baseline instead

Run from the project root.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# ── configuration ───────────────────────────────────────────────────────────────
# Length buckets in characters. "short" posts are where we expect length-cue overfitting.
SHORT_MAX = 60
MEDIUM_MAX = 200

# Emoji / symbol detection — covers the common emotive ranges seen in match-thread reactions.
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF]"
)
# "Evidence markers" — digits often signal stats/scores, a proxy for analytical content.
DIGIT_RE = re.compile(r"\d")


# ── loading ─────────────────────────────────────────────────────────────────────
def _load_predictions(csv_path: Path):
    """
    Load the predictions CSV.

    Args:
        csv_path: path to a CSV with columns text, true_label, and a prediction column.

    Returns:
        (DataFrame, error_message). On any failure the DataFrame is None and the message
        explains why — the caller decides whether to fall back to the demo, no raise.
    """
    if not csv_path.exists():
        return None, f"Predictions file not found at '{csv_path}'."
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return None, f"Could not parse '{csv_path}': {exc}"
    return df, None


# ── structural attributes ─────────────────────────────────────────────────────--
def _length_bucket(text):
    """Classify a post into a coarse length band."""
    n = len(str(text))
    if n <= SHORT_MAX:
        return "short"
    if n <= MEDIUM_MAX:
        return "medium"
    return "long"


def _markers(text):
    """Return the set of structural markers present in a post."""
    text = str(text)
    present = set()
    if "!" in text:
        present.add("exclamation")
    if "?" in text:
        present.add("question")
    if EMOJI_RE.search(text):
        present.add("emoji")
    if DIGIT_RE.search(text):
        present.add("digits")  # proxy for a stat/score being cited
    if not present:
        present.add("plain")  # no explicit markers at all
    return present


# ── analysis ────────────────────────────────────────────────────────────────────
def analyze(df, model_col: str) -> dict:
    """
    Group misclassifications by structural attribute.

    Args:
        df: predictions DataFrame (text, true_label, <model_col>).
        model_col: which prediction column to evaluate ('ft_pred' or 'baseline_pred').

    Returns:
        A dict of grouped error counts. Returns an empty-but-valid report (with
        total_errors=0) if there are no errors — never raises.
    """
    required = {"text", "true_label", model_col}
    missing = required - set(df.columns)
    if missing:
        return {"error": f"Missing column(s): {sorted(missing)}"}

    errors = df[df["true_label"] != df[model_col]].copy()
    report = {
        "model_col": model_col,
        "total_rows": len(df),
        "total_errors": len(errors),
        "accuracy": round(1 - len(errors) / len(df), 3) if len(df) else 0.0,
        "by_length": {},
        "by_marker": {},
        "by_confusion": {},
    }
    if errors.empty:
        return report

    # By length band.
    errors["_len_bucket"] = errors["text"].map(_length_bucket)
    report["by_length"] = errors["_len_bucket"].value_counts().to_dict()

    # By structural marker (a post can contribute to several markers).
    marker_counts = {}
    for text in errors["text"]:
        for m in _markers(text):
            marker_counts[m] = marker_counts.get(m, 0) + 1
    report["by_marker"] = dict(sorted(marker_counts.items(), key=lambda kv: -kv[1]))

    # By confusion pair (true -> predicted); these are the systematic mistakes.
    pairs = errors["true_label"] + " -> " + errors[model_col]
    report["by_confusion"] = pairs.value_counts().to_dict()

    return report


# ── reporting ─────────────────────────────────────────────────────────────────--
def print_report(report: dict):
    """Pretty-print a report dict to stdout."""
    if "error" in report:
        print(f"❌ {report['error']}")
        return

    print(f"=== Error-pattern analysis: '{report['model_col']}' ===")
    print(f"Rows: {report['total_rows']}  |  Errors: {report['total_errors']}  |  "
          f"Accuracy: {report['accuracy']:.3f}")
    if report["total_errors"] == 0:
        print("No misclassifications — nothing to group.")
        return

    def _block(title, mapping):
        print(f"\n{title}")
        if not mapping:
            print("  (none)")
            return
        for key, count in mapping.items():
            share = count / report["total_errors"]
            print(f"  {key:<22} {count:>3}  ({share:.0%} of errors)")

    _block("Errors by post length:", report["by_length"])
    _block("Errors by structural marker:", report["by_marker"])
    _block("Errors by confusion pair (true -> predicted):", report["by_confusion"])

    # Surface the headline pattern we expect to report in the README.
    conf = report["by_confusion"]
    a_vs_h = conf.get("analysis -> hot_take", 0) + conf.get("hot_take -> analysis", 0)
    if a_vs_h:
        print(f"\nNote: {a_vs_h} of {report['total_errors']} errors are analysis<->hot_take "
              "confusion — the expected hard boundary.")


# ── built-in demo (smoke test) ──────────────────────────────────────────────────
def _demo_frame():
    """A tiny synthetic prediction set so the file runs without the real CSV."""
    return pd.DataFrame([
        {"text": "City dominated, xG was 2.8 to 0.4 across the 90.",
         "true_label": "analysis", "ft_pred": "hot_take", "baseline_pred": "analysis"},
        {"text": "Arteta out, simple as that.",
         "true_label": "hot_take", "ft_pred": "hot_take", "baseline_pred": "reaction"},
        {"text": "WHAT A GOAL!!! 🔥",
         "true_label": "reaction", "ft_pred": "reaction", "baseline_pred": "reaction"},
        {"text": "Best midfield in the league, no contest.",
         "true_label": "hot_take", "ft_pred": "analysis", "baseline_pred": "hot_take"},
    ])


# ── entry point ───────────────────────────────────────────────────────────────--
def main():
    parser = argparse.ArgumentParser(description="Systematic error-pattern analysis.")
    parser.add_argument("--csv", default="test_predictions.csv",
                        help="Predictions CSV (default: test_predictions.csv).")
    parser.add_argument("--model", default="ft_pred",
                        help="Prediction column to analyze (ft_pred or baseline_pred).")
    args = parser.parse_args()

    df, err = _load_predictions(Path(args.csv))
    if err:
        # Fall back to the demo so the script is runnable in isolation for a sanity check.
        print(f"⚠️  {err} Running the built-in demo instead.\n")
        df = _demo_frame()

    report = analyze(df, args.model)
    print_report(report)
    sys.exit(0 if "error" not in report else 1)


if __name__ == "__main__":
    main()
