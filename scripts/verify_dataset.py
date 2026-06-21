"""
verify_dataset.py

Annotation verification for the TakeMeter dataset. Validates that the compiled CSV is
fit for training before it ever reaches Colab: enough rows, no dominant class, and no
null or malformed fields. Degrades gracefully — it never raises on a data problem; it
collects every issue and reports them together, exiting non-zero if any are fatal.

Usage:
    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --csv data/takemeter_dataset.csv

Run from the project root. Exits 0 if the dataset passes all hard checks, 1 otherwise.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── configuration ───────────────────────────────────────────────────────────────
# The taxonomy locked in planning.md. Any label outside this set is malformed.
VALID_LABELS = {"analysis", "hot_take", "reaction"}
REQUIRED_COLUMNS = ["text", "label", "annotation_notes"]

MIN_ROWS = 200          # project floor: at least 200 annotated rows
MAX_CLASS_SHARE = 0.70  # no single label may exceed 70% of the data
MIN_TEXT_LEN = 3        # text shorter than this is almost certainly malformed/empty


# ── loading ─────────────────────────────────────────────────────────────────────
def _load_csv(csv_path: Path):
    """
    Load the dataset CSV.

    Args:
        csv_path: Path to the dataset CSV.

    Returns:
        A (DataFrame, error_message) tuple. On any load failure the DataFrame is None
        and error_message explains why — the caller decides how to stop, we don't raise.
    """
    if not csv_path.exists():
        return None, f"CSV not found at '{csv_path}'. Create it or pass --csv."
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # malformed CSV, encoding issues, etc.
        return None, f"Could not parse '{csv_path}': {exc}"
    return df, None


# ── checks ──────────────────────────────────────────────────────────────────────
def _check_columns(df):
    """Return a list of issues if any required column is missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"Missing required column(s): {', '.join(missing)}"]
    return []


def _check_row_count(df):
    """Return a list of issues if the row count is below the project floor."""
    if len(df) < MIN_ROWS:
        return [f"Only {len(df)} rows — need at least {MIN_ROWS}."]
    return []


def _check_nulls_and_malformed(df):
    """
    Find null/blank text or labels, invalid labels, and suspiciously short text.
    `annotation_notes` is allowed to be blank (notes are optional), so it is not
    checked for nulls here.
    """
    issues = []

    # text: must be present and non-trivial
    null_text = df["text"].isna() | (df["text"].astype(str).str.strip().str.len() < MIN_TEXT_LEN)
    if null_text.any():
        rows = (df.index[null_text] + 2).tolist()  # +2 → 1-based + header row
        issues.append(f"{null_text.sum()} row(s) with empty/too-short text (CSV lines: {rows})")

    # label: must be present and one of the valid taxonomy labels
    null_label = df["label"].isna() | (df["label"].astype(str).str.strip() == "")
    if null_label.any():
        rows = (df.index[null_label] + 2).tolist()
        issues.append(f"{null_label.sum()} row(s) with missing label (CSV lines: {rows})")

    present = df["label"].astype(str).str.strip()
    invalid = ~present.isin(VALID_LABELS) & ~null_label
    if invalid.any():
        bad_values = sorted(present[invalid].unique())
        issues.append(
            f"{invalid.sum()} row(s) with invalid label(s): {bad_values}. "
            f"Allowed: {sorted(VALID_LABELS)}"
        )

    return issues


def _check_class_balance(df):
    """Return (issues, distribution). Fatal if any class exceeds MAX_CLASS_SHARE."""
    issues = []
    valid = df["label"].astype(str).str.strip()
    valid = valid[valid.isin(VALID_LABELS)]
    if valid.empty:
        return ["No valid labels to compute a distribution."], {}

    counts = valid.value_counts()
    shares = (counts / len(valid)).to_dict()
    for label, share in shares.items():
        if share > MAX_CLASS_SHARE:
            issues.append(
                f"Class '{label}' is {share:.1%} of the data — exceeds the "
                f"{MAX_CLASS_SHARE:.0%} cap. Collect more of the other classes."
            )
    return issues, {lbl: (int(counts[lbl]), shares[lbl]) for lbl in counts.index}


# ── reporting ─────────────────────────────────────────────────────────────────--
def _print_distribution(distribution):
    """Pretty-print the label distribution so balance is visible at a glance."""
    if not distribution:
        return
    print("\nLabel distribution:")
    for label, (count, share) in sorted(distribution.items()):
        bar = "█" * int(share * 40)  # simple visual gauge, 40 chars = 100%
        print(f"  {label:<10} {count:>4}  {share:>6.1%}  {bar}")


def verify(csv_path: Path) -> bool:
    """
    Run all checks and print a report.

    Args:
        csv_path: Path to the dataset CSV.

    Returns:
        True if the dataset passes every hard check, False otherwise. Never raises on a
        data problem — load/parse failures are reported and treated as a failed check.
    """
    df, load_error = _load_csv(csv_path)
    if load_error:
        print(f"❌ {load_error}")
        return False

    # Column check is a prerequisite for everything else; bail early if it fails.
    column_issues = _check_columns(df)
    if column_issues:
        for issue in column_issues:
            print(f"❌ {issue}")
        return False

    issues = []
    issues += _check_row_count(df)
    issues += _check_nulls_and_malformed(df)
    balance_issues, distribution = _check_class_balance(df)
    issues += balance_issues

    print(f"Checked '{csv_path}' — {len(df)} rows.")
    _print_distribution(distribution)

    if issues:
        print(f"\n❌ {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  • {issue}")
        return False

    print("\n✅ Dataset passes all checks: ready for the Colab pipeline.")
    return True


# ── entry point ───────────────────────────────────────────────────────────────--
def main():
    parser = argparse.ArgumentParser(description="Validate the TakeMeter dataset CSV.")
    parser.add_argument(
        "--csv",
        default="data/takemeter_dataset.csv",
        help="Path to the dataset CSV (default: data/takemeter_dataset.csv).",
    )
    args = parser.parse_args()
    ok = verify(Path(args.csv))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
