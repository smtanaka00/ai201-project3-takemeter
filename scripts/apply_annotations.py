"""
apply_annotations.py

Applies the human-reviewed annotation decisions to the raw harvested comments and writes
the clean training dataset. The label decisions live here as explicit index sets so the
labeling is transparent and auditable (and easy to correct on review) — this is the
"annotation assistance" step from planning.md § 7, where AI drafts and a human confirms.

Decision rules (from planning.md):
  • analysis  — cites specific, checkable evidence (stats, named sequences, a tactical
                mechanism explained with concrete support).
  • hot_take  — a bold opinion/claim/prediction asserted without checkable evidence,
                even when it gives a bare reason.
  • reaction  — purely emotional/expressive, a joke, or an insult with no claim to defend.
  • DROP      — off-topic for the taxonomy (the recurring "Reddit app idea" meta-thread,
                pure conversational filler, info-only questions) or a near-duplicate row.

Run with:
    python scripts/apply_annotations.py
Produces data/takemeter_dataset.csv (text, label, annotation_notes).
"""

from pathlib import Path

import pandas as pd

RAW = Path("data/raw_posts.csv")
RAW_REACTIONS = Path("data/raw_reactions.csv")  # targeted harvest to balance the reaction class
OUT = Path("data/takemeter_dataset.csv")

# ── label decisions by raw-row index ────────────────────────────────────────────
ANALYSIS = {
    4, 9, 26, 27, 30, 34, 47, 51, 54, 59, 64, 65, 74, 77, 80, 91, 92, 94, 95, 96, 97,
    99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
    117, 118, 119, 121, 123, 124, 131, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 148, 149, 150, 151, 153, 156, 158, 159, 161, 162, 164, 165, 167, 179,
    184, 186, 188, 189, 191, 193, 194, 195, 198, 199, 202, 204, 206, 207, 208, 209, 214,
    215, 217, 218, 222, 223, 225, 227, 229, 231, 232, 236, 241,
}

REACTION = {5, 14, 22, 33, 35, 36, 46, 49, 62, 73, 76, 85, 88, 90, 155, 160, 230, 242}

# Reaction-class rows hand-picked from the targeted harvest (data/raw_reactions.csv). The
# broad/analysis queries surfaced almost no pure reactions, so this boost keeps the class
# from being too thin to evaluate. Only genuinely expressive one-liners are kept — rows that
# carried a real argument were left out (they'd be hot_take/analysis, not reaction).
REACTION_BOOST = {
    9, 14, 19, 21, 27, 29, 32, 36, 40, 41, 42, 47, 51, 54, 57, 75, 79, 83, 84, 85, 90, 91,
    95, 97, 99, 102, 103, 104, 108, 109, 112, 113, 114, 116, 117,
}

# Off-topic (app-idea meta-thread, pure filler, info-only questions) and near-duplicates.
DROP = {
    0, 1, 8, 11, 15, 17, 28, 29, 32, 37, 38, 40, 45, 48, 53, 56, 58, 61, 67, 71, 78, 79,
    82, 83, 84, 87, 133, 171, 212, 247,            # off-topic / filler / question-only
    147, 163, 166, 190, 197, 200, 210, 213, 224,   # near-duplicate of an earlier row
}

# Default rationale per label (drafts; refine on review). A few rows carry edge-case notes.
DEFAULT_NOTES = {
    "analysis": "Cites specific evidence (stats / named sequence / tactical mechanism).",
    "hot_take": "Bold opinion or claim asserted without checkable evidence.",
    "reaction": "Emotional/expressive, joke, or insult with no claim to defend.",
}
NOTE_OVERRIDES = {
    9: "xG breakdown with concrete per-player numbers -> evidence-backed analysis.",
    118: "Cites full stat line (possession, xG, shots, big chances) -> analysis.",
    93: "Edge case A: gives a reason ('different sites') but no checkable evidence -> hot_take.",
    160: "Edge case B: mentions xG but dominant intent is emotional ('feels like', emoji) -> reaction.",
    24: "Edge case C: rhetorical question carrying an opinion -> hot_take.",
    189: "Explains a pressing mechanism and points to a named goal as evidence -> analysis.",
}


def _label_for(i):
    """Return the label for a raw-row index, or None to drop it."""
    if i in DROP:
        return None
    if i in ANALYSIS:
        return "analysis"
    if i in REACTION:
        return "reaction"
    return "hot_take"  # default: argumentative opinion without hard evidence


def main():
    df = pd.read_csv(RAW)
    rows = []
    for i, r in df.iterrows():
        label = _label_for(i)
        if label is None:
            continue
        rows.append({
            "text": r["text"],
            "label": label,
            "annotation_notes": NOTE_OVERRIDES.get(i, DEFAULT_NOTES[label]),
        })

    # Fold in the hand-picked reaction-class rows from the targeted harvest.
    if RAW_REACTIONS.exists():
        rdf = pd.read_csv(RAW_REACTIONS)
        seen = {r["text"] for r in rows}  # avoid any accidental cross-file duplicate
        for i in sorted(REACTION_BOOST):
            if i in rdf.index:
                text = rdf.loc[i, "text"]
                if text not in seen:
                    seen.add(text)
                    rows.append({"text": text, "label": "reaction",
                                 "annotation_notes": DEFAULT_NOTES["reaction"]})

    out = pd.DataFrame(rows, columns=["text", "label", "annotation_notes"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"Wrote {len(out)} labeled rows to '{OUT}' (dropped {len(df) - len(out)}).")
    dist = out["label"].value_counts()
    for lbl, n in dist.items():
        print(f"  {lbl:<10} {n:>4}  {n / len(out):.1%}")


if __name__ == "__main__":
    main()
