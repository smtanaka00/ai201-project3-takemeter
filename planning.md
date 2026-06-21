# TakeMeter — Planning & Design Spec

> **What this is.** The design contract for TakeMeter, written before any data collection
> or code. It defines the community, the label taxonomy, the data plan, the evaluation
> approach, and how AI tools are used. Code and data should match this document; any
> divergence is recorded in the README's reflection section.

---

## 1. Community

**Name:** r/PremierLeague — the primary subreddit for English Premier League football.

**Description:** An active, high-volume community where fans discuss matches, transfers,
tactics, refereeing, and club narratives. Content ranges from long post-match tactical
write-ups to one-line emotional outbursts during live match threads. Daily discussion
threads, post-match threads, and news posts produce a steady stream of text at every level
of depth.

**Why its discourse is interesting:** Football fandom sits at the intersection of genuine
analytical expertise and pure tribal emotion. The *same event* — say, a late winner — will
produce a measured xG-backed breakdown, a sweeping "we're winning the league now" claim, and
a wordless scream of joy, often in the same thread. That natural spread of depth is exactly
what makes the community a good testbed for a discourse-quality classifier: the categories
are real, frequent, and meaningfully distinct, but the boundaries between them are genuinely
fuzzy in a way that challenges both a zero-shot LLM and a fine-tuned model.

---

## 2. Labels

The taxonomy has three mutually exclusive, collectively exhaustive labels. They are ordered
by analytical depth: `analysis` (most reasoned) → `hot_take` (asserted) → `reaction`
(expressive). Targeted coverage of real posts is ≥ 90%.

### `analysis`
A reasoned argument about tactics, performance, or trends that **cites specific, checkable
evidence** — a named sequence of play, a formation, a statistic (xG, possession, pass maps),
a player's defined role, or an explicit cause-and-effect chain that could be verified by
rewatching or checking data.

- *"Arsenal's high line held up because Saliba's recovery pace covered the space in behind —
  you can see City's second-half xG drop once they stopped pressing the half-spaces."*
- *"United keep getting overrun in midfield because Casemiro drops too deep, leaving the
  single pivot exposed on transitions; the last three games' heatmaps show the same gap."*

### `hot_take`
A bold, declarative opinion or prediction stated **without verifiable support**. It may
sound like a reason ("because X"), but the support is an unchecked assertion or a sweeping
generalization rather than concrete evidence.

- *"Haaland is overrated, he'd be useless in any other league."*
- *"Arteta is the best manager in the league right now, end of discussion."*

### `reaction`
A purely emotional or expressive response to a match, goal, result, or piece of news. There
is no claim to defend — the post's function is to vent, celebrate, or despair.

- *"WHAT A GOAL!!! I'm actually shaking 😭🔥"*
- *"We're going down. I genuinely can't watch this anymore."*

---

## 3. Hard edge cases & exact rules

These rules are applied verbatim during annotation to keep label boundaries consistent.

### Edge case A — a claim that *gives a reason* but cites no evidence
*"Haaland is overrated because he only scores against weak teams."*

This has a "because" clause, which tempts an `analysis` label. **Rule:** a post is
`analysis` **only if it cites specific, checkable evidence** (a stat, a named match/sequence,
or data). A bare assertion or unverifiable generalization — even phrased as "because X" —
remains a **`hot_take`**. → This example is **`hot_take`**.

### Edge case B — emotion *and* a claim in the same post
*"Unreal save, De Gea is still elite!"*

Mixes an emotional exclamation with an opinion. **Rule:** label by **dominant intent**. If
the post is primarily expressive and any claim is incidental, it is `reaction`. If it makes a
standalone, defensible claim, it is `hot_take`. → Here the claim "still elite" is defensible
and central, so this leans **`hot_take`**; a version like *"UNREAL SAVE!!! 😱"* would be
**`reaction`**.

### Edge case C — a question that contains an implied take
*"How is Maguire still starting over Lisandro? Makes no sense."*

A rhetorical question carrying an opinion. **Rule:** if the post's real function is to assert
an opinion (the question is rhetorical), label it `hot_take`. A genuine information-seeking
question with no embedded stance falls outside the taxonomy and is excluded from the dataset
(kept under 10% of sampled content, consistent with the ≥90% coverage target).

### Difficult cases log (from real data inspection)

Three genuinely ambiguous rows met while annotating the harvested data, and how the rules
resolved them:

1. *"The problem with XG is that it is a very dodgy stat. For instance, I have just looked at
   three different websites and each one has different XG for each player."* — It references
   evidence ("three websites"), which tempts `analysis`. But the evidence is vague and the
   point is an opinion *about* the stat, not an evidence-backed argument. By Edge-case-A,
   labeled **`hot_take`**.
2. *"...feels like almost every game that we under scored our xG 😭"* — Mentions xG (an
   analytical term) but the dominant intent is emotional venting (the 😭 and "feels like").
   By Edge-case-B (label by dominant intent), labeled **`reaction`**.
3. *"Dude we won the league by a mile with the highest XG meaning we're creating loads of
   chances. Honestly this feels like a troll post."* — Opens with a real stat-backed point
   (highest xG → chance creation) but collapses into a dismissive jab. The standalone claim
   is unbacked beyond the one figure and the tone is argumentative, so it was labeled
   **`hot_take`** rather than `analysis`. This is the classic `analysis`↔`hot_take` boundary
   the model is expected to struggle with.

---

## 4. Data plan

**Sourcing method:** Automated harvest from r/PremierLeague via the **PullPush API** (the
open Pushshift successor — no credentials, and reachable where Reddit's own endpoints block
datacenter IPs). `scripts/scrape_reddit.py` round-robins across a query set to capture all
three depth levels:
- **Broad recent sample** (no query) → the natural mix, rich in `hot_take` and `reaction`.
- **Evidence-laden queries** (`xG`, `pressing`, `formation`, `tactically`, ...) → bias the
  pull toward `analysis`, the rarest and hardest-to-find class.

Each row is one comment, cleaned to a single line, then **human-annotated** against the rules
above (AI may draft a label, but every gold label is human-confirmed — see § 7).

**Target counts (≥ 200 rows total, deliberately balanced):**

| Label | Target | Share |
|-------|-------:|------:|
| `analysis`  | ~80 | ~33% |
| `hot_take`  | ~80 | ~33% |
| `reaction`  | ~80 | ~33% |
| **Total**   | **~240** | **100%** |

**Class-imbalance mitigation (no class > 70%):**
- `reaction` is the easiest to over-collect (match threads are full of it) and `analysis` the
  hardest, so collection is **quota-driven**: we stop adding `reaction` rows once the quota is
  met and keep mining tactical threads for `analysis` until its quota is hit.
- The verification script (Milestone 3) hard-fails if any label exceeds 70%, catching skew
  before training.
- The train/val/test split is **stratified by label** so each partition preserves the balance.

---

## 5. Evaluation metrics

Two systems (Groq zero-shot baseline and fine-tuned DistilBERT) are scored on the **same
locked test set**.

| Metric | Why it's needed |
|--------|-----------------|
| **Overall accuracy** | Headline single-number comparison between the two systems. |
| **Per-class Precision** | Of posts predicted as label X, how many were correct — exposes a model that over-applies a popular class. |
| **Per-class Recall** | Of true label-X posts, how many were caught — exposes a class the model systematically misses (likely `analysis`). |
| **Per-class F1** | Harmonic mean; the fairest per-class summary and the metric we select the best checkpoint on (weighted F1). |
| **Confusion matrix** | Shows *which* labels get confused for which — we expect `analysis`↔`hot_take` confusion, and the matrix is the input to the stretch error analysis. |

Accuracy alone is insufficient because the classes, while balanced in our dataset, are not
equally easy: a model could score well on `reaction` and `hot_take` while failing `analysis`.
Per-class metrics make that failure visible.

---

## 6. Success criteria

The classifier is considered to have **real-world utility** if, on the locked test set:

1. **Fine-tuned overall accuracy ≥ 0.75**, and
2. **Fine-tuned weighted F1 ≥ 0.70**, and
3. **The fine-tuned model beats the Groq zero-shot baseline** on overall accuracy, and
4. **No single class has F1 < 0.55** (i.e. the model is usable on every category, not just the
   easy ones).

Hitting (1)–(3) but missing (4) is the expected "good but flawed" outcome and becomes the
core of the error analysis: it would indicate the model leans on easy lexical/length cues and
struggles on the `analysis`↔`hot_take` boundary.

---

## 7. AI tool plan

AI is used at three specific points, each with a defined input, expected output, and a
verification step before the output is trusted.

| Workflow | AI tool | Input | Expected output | How it's verified |
|----------|---------|-------|-----------------|-------------------|
| **Label stress-testing** | Claude (chat) | The taxonomy definitions + 10–15 borderline real posts | A proposed label per post + a flag on any post the rules don't cleanly resolve | I manually adjudicate every flagged post and tighten the §3 rules; the AI never sets a label unreviewed. |
| **Annotation assistance** | Claude (chat/batch) | Batches of unlabeled posts + the locked taxonomy + edge-case rules | Suggested label + one-line `annotation_notes` rationale per post | I review 100% of suggestions; any I disagree with is corrected by hand and the disagreement is noted. AI suggestions are a draft, not ground truth. |
| **Zero-shot baseline** | Groq `llama-3.3-70b-versatile` | Test-set post + taxonomy-only prompt | A single predicted label | This is a *measured system*, not a labeling aid — its outputs are scored against my gold labels, never merged into the dataset. |
| **Automated error-pattern ID** | `src/error_analysis.py` (+ Claude for summary) | Misclassified val/test rows | Errors grouped by structural attributes (length, punctuation/markers, class confusion) | Group definitions are deterministic code; Claude only drafts the prose summary of the resulting buckets, which I check against the numbers. |

**Guardrail:** AI assists annotation and analysis but never owns the gold labels. Every label
in `data/takemeter_dataset.csv` is human-confirmed.

---

## 8. A complete interaction, end to end

A concrete walkthrough of one post moving through the whole pipeline:

1. **Collect.** During a post-match thread I copy: *"City's build-up broke down because Rodri
   was isolated — once United man-marked him, their progressive passes into the final third
   dropped off a cliff."*
2. **Annotate.** It cites a specific cause (Rodri isolated / man-marked) and a checkable trend
   (progressive passes dropped). By the §2 `analysis` definition and Edge-case-A rule, I label
   it `analysis` and note `"cites cause + checkable trend"` in `annotation_notes`.
3. **Verify.** `scripts/verify_dataset.py` confirms the row is non-null, the label is valid,
   total rows ≥ 200, and `analysis` ≤ 70% of the set.
4. **Split.** In Colab, the stratified 70/15/15 split places this row in (say) the training
   set, preserving label balance across partitions.
5. **Train / baseline.** DistilBERT trains on it (if in train) or it's held out for scoring
   (if in test). The Groq baseline, separately, predicts a label for every test row.
6. **Evaluate.** If this row is in the test set, both systems' predictions for it feed the
   accuracy, per-class P/R/F1, and confusion-matrix computations.
7. **Error analysis.** If a model misclassifies it (e.g. predicts `hot_take`), the error
   parser buckets it — here likely under "long-but-confused-with-hot_take" — and the README
   reports the pattern.
