"""
scrape_reddit.py

Pulls real r/PremierLeague comments into a raw CSV for annotation — using the **PullPush**
API (api.pullpush.io), the open Pushshift successor. PullPush needs **no credentials** and,
unlike Reddit's own endpoints, is reachable from datacenter IPs that Reddit blocks. It also
supports keyword search, which we use to deliberately surface the rare `analysis` class
(see planning.md § 4): a broad query captures the natural mix (lots of `hot_take` /
`reaction`), and evidence-laden queries (xG, formations, pressing...) boost `analysis`.

It writes unlabeled rows — `label` and `annotation_notes` are left blank for you (with
optional AI assistance) to fill in. The script degrades gracefully: a query that fails is
skipped with a note, HTTP 429 is retried with backoff, and partial results are still saved.

Setup:
    pip install -r requirements.txt   # needs pandas + certifi; the fetch layer is stdlib

Usage:
    python scripts/scrape_reddit.py                       # balanced harvest, target ~250
    python scripts/scrape_reddit.py --target 300
    python scripts/scrape_reddit.py --queries "xG,pressing,formation" --target 120

Output:
    data/raw_posts.csv  with columns: text, label, annotation_notes, permalink, query
    (verify_dataset.py ignores the extra permalink/query columns; drop them once the set
     is annotated and finalized.)
"""

import argparse
import json
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

# Build a TLS context from certifi's CA bundle if present. Stock macOS Python often can't
# find system CAs (SSL: CERTIFICATE_VERIFY_FAILED); certifi gives a portable, verified bundle.
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


# ── configuration ───────────────────────────────────────────────────────────────
SUBREDDIT = "PremierLeague"
ENDPOINT = "https://api.pullpush.io/reddit/search/comment/"
USER_AGENT = "Mozilla/5.0 (compatible; takemeter-educational/0.3)"

# Be a good citizen toward the free PullPush service.
REQUEST_DELAY_S = 1.0
MAX_RETRIES = 4
PER_REQUEST = 100  # PullPush caps a single request at 100 items

# Default balanced query set. "" = broad recent sample (natural mix); the rest are
# evidence-bearing terms that bias toward `analysis`, our rarest class.
DEFAULT_QUERIES = ["", "xG", "pressing", "formation", "tactically", "because", "stats",
                   "underperform", "build-up", "transition"]

# Length window: drop one-word noise and giant copy-pastes. Reactions are short, so the
# floor is deliberately low; the cap keeps tokenization sane for DistilBERT later.
MIN_CHARS = 8
MAX_CHARS = 1000

# Authors/markers we never want as data rows.
SKIP_AUTHORS = {"AutoModerator", "[deleted]", "PremierLeague-ModTeam"}
SKIP_MARKERS = ("[removed]", "[deleted]")


# ── fetch layer ───────────────────────────────────────────────────────────────--
def _fetch_json(url):
    """
    GET a PullPush URL and parse it.

    Args:
        url: a fully-formed PullPush search URL with query string.

    Returns:
        (parsed_json, error_message). On any failure the json is None and the message
        explains why. HTTP 429 (rate limit) is retried with exponential backoff before
        giving up — we never raise out of this function.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=45, context=_SSL_CTX) as resp:
                return json.loads(resp.read().decode("utf-8")), None
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 502, 503) and attempt < MAX_RETRIES - 1:
                wait = REQUEST_DELAY_S * (2 ** attempt)  # back off on rate limit / gateway blip
                print(f"    {exc.code}; waiting {wait:.1f}s...")
                time.sleep(wait)
                continue
            return None, f"HTTP {exc.code} for {url}"
        except (urllib.error.URLError, ValueError, TimeoutError) as exc:
            return None, f"request failed: {exc}"
    return None, f"gave up after {MAX_RETRIES} retries"


# ── extraction ────────────────────────────────────────────────────────────────--
def _clean(text):
    """Normalize whitespace and strip markdown line breaks to one flat string."""
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def _is_usable(text, author):
    """Return True if a comment is a plausible dataset row (length + not junk)."""
    if author in SKIP_AUTHORS:
        return False
    if any(marker in text for marker in SKIP_MARKERS):
        return False
    return MIN_CHARS <= len(text) <= MAX_CHARS


def _fetch_query(query, before, seen):
    """
    Fetch one page of comments for a query and return usable, deduped rows.

    Args:
        query: the search term ("" for a broad sample).
        before: a created_utc cursor for pagination (None for the first page).
        seen: set of already-collected texts, used to dedup across queries/pages.

    Returns:
        (rows, oldest_utc, error). `oldest_utc` is the cursor for the next page (None if
        the page was empty). Per-query failures return ([], None, error), never raise.
    """
    params = {"subreddit": SUBREDDIT, "size": PER_REQUEST,
              "sort": "desc", "sort_type": "created_utc"}
    if query:
        params["q"] = query
    if before:
        params["before"] = before
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)

    data, err = _fetch_json(url)
    if err:
        return [], None, err

    items = data.get("data", [])
    if not items:
        return [], None, None

    rows = []
    oldest = None
    for c in items:
        oldest = c.get("created_utc", oldest)  # track cursor for pagination
        text = _clean(c.get("body", ""))
        author = c.get("author", "[deleted]")
        if _is_usable(text, author) and text not in seen:
            seen.add(text)
            rows.append({
                "text": text,
                "permalink": "https://reddit.com" + c.get("permalink", ""),
                "query": query or "(broad)",
            })
    return rows, oldest, None


# ── orchestration ─────────────────────────────────────────────────────────────--
def scrape(target: int, queries, out_path: Path, max_pages: int = 3) -> bool:
    """
    Harvest comments across the query set until the target row count is reached.

    Cycles through the queries **round-robin** — one page from each query per round —
    rather than exhausting the broad query first. This guarantees the evidence-laden
    queries (which surface the rare `analysis` class) actually contribute to the pool.

    Args:
        target: how many usable, deduped rows to aim for.
        queries: list of search terms to cycle through ("" = broad sample).
        out_path: where to write the raw CSV.
        max_pages: how many paginated pages to pull per query before moving on.

    Returns:
        True if any rows were collected, False otherwise. Never raises on a per-query
        error — those are logged and skipped, and whatever was collected is still saved.
    """
    print(f"Harvesting r/{SUBREDDIT} via PullPush — target {target} rows, "
          f"{len(queries)} queries (round-robin)...")
    seen = set()
    rows = []

    # Per-query pagination cursors and an "exhausted" flag so a dry query drops out.
    cursors = {q: None for q in queries}
    exhausted = {q: False for q in queries}

    for page in range(max_pages):
        if len(rows) >= target or all(exhausted.values()):
            break
        for q in queries:
            if len(rows) >= target or exhausted[q]:
                continue
            label = q or "(broad)"
            page_rows, oldest, err = _fetch_query(q, cursors[q], seen)
            if err:
                print(f"  ! query '{label}' round {page + 1} skipped: {err}")
                exhausted[q] = True
                continue
            rows.extend(page_rows)
            print(f"  round {page + 1} · query '{label}': +{len(page_rows)} rows "
                  f"(total {len(rows)})")
            if oldest is None:
                exhausted[q] = True  # no more results for this query
            else:
                cursors[q] = oldest
            time.sleep(REQUEST_DELAY_S)

    if not rows:
        print("❌ No usable rows collected. Try --target lower or different --queries.")
        return False

    df = pd.DataFrame(rows[:target])
    # Add the empty annotation columns so the file is ready to label in place.
    df.insert(1, "label", "")
    df.insert(2, "annotation_notes", "")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n✅ Wrote {len(df)} unlabeled rows to '{out_path}'.")
    print("   Per-query spread (a rough proxy for class mix before you annotate):")
    for q, n in df["query"].value_counts().items():
        print(f"     {q:<14} {n}")
    print("   Next: annotate the 'label' and 'annotation_notes' columns, then run "
          "verify_dataset.py.")
    return True


# ── entry point ───────────────────────────────────────────────────────────────--
def main():
    parser = argparse.ArgumentParser(
        description="Harvest r/PremierLeague comments via PullPush (no credentials).")
    parser.add_argument("--target", type=int, default=250,
                        help="Number of usable rows to collect (default: 250).")
    parser.add_argument("--queries", default=None,
                        help="Comma-separated search terms. Default: a balanced built-in set "
                             "that boosts the rare 'analysis' class.")
    parser.add_argument("--max-pages", type=int, default=3,
                        help="Paginated pages to pull per query (default: 3).")
    parser.add_argument("--out", default="data/raw_posts.csv",
                        help="Output CSV path (default: data/raw_posts.csv).")
    args = parser.parse_args()

    if args.queries:
        queries = [q.strip() for q in args.queries.split(",")]
    else:
        queries = DEFAULT_QUERIES

    ok = scrape(args.target, queries, Path(args.out), max_pages=args.max_pages)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
