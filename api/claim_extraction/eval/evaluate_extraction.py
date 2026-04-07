"""
Evaluation script for the claim extraction module.

Measures extraction *quality* against a hand-annotated gold set using
semantic similarity

Metrics reported per passage and overall:
  - Semantic Recall    : fraction of gold reference claims recovered
  - Semantic Precision : fraction of extracted claims that match a gold claim
  - Semantic F1        : harmonic mean of the above
  - Avg latency (s)    : mean extraction time per passage

A reference claim is considered "recovered" if at least one extracted claim
has cosine similarity >= SIMILARITY_THRESHOLD using sentence-transformers.

Usage (from repo root):
    # Run against the small hand-annotated gold set (default):
    python api/claim_extraction/eval/evaluate_extraction.py

    # Run against the first 50 rows of FEVERFact:
    python api/claim_extraction/eval/evaluate_extraction.py --feverfact api/claim_extraction/eval/feverfact.jsonl --limit 50 --use-context

    # Run against all 444 FEVERFact rows:
    python api/claim_extraction/eval/evaluate_extraction.py --feverfact api/claim_extraction/eval/feverfact.jsonl --use-context

    # Test a different model:
    python api/claim_extraction/eval/evaluate_extraction.py --model llama3

    # Lower the similarity threshold (more lenient matching):
    python api/claim_extraction/eval/evaluate_extraction.py --threshold 0.80

Requirements:
    pip install sentence-transformers
    ollama pull mistral   (or whichever model you choose)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Add repo root to sys.path so `api.*` imports work when script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# ---------------------------------------------------------------------------
# Optional rich output — falls back gracefully if not installed
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def _print(msg: str = "") -> None:
    if HAS_RICH:
        console.print(msg)
    else:
        print(msg)


# ---------------------------------------------------------------------------
# Config / defaults
# ---------------------------------------------------------------------------

GOLD_PATH = Path(__file__).parent / "gold_annotations.json"
DEFAULT_MODEL = "mixtral"
DEFAULT_THRESHOLD = 0.80
DEFAULT_MAX_CLAIMS = 10
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_gold_annotations(path: Path) -> List[dict]:
    """
    Load the small hand-annotated gold set (JSON array).

    Expected format:
        [{"id": "apollo_01", "passage": "...", "reference_claims": ["...", ...]}, ...]
    """
    with open(path) as f:
        data = json.load(f)
    return data


def load_feverfact(path: Path, limit: int = None, use_context: bool = False) -> List[dict]:
    """
    Load FEVERFact entries from a JSONL file and convert to the internal format.

    FEVERFact row fields used:
        sentence_id      -> used as the entry id
        source           -> Wikipedia article title (used in id for readability)
        sentence         -> the single source sentence the claims were extracted from
        sentence_context -> list of 3 strings: [prev_sentence, source_sentence, next_sentence]
        claims           -> list of gold reference claim strings

    Args:
        path:        Path to the .jsonl file.
        limit:       If set, only load the first N rows.
        use_context: If True, pass the full 3-sentence context as the passage.
                     If False (default), pass only the single source sentence.
                     The single sentence is the fair default because the reference
                     claims only cover that sentence — using context causes the
                     model to extract additional valid claims that have no
                     reference to match against, artificially lowering precision.

    Returns:
        List of dicts in the same format as gold_annotations.json.
    """
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            # Build a readable id from source title + sentence_id
            source_slug = row["source"].replace(" ", "_")[:30]
            entry_id = f"{source_slug}_{row['sentence_id']}"

            if use_context:
                context = row.get("sentence_context", [])
                passage = " ".join(context) if isinstance(context, list) else row["sentence"]
            else:
                passage = row["sentence"]

            entries.append({
                "id": entry_id,
                "passage": passage,
                "reference_claims": row["claims"],
            })

            if limit is not None and len(entries) >= limit:
                break

    return entries


# ---------------------------------------------------------------------------
# Lazy import of heavy deps
# ---------------------------------------------------------------------------

def _load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        _print("[red]sentence-transformers is not installed.[/red]")
        _print("Run:  pip install sentence-transformers")
        sys.exit(1)
    _print(f"[dim]Loading embedding model ({EMBEDDING_MODEL})…[/dim]")
    return SentenceTransformer(EMBEDDING_MODEL)


def _load_claim_extractor():
    try:
        from api.claim_extraction.claim_extraction import extract_claims
        return extract_claims
    except ImportError as exc:
        _print(f"[red]Could not import extract_claims: {exc}[/red]")
        _print("Make sure you are running from the repo root.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Semantic matching helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a, b) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _best_similarity(candidate_emb, reference_embs) -> float:
    return max(_cosine_similarity(candidate_emb, ref) for ref in reference_embs)


def compute_prf(
    extracted: List[str],
    reference: List[str],
    embedder,
    threshold: float,
) -> Tuple[float, float, float]:
    if not reference:
        return (1.0, 1.0, 1.0) if not extracted else (0.0, 1.0, 0.0)
    if not extracted:
        return (0.0, 0.0, 0.0)

    ref_embs = embedder.encode(reference, convert_to_numpy=True).tolist()
    ext_embs = embedder.encode(extracted, convert_to_numpy=True).tolist()

    recovered = sum(
        1 for ref_emb in ref_embs
        if _best_similarity(ref_emb, ext_embs) >= threshold
    )
    recall = recovered / len(reference)

    matched = sum(
        1 for ext_emb in ext_embs
        if _best_similarity(ext_emb, ref_embs) >= threshold
    )
    precision = matched / len(extracted)

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


# ---------------------------------------------------------------------------
# Per-passage evaluation
# ---------------------------------------------------------------------------

def evaluate_passage(
    entry: dict,
    extract_claims,
    embedder,
    model: str,
    max_claims: int,
    threshold: float,
) -> dict:
    passage_id = entry["id"]
    passage = entry["passage"]
    reference = entry["reference_claims"]

    t0 = time.perf_counter()
    try:
        extracted = extract_claims(passage, model=model, max_claims=max_claims)
        error = None
    except RuntimeError as exc:
        extracted = []
        error = str(exc)
    latency = time.perf_counter() - t0

    precision, recall, f1 = compute_prf(extracted, reference, embedder, threshold)

    return {
        "id": passage_id,
        "passage_snippet": passage[:80] + "…" if len(passage) > 80 else passage,
        "n_reference": len(reference),
        "n_extracted": len(extracted),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency_s": latency,
        "error": error,
        "extracted_claims": extracted,
        "reference_claims": reference,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(value: float) -> str:
    return f"{value:.3f}"


def print_results_table(results: List[dict]) -> None:
    # Only print the per-row table for small runs — skip for large FEVERFact runs
    if len(results) > 20:
        _print(f"[dim](Skipping per-row table for {len(results)} entries — see saved JSON for details)[/dim]")
        return

    if HAS_RICH:
        table = Table(title="Claim Extraction Evaluation Results", show_lines=True)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Ref", justify="right")
        table.add_column("Ext", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("Latency (s)", justify="right")
        table.add_column("Error", style="red")

        for r in results:
            table.add_row(
                r["id"],
                str(r["n_reference"]),
                str(r["n_extracted"]),
                _fmt(r["precision"]),
                _fmt(r["recall"]),
                _fmt(r["f1"]),
                f"{r['latency_s']:.2f}",
                r["error"] or "",
            )
        console.print(table)
    else:
        header = f"{'ID':<35} {'Ref':>4} {'Ext':>4} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Lat(s)':>7}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r['id']:<35} {r['n_reference']:>4} {r['n_extracted']:>4} "
                f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} "
                f"{r['latency_s']:>7.2f}"
                + (f"  ERROR: {r['error']}" if r["error"] else "")
            )


def print_summary(results: List[dict], model: str, threshold: float, dataset_label: str) -> None:
    valid = [r for r in results if r["error"] is None]
    n_total = len(results)
    n_errors = sum(1 for r in results if r["error"])

    if not valid:
        _print("[red]All passages failed — no summary available.[/red]")
        return

    avg_precision = sum(r["precision"] for r in valid) / len(valid)
    avg_recall    = sum(r["recall"]    for r in valid) / len(valid)
    avg_f1        = sum(r["f1"]        for r in valid) / len(valid)
    avg_latency   = sum(r["latency_s"] for r in valid) / len(valid)
    max_latency   = max(r["latency_s"] for r in valid)

    _print()
    _print("=" * 55)
    _print(f"  Dataset            : {dataset_label}")
    _print(f"  Model              : {model}")
    _print(f"  Similarity threshold: {threshold}")
    _print(f"  Passages evaluated : {n_total}  ({n_errors} errors)")
    _print("-" * 55)
    _print(f"  Avg Semantic Precision : {avg_precision:.3f}")
    _print(f"  Avg Semantic Recall    : {avg_recall:.3f}  (target ≥ 0.75)")
    _print(f"  Avg Semantic F1        : {avg_f1:.3f}")
    _print(f"  Avg latency            : {avg_latency:.2f}s  (target < 10s)")
    _print(f"  Max latency            : {max_latency:.2f}s")
    _print("=" * 55)

    if avg_recall < 0.75:
        _print(f"[yellow]  ⚠ Recall {avg_recall:.3f} is below the 0.75 target.[/yellow]")
    else:
        _print(f"[green]  ✓ Recall target (≥ 0.75) met.[/green]")

    if max_latency > 10.0:
        _print(f"[yellow]  ⚠ Max latency {max_latency:.2f}s exceeds the 10s target.[/yellow]")
    else:
        _print(f"[green]  ✓ Latency target (< 10s) met.[/green]")


def save_results(results: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    _print(f"\n[dim]Full results saved to {output_path}[/dim]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate claim extraction quality against gold annotations or FEVERFact."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama model name to evaluate (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Cosine similarity threshold for a claim match (default: {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--max-claims", type=int, default=DEFAULT_MAX_CLAIMS,
        help=f"Max claims to extract per passage (default: {DEFAULT_MAX_CLAIMS})"
    )

    # Dataset selection — mutually exclusive
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--gold", type=Path, default=None,
        help="Path to hand-annotated gold JSON file (default if --feverfact not set)"
    )
    dataset_group.add_argument(
        "--feverfact", type=Path, default=None,
        help="Path to FEVERFact JSONL file (e.g. eval/feverfact.jsonl)"
    )

    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of passages to evaluate (useful for quick FEVERFact runs, e.g. --limit 50)"
    )
    parser.add_argument(
        "--use-context", action="store_true", default=False,
        help=(
            "FEVERFact only: pass the full 3-sentence context as the passage instead of "
            "just the source sentence. Not recommended for standard evaluation because "
            "reference claims only cover the middle sentence — the model will extract valid "
            "claims from surrounding sentences that have no reference to match against, "
            "artificially inflating extracted count and lowering precision."
        )
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Where to write full results JSON (auto-named by dataset if not set)"
    )
    parser.add_argument(
        "--ids", nargs="*",
        help="Evaluate only specific passage IDs — gold set only (e.g. --ids apollo_01 climate_01)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve dataset
    if args.feverfact is not None:
        if not args.feverfact.exists():
            _print(f"[red]FEVERFact file not found: {args.feverfact}[/red]")
            sys.exit(1)
        use_context = args.use_context
        data = load_feverfact(args.feverfact, limit=args.limit, use_context=use_context)
        context_note = " [context]" if use_context else " [sentence only]"
        dataset_label = f"FEVERFact ({args.feverfact.name}, n={len(data)}{context_note})"
        default_output = Path("api/claim_extraction/eval/results_feverfact.json")
    else:
        gold_path = args.gold or GOLD_PATH
        if not gold_path.exists():
            _print(f"[red]Gold annotations file not found: {gold_path}[/red]")
            sys.exit(1)
        data = load_gold_annotations(gold_path)
        if args.limit:
            data = data[:args.limit]
        if args.ids:
            data = [e for e in data if e["id"] in args.ids]
            if not data:
                _print(f"[red]No entries matched IDs: {args.ids}[/red]")
                sys.exit(1)
        dataset_label = f"Gold annotations ({gold_path.name}, n={len(data)})"
        default_output = Path("api/claim_extraction/eval/results_gold.json")

    output_path = args.output or default_output

    _print(f"\n[bold]Claim Extraction Evaluation[/bold]")
    _print(f"Dataset: [cyan]{dataset_label}[/cyan]")
    _print(f"Model: [cyan]{args.model}[/cyan] | Threshold: [cyan]{args.threshold}[/cyan]\n")

    embedder = _load_embedder()
    extract_claims = _load_claim_extractor()

    results = []
    for i, entry in enumerate(data, 1):
        _print(f"[dim]({i}/{len(data)}) Evaluating: {entry['id']}…[/dim]")
        result = evaluate_passage(
            entry,
            extract_claims,
            embedder,
            model=args.model,
            max_claims=args.max_claims,
            threshold=args.threshold,
        )
        results.append(result)

        if result["error"]:
            _print(f"  [red]Error: {result['error']}[/red]")
        else:
            _print(
                f"  Precision={result['precision']:.3f}  "
                f"Recall={result['recall']:.3f}  "
                f"F1={result['f1']:.3f}  "
                f"({result['latency_s']:.1f}s)"
            )

    _print()
    print_results_table(results)
    print_summary(results, model=args.model, threshold=args.threshold, dataset_label=dataset_label)
    save_results(results, output_path)


if __name__ == "__main__":
    main()