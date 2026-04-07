import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

from api.claim_verification.retrieval.colbert_rerank import ClaimEvidenceRetriever


def load_claims(csv_path: str, limit: Optional[int] = None) -> List[str]:
    df = pd.read_csv(csv_path)
    if "claim" not in df.columns:
        raise ValueError(f"'claim' column not found in {csv_path}")

    claims = [str(value).strip() for value in df["claim"].tolist() if str(value).strip()]
    if limit is not None:
        claims = claims[:limit]
    return claims


def run_benchmark(
    claims: List[str],
    retriever: ClaimEvidenceRetriever,
    output_csv: Optional[str] = None,
) -> None:
    rows = []

    for i, claim in enumerate(claims, start=1):
        start = time.perf_counter()
        result = retriever.retrieve(claim)
        elapsed = time.perf_counter() - start

        rows.append(
            {
                "claim": claim,
                "latency_seconds": elapsed,
                "evidence_count": len(result.get("evidence", [])),
                "confidence": bool(result.get("confidence", False)),
                "force_nei": bool(result.get("force_nei", False)),
                "source": result.get("source", "unknown"),
                "claim_length_chars": len(claim),
                "claim_length_words": len(claim.split()),
            }
        )
        print(
            f"[{i}/{len(claims)}] {elapsed:.2f}s | "
            f"evidence={rows[-1]['evidence_count']} | source={rows[-1]['source']}"
        )

    latencies = [row["latency_seconds"] for row in rows]
    under_budget = sum(1 for value in latencies if value <= 30.0)

    print("\nER latency summary")
    print(f"claims tested: {len(rows)}")
    print(f"mean latency: {statistics.mean(latencies):.2f}s")
    print(f"median latency: {statistics.median(latencies):.2f}s")
    print(f"max latency: {max(latencies):.2f}s")
    print(f"within 30s budget: {under_budget}/{len(rows)}")

    if output_csv:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "claim",
                    "latency_seconds",
                    "evidence_count",
                    "confidence",
                    "force_nei",
                    "source",
                    "claim_length_chars",
                    "claim_length_words",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved benchmark report to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark evidence retrieval latency.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="api/claim_verification/data/processed/averitec.csv",
        help="CSV containing a 'claim' column.",
    )
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument(
        "--output_csv",
        type=str,
        default="test_results/test_results_ER/er_latency_report.csv",
    )
    parser.add_argument(
        "--bm25_index",
        type=str,
        default="api/claim_verification/data/wiki_corpus/bm25_index",
    )
    parser.add_argument("--checkpoint", type=str, default="colbert-ir/colbertv2.0")
    parser.add_argument("--bm25_k", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--evidence_k", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--enable_live_news", action="store_true")
    args = parser.parse_args()

    claims = load_claims(args.input_csv, limit=args.limit)
    retriever = ClaimEvidenceRetriever(
        bm25_index=args.bm25_index,
        checkpoint=args.checkpoint,
        use_gpu=(not args.cpu),
        batch_size=args.batch_size,
        bm25_k=args.bm25_k,
        top_k=args.top_k,
        evidence_k=args.evidence_k,
        enable_live_news=args.enable_live_news,
    )
    run_benchmark(claims, retriever, output_csv=args.output_csv)


if __name__ == "__main__":
    main()
