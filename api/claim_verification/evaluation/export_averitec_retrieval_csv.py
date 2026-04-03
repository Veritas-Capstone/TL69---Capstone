import argparse
import ast
import csv
import json
from pathlib import Path
from typing import List, Dict, Any

from api.claim_verification.retrieval.bm25 import BM25Retriever
from api.claim_verification.retrieval.colbert_rerank import ColBERTReranker


def parse_evidence_cell(cell: str) -> List[str]:
    """
    Kept here only in case you want to inspect original evidence columns later.
    Not required for retrieval export, but harmless to keep.
    """
    if cell is None:
        return []

    cell = cell.strip()
    if not cell:
        return []

    try:
        val = json.loads(cell)
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
    except Exception:
        pass

    try:
        val = ast.literal_eval(cell)
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
    except Exception:
        pass

    return []


def load_claims_from_csv(path: str) -> List[str]:
    """
    Load claims from a CSV file.
    Supports:
    - headered CSV with a 'claim' column
    - fallback to first column if no header is found
    """
    claims: List[str] = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return claims

    header = [h.strip().lower() for h in rows[0]]
    has_header = "claim" in header

    if has_header:
        claim_idx = header.index("claim")
        for row in rows[1:]:
            if len(row) <= claim_idx:
                continue
            claim = row[claim_idx].strip()
            if claim:
                claims.append(claim)
    else:
        for row in rows:
            if not row:
                continue
            claim = row[0].strip()
            if claim:
                claims.append(claim)

    return claims


def export_retrieval_csv(
    claims: List[str],
    output_csv: str,
    bm25_index: str,
    checkpoint: str,
    bm25_k: int,
    top_k: int,
    evidence_k: int,
    batch_size: int,
    use_gpu: bool,
) -> None:
    bm25 = BM25Retriever(index_path=bm25_index, k=bm25_k)
    reranker = ColBERTReranker(
        checkpoint=checkpoint,
        query_maxlen=64,
        doc_maxlen=180,
        use_gpu=use_gpu,
        batch_size=batch_size,
    )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["claim", "evidence"])

        for i, claim in enumerate(claims, start=1):
            candidates = bm25.retrieve(claim)
            reranked = reranker.rerank(claim, candidates, top_k=top_k)

            evidence_list = [
                r["text"]
                for r in reranked[:evidence_k]
                if r.get("text")
            ]

            writer.writerow([claim, json.dumps(evidence_list, ensure_ascii=False)])

            print(
                f"[{i}/{len(claims)}] exported claim='{claim[:70]}...' "
                f"with {len(evidence_list)} evidence passages"
            )

    print(f"\nSaved retrieval CSV to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export ColBERT retrieval results for Averitec claims into Averitec-style CSV format."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to Averitec CSV file containing claims.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="api/claim_verification/evaluation/averitec_retrieved_evidence.csv",
        help="Where to save the exported retrieval CSV.",
    )
    parser.add_argument(
        "--bm25_index",
        type=str,
        default="api/claim_verification/data/wiki_corpus/bm25_index",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="colbert-ir/colbertv2.0",
    )
    parser.add_argument("--bm25_k", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--evidence_k",
        type=int,
        default=2,
        help="How many top reranked evidence passages to write per claim.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--max_claims",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )

    args = parser.parse_args()

    claims = load_claims_from_csv(args.input_csv)

    if args.max_claims is not None:
        claims = claims[: args.max_claims]

    print(f"Loaded {len(claims)} claims from {args.input_csv}")

    export_retrieval_csv(
        claims=claims,
        output_csv=args.output_csv,
        bm25_index=args.bm25_index,
        checkpoint=args.checkpoint,
        bm25_k=args.bm25_k,
        top_k=args.top_k,
        evidence_k=args.evidence_k,
        batch_size=args.batch_size,
        use_gpu=(not args.cpu),
    )


if __name__ == "__main__":
    main()