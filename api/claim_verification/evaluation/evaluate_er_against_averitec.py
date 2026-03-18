import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

from sentence_transformers import SentenceTransformer, util

from api.claim_verification.retrieval.bm25 import BM25Retriever
from api.claim_verification.retrieval.colbert_rerank import ColBERTReranker


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = text.replace("-lrb-", " ").replace("-rrb-", " ")
    text = text.replace("-lsb-", " ").replace("-rsb-", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def jaccard_similarity(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def overlap_recall(gold: str, pred: str) -> float:
    """
    Fraction of gold tokens covered by pred tokens.
    Useful when the retrieved passage is longer than the gold evidence.
    """
    tg = set(tokenize(gold))
    tp = set(tokenize(pred))
    if not tg:
        return 0.0
    return len(tg & tp) / len(tg)


def lexical_match(
    gold: str,
    pred: str,
    jaccard_threshold: float = 0.35,
    recall_threshold: float = 0.6,
) -> bool:
    """
    Lexical fuzzy match.
    """
    ng = normalize_text(gold)
    np = normalize_text(pred)

    if not ng or not np:
        return False

    if ng in np or np in ng:
        return True

    if jaccard_similarity(ng, np) >= jaccard_threshold:
        return True

    if overlap_recall(ng, np) >= recall_threshold:
        return True

    return False


def parse_evidence_cell(cell: str) -> List[str]:
    """
    Parse Averitec evidence column, which may look like:
    ["evidence 1", "evidence 2"]
    or similar Python-list-like text.
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


def load_averitec_csv(path: str) -> List[Dict[str, Any]]:
    """
    Supports:
    - headered CSV with columns including claim,evidence
    - headerless CSV where row[0]=claim, row[1]=evidence, row[2]=label(optional)
    """
    rows: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        raw_rows = list(reader)

    if not raw_rows:
        return rows

    header = [h.strip().lower() for h in raw_rows[0]]
    has_header = "claim" in header and "evidence" in header

    if has_header:
        claim_idx = header.index("claim")
        evidence_idx = header.index("evidence")

        for row in raw_rows[1:]:
            if not row or len(row) <= max(claim_idx, evidence_idx):
                continue

            claim = row[claim_idx].strip()
            evidence = parse_evidence_cell(row[evidence_idx])

            if not claim:
                continue

            rows.append(
                {
                    "claim": claim,
                    "gold_evidence": evidence,
                }
            )
    else:
        for row in raw_rows:
            if len(row) < 2:
                continue

            claim = row[0].strip()
            evidence = parse_evidence_cell(row[1])

            if not claim:
                continue

            rows.append(
                {
                    "claim": claim,
                    "gold_evidence": evidence,
                }
            )

    return rows


def match_gold_against_retrieved(
    gold_evidence: List[str],
    retrieved_evidence: List[str],
    model: SentenceTransformer,
    semantic_threshold: float = 0.72,
) -> Tuple[int, List[str], List[Dict[str, Any]]]:
    """
    Match gold evidence against retrieved evidence using:
    1. lexical match first
    2. semantic similarity fallback

    Returns:
    - matched_count
    - matched_gold_evidence
    - match_details
    """
    if not gold_evidence or not retrieved_evidence:
        return 0, [], []

    matched_count = 0
    matched_gold = []
    match_details = []

    # Precompute embeddings once per claim for efficiency
    gold_emb = model.encode(gold_evidence, convert_to_tensor=True, normalize_embeddings=True)
    pred_emb = model.encode(retrieved_evidence, convert_to_tensor=True, normalize_embeddings=True)

    sim_matrix = util.cos_sim(gold_emb, pred_emb)

    for i, gold in enumerate(gold_evidence):
        found = False
        best_sim = -1.0
        best_pred = None
        best_match_type = None

        # First try lexical matching
        for pred in retrieved_evidence:
            if lexical_match(gold, pred):
                found = True
                best_pred = pred
                best_match_type = "lexical"
                best_sim = 1.0
                break

        # If lexical failed, try semantic matching
        if not found:
            row = sim_matrix[i]
            best_idx = int(row.argmax().item())
            best_sim = float(row[best_idx].item())
            best_pred = retrieved_evidence[best_idx]

            if best_sim >= semantic_threshold:
                found = True
                best_match_type = "semantic"

        if found:
            matched_count += 1
            matched_gold.append(gold)

        match_details.append(
            {
                "gold_evidence": gold,
                "matched": found,
                "match_type": best_match_type,
                "best_similarity": round(best_sim, 4) if best_sim >= 0 else None,
                "best_retrieved_evidence": best_pred,
            }
        )

    return matched_count, matched_gold, match_details


def evaluate_claims(
    claims_data: List[Dict[str, Any]],
    bm25_index: str,
    checkpoint: str,
    bm25_k: int,
    top_k: int,
    batch_size: int,
    use_gpu: bool,
    semantic_threshold: float,
) -> List[Dict[str, Any]]:
    bm25 = BM25Retriever(index_path=bm25_index, k=bm25_k)
    reranker = ColBERTReranker(
        checkpoint=checkpoint,
        query_maxlen=64,
        doc_maxlen=180,
        use_gpu=use_gpu,
        batch_size=batch_size,
    )
    semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    results: List[Dict[str, Any]] = []

    for i, item in enumerate(claims_data, start=1):
        claim = item["claim"]
        gold_evidence = item["gold_evidence"]

        if not gold_evidence:
            results.append(
                {
                    "claim": claim,
                    "percentage_matched": None,
                    "matched_gold_count": 0,
                    "total_gold_count": 0,
                    "matched_gold_evidence": [],
                    "retrieved_evidence": [],
                    "match_details": [],
                }
            )
            continue

        candidates = bm25.retrieve(claim)
        reranked = reranker.rerank(claim, candidates, top_k=top_k)
        retrieved_evidence = [r["text"] for r in reranked if r.get("text")]

        matched_count, matched_gold, match_details = match_gold_against_retrieved(
            gold_evidence,
            retrieved_evidence,
            model=semantic_model,
            semantic_threshold=semantic_threshold,
        )

        percentage_matched = 100.0 * matched_count / len(gold_evidence)

        results.append(
            {
                "claim": claim,
                "percentage_matched": round(percentage_matched, 2),
                "matched_gold_count": matched_count,
                "total_gold_count": len(gold_evidence),
                "matched_gold_evidence": matched_gold,
                "gold_evidence": gold_evidence,
                "retrieved_evidence": retrieved_evidence,
                "match_details": match_details,
            }
        )

        print(
            f"[{i}/{len(claims_data)}] "
            f"claim='{claim[:70]}...' matched={matched_count}/{len(gold_evidence)} "
            f"({percentage_matched:.2f}%)"
        )

    return results


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in results if r["percentage_matched"] is not None]

    if not scored:
        return {
            "num_claims": len(results),
            "num_scored_claims": 0,
            "average_percentage_matched": None,
            "full_match_rate": None,
            "nonzero_match_rate": None,
        }

    avg = sum(r["percentage_matched"] for r in scored) / len(scored)
    full_match_rate = 100.0 * sum(1 for r in scored if r["percentage_matched"] == 100.0) / len(scored)
    nonzero_match_rate = 100.0 * sum(1 for r in scored if r["percentage_matched"] > 0.0) / len(scored)

    return {
        "num_claims": len(results),
        "num_scored_claims": len(scored),
        "average_percentage_matched": round(avg, 2),
        "full_match_rate": round(full_match_rate, 2),
        "nonzero_match_rate": round(nonzero_match_rate, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ER pipeline against Averitec gold evidence."
    )
    parser.add_argument(
        "--averitec_file",
        type=str,
        required=True,
        help="Path to Averitec CSV file (sample or full).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="api/claim_verification/evaluation/averitec_er_eval.json",
        help="Where to save per-claim evaluation results.",
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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--max_claims",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )
    parser.add_argument(
        "--semantic_threshold",
        type=float,
        default=0.72,
        help="Cosine similarity threshold for semantic evidence matching.",
    )

    args = parser.parse_args()

    claims_data = load_averitec_csv(args.averitec_file)

    if args.max_claims is not None:
        claims_data = claims_data[: args.max_claims]

    print(f"Loaded {len(claims_data)} claims from {args.averitec_file}")

    results = evaluate_claims(
        claims_data=claims_data,
        bm25_index=args.bm25_index,
        checkpoint=args.checkpoint,
        bm25_k=args.bm25_k,
        top_k=args.top_k,
        batch_size=args.batch_size,
        use_gpu=(not args.cpu),
        semantic_threshold=args.semantic_threshold,
    )

    summary = summarize_results(results)

    output = {
        "config": {
            "averitec_file": args.averitec_file,
            "bm25_k": args.bm25_k,
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "semantic_threshold": args.semantic_threshold,
            "max_claims": args.max_claims,
        },
        "summary": summary,
        "results": results,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved evaluation results to: {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()