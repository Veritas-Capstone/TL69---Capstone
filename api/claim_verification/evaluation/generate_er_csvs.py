import csv
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from api.claim_verification.retrieval.colbert_rerank import ClaimEvidenceRetriever


def load_claim_rows(csv_path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)

    if "claim" not in df.columns:
        raise ValueError(f"'claim' column not found in {csv_path}")

    rows = []
    for _, row in df.iterrows():
        rows.append({"claim": str(row["claim"]).strip()})
    return rows


def write_er_csv(
    input_csv: str,
    output_csv: str,
    retriever: ClaimEvidenceRetriever,
    progress_every: int = 100,
):
    rows = load_claim_rows(input_csv)
    total = len(rows)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["claim", "evidence"])

        for i, row in enumerate(rows, start=1):
            claim = row["claim"]

            try:
                result = retriever.retrieve(claim)
            except Exception as e:
                print(f"[ERROR] Claim failed: {claim[:80]}... | {e}")
                continue

            writer.writerow(
                [
                    claim,
                    json.dumps(result["evidence"], ensure_ascii=False),
                ]
            )

            f.flush()

            if progress_every > 0 and (i % progress_every == 0 or i == total):
                print(f"[{output_csv}] {i}/{total} done")


def main():
    # -----------------------------
    # CONFIG
    # -----------------------------
    BM25_INDEX = "api/claim_verification/data/wiki_corpus/bm25_index"
    COLBERT_CHECKPOINT = "colbert-ir/colbertv2.0"

    BATCH_SIZE = 32
    BM25_K = 50
    TOP_K = 5
    EVIDENCE_K = 2

    ENABLE_LIVE_NEWS = False
    PROGRESS_EVERY = 100
    NEWS_LIMIT = 3
    NEWS_HOURS_BACK = 168
    NEWS_LANGUAGE = "en"
    ALLOW_UNTRUSTED_DOMAINS = False

    # -----------------------------
    # DATASETS
    # -----------------------------
    RUN_AVERITEC = False
    RUN_FEVER = True

    AVERITEC_INPUT = "api/claim_verification/data/processed/averitec.csv"
    AVERITEC_OUTPUT = "api/claim_verification/evaluation/averitec_full_er.csv"

    FEVER_INPUT = "api/claim_verification/data/processed/fever_train_claims.csv"
    FEVER_OUTPUT = "api/claim_verification/evaluation/fever_full_er.csv"

    retriever = ClaimEvidenceRetriever(
        bm25_index=BM25_INDEX,
        checkpoint=COLBERT_CHECKPOINT,
        use_gpu=True,
        batch_size=BATCH_SIZE,
        bm25_k=BM25_K,
        top_k=TOP_K,
        evidence_k=EVIDENCE_K,
        enable_live_news=ENABLE_LIVE_NEWS,
        news_limit=NEWS_LIMIT,
        news_hours_back=NEWS_HOURS_BACK,
        news_language=NEWS_LANGUAGE,
        allow_untrusted_domains=ALLOW_UNTRUSTED_DOMAINS,
    )

    if RUN_AVERITEC:
        print("\n=== Running Averitec ===")
        write_er_csv(
            input_csv=AVERITEC_INPUT,
            output_csv=AVERITEC_OUTPUT,
            retriever=retriever,
            progress_every=PROGRESS_EVERY,
        )

    if RUN_FEVER:
        print("\n=== Running FEVER ===")
        write_er_csv(
            input_csv=FEVER_INPUT,
            output_csv=FEVER_OUTPUT,
            retriever=retriever,
            progress_every=PROGRESS_EVERY,
        )


if __name__ == "__main__":
    main()