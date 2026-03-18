from __future__ import annotations

from pyserini.search.lucene import LuceneSearcher
import argparse
import json
from typing import Any, Dict, List


class BM25Retriever:
    """
    Thin wrapper around Pyserini LuceneSearcher.

    Contract:
      retrieve() ALWAYS returns dicts with:
        - docid (str)
        - score (float)         # bm25 score
        - text (str)            # passage text to feed rerankers / NLI
        - title_or_id (str)     # convenience identifier
        - raw (dict)            # parsed raw JSON stored in the index
    """
    def __init__(self, index_path: str, k: int = 200):
        self.searcher = LuceneSearcher(index_path)
        self.k = k

    def retrieve(self, query: str, k: int | None = None) -> List[Dict[str, Any]]:
        k = self.k if k is None else k
        hits = self.searcher.search(query, k)

        results: List[Dict[str, Any]] = []
        for h in hits:
            doc = self.searcher.doc(h.docid)

            # doc.raw() is a JSON string like: {"id": "...", "contents": "..."}
            raw_str = doc.raw() if doc is not None else "{}"
            try:
                raw = json.loads(raw_str) if raw_str else {}
            except json.JSONDecodeError:
                raw = {}

            contents = raw.get("contents", "") or ""

            # This is the key change: ALWAYS provide "text"
            results.append({
                "docid": h.docid,
                "score": float(h.score),
                "title_or_id": raw.get("id", h.docid),
                "text": contents,
                "raw": raw,
            })

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 retrieval over Wikipedia corpus (Pyserini)")
    parser.add_argument("query", type=str, help="Claim / query string")
    parser.add_argument(
        "--index",
        type=str,
        default="api/claim_verification/data/wiki_corpus/bm25_index",
        help="Path to BM25 Lucene index",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of documents to retrieve")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON output instead of text preview",
    )
    parser.add_argument(
        "--chars",
        type=int,
        default=300,
        help="How many characters of each passage to preview (non-json mode)",
    )
    args = parser.parse_args()

    retriever = BM25Retriever(index_path=args.index, k=args.k)
    hits = retriever.retrieve(args.query)

    if args.json:
        print(json.dumps(hits, indent=2))
        return

    for i, h in enumerate(hits, 1):
        print(f"\n#{i} docid={h['docid']} score={h['score']:.3f}")
        print(h["text"][: args.chars])


if __name__ == "__main__":
    main()
    