import argparse
import re
from typing import List, Dict, Any

import torch

from api.claim_verification.retrieval.bm25 import BM25Retriever
from api.claim_verification.retrieval.confidence import has_sufficient_evidence
from api.claim_verification.retrieval.live_news import live_news_passages

from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint


def _safe_text(c: Dict[str, Any]) -> str:
    t = c.get("contents") or c.get("text") or c.get("raw") or ""
    if t is None:
        return ""
    if not isinstance(t, str):
        t = str(t)
    return t.strip()


_URL_LINE_RE = re.compile(r"^\s*URL:\s*https?://\S+\s*$", re.IGNORECASE)
_BARE_URL_RE = re.compile(r"^\s*https?://\S+\s*$", re.IGNORECASE)


def _strip_url_lines(text: str) -> str:
    if not text:
        return ""

    cleaned_lines = []
    for line in text.splitlines():
        if _URL_LINE_RE.match(line) or _BARE_URL_RE.match(line):
            continue
        cleaned_lines.append(line.rstrip())

    return "\n".join(cleaned_lines).strip()


class ColBERTReranker:
    def __init__(
        self,
        checkpoint: str = "colbert-ir/colbertv2.0",
        query_maxlen: int = 64,
        doc_maxlen: int = 180,
        use_gpu: bool = True,
        batch_size: int = 16,
    ):
        self.checkpoint_name = checkpoint
        self.batch_size = max(1, int(batch_size))

        self.config = ColBERTConfig(
            query_maxlen=int(query_maxlen),
            doc_maxlen=int(doc_maxlen),
        )

        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

        self.ckpt = Checkpoint(self.checkpoint_name, colbert_config=self.config)
        self.ckpt = self.ckpt.to(self.device)
        self.ckpt.eval()

    @torch.no_grad()
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 10):
        if not candidates:
            return []

        texts: List[str] = []
        pid2cid: List[int] = []
        for cid, c in enumerate(candidates):
            t = _safe_text(c)
            if not t:
                continue
            texts.append(t)
            pid2cid.append(cid)

        if not texts:
            return []

        scores: List[float] = []
        bs = self.batch_size

        for i in range(0, len(texts), bs):
            batch_docs = texts[i:i + bs]
            batch_queries = [query] * len(batch_docs)

            Q = self.ckpt.queryFromText(batch_queries, bsize=len(batch_queries))
            D = self.ckpt.docFromText(batch_docs, bsize=len(batch_docs))

            if isinstance(Q, (tuple, list)):
                Q = Q[0]
            if isinstance(D, (tuple, list)):
                D = D[0]

            Q = Q.to(self.device).float()
            D = D.to(self.device).float()

            sim = torch.einsum("bqd,bkd->bqk", Q, D)
            max_sim, _ = sim.max(dim=2)
            s = max_sim.sum(dim=1)

            scores.extend([float(x) for x in s.detach().cpu().tolist()])

        reranked: List[Dict[str, Any]] = []
        for pid, score in enumerate(scores):
            cid = pid2cid[pid]
            c = candidates[cid]
            docid = c.get("docid", f"pid={pid}")

            reranked.append(
                {
                    "docid": docid,
                    "text": _strip_url_lines(_safe_text(c)),
                    "bm25_score": float(c.get("score", 0.0)),
                    "colbert_score": float(score),
                    "source_type": c.get("source_type", "local"),
                    "provider": c.get("provider"),
                    "url": c.get("url"),
                    "domain": c.get("domain"),
                    "published_at": c.get("published_at"),
                    "title": c.get("title"),
                    "news_source": c.get("news_source"),
                }
            )

        reranked.sort(key=lambda x: x["colbert_score"], reverse=True)
        return reranked[: int(top_k)]


class ClaimEvidenceRetriever:
    """
    Reusable ER pipeline:
      BM25 -> ColBERT -> confidence gate -> optional live news fallback

    Main callable:
      retrieve(claim) -> {
          "claim": str,
          "evidence": List[str],
          "confidence": bool,
          "force_nei": bool,
          "source": str,
          "reranked": List[Dict[str, Any]]
      }
    """

    def __init__(
        self,
        bm25_index: str = "api/claim_verification/data/wiki_corpus/bm25_index",
        checkpoint: str = "colbert-ir/colbertv2.0",
        query_maxlen: int = 64,
        doc_maxlen: int = 180,
        use_gpu: bool = True,
        batch_size: int = 16,
        bm25_k: int = 200,
        top_k: int = 10,
        evidence_k: int = 5,
        enable_live_news: bool = True,
        news_limit: int = 8,
        news_hours_back: int = 72,
        news_language: str = "en",
        allow_untrusted_domains: bool = False,
    ):
        self.bm25 = BM25Retriever(index_path=bm25_index, k=bm25_k)
        self.reranker = ColBERTReranker(
            checkpoint=checkpoint,
            query_maxlen=query_maxlen,
            doc_maxlen=doc_maxlen,
            use_gpu=use_gpu,
            batch_size=batch_size,
        )

        self.top_k = int(top_k)
        self.evidence_k = int(evidence_k)

        self.enable_live_news = enable_live_news
        self.news_limit = int(news_limit)
        self.news_hours_back = int(news_hours_back)
        self.news_language = news_language
        self.allow_untrusted_domains = allow_untrusted_domains

    def retrieve(self, claim: str) -> Dict[str, Any]:
        candidates = self.bm25.retrieve(claim)
        reranked = self.reranker.rerank(claim, candidates, top_k=self.top_k)
        confident = has_sufficient_evidence(reranked)

        source = "local"
        final_reranked = reranked

        if (not confident) and self.enable_live_news:
            try:
                web_candidates = live_news_passages(
                    claim,
                    language=self.news_language,
                    limit=self.news_limit,
                    hours_back=self.news_hours_back,
                    trusted_only=(not self.allow_untrusted_domains),
                    max_chunks_per_article=6,
                )
            except Exception as exc:
                # Keep retrieval resilient: if live-news lookup fails,
                # continue with local corpus evidence instead of crashing.
                print(f"[ClaimEvidenceRetriever] Live-news fallback failed: {exc}")
                web_candidates = []

            if web_candidates:
                merged_candidates = candidates + web_candidates
                final_reranked = self.reranker.rerank(
                    claim,
                    merged_candidates,
                    top_k=self.top_k,
                )
                confident = has_sufficient_evidence(final_reranked)
                source = "local+web"

        evidence = [
            r["text"] for r in final_reranked[: self.evidence_k] if r.get("text")
        ]

        return {
            "claim": claim,
            "evidence": evidence,
            "confidence": bool(confident),
            "force_nei": (not confident),
            "source": source,
            "reranked": final_reranked,
        }


def print_results(reranked: List[Dict[str, Any]], header: str):
    print(f"\n=== {header} ===")
    for i, r in enumerate(reranked, 1):
        extra = ""
        if r.get("url"):
            extra = f" source={r.get('domain', '')} url={r['url']}"
        print(
            f"\n#{i} docid={r['docid']} bm25={r['bm25_score']:.3f} "
            f"colbert={r['colbert_score']:.3f} type={r.get('source_type', 'local')}{extra}"
        )
        print(r["text"][:350])


def main():
    parser = argparse.ArgumentParser(description="BM25 -> ColBERTv2 reranking with live-news fallback")
    parser.add_argument("query", type=str, help="Claim/query")

    parser.add_argument(
        "--bm25_index",
        type=str,
        default="api/claim_verification/data/wiki_corpus/bm25_index",
    )
    parser.add_argument("--bm25_k", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--evidence_k", type=int, default=5)

    parser.add_argument("--checkpoint", type=str, default="colbert-ir/colbertv2.0")
    parser.add_argument("--query_maxlen", type=int, default=64)
    parser.add_argument("--doc_maxlen", type=int, default=180)

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--enable_live_news", action="store_true")
    parser.add_argument("--news_limit", type=int, default=8)
    parser.add_argument("--news_hours_back", type=int, default=72)
    parser.add_argument("--news_language", type=str, default="en")
    parser.add_argument("--allow_untrusted_domains", action="store_true")

    args = parser.parse_args()

    retriever = ClaimEvidenceRetriever(
        bm25_index=args.bm25_index,
        checkpoint=args.checkpoint,
        query_maxlen=args.query_maxlen,
        doc_maxlen=args.doc_maxlen,
        use_gpu=(not args.cpu),
        batch_size=args.batch_size,
        bm25_k=args.bm25_k,
        top_k=args.top_k,
        evidence_k=args.evidence_k,
        enable_live_news=args.enable_live_news,
        news_limit=args.news_limit,
        news_hours_back=args.news_hours_back,
        news_language=args.news_language,
        allow_untrusted_domains=args.allow_untrusted_domains,
    )

    result = retriever.retrieve(args.query)

    print("\nFinal evidence confidence:", result["confidence"])
    print("Force NEI:", result["force_nei"])
    print("Source:", result["source"])
    print_results(result["reranked"], "Top results after ER pipeline")


if __name__ == "__main__":
    main()
