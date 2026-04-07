import time

from api.claim_verification.retrieval.colbert_rerank import (
    ClaimEvidenceRetriever,
)


class _FakeBM25:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, claim):
        self.calls.append(claim)
        return list(self.results)


class _FakeReranker:
    def __init__(self, first_pass, second_pass=None):
        self.first_pass = list(first_pass)
        self.second_pass = None if second_pass is None else list(second_pass)
        self.calls = []

    def rerank(self, query, candidates, top_k=10):
        self.calls.append(
            {
                "query": query,
                "candidate_count": len(candidates),
                "top_k": top_k,
            }
        )
        if len(self.calls) == 1 or self.second_pass is None:
            return list(self.first_pass[:top_k])
        return list(self.second_pass[:top_k])


def _make_retriever(
    *,
    bm25_results,
    first_pass,
    second_pass=None,
    evidence_k=2,
    enable_live_news=False,
):
    retriever = ClaimEvidenceRetriever.__new__(ClaimEvidenceRetriever)
    retriever.bm25 = _FakeBM25(bm25_results)
    retriever.reranker = _FakeReranker(first_pass, second_pass)
    retriever.top_k = 10
    retriever.evidence_k = evidence_k
    retriever.enable_live_news = enable_live_news
    retriever.news_limit = 3
    retriever.news_hours_back = 72
    retriever.news_language = "en"
    retriever.allow_untrusted_domains = False
    return retriever

def test_retrieve_returns_expected_contract(monkeypatch):
    monkeypatch.setattr(
        "api.claim_verification.retrieval.colbert_rerank.has_sufficient_evidence",
        lambda reranked: True,
    )

    reranked = [
        {
            "docid": "doc-1",
            "text": "Passage one",
            "bm25_score": 12.3,
            "colbert_score": 0.91,
            "source_type": "local",
        },
        {
            "docid": "doc-2",
            "text": "Passage two",
            "bm25_score": 10.1,
            "colbert_score": 0.77,
            "source_type": "local",
        },
    ]
    retriever = _make_retriever(
        bm25_results=[{"docid": "raw-1", "text": "raw passage", "score": 3.5}],
        first_pass=reranked,
        evidence_k=2,
    )

    result = retriever.retrieve("The sky is blue")

    assert result["claim"] == "The sky is blue"
    assert result["confidence"] is True
    assert result["force_nei"] is False
    assert result["source"] == "local"
    assert result["evidence"] == ["Passage one", "Passage two"]
    assert len(result["reranked"]) == 2
    assert result["reranked"][0]["docid"] == "doc-1"
    assert isinstance(result["reranked"][0]["bm25_score"], float)
    assert isinstance(result["reranked"][0]["colbert_score"], float)


def test_retrieve_handles_empty_results(monkeypatch):
    monkeypatch.setattr(
        "api.claim_verification.retrieval.colbert_rerank.has_sufficient_evidence",
        lambda reranked: False,
    )

    retriever = _make_retriever(
        bm25_results=[],
        first_pass=[],
        evidence_k=3,
    )

    result = retriever.retrieve("")

    assert result["claim"] == ""
    assert result["evidence"] == []
    assert result["confidence"] is False
    assert result["force_nei"] is True
    assert result["source"] == "local"
    assert result["reranked"] == []


def test_retrieve_uses_live_news_fallback_when_local_evidence_is_not_confident(monkeypatch):
    confidence_values = iter([False, True])
    monkeypatch.setattr(
        "api.claim_verification.retrieval.colbert_rerank.has_sufficient_evidence",
        lambda reranked: next(confidence_values),
    )
    monkeypatch.setattr(
        "api.claim_verification.retrieval.colbert_rerank.live_news_passages",
        lambda *args, **kwargs: [
            {
                "docid": "news-1",
                "text": "Fresh article passage",
                "score": 1.1,
                "source_type": "news",
                "url": "https://news.example/story",
            }
        ],
    )

    retriever = _make_retriever(
        bm25_results=[{"docid": "wiki-1", "text": "Local passage", "score": 2.0}],
        first_pass=[
            {
                "docid": "wiki-1",
                "text": "Local passage",
                "bm25_score": 2.0,
                "colbert_score": 0.2,
                "source_type": "local",
            }
        ],
        second_pass=[
            {
                "docid": "news-1",
                "text": "Fresh article passage",
                "bm25_score": 1.1,
                "colbert_score": 0.95,
                "source_type": "news",
            }
        ],
        evidence_k=1,
        enable_live_news=True,
    )

    result = retriever.retrieve("A recent breaking claim")

    assert result["confidence"] is True
    assert result["force_nei"] is False
    assert result["source"] == "local+web"
    assert result["evidence"] == ["Fresh article passage"]
    assert len(retriever.reranker.calls) == 2
    assert retriever.reranker.calls[1]["candidate_count"] == 2


def test_retrieve_respects_evidence_k(monkeypatch):
    monkeypatch.setattr(
        "api.claim_verification.retrieval.colbert_rerank.has_sufficient_evidence",
        lambda reranked: True,
    )

    reranked = [
        {
            "docid": f"doc-{i}",
            "text": f"Passage {i}",
            "bm25_score": float(i),
            "colbert_score": float(10 - i),
            "source_type": "local",
        }
        for i in range(5)
    ]
    retriever = _make_retriever(
        bm25_results=[{"docid": "raw-1", "text": "raw passage", "score": 3.5}],
        first_pass=reranked,
        evidence_k=2,
    )

    result = retriever.retrieve("A claim with many hits")

    assert result["evidence"] == ["Passage 0", "Passage 1"]


def test_latency_budget_with_stubbed_components(monkeypatch):
    monkeypatch.setattr(
        "api.claim_verification.retrieval.colbert_rerank.has_sufficient_evidence",
        lambda reranked: True,
    )

    reranked = [
        {
            "docid": "doc-1",
            "text": "Passage one",
            "bm25_score": 5.0,
            "colbert_score": 0.88,
            "source_type": "local",
        }
    ]
    retriever = _make_retriever(
        bm25_results=[{"docid": "raw-1", "text": "raw passage", "score": 3.5}],
        first_pass=reranked,
        evidence_k=1,
    )

    start = time.perf_counter()
    result = retriever.retrieve("Latency test claim")
    elapsed = time.perf_counter() - start

    assert result["evidence"] == ["Passage one"]
    assert elapsed < 30.0
