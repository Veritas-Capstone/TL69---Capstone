import torch

from api.claim_verification.retrieval.colbert_rerank import ColBERTReranker


class _FakeCheckpoint:
    def __init__(self, checkpoint_name, colbert_config=None):
        self.checkpoint_name = checkpoint_name
        self.colbert_config = colbert_config
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def queryFromText(self, batch_queries, bsize=None):
        return torch.tensor([[[1.0, 1.0]]] * len(batch_queries))

    def docFromText(self, batch_docs, bsize=None):
        docs = []
        for text in batch_docs:
            if "strong" in text.lower():
                docs.append([[2.0, 2.0]])
            elif "medium" in text.lower():
                docs.append([[1.5, 1.5]])
            else:
                docs.append([[0.5, 0.5]])
        return torch.tensor(docs)


class _FakeConfig:
    def __init__(self, query_maxlen, doc_maxlen):
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen


def test_colbert_rerank_returns_sorted_expected_fields(monkeypatch):
    monkeypatch.setattr(
        "api.claim_verification.retrieval.colbert_rerank.Checkpoint",
        _FakeCheckpoint,
    )
    monkeypatch.setattr(
        "api.claim_verification.retrieval.colbert_rerank.ColBERTConfig",
        _FakeConfig,
    )
    monkeypatch.setattr(
        "api.claim_verification.retrieval.colbert_rerank.torch.cuda.is_available",
        lambda: False,
    )

    reranker = ColBERTReranker(checkpoint="fake-ckpt", use_gpu=False, batch_size=2)
    candidates = [
        {"docid": "doc-weak", "text": "weak evidence", "score": 1.0},
        {"docid": "doc-strong", "text": "strong evidence", "score": 3.0},
        {"docid": "doc-medium", "text": "medium evidence", "score": 2.0},
    ]

    results = reranker.rerank("test claim", candidates, top_k=3)

    assert [item["docid"] for item in results] == [
        "doc-strong",
        "doc-medium",
        "doc-weak",
    ]
    assert results[0]["source_type"] == "local"
    assert isinstance(results[0]["bm25_score"], float)
    assert isinstance(results[0]["colbert_score"], float)

def test_colbert_rerank_skips_empty_candidates(monkeypatch):
    monkeypatch.setattr(
        "api.claim_verification.retrieval.colbert_rerank.Checkpoint",
        _FakeCheckpoint,
    )
    monkeypatch.setattr(
        "api.claim_verification.retrieval.colbert_rerank.ColBERTConfig",
        _FakeConfig,
    )
    monkeypatch.setattr(
        "api.claim_verification.retrieval.colbert_rerank.torch.cuda.is_available",
        lambda: False,
    )

    reranker = ColBERTReranker(checkpoint="fake-ckpt", use_gpu=False, batch_size=2)
    candidates = [
        {"docid": "empty-1", "text": "   ", "score": 2.0},
        {"docid": "good-1", "text": "strong evidence", "score": 1.0},
        {"docid": "empty-2", "text": None, "score": 3.0},
    ]

    results = reranker.rerank("claim", candidates, top_k=5)

    assert len(results) == 1
    assert results[0]["docid"] == "good-1"
