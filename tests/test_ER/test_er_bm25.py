import json

from api.claim_verification.retrieval.bm25 import BM25Retriever


class _FakeDoc:
    def __init__(self, raw_text):
        self._raw_text = raw_text

    def raw(self):
        return self._raw_text


class _FakeHit:
    def __init__(self, docid, score):
        self.docid = docid
        self.score = score


class _FakeLuceneSearcher:
    def __init__(self, index_path):
        self.index_path = index_path
        self.searched = []

    def search(self, query, k):
        self.searched.append((query, k))
        return [
            _FakeHit("doc-1", 12.5),
            _FakeHit("doc-2", 9.25),
        ]

    def doc(self, docid):
        if docid == "doc-1":
            return _FakeDoc(
                json.dumps(
                    {
                        "id": "passage-1",
                        "contents": "Passage one body",
                    }
                )
            )
        if docid == "doc-2":
            return _FakeDoc("{bad json")
        return None


def test_bm25_retrieve_returns_expected_fields(monkeypatch):
    monkeypatch.setattr(
        "api.claim_verification.retrieval.bm25.LuceneSearcher",
        _FakeLuceneSearcher,
    )

    retriever = BM25Retriever(index_path="fake-index", k=7)
    results = retriever.retrieve("climate claim")

    assert len(results) == 2
    assert results[0]["docid"] == "doc-1"
    assert results[0]["score"] == 12.5
    assert results[0]["title_or_id"] == "passage-1"
    assert results[0]["text"] == "Passage one body"
    assert results[0]["raw"]["id"] == "passage-1"


def test_bm25_retrieve_handles_invalid_raw_json(monkeypatch):
    monkeypatch.setattr(
        "api.claim_verification.retrieval.bm25.LuceneSearcher",
        _FakeLuceneSearcher,
    )

    retriever = BM25Retriever(index_path="fake-index", k=3)
    results = retriever.retrieve("broken json claim")

    assert results[1]["docid"] == "doc-2"
    assert results[1]["raw"] == {}
    assert results[1]["text"] == ""
    assert results[1]["title_or_id"] == "doc-2"


def test_bm25_retrieve_uses_default_k_when_not_overridden(monkeypatch):
    monkeypatch.setattr(
        "api.claim_verification.retrieval.bm25.LuceneSearcher",
        _FakeLuceneSearcher,
    )

    retriever = BM25Retriever(index_path="fake-index", k=11)
    retriever.retrieve("default k claim")

    assert retriever.searcher.searched == [("default k claim", 11)]
