from api.claim_verification.retrieval.confidence import has_sufficient_evidence


def test_confidence_returns_false_for_empty_results():
    assert has_sufficient_evidence([]) is False


def test_confidence_returns_true_for_single_result():
    reranked = [{"docid": "doc-1", "colbert_score": 0.42}]
    assert has_sufficient_evidence(reranked) is True


def test_confidence_returns_false_when_scores_are_identical():
    reranked = [
        {"docid": "doc-1", "colbert_score": 0.5},
        {"docid": "doc-2", "colbert_score": 0.5},
    ]
    assert has_sufficient_evidence(reranked) is False


def test_confidence_returns_false_when_top_gap_is_too_small():
    reranked = [
        {"docid": "doc-1", "colbert_score": 1.00},
        {"docid": "doc-2", "colbert_score": 0.95},
        {"docid": "doc-3", "colbert_score": 0.00},
    ]
    assert has_sufficient_evidence(reranked, min_gap=0.08) is False


def test_confidence_returns_true_for_clear_top_hit():
    reranked = [
        {"docid": "doc-1", "colbert_score": 0.95},
        {"docid": "doc-2", "colbert_score": 0.30},
        {"docid": "doc-3", "colbert_score": 0.10},
    ]
    assert has_sufficient_evidence(reranked) is True
