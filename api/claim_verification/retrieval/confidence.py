from typing import List, Dict, Any


def has_sufficient_evidence(
    reranked: List[Dict[str, Any]],
    k: int = 5,
    min_top1: float = 0.75,
    min_gap: float = 0.08,
) -> bool:
    """
    Returns True if reranked evidence appears reliable enough to verify a claim.

    Stability improvements:
    1. Uses top-k only.
    2. Min-max normalizes scores within top-k.
    3. Checks whether the best hit is strong after normalization.
    4. Checks whether the best hit is separated from the runner-up.
    """
    if not reranked:
        return False

    topk = reranked[: min(k, len(reranked))]
    scores = [r.get("colbert_score") for r in topk]
    scores = [float(s) for s in scores if s is not None]

    if len(scores) == 0:
        return False

    if len(scores) == 1:
        return True

    smax = max(scores)
    smin = min(scores)

    # Avoid divide-by-zero if all scores are identical
    if abs(smax - smin) < 1e-8:
        return False

    norm = [(s - smin) / (smax - smin) for s in scores]

    top1 = norm[0]
    top2 = norm[1]
    gap = top1 - top2

    if top1 < min_top1:
        return False

    if gap < min_gap:
        return False

    return True