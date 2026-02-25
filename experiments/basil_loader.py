"""
BASIL Dataset Loader
====================
Loads 300 articles (100 events × 3 sources) from the BASIL repository.
Maps source outlets to Left/Center/Right bias labels.

Source mapping:
  - hpo / HPO (Huffington Post)  →  Left
  - nyt / NYT (New York Times)   →  Center
  - fox / FOX (Fox News)         →  Right

Article-level annotations provide `relative_stance`:
  liberal, left       →  Left
  center              →  Center
  conservative, right →  Right
"""

import json
import glob
import os
from typing import List, Dict, Optional

# ── Source → label mapping ──
SOURCE_MAP = {
    "fox": "Right", "FOX": "Right",
    "hpo": "Left",  "HPO": "Left",
    "nyt": "Center", "NYT": "Center",
}

# ── Annotation stance → label mapping ──
STANCE_MAP = {
    "liberal": "Left",
    "left": "Left",
    "center": "Center",
    "conservative": "Right",
    "right": "Right",
}


def flatten_body(body_paragraphs) -> str:
    """Flatten nested body-paragraphs arrays into a single text string."""
    parts = []
    if isinstance(body_paragraphs, list):
        for item in body_paragraphs:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, list):
                # Nested paragraph arrays
                for sub in item:
                    if isinstance(sub, str):
                        parts.append(sub)
                    elif isinstance(sub, list):
                        parts.extend(s for s in sub if isinstance(s, str))
    return " ".join(parts).strip()


def load_basil(basil_dir: str = "BASIL",
               label_source: str = "source",
               min_body_length: int = 50) -> List[Dict]:
    """
    Load BASIL articles with ground truth labels.

    Args:
        basil_dir:        Path to cloned BASIL repo
        label_source:     "source" (outlet-based) or "annotation" (stance-based)
        min_body_length:  Skip articles with body text shorter than this

    Returns:
        List of article dicts with keys:
            body, headline, true_label, source, event_id,
            file_suffix, annotation_stance (if available)
    """
    articles_dir = os.path.join(basil_dir, "articles")
    annotations_dir = os.path.join(basil_dir, "annotations")

    if not os.path.isdir(articles_dir):
        raise FileNotFoundError(f"BASIL articles directory not found: {articles_dir}")

    articles = []
    skipped = {"no_source": 0, "short_body": 0, "no_label": 0, "parse_error": 0}

    # Find all article JSON files (exclude annotation files)
    article_files = sorted(glob.glob(
        os.path.join(articles_dir, "**", "*.json"), recursive=True
    ))

    for article_path in article_files:
        try:
            with open(article_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            skipped["parse_error"] += 1
            continue

        # Extract fields
        source = data.get("source", "").strip()
        title = data.get("title", "")
        body_paragraphs = data.get("body-paragraphs", [])
        body = flatten_body(body_paragraphs)

        # Determine label from source
        source_label = SOURCE_MAP.get(source)
        if not source_label:
            skipped["no_source"] += 1
            continue

        if len(body) < min_body_length:
            skipped["short_body"] += 1
            continue

        # Extract event ID and suffix from filename
        basename = os.path.basename(article_path).replace(".json", "")
        parts = basename.rsplit("_", 1)
        event_id = parts[0] if len(parts) == 2 else basename
        file_suffix = parts[1] if len(parts) == 2 else ""

        # Try to load annotation
        year_dir = os.path.basename(os.path.dirname(article_path))
        ann_path = os.path.join(annotations_dir, year_dir, f"{basename}_ann.json")
        annotation_stance = None
        annotation_label = None

        if os.path.isfile(ann_path):
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    ann_data = json.load(f)
                annotation_stance = ann_data.get(
                    "article-level-annotations", {}
                ).get("relative_stance", "")
                annotation_label = STANCE_MAP.get(annotation_stance)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # Choose ground truth label
        if label_source == "annotation":
            true_label = annotation_label
            if not true_label:
                skipped["no_label"] += 1
                continue
        else:
            true_label = source_label

        articles.append({
            "body": body,
            "headline": title,
            "true_label": true_label,
            "source": source,
            "source_label": source_label,
            "annotation_stance": annotation_stance,
            "annotation_label": annotation_label,
            "event_id": event_id,
            "file_suffix": file_suffix,
            "bias_detail": source,  # For compatibility with pipeline_v2
            "url": data.get("url", ""),
        })

    print(f"BASIL loaded: {len(articles)} articles")
    print(f"  Skipped: {skipped}")
    if articles:
        from collections import Counter
        labels = Counter(a["true_label"] for a in articles)
        print(f"  Label distribution: {dict(labels)}")

    return articles


def load_basil_with_sentence_annotations(basil_dir: str = "BASIL") -> List[Dict]:
    """
    Load BASIL articles with full sentence-level bias annotations.
    Useful for validating Gradient×Input explainability.

    Returns articles with an additional 'sentence_annotations' field.
    """
    articles = load_basil(basil_dir, label_source="source")
    annotations_dir = os.path.join(basil_dir, "annotations")

    for article in articles:
        year_dir = None
        # Find the year directory
        for year in range(2010, 2020):
            test_path = os.path.join(
                annotations_dir, str(year),
                f"{article['event_id']}_{article['file_suffix']}_ann.json"
            )
            if os.path.isfile(test_path):
                year_dir = str(year)
                break

        if year_dir:
            ann_path = os.path.join(
                annotations_dir, year_dir,
                f"{article['event_id']}_{article['file_suffix']}_ann.json"
            )
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    ann_data = json.load(f)
                article["sentence_annotations"] = ann_data.get(
                    "phrase-level-annotations", []
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                article["sentence_annotations"] = []
        else:
            article["sentence_annotations"] = []

    return articles


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and inspect BASIL dataset")
    parser.add_argument("--basil-dir", default="BASIL", help="Path to BASIL repo")
    parser.add_argument("--label-source", default="source",
                        choices=["source", "annotation"],
                        help="Ground truth source: outlet or annotation")
    args = parser.parse_args()

    articles = load_basil(args.basil_dir, label_source=args.label_source)

    # Show source vs annotation agreement
    agree = disagree = no_ann = 0
    for a in articles:
        if a["annotation_label"]:
            if a["source_label"] == a["annotation_label"]:
                agree += 1
            else:
                disagree += 1
        else:
            no_ann += 1

    print(f"\nSource ↔ Annotation agreement:")
    print(f"  Agree: {agree}, Disagree: {disagree}, No annotation: {no_ann}")
    if agree + disagree > 0:
        print(f"  Agreement rate: {agree / (agree + disagree):.1%}")

    # Show a few samples
    print(f"\nSample articles:")
    for a in articles[:3]:
        print(f"  [{a['source']}→{a['source_label']}] "
              f"ann:{a['annotation_stance']}→{a['annotation_label']}  "
              f"{a['headline'][:60]}")
