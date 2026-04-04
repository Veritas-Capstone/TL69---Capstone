"""
BASIL Dataset Loader
"""

import json
import glob
import os
from collections import Counter

SOURCE_MAP = {
    "fox": "Right", "FOX": "Right",
    "hpo": "Left",  "HPO": "Left",
    "nyt": "Center", "NYT": "Center",
}

STANCE_MAP = {
    "liberal": "Left", "left": "Left",
    "center": "Center",
    "conservative": "Right", "right": "Right",
}

BASIL_DIR = "BASIL"


def flatten_body(body_paragraphs):
    parts = []
    if isinstance(body_paragraphs, list):
        for item in body_paragraphs:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, str):
                        parts.append(sub)
                    elif isinstance(sub, list):
                        parts.extend(s for s in sub if isinstance(s, str))
    return " ".join(parts).strip()


def load_basil(basil_dir=BASIL_DIR, label_source="source", min_body_length=50):
    articles_dir = os.path.join(basil_dir, "articles")
    annotations_dir = os.path.join(basil_dir, "annotations")

    if not os.path.isdir(articles_dir):
        raise FileNotFoundError(f"BASIL articles directory not found: {articles_dir}")

    articles = []
    skipped = {"no_source": 0, "short_body": 0, "no_label": 0, "parse_error": 0}

    article_files = sorted(glob.glob(os.path.join(articles_dir, "**", "*.json"), recursive=True))

    for article_path in article_files:
        try:
            with open(article_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            skipped["parse_error"] += 1
            continue

        source = data.get("source", "").strip()
        title = data.get("title", "")
        body = flatten_body(data.get("body-paragraphs", []))

        source_label = SOURCE_MAP.get(source)
        if not source_label:
            skipped["no_source"] += 1
            continue
        if len(body) < min_body_length:
            skipped["short_body"] += 1
            continue

        # extract event ID from filename
        basename = os.path.basename(article_path).replace(".json", "")
        parts = basename.rsplit("_", 1)
        event_id = parts[0] if len(parts) == 2 else basename
        file_suffix = parts[1] if len(parts) == 2 else ""

        # try to load annotation stance
        year_dir = os.path.basename(os.path.dirname(article_path))
        ann_path = os.path.join(annotations_dir, year_dir, f"{basename}_ann.json")
        annotation_stance = None
        annotation_label = None

        if os.path.isfile(ann_path):
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    ann_data = json.load(f)
                annotation_stance = ann_data.get("article-level-annotations", {}).get("relative_stance", "")
                annotation_label = STANCE_MAP.get(annotation_stance)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # choose ground truth label
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
            "bias_detail": source,
            "url": data.get("url", ""),
        })

    print(f"BASIL loaded: {len(articles)} articles")
    print(f"  Skipped: {skipped}")
    if articles:
        labels = Counter(a["true_label"] for a in articles)
        print(f"  Label distribution: {dict(labels)}")

    return articles


if __name__ == "__main__":
    articles = load_basil(BASIL_DIR, label_source="source")

    # check source vs annotation agreement
    agree = disagree = no_ann = 0
    for a in articles:
        if a["annotation_label"]:
            if a["source_label"] == a["annotation_label"]:
                agree += 1
            else:
                disagree += 1
        else:
            no_ann += 1

    print(f"\nSource vs Annotation agreement:")
    print(f"  Agree: {agree}, Disagree: {disagree}, No annotation: {no_ann}")
    if agree + disagree > 0:
        print(f"  Agreement rate: {agree / (agree + disagree):.1%}")

    for a in articles[:3]:
        print(f"  [{a['source']}->{a['source_label']}] ann:{a['annotation_stance']}->{a['annotation_label']}  {a['headline'][:60]}")