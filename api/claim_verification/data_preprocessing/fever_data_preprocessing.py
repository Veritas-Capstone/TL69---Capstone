import os
import json
import csv

FEVER_TRAIN_JSONL = "../data/raw/fever.jsonl"
WIKI_PAGES_DIR    = "../data/raw/wiki-pages"
OUTPUT_TRAIN_CSV  = "../data/processed/fever_train_claims.csv"
OUTPUT_SAMPLE_CSV = "../data/processed/fever_train_claims_sample.csv"

LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "REFUTED",
    "NOT ENOUGH INFO": "NOT ENOUGH INFO",
}

# ----------------------------------------------------------
# Load all wiki evidence shards and make a lookup table
# ----------------------------------------------------------
def load_fever_wiki_jsonl(dir_path):
    index = {}

    files = [f for f in os.listdir(dir_path) if f.endswith(".jsonl")]
    files.sort()

    total_lines = 0
    for fname in files:
        fpath = os.path.join(dir_path, fname)
        print(f"Loading wiki shard: {fpath}")

        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                ex = json.loads(line)

                page_title = ex["id"]
                lines = ex["lines"].split("\n")

                for l in lines:
                    if not l.strip():
                        continue
                    try:
                        idx_str, sent = l.split("\t", 1)
                        idx = int(idx_str)
                    except ValueError:
                        continue

                    # some lines have multiple tabs; only take first part
                    sent_text = sent.split("\t")[0]
                    index[(page_title, idx)] = sent_text

    print(f"Wiki index loaded: {len(index)} sentences ({total_lines} page-lines read)")
    return index

def extract_evidence_sentences(example, wiki_index):
    label = example.get("label")

    # NEI → no gold evidence
    if label == "NOT ENOUGH INFO":
        return []

    collected = []
    seen = set()

    for evid_set in example.get("evidence", []):
        for item in evid_set:
            if len(item) < 4:
                continue

            _, _, page_title, sent_idx = item
            if page_title is None or sent_idx is None:
                continue

            key = (page_title, sent_idx)
            if key in wiki_index and key not in seen:
                collected.append(wiki_index[key])
                seen.add(key)

    return collected

def process_fever_split(jsonl_path, wiki_index):
    rows = []
    n_total = 0
    n_missing = 0

    print(f"Processing FEVER split: {jsonl_path}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            n_total += 1
            ex = json.loads(line)

            claim = ex["claim"].replace("\n", " ").strip()
            label = ex["label"]

            if label not in LABEL_MAP:
                continue

            mapped_label = LABEL_MAP[label]

            evid_sentences = extract_evidence_sentences(ex, wiki_index)

            # Count missing gold evidence for SUPPORTED/REFUTED
            if mapped_label != "NOT ENOUGH INFO" and not evid_sentences:
                n_missing += 1

            rows.append({
                "claim": claim,
                "evidence": evid_sentences,
                "label": mapped_label
            })

    print(f"Total examples: {n_total}")
    print(f"Missing evidence for verifiable claims: {n_missing}")
    print(f"Final rows: {len(rows)}")
    return rows


def save_rows_to_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["claim", "evidence", "label"])

        for r in rows:
            writer.writerow([
                r["claim"],
                json.dumps(r["evidence"], ensure_ascii=False),
                r["label"]
            ])

    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    wiki_index = load_fever_wiki_jsonl(WIKI_PAGES_DIR)
    train_rows = process_fever_split(FEVER_TRAIN_JSONL, wiki_index)

    save_rows_to_csv(train_rows, OUTPUT_TRAIN_CSV)

    # Create a small sample file for testing
    if os.path.exists(OUTPUT_SAMPLE_CSV):
        os.remove(OUTPUT_SAMPLE_CSV)

    save_rows_to_csv(train_rows[:50], OUTPUT_SAMPLE_CSV)
