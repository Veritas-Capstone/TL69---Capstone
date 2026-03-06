import json
from pathlib import Path

IN_PATH  = Path("api/claim_verification/data/wiki_corpus/wiki_passages.jsonl")
OUT_PATH = Path("api/claim_verification/data/wiki_corpus/wiki_passages_pyserini.jsonl")

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with IN_PATH.open("r", encoding="utf-8") as r, OUT_PATH.open("w", encoding="utf-8") as w:
        for line in r:
            ex = json.loads(line)
            pid = ex["passage_id"]
            title = ex.get("title", "")
            text  = ex.get("text", "")
            url   = ex.get("url", "")

            contents = f"{title}\n{text}\nURL: {url}"
            w.write(json.dumps({"id": pid, "contents": contents}, ensure_ascii=False) + "\n")

            n += 1
            if n % 500000 == 0:
                print(f"wrote {n:,}")
    print(f"done: {n:,} docs -> {OUT_PATH}")

if __name__ == "__main__":
    main()
