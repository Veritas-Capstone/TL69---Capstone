import json
import re
from pathlib import Path
from typing import Iterable, List

# ---- Chunking params (good defaults for BM25 + ColBERT) ----
MAX_TOKENS = 220
OVERLAP = 40
MIN_CHARS = 80

def ws_tokenize(text: str) -> List[str]:
    # simple whitespace tokenization (fast, robust)
    return re.findall(r"\S+", text)

def detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)

def chunk_tokens(tokens: List[str], max_tokens: int, overlap: int) -> Iterable[List[str]]:
    step = max_tokens - overlap
    if step <= 0:
        raise ValueError("overlap must be < max_tokens")
    for i in range(0, len(tokens), step):
        yield tokens[i:i + max_tokens]

def clean_text(t: str) -> str:
    # normalize whitespace + remove weird control chars
    t = t.replace("\u0000", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def title_to_url(title: str) -> str:
    # wiki ids in these dumps are usually Title_With_Underscores
    return f"https://en.wikipedia.org/wiki/{title}"

def build_passages(
    wiki_pages_dir: Path,
    out_jsonl: Path,
    max_tokens: int = MAX_TOKENS,
    overlap: int = OVERLAP,
):
    files = sorted(wiki_pages_dir.glob("wiki-*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No wiki-*.jsonl files found in {wiki_pages_dir}")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n_pages = 0
    n_passages = 0

    with out_jsonl.open("w", encoding="utf-8") as w:
        for fp in files:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    ex = json.loads(line)

                    # These FEVER-style dumps usually store:
                    # {"id": "Some_Page_Title", "text": "..."}
                    title = (ex.get("id") or "").strip()
                    text  = clean_text(ex.get("text") or "")

                    if not title or not text:
                        continue

                    n_pages += 1

                    doc_id = f"wiki:{title}"
                    url = title_to_url(title)

                    toks = ws_tokenize(text)
                    for j, chunk in enumerate(chunk_tokens(toks, max_tokens, overlap)):
                        passage = detokenize(chunk).strip()
                        if len(passage) < MIN_CHARS:
                            continue

                        rec = {
                            "doc_id": doc_id,
                            "passage_id": f"{doc_id}#{j:05d}",
                            "title": title.replace("_", " "),
                            "url": url,
                            "text": passage,
                            "source": "wikipedia"
                        }
                        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_passages += 1

            print(f"[Done] {fp.name}")

    print(f"\n✅ Finished.")
    print(f"Pages processed:   {n_pages}")
    print(f"Passages written:  {n_passages}")
    print(f"Output: {out_jsonl}")

def main():
    wiki_pages_dir = Path("api/claim_verification/data/raw/wiki-pages")
    out_jsonl = Path("api/claim_verification/data/wiki_corpus/wiki_passages.jsonl")
    build_passages(wiki_pages_dir, out_jsonl)

if __name__ == "__main__":
    main()
