#!/usr/bin/env python3
"""
content/build_index.py

Robust builder: reads scraped JSON/HTML/text files, optionally filters by domain,
(optional) chunks long pages, embeds with sentence-transformers (or OpenAI),
builds FAISS index (IndexFlatIP for cosine), and writes outputs:

  <persist_dir>/pages.json       # list of docs (id matches index position)
  <persist_dir>/pages_meta.json
  <persist_dir>/pages.faiss
  <persist_dir>/embeddings.npy   # optional raw embeddings

Usage examples:
  python content/build_index.py --content-dirs ./scrape/content ./content --persist-dir ./embeddings
  python content/build_index.py --content-dirs ./scrape/content --persist-dir ./embeddings --include-domains scriptbees.com --exclude-domains volunteermark.com
  python content/build_index.py --content-dirs ./scrape/content --persist-dir ./embeddings --chunk-size 800 --overlap 100

Optional OpenAI embeddings (requires OPENAI_API_KEY in env):
  python content/build_index.py --use-openai --openai-model text-embedding-ada-002
"""
from pathlib import Path
import argparse
import json
import os
from typing import List, Dict, Optional, Iterable
import re

import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup

# Optional dependencies (raise friendly errors if missing)
try:
    from sentence_transformers import SentenceTransformer
    HAS_S2 = True
except Exception:
    HAS_S2 = False

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False


# ----------------------
# Utilities
# ----------------------
def load_text_from_file(path: Path) -> str:
    try:
        raw = path.read_text(encoding="utf8", errors="ignore")
    except Exception:
        return ""
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm"}:
        soup = BeautifulSoup(raw, "html.parser")
        for s in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            s.decompose()
        text = soup.get_text(separator=" ")
    elif suffix == ".json":
        # Try to parse JSON and extract common text fields if possible
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                # common fields
                for k in ("text", "content", "body", "page_text"):
                    if k in data and isinstance(data[k], str) and data[k].strip():
                        text = data[k]
                        break
                else:
                    # fallback: stringify values
                    text = " ".join(str(v) for v in data.values() if isinstance(v, str))
            elif isinstance(data, list):
                # join list items if list of strings/dicts
                pieces = []
                for item in data:
                    if isinstance(item, str):
                        pieces.append(item)
                    elif isinstance(item, dict):
                        for k in ("text", "content", "body"):
                            if k in item and isinstance(item[k], str):
                                pieces.append(item[k])
                                break
                text = " ".join(pieces)
            else:
                text = ""
        except Exception:
            text = ""
    else:
        text = raw

    text = " ".join(text.split())
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> Iterable[str]:
    if chunk_size <= 0:
        yield text
        return
    if len(text) <= chunk_size:
        yield text
        return
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        yield text[start : start + chunk_size]
        start += step


def domain_list_from_arg(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip().lower() for x in s.split(",") if x.strip()]


# ----------------------
# Document discovery & normalization
# ----------------------
def discover_documents(dirs: List[Path], include_domains: List[str], exclude_domains: List[str], exts=None, min_len=50) -> List[Dict]:
    if exts is None:
        exts = {".txt", ".md", ".html", ".htm", ".json"}

    docs = []
    seen = set()
    for d in dirs:
        if not d:
            continue
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            txt = load_text_from_file(p)
            if not txt or len(txt) < min_len:
                continue

            # try to extract URL from JSON content if present
            url = ""
            if p.suffix.lower() == ".json":
                try:
                    obj = json.loads(p.read_text(encoding="utf8", errors="ignore"))
                    if isinstance(obj, dict):
                        url = (obj.get("url") or obj.get("source") or obj.get("link") or "").strip()
                except Exception:
                    pass

            # fallback: derive url from filename if looks like a url
            filename = p.name.lower()
            if not url:
                # naive check: filename contains domain-like string
                m = re.search(r"(https?[_\-\.]?[:/]*\S+)", p.name)
                # not reliable; skip
            url_low = (url or "").lower()

            # exclude/inclusion rules
            if exclude_domains and any(s in url_low for s in exclude_domains):
                continue
            if include_domains and not any(s in url_low for s in include_domains):
                # if URL blank, still include since include filter is specified for targeted domains only
                if url_low:
                    continue

            # dedupe by text snippet and url
            key = url_low or txt[:250].strip().lower()
            if key in seen:
                continue
            seen.add(key)

            docs.append({"path": str(p), "url": url, "text": txt})
    return docs


# ----------------------
# Embedding helpers
# ----------------------
def embed_with_sentence_transformers(texts: List[str], model_name: str):
    if not HAS_S2:
        raise RuntimeError("sentence-transformers not installed. Install: pip install sentence-transformers")
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return np.asarray(emb, dtype=np.float32)


def embed_with_openai(texts: List[str], model_name: str, batch: int = 16):
    if not HAS_OPENAI:
        raise RuntimeError("openai not installed. Install: pip install openai")
    if os.getenv("OPENAI_API_KEY") is None:
        raise RuntimeError("OPENAI_API_KEY environment variable not found")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    out = []
    for i in tqdm(range(0, len(texts), batch), desc="OpenAI embeddings"):
        batch_texts = texts[i : i + batch]
        resp = openai.Embedding.create(input=batch_texts, model=model_name)
        for r in resp["data"]:
            out.append(r["embedding"])
    return np.asarray(out, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray):
    if not HAS_FAISS:
        raise RuntimeError("faiss not installed. Install: pip install faiss-cpu (or conda)")
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D")
    # Normalize for cosine-sim (inner product)
    emb_norm = embeddings.copy()
    faiss.normalize_L2(emb_norm)
    d = emb_norm.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb_norm)
    return index


# ----------------------
# Main
# ----------------------
def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from scraped content.")
    parser.add_argument("--content-dirs", nargs="+", default=["./content"], help="One or more content directories to scan.")
    parser.add_argument("--persist-dir", default="./embeddings", help="Where to write index/docs/embeddings.")
    parser.add_argument("--include-domains", default="", help="Comma-separated substrings to include (e.g. scriptbees.com)")
    parser.add_argument("--exclude-domains", default="volunteermark,volunteermark.com,volunteerMark", help="Comma-separated substrings to exclude")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformers model name (default) or ignored when using OpenAI.")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI embeddings (requires OPENAI_API_KEY)")
    parser.add_argument("--openai-model", default="text-embedding-ada-002", help="OpenAI embedding model")
    parser.add_argument("--chunk-size", type=int, default=0, help="If >0, chunk long documents into this char size.")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap in characters.")
    parser.add_argument("--min-len", type=int, default=50, help="Minimum text length to include.")
    parser.add_argument("--exts", default=".txt,.md,.html,.htm,.json", help="File extensions to scan (comma-separated).")
    args = parser.parse_args()

    content_dirs = [Path(p) for p in args.content_dirs]
    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    include_domains = domain_list_from_arg(args.include_domains)
    exclude_domains = domain_list_from_arg(args.exclude_domains)

    exts = {s.strip().lower() if s.strip().startswith(".") else "." + s.strip().lower() for s in args.exts.split(",")}

    print("Scanning content dirs:", content_dirs)
    print("Persist dir:", persist_dir)
    print("Include domains:", include_domains)
    print("Exclude domains:", exclude_domains)
    print("Extensions:", exts)

    # Discover docs
    raw_docs = discover_documents(content_dirs, include_domains=include_domains, exclude_domains=exclude_domains, exts=exts, min_len=args.min_len)
    if not raw_docs:
        print("No scraped documents found. Check content directories and filters.")
        return

    # Optionally chunk
    documents = []
    for doc in raw_docs:
        if args.chunk_size and len(doc["text"]) > args.chunk_size:
            for i, ch in enumerate(chunk_text(doc["text"], chunk_size=args.chunk_size, overlap=args.overlap)):
                documents.append({
                    "url": doc.get("url", ""),
                    "path": doc.get("path", ""),
                    "text": ch,
                    "chunk_id": i
                })
        else:
            documents.append({"url": doc.get("url", ""), "path": doc.get("path", ""), "text": doc.get("text", ""), "chunk_id": None})

    print(f"Discovered {len(raw_docs)} source files -> {len(documents)} documents (after chunking).")

    # Build texts list (order matters)
    texts = [d["text"] for d in documents]

    # Embed
    if args.use_openai:
        print("Using OpenAI embeddings model:", args.openai_model)
        embeddings = embed_with_openai(texts, args.openai_model)
    else:
        print("Using sentence-transformers model:", args.model)
        embeddings = embed_with_sentence_transformers(texts, args.model)

    embeddings = np.asarray(embeddings, dtype=np.float32)
    print("Embeddings shape:", embeddings.shape)

    # Build FAISS
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    # Save docs.json with numeric sequential ids (index position == id)
    out_docs = []
    for i, d in enumerate(documents):
        out_docs.append({
            "id": i,
            "url": d.get("url", ""),
            "path": d.get("path", ""),
            "chunk_id": d.get("chunk_id"),
            # store a preview; keep full text if you want but large JSON may be heavy
            "text": d.get("text", "")[:2000]
        })

    with open(persist_dir / "pages.json", "w", encoding="utf8") as f:
        json.dump(out_docs, f, ensure_ascii=False, indent=2)

    with open(persist_dir / "pages_meta.json", "w", encoding="utf8") as f:
        json.dump([{"id": d["id"], "url": d["url"], "path": d["path"], "chunk_id": d["chunk_id"]} for d in out_docs], f, ensure_ascii=False, indent=2)

    np.save(persist_dir / "embeddings.npy", embeddings)
    faiss.write_index(index, str(persist_dir / "pages.faiss"))

    print("Saved outputs to:", persist_dir)
    print(" - pages.json")
    print(" - pages_meta.json")
    print(" - pages.faiss")
    print(" - embeddings.npy")
    print("Done.")


if __name__ == "__main__":
    main()
