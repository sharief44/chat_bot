#!/usr/bin/env python3
"""
embeddings/embedder.py - Minimal embedder for ScriptBees RAG

Purpose:
 - Read scraped pages (content/pages.json) OR scan content/ for text files
 - Produce backend/embeddings/pages.json, pages_meta.json and pages.faiss
 - Supports sentence-transformers (default) or OpenAI embeddings (--use-openai)

Usage:
  python embeddings/embedder.py --content-dir ./content --persist-dir ./backend/embeddings
  python embeddings/embedder.py --use-openai --openai-model text-embedding-ada-002
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
from tqdm import tqdm

# optional libs
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

# -------------------------
# Utilities
# -------------------------
def load_pages_json(path: Path) -> List[Dict]:
    """Load pages.json produced by scraper if present."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure each entry has id, url, title, text
        pages = []
        for i, p in enumerate(data):
            pages.append({
                "id": int(p.get("id", i)),
                "url": p.get("url", ""),
                "title": p.get("title", "") or "",
                "text": (p.get("text") or "")[:20000]  # cap to avoid huge texts
            })
        return pages
    except Exception:
        return []


def gather_from_files(content_dir: Path, exts=None, min_len=50) -> List[Dict]:
    """Scan content_dir for files and extract text (simple)."""
    if exts is None:
        exts = {".html", ".htm", ".md", ".txt", ".json"}
    docs = []
    for p in sorted(content_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                txt = " ".join(txt.split())
                if len(txt) < min_len:
                    continue
                docs.append({
                    "id": len(docs),
                    "url": str(p),
                    "title": p.stem,
                    "text": txt[:20000]
                })
            except Exception:
                continue
    return docs


# -------------------------
# Embedding backends
# -------------------------
def embed_with_sentence_transformers(texts: List[str], model_name: str = "all-MiniLM-L6-v2"):
    if not HAS_S2:
        raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return np.asarray(emb, dtype=np.float32)


def embed_with_openai(texts: List[str], model_name: str = "text-embedding-ada-002"):
    if not HAS_OPENAI:
        raise RuntimeError("openai package not installed. pip install openai")
    out = []
    B = 16
    for i in tqdm(range(0, len(texts), B), desc="OpenAI embeddings"):
        batch = texts[i : i + B]
        resp = openai.Embedding.create(input=batch, model=model_name)
        for r in resp["data"]:
            out.append(r["embedding"])
    return np.asarray(out, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray):
    if not HAS_FAISS:
        raise RuntimeError("faiss not installed. pip install faiss-cpu")
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D (n, d)")
    emb_norm = embeddings.copy()
    faiss.normalize_L2(emb_norm)
    d = emb_norm.shape[1]
    index = faiss.IndexFlatIP(d)  # inner-product on normalized vectors == cosine
    index.add(emb_norm)
    return index


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-dir", type=str, default="./content")
    parser.add_argument("--persist-dir", type=str, default="./backend/embeddings")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI embeddings (requires OPENAI_API_KEY)")
    parser.add_argument("--openai-model", type=str, default="text-embedding-ada-002")
    parser.add_argument("--min-len", type=int, default=50)
    args = parser.parse_args()

    content_dir = Path(args.content_dir)
    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # 1) Prefer scraper-produced pages.json (content/pages.json)
    scraper_pages_path = content_dir / "pages.json"
    if scraper_pages_path.exists():
        print("Loading pages from", scraper_pages_path)
        pages = load_pages_json(scraper_pages_path)
    else:
        print("No pages.json found; scanning files under", content_dir)
        pages = gather_from_files(content_dir, min_len=args.min_len)

    if not pages:
        raise SystemExit("No documents found. Run scraper first or check content directory.")

    # Prepare texts aligned with index positions
    texts = [p["text"] for p in pages]

    # Choose embeddings backend
    use_openai = args.use_openai or (os.getenv("OPENAI_API_KEY") is not None and args.use_openai)
    if use_openai:
        if not HAS_OPENAI:
            raise RuntimeError("OpenAI SDK not installed. pip install openai")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        print("Using OpenAI embeddings model:", args.openai_model)
        embeddings = embed_with_openai(texts, args.openai_model)
    else:
        print("Using sentence-transformers model:", args.model)
        embeddings = embed_with_sentence_transformers(texts, args.model)

    embeddings = np.asarray(embeddings, dtype=np.float32)
    index = build_faiss_index(embeddings)

    # Save outputs with names main.py expects
    pages_out = []
    meta_out = []
    for i, p in enumerate(pages):
        pid = int(p.get("id", i))
        url = p.get("url", "") or ""
        title = p.get("title", "") or ""
        text_preview = (p.get("text") or "")[:4000]  # keep a preview to save space
        pages_out.append({
            "id": pid,
            "url": url,
            "title": title,
            "text": text_preview
        })
        meta_out.append({"id": pid, "url": url, "title": title})

    # filenames expected by backend/main.py
    pages_path = persist_dir / "pages.json"
    meta_path = persist_dir / "pages_meta.json"
    index_path = persist_dir / "pages.faiss"

    with open(pages_path, "w", encoding="utf-8") as f:
        json.dump(pages_out, f, ensure_ascii=False, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    faiss.write_index(index, str(index_path))

    print(f"\nSaved {len(pages_out)} pages to {pages_path}")
    print(f"Saved metadata to {meta_path}")
    print(f"Saved FAISS index to {index_path}")
    print("Done.")


if __name__ == "__main__":
    main()
