#!/usr/bin/env python3
"""
embeddings/embedder.py

Build embeddings + FAISS index for a content folder.

Outputs in persist-dir:
 - index.faiss
 - docs.json   (list, order matches index positions)
 - embeddings.npy

Usage:
  python embeddings/embedder.py --content-dir ./content --persist-dir ./embeddings --model all-MiniLM-L6-v2
  python embeddings/embedder.py --content-dir ./content --persist-dir ./embeddings --use-openai --openai-model text-embedding-ada-002

Notes:
 - By default uses sentence-transformers. To use OpenAI embeddings set OPENAI_API_KEY env var and pass --use-openai.
 - Uses IndexFlatIP (cosine) with normalized vectors.
 - Documents are stored as a list. Index positions correspond to list order.
"""
import os
import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# optional imports
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


def load_text_from_file(path: Path) -> str:
    text = ""
    suffix = path.suffix.lower()
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if suffix in {".html", ".htm"}:
        soup = BeautifulSoup(content, "html.parser")
        for s in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            s.decompose()
        text = soup.get_text(separator=" ")
    else:
        text = content
    text = " ".join(text.split())
    return text


def gather_documents(content_dir: Path, exts=None, min_len=50) -> List[Dict]:
    if exts is None:
        exts = {".txt", ".md", ".html", ".htm", ".json"}
    docs = []
    for p in sorted(content_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            txt = load_text_from_file(p)
            if not txt or len(txt) < min_len:
                continue
            docs.append({
                # DO NOT use file path as id; we'll use list index as index-id
                "path": str(p.relative_to(content_dir)),
                "text": txt
            })
    return docs


def embed_with_sentence_transformers(texts: List[str], model_name: str):
    if not HAS_S2:
        raise RuntimeError("sentence-transformers not installed.")
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return np.asarray(emb, dtype=np.float32)


def embed_with_openai(texts: List[str], model_name: str):
    if not HAS_OPENAI:
        raise RuntimeError("openai package not installed.")
    # batch in chunks to avoid very large requests
    B = 16
    out = []
    for i in tqdm(range(0, len(texts), B), desc="OpenAI embeddings"):
        batch = texts[i : i + B]
        resp = openai.Embedding.create(input=batch, model=model_name)
        for r in resp["data"]:
            out.append(r["embedding"])
    return np.asarray(out, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray):
    if not HAS_FAISS:
        raise RuntimeError("faiss not installed.")
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D array (n, d)")
    # normalize for cosine (inner product)
    emb_norm = embeddings.copy()
    faiss.normalize_L2(emb_norm)
    d = emb_norm.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb_norm)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-dir", type=str, default="./content")
    parser.add_argument("--persist-dir", type=str, default="./embeddings")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="sentence-transformers model name (default) or OpenAI model if --use-openai")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI embeddings (requires OPENAI_API_KEY)")
    parser.add_argument("--openai-model", type=str, default="text-embedding-ada-002", help="OpenAI embedding model")
    parser.add_argument("--min-len", type=int, default=50, help="Minimum text length to include")
    args = parser.parse_args()

    load_dotenv()
    content_dir = Path(args.content_dir)
    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    if not content_dir.exists():
        raise SystemExit(f"Content dir not found: {content_dir}")

    docs = gather_documents(content_dir, min_len=args.min_len)
    if not docs:
        raise SystemExit("No documents found. Check content directory.")

    texts = [d["text"] for d in docs]

    # choose embedding backend
    use_openai = args.use_openai or (os.getenv("OPENAI_API_KEY") is not None and args.use_openai)
    if use_openai:
        if not HAS_OPENAI:
            raise RuntimeError("OpenAI SDK not installed. pip install openai")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        print("Using OpenAI embeddings with model:", args.openai_model)
        embeddings = embed_with_openai(texts, args.openai_model)
    else:
        if not HAS_S2:
            raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
        print("Using sentence-transformers model:", args.model)
        embeddings = embed_with_sentence_transformers(texts, args.model)

    embeddings = np.asarray(embeddings, dtype=np.float32)
    # build index
    index = build_faiss_index(embeddings)

    # Save outputs: docs.json is the list of docs in the SAME ORDER as embeddings/index positions!
    out_docs = []
    for i, d in enumerate(docs):
        out_docs.append({
            "id": i,            # numeric index id -> matches FAISS position
            "path": d["path"],
            "text": d["text"]   # optionally you can store a shorter preview if too large
        })

    with open(persist_dir / "docs.json", "w", encoding="utf-8") as f:
        json.dump(out_docs, f, ensure_ascii=False, indent=2)

    np.save(persist_dir / "embeddings.npy", embeddings)
    # write index file
    faiss.write_index(index, str(persist_dir / "index.faiss"))

    print(f"Saved {len(out_docs)} docs to {persist_dir}/docs.json")
    print(f"Saved FAISS index to {persist_dir}/index.faiss")
    print("Done.")


if __name__ == "__main__":
    main()
