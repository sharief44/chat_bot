import os
import json
import time
import hashlib
import logging
from typing import List
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------
# Environment Config
# ---------------------
API_KEY = os.getenv("RAG_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 2
MAX_TOKENS = 200
TEMPERATURE = 0.3

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

INDEX_PATH = str(REPO_ROOT / "backend" / "embeddings" / "pages.faiss")
PAGES_PATH = str(REPO_ROOT / "backend" / "embeddings" / "pages.json")
META_PATH = str(REPO_ROOT / "backend" / "embeddings" / "pages_meta.json")

# ---------------------
# API Models
# ---------------------
class AskRequest(BaseModel):
    question: str = Field(...)

class Source(BaseModel):
    url: str
    title: str
    score: float

class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieved: List[Source]
    response_time_seconds: float
    cached: bool = False

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(request: Request, api_key: str = Security(api_key_header)):
    if not API_KEY:
        return ""
    incoming = (api_key or "").strip()
    if not incoming:
        incoming = request.headers.get("Authorization", "").replace("Bearer ", "")
    if incoming != API_KEY:
        raise HTTPException(403, "Invalid API Key")
    return incoming

# ---------------------
# Cache
# ---------------------
cache = {}
def cache_key(q: str): return hashlib.md5(q.lower().encode()).hexdigest()

# ---------------------
# Globals
# ---------------------
retriever = None

# ---------------------
# Lifespan
# ---------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    logger.info("Loading retriever...")
    if not os.path.exists(INDEX_PATH) or not os.path.exists(PAGES_PATH):
        raise FileNotFoundError("FAISS index or pages.json missing.")

    class Retriever:
        def __init__(self):
            self.model = SentenceTransformer(MODEL_NAME)
            self.index = faiss.read_index(INDEX_PATH)

            with open(PAGES_PATH, "r") as f:
                self.pages = json.load(f)
            with open(META_PATH, "r") as f:
                self.meta = json.load(f)

        def retrieve(self, q):
            emb = self.model.encode([q])
            D, I = self.index.search(emb.astype("float32"), TOP_K)
            out = []
            for score, idx in zip(D[0], I[0]):
                m = self.meta[idx]
                out.append({
                    "id": idx,
                    "url": m["url"],
                    "title": m["title"],
                    "text": self.pages[idx]["text"][:800],
                    "score": float(score)
                })
            return out

    retriever = Retriever()
    logger.info("Retriever loaded.")
    yield
    logger.info("Shutting down.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------
# Prompt Builder
# ---------------------
def build_prompt(q, docs):
    ctx = "\n\n".join(
        f"[{i+1}] {d['text']}\nSource: {d['url']}"
        for i, d in enumerate(docs)
    )
    return f"""Use ONLY the context below.

{ctx}

Question: {q}

Answer concisely with 3–5 sentences.
"""

# ---------------------
# Ask Endpoint
# ---------------------
@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest, api_key: str = Depends(verify_api_key)):
    q = req.question.strip()
    t0 = time.time()

    # cache
    if cache_key(q) in cache:
        c = cache[cache_key(q)]
        return AskResponse(**c, cached=True)

    docs = retriever.retrieve(q)

    # OpenAI answer
    answer = ""
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            prompt = build_prompt(q, docs)

            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            answer = resp.choices[0].message["content"]
        except Exception as e:
            logger.error("OpenAI error: %s", e)
            answer = docs[0]["text"][:400] + "..."

    else:
        # fallback
        answer = docs[0]["text"][:400] + "..."

    out = {
        "answer": answer,
        "sources": [d["url"] for d in docs],
        "retrieved": [Source(url=d["url"], title=d["title"], score=d["score"]) for d in docs],
        "response_time_seconds": time.time() - t0,
    }
    cache[cache_key(q)] = out
    return AskResponse(**out)

@app.get("/health")
def health():
    return {"ok": True}
