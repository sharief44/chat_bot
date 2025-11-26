"""
SCRIPTBEES RAG CHATBOT - PRODUCTION OPTIMIZED
8-15 Second Response Times

This file is an updated version (env-configurable paths, robust startup, fixed embedding calls).
"""

import os
import json
import logging
import time
from typing import List, Optional
from pathlib import Path
import hashlib

from dotenv import load_dotenv

# find .env up the tree (same helper you had)
def find_env_file(start_path: Optional[Path] = None) -> Optional[Path]:
    if start_path is None:
        start_path = Path(__file__).resolve().parent
    cur = start_path
    for _ in range(20):
        candidate = cur / ".env"
        if candidate.exists():
            return candidate
        if cur == cur.parent:
            break
        cur = cur.parent
    return None

_env_path = find_env_file()
if _env_path:
    load_dotenv(dotenv_path=_env_path, override=False)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CRITICAL SPEED SETTINGS (can be overridden via .env)
API_KEY = os.getenv("RAG_API_KEY", os.getenv("RAG_API_KEY", "X2Cli1ZSPhHHAHlfZkOEPRWIqtd1TQD9ErH705-HMc4"))
CONTENT_DIR = os.getenv("CONTENT_DIR", "content")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "orca-mini-3b-gguf2-q4_0.gguf")

# ULTRA-FAST SETTINGS (tune via .env)
TOP_K = int(os.getenv("TOP_K", "2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "100"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
N_THREADS = int(os.getenv("N_THREADS", "8"))

# Paths (env override recommended). Default points to embeddings/ to match your build output.
INDEX_PATH = os.getenv("FAISS_INDEX", os.path.join("embeddings", "pages.faiss"))
PAGES_PATH = os.getenv("PAGES_PATH", os.path.join("embeddings", "pages.json"))
META_PATH = os.getenv("META_PATH", os.path.join("embeddings", "pages_meta.json"))

from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

# Pydantic models
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)

class Source(BaseModel):
    url: str
    title: str
    score: float

class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieved: Optional[List[Source]] = None
    cached: bool = False
    response_time_seconds: Optional[float] = None

# Auth
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(request: Request, api_key: str = Security(api_key_header)):
    incoming = (api_key or "").strip()
    if not incoming:
        auth = request.headers.get("authorization", "")
        if auth and auth.lower().startswith("bearer "):
            incoming = auth.split(" ", 1)[1].strip()
    expected = API_KEY
    if not incoming or incoming != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return incoming

# Aggressive in-memory cache (simple LRU-like eviction)
response_cache = {}
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "300"))

def get_cache_key(question: str) -> str:
    return hashlib.md5(question.lower().strip().encode()).hexdigest()

def get_cached_response(question: str):
    return response_cache.get(get_cache_key(question))

def cache_response(question: str, response: dict):
    key = get_cache_key(question)
    if len(response_cache) >= MAX_CACHE_SIZE:
        # pop oldest insertion (dict preserves insertion order)
        oldest = next(iter(response_cache))
        del response_cache[oldest]
    response_cache[key] = response

retriever = None
generator = None

app = FastAPI(
    title="ScriptBees RAG Chatbot",
    version="4.0",
    description="AI-powered chatbot for ScriptBees.com content"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global retriever, generator
    logger.info("="*70)
    logger.info("🐝 SCRIPTBEES RAG CHATBOT - PRODUCTION SERVER STARTING")
    logger.info("="*70)
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        # gpt4all import; keep lazy so startup fails with helpful message if missing
        from gpt4all import GPT4All

        # Validate presence of index + pages (give helpful message + hints)
        if not os.path.exists(INDEX_PATH) or not os.path.exists(PAGES_PATH):
            hint = (
                f"Missing files. Expected:\n  {INDEX_PATH}\n  {PAGES_PATH}\n\n"
                "Run scraping then build index:\n"
                "  python scrape/scraper.py --start-url https://scriptbees.com\n"
                "  python embeddings/embedder.py --content-dir ./content --persist-dir ./embeddings\n"
                "Or set env vars FAISS_INDEX/PAGES_PATH/META_PATH to the correct files."
            )
            raise FileNotFoundError(hint)

        class VectorRetriever:
            def __init__(self, index_path, pages_path, meta_path, model_name):
                logger.info("📦 Loading retriever...")
                self.model = SentenceTransformer(model_name)
                self.index = faiss.read_index(index_path)

                # Load pages (list)
                with open(pages_path, 'r', encoding='utf-8') as f:
                    pages = json.load(f)

                # Load metadata if available; otherwise derive it
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                else:
                    logger.warning(f"No meta file at {meta_path} — deriving meta from pages.json")
                    self.metadata = [{"id": p.get("id", idx), "url": p.get("url", ""), "title": p.get("title", "")} for idx, p in enumerate(pages)]

                # Build pages dict keyed by numeric id
                self.pages_dict = {}
                for p in pages:
                    pid = p.get("id")
                    # ensure numeric id fallback to index if missing
                    if pid is None:
                        pid = pages.index(p)
                    self.pages_dict[int(pid)] = p

                logger.info(f"✓ Indexed: {self.index.ntotal} ScriptBees documents")

                # Quick duplicate check
                try:
                    unique_texts = set()
                    for page in pages:
                        unique_texts.add((page.get('text') or "")[:120])
                    if len(unique_texts) < len(pages):
                        logger.warning(f"⚠️  Found {len(pages) - len(unique_texts)} duplicate/near-duplicate pages")
                except Exception:
                    pass

            def retrieve(self, query: str, top_k: int = TOP_K):
                # Encode query
                query_emb = self.model.encode([query], convert_to_numpy=True)
                # Ensure dtype float32
                import numpy as _np
                q = _np.asarray(query_emb, dtype=_np.float32)
                # Normalize for cosine search with IndexFlatIP
                try:
                    faiss.normalize_L2(q)
                except Exception:
                    # if faiss not available in this scope, import
                    import faiss as _faiss
                    _faiss.normalize_L2(q)

                D, I = self.index.search(q, top_k)
                results = []
                for score, idx in zip(D[0], I[0]):
                    if idx == -1:
                        continue
                    meta = None
                    try:
                        meta = next((m for m in self.metadata if int(m.get("id", -1)) == int(idx)), None)
                    except Exception:
                        meta = None
                    # Fallback to pages dict
                    page = self.pages_dict.get(int(idx))
                    if not page and meta:
                        page = self.pages_dict.get(int(meta.get("id", idx)))
                    if page:
                        results.append({
                            'id': int(idx),
                            'url': meta.get('url', page.get('url', '')) if meta else page.get('url', ''),
                            'title': meta.get('title', page.get('title', '')) if meta else page.get('title', ''),
                            'text': (page.get('text') or "")[:500],
                            'score': float(score)
                        })
                return results

        class OptimizedLLMGenerator:
            def __init__(self, model_name):
                logger.info(f"🤖 Loading LLM: {model_name}")
                logger.info(f"   Settings: MAX_TOKENS={MAX_TOKENS}, TEMP={TEMPERATURE}, THREADS={N_THREADS}")
                # GPT4All model wrapper; adjust init args as needed for your local runtime
                self.llm = GPT4All(model_name, n_threads=N_THREADS)
                logger.info("✓ LLM ready")

            def generate(self, query: str, context_docs: List[dict]) -> str:
                # Minimal context for speed
                context_parts = []
                for i, doc in enumerate(context_docs[:2], 1):
                    text = (doc.get('text') or "")[:300]
                    context_parts.append(f"[{i}] {text}")

                context = "\n".join(context_parts)

                prompt = f"""Based on ScriptBees documentation:

{context}

Question: {query}

Answer (brief and accurate):"""

                try:
                    answer = self.llm.generate(
                        prompt,
                        max_tokens=MAX_TOKENS,
                        temp=TEMPERATURE,
                        top_k=30,
                        top_p=0.9,
                        repeat_penalty=1.15
                    )
                    # GPT4All's generate may return list or str depending on wrapper
                    if isinstance(answer, (list, tuple)):
                        answer = " ".join([str(a) for a in answer])
                    answer = answer.strip()
                    if not any(w in answer.lower() for w in ['source', 'based', 'according', 'scriptbees']):
                        if context_docs:
                            answer += f"\n\n[Source: {context_docs[0].get('url')}]"
                    return answer
                except Exception as e:
                    logger.error(f"LLM error: {e}", exc_info=True)
                    if context_docs:
                        doc = context_docs[0]
                        return f"{(doc.get('text') or '')[:200]}...\n\n[Source: {doc.get('url')}]"
                    return "Sorry — I couldn't generate an answer right now."

        # Instantiate retriever & generator
        retriever = VectorRetriever(INDEX_PATH, PAGES_PATH, META_PATH, MODEL_NAME)
        generator = OptimizedLLMGenerator(LLM_MODEL)

        logger.info("="*70)
        logger.info("✅ SCRIPTBEES RAG SERVER READY")
        logger.info(f"   Target response: 8-15 seconds")
        logger.info(f"   Cached response: <0.1 seconds")
        logger.info(f"   ScriptBees documents: {retriever.index.ntotal}")
        logger.info(f"   Cache: {MAX_CACHE_SIZE} entries")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        # re-raise so uvicorn will show the traceback and shut down
        raise

@app.get("/")
async def root():
    return {
        "service": "ScriptBees RAG Chatbot",
        "version": "4.0",
        "status": "online",
        "company": "ScriptBees IT Pvt Ltd",
        "description": "AI-powered chatbot for ScriptBees.com content",
        "optimizations": ["ultra_fast_tokens", "high_threads", "aggressive_cache"],
        "endpoints": {
            "health": "/health",
            "ask": "/api/ask"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "ScriptBees RAG Chatbot",
        "retriever_loaded": retriever is not None,
        "generator_loaded": generator is not None,
        "num_documents": retriever.index.ntotal if retriever else 0,
        "cache_size": len(response_cache),
        "settings": {
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "threads": N_THREADS,
            "top_k": TOP_K
        }
    }

@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest, api_key: str = Depends(verify_api_key)):
    start_time = time.time()
    try:
        question = request.question.strip()
        logger.info(f"🐝 Q: {question[:120]}")
        # Cache check
        cached = get_cached_response(question)
        if cached:
            elapsed = time.time() - start_time
            logger.info(f"✓ Cache hit ({elapsed:.3f}s)")
            cached['cached'] = True
            cached['response_time_seconds'] = elapsed
            return AskResponse(**cached)

        if not retriever or not generator:
            raise HTTPException(status_code=503, detail="Service not ready - models still loading")

        # Retrieve
        retrieved = retriever.retrieve(question, TOP_K)
        if not retrieved:
            resp = {
                'answer': "I couldn't find relevant information about that topic in ScriptBees' documentation. Please try rephrasing your question or ask about ScriptBees' services, expertise, or projects.",
                'sources': [],
                'retrieved': [],
                'response_time_seconds': time.time() - start_time
            }
            return AskResponse(**resp)

        logger.info(f"✓ Retrieved {len(retrieved)} docs from ScriptBees content")

        # Generate
        answer = generator.generate(question, retrieved)

        sources = [doc['url'] for doc in retrieved]
        retrieved_info = [
            Source(url=doc['url'], title=doc['title'], score=doc['score'])
            for doc in retrieved
        ]

        elapsed = time.time() - start_time

        response = {
            'answer': answer,
            'sources': sources,
            'retrieved': retrieved_info,
            'response_time_seconds': elapsed
        }

        # Cache it
        cache_response(question, response)

        logger.info(f"✓ Answer in {elapsed:.3f}s")

        return AskResponse(**response)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled exception in /api/ask")
        raise HTTPException(status_code=500, detail=str(e))

# Handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(status_code=404, content={"error": "Not Found"})

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.exception("Internal error")
    return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))

    print("\n" + "="*70)
    print("🐝 SCRIPTBEES RAG CHATBOT - PRODUCTION SERVER")
    print(f"   MAX_TOKENS: {MAX_TOKENS} (2x faster)")
    print(f"   TEMPERATURE: {TEMPERATURE} (faster)")
    print(f"   THREADS: {N_THREADS} (2x parallelism)")
    print(f"   PORT: {port}")
    print(f"   INDEX_PATH: {INDEX_PATH}")
    print(f"   PAGES_PATH: {PAGES_PATH}")
    print(f"   META_PATH: {META_PATH}")
    print("="*70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
