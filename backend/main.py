"""
backend/main.py

ScriptBees RAG Chatbot - Production-ready main file.

Behavior:
- Loads FAISS index + pages.json from embeddings/ (defaults to repo_root/embeddings)
- Uses SentenceTransformer for retrieval
- Tries to use gpt4all for local LLM; if not installed, falls back to OpenAI API if OPENAI_API_KEY is set.
- If neither LLM is available, uses a safe stub generator (returns retrieved text).
- Lifespan handler used for startup/shutdown.
"""

import os
import json
import logging
import time
import hashlib
from typing import List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

# Optional dotenv
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# helper to find .env up the tree (useful in dev)
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
if _env_path and load_dotenv:
    load_dotenv(dotenv_path=_env_path, override=False)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------
# Configurable settings via env (safe defaults)
# -------------------------
API_KEY = os.getenv("RAG_API_KEY", "").strip()  # recommend setting in production (empty disables auth)
CONTENT_DIR = os.getenv("CONTENT_DIR", "content")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # sentence-transformers model for embeddings
LLM_MODEL = os.getenv("LLM_MODEL", "orca-mini-3b-gguf2-q4_0.gguf")  # default local gpt4all model name / path

TOP_K = int(os.getenv("TOP_K", "2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "150"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
N_THREADS = int(os.getenv("N_THREADS", "8"))

MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "300"))

# -------------------------
# Path resolution (resolve relative env values against repo root)
# -------------------------
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent  # assume backend/ inside repo root
def _abs_from_env(name: str, default_path: Path) -> str:
    v = os.getenv(name)
    if v:
        if os.path.isabs(v):
            return str(Path(v).resolve())
        return str((REPO_ROOT / v).resolve())
    return str(default_path.resolve())

INDEX_PATH = _abs_from_env("FAISS_INDEX", REPO_ROOT / "backend" / "embeddings" / "pages.faiss")
PAGES_PATH = _abs_from_env("PAGES_PATH",  REPO_ROOT / "backend" / "embeddings" / "pages.json")
META_PATH  = _abs_from_env("META_PATH",   REPO_ROOT / "backend" / "embeddings" / "pages_meta.json")


logger.info(f"Resolved INDEX_PATH={INDEX_PATH}")
logger.info(f"Resolved PAGES_PATH={PAGES_PATH}")
logger.info(f"Resolved META_PATH={META_PATH}")

# -------------------------
# FastAPI and models
# -------------------------
from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

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

# API Key header support (optional - if API_KEY is empty, skip enforcement)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
def verify_api_key(request: Request, api_key: str = Security(api_key_header)):
    if not API_KEY:
        # no server-side API key set -> allow all (not recommended for prod)
        return ""
    incoming = (api_key or "").strip()
    if not incoming:
        auth = request.headers.get("authorization", "")
        if auth and auth.lower().startswith("bearer "):
            incoming = auth.split(" ", 1)[1].strip()
    if not incoming or incoming != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return incoming

# in-memory cache
response_cache = {}
def get_cache_key(question: str) -> str:
    return hashlib.md5(question.lower().strip().encode()).hexdigest()

def get_cached_response(question: str):
    return response_cache.get(get_cache_key(question))

def cache_response(question: str, response: dict):
    key = get_cache_key(question)
    if len(response_cache) >= MAX_CACHE_SIZE:
        oldest = next(iter(response_cache))
        del response_cache[oldest]
    response_cache[key] = response

# Globals to be set in startup
retriever = None
generator = None

app = FastAPI(title="ScriptBees RAG Chatbot", version="4.0", description="AI-powered chatbot for ScriptBees.com content")

# CORS - in production set explicit origins via env var FRONTEND_ORIGIN (comma separated)
_frontend_origin = os.getenv("FRONTEND_ORIGIN", "*")
allow_origins = [_frontend_origin] if _frontend_origin != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Utility: prompt builder
# -------------------------
def construct_prompt(query: str, context_docs: List[dict]) -> str:
    # Build a short, focused prompt to keep latency low
    ctx_parts = []
    for i, d in enumerate(context_docs[:3], 1):
        snippet = (d.get("text") or "")[:300].strip()
        ctx_parts.append(f"[{i}] {snippet}\nSource: {d.get('url')}")
    context = "\n\n".join(ctx_parts)
    prompt = f"""You are an assistant answering questions about ScriptBees.com using only the provided context. Reply concisely and accurately.

Context:
{context}

Question: {query}

Answer (brief, 2-6 sentences). If unsure, say you don't know and suggest rephrasing."""
    return prompt

# -------------------------
# Lifespan startup: load index, embeddings, and LLMs
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, generator
    logger.info("="*70)
    logger.info("🐝 SCRIPTBEES RAG CHATBOT - PRODUCTION SERVER STARTING")
    logger.info("="*70)
    try:
        # lazy imports so missing deps show helpful tracebacks
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        # Validate presence of index + pages
        if not os.path.exists(INDEX_PATH) or not os.path.exists(PAGES_PATH):
            hint = (
                f"Missing files. Expected:\n  {INDEX_PATH}\n  {PAGES_PATH}\n\n"
                "Run scraping then build index:\n"
                "  python scrape/scraper.py --start-url https://scriptbees.com\n"
                "  python embeddings/embedder.py --content-dir ./content --persist-dir ./embeddings\n"
                "Or set env vars FAISS_INDEX/PAGES_PATH/META_PATH to the correct files."
            )
            raise FileNotFoundError(hint)

        # ------- Retriever class -------
        class VectorRetriever:
            def __init__(self, index_path, pages_path, meta_path, model_name):
                logger.info("📦 Loading retriever...")
                self.model = SentenceTransformer(model_name)
                # faiss index
                self.index = faiss.read_index(index_path)

                with open(pages_path, 'r', encoding='utf-8') as f:
                    pages = json.load(f)

                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                else:
                    logger.warning(f"No meta file at {meta_path} — deriving meta from pages.json")
                    self.metadata = [{"id": p.get("id", idx), "url": p.get("url", ""), "title": p.get("title", "")} for idx, p in enumerate(pages)]

                # build pages dict keyed by numeric id
                self.pages_dict = {}
                for idx, p in enumerate(pages):
                    pid = p.get("id")
                    if pid is None:
                        pid = idx
                    self.pages_dict[int(pid)] = p

                total = getattr(self.index, "ntotal", None)
                logger.info(f"✓ Indexed: {total} ScriptBees documents")

                # quick dup check (best-effort)
                try:
                    unique = set()
                    for p in pages:
                        unique.add((p.get('text') or "")[:120])
                    if len(unique) < len(pages):
                        logger.warning(f"⚠️ Found possible duplicates: {len(pages)-len(unique)}")
                except Exception:
                    pass

            def retrieve(self, query: str, top_k: int = TOP_K):
                query_emb = self.model.encode([query], convert_to_numpy=True)
                import numpy as _np
                q = _np.asarray(query_emb, dtype=_np.float32)
                # normalize for cosine search
                try:
                    faiss.normalize_L2(q)
                except Exception:
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
                    page = self.pages_dict.get(int(idx))
                    if not page and meta:
                        page = self.pages_dict.get(int(meta.get("id", idx)))
                    if page:
                        results.append({
                            'id': int(idx),
                            'url': meta.get('url', page.get('url', '')) if meta else page.get('url', ''),
                            'title': meta.get('title', page.get('title', '')) if meta else page.get('title', ''),
                            'text': (page.get('text') or "")[:800],
                            'score': float(score)
                        })
                return results

        # ------- LLM generator (gpt4all primary, OpenAI fallback, stub fallback) -------
        use_gpt4all = False
        use_openai = False
        gpt4all_mod = None

        try:
            # Try to import gpt4all (preferred for local LLM)
            from gpt4all import GPT4All  # type: ignore
            gpt4all_mod = GPT4All
            use_gpt4all = True
            logger.info("gpt4all available — will use local model if model file present.")
        except Exception as e:
            logger.warning("gpt4all not available in environment: %s", e)
            # try OpenAI fallback
            if os.getenv("OPENAI_API_KEY"):
                try:
                    import openai  # type: ignore
                    use_openai = True
                    logger.info("OPENAI_API_KEY found — will use OpenAI as fallback LLM.")
                except Exception as ex:
                    logger.warning("openai module not available: %s", ex)

        class OptimizedLLMGenerator:
            def __init__(self, model_name):
                self.model_name = model_name
                self.kind = "stub"
                self._gpt4all = None
                self._openai = None

                if use_gpt4all:
                    try:
                        self._gpt4all = gpt4all_mod(model_name, n_threads=N_THREADS)
                        self.kind = "gpt4all"
                        logger.info(f"🤖 Loaded gpt4all model: {model_name}")
                    except Exception as e:
                        logger.error("Failed to load gpt4all model: %s", e, exc_info=True)
                        self._gpt4all = None
                        # if openai available, we'll fallback next
                if self._gpt4all is None and use_openai:
                    try:
                        import openai  # type: ignore
                        openai.api_key = os.getenv("OPENAI_API_KEY")
                        self._openai = openai
                        self.kind = "openai"
                        logger.info("🤖 Using OpenAI API as LLM backend")
                    except Exception as e:
                        logger.error("Failed to initialize OpenAI client: %s", e, exc_info=True)
                        self._openai = None

                if self.kind == "stub":
                    logger.warning("LLM stub active — returning retrieved context as answer (safe fallback).")

            def generate(self, query: str, context_docs: List[dict]) -> str:
                prompt = construct_prompt(query, context_docs)

                if self.kind == "gpt4all" and self._gpt4all:
                    try:
                        # gpt4all generate API varies by wrapper; this assumes .generate exists
                        out = self._gpt4all.generate(
                            prompt,
                            max_tokens=MAX_TOKENS,
                            temp=TEMPERATURE,
                            top_k=30,
                            top_p=0.9,
                            repeat_penalty=1.15
                        )
                        if isinstance(out, (list, tuple)):
                            out = " ".join(map(str, out))
                        answer = str(out).strip()
                        if not answer and context_docs:
                            return f"{(context_docs[0].get('text') or '')[:400]}...\n\n[Source: {context_docs[0].get('url')}]"
                        return answer
                    except Exception as e:
                        logger.error("gpt4all generation error: %s", e, exc_info=True)
                        # fallback to openai if available
                        if self._openai:
                            self.kind = "openai"

                if self.kind == "openai" and self._openai:
                    try:
                        # Use ChatCompletion if available, else Completion
                        client = self._openai
                        # prefer ChatCompletion
                        try:
                            resp = client.ChatCompletion.create(
                                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                                messages=[{"role":"system","content":"You answer concisely based on given context."},
                                          {"role":"user","content":prompt}],
                                max_tokens=MAX_TOKENS,
                                temperature=float(os.getenv("OPENAI_TEMP", TEMPERATURE))
                            )
                            # extract text
                            choice = resp.choices[0]
                            if hasattr(choice, "message"):
                                return choice.message.get("content", "").strip()
                            else:
                                return getattr(choice, "text", str(resp)).strip()
                        except Exception:
                            # fallback older API
                            resp = client.Completion.create(
                                engine=os.getenv("OPENAI_MODEL", "text-davinci-003"),
                                prompt=prompt,
                                max_tokens=MAX_TOKENS,
                                temperature=float(os.getenv("OPENAI_TEMP", TEMPERATURE))
                            )
                            return getattr(resp.choices[0], "text", "").strip()
                    except Exception as e:
                        logger.error("OpenAI generation error: %s", e, exc_info=True)
                        # fall through to stub

                # Stub fallback: return first retrieved doc text + source
                if context_docs:
                    doc = context_docs[0]
                    return f"{(doc.get('text') or '')[:500].strip()}...\n\n[Source: {doc.get('url')}]"
                return "Sorry — I couldn't generate an answer right now."

        # instantiate retriever & generator
        retriever = VectorRetriever(INDEX_PATH, PAGES_PATH, META_PATH, MODEL_NAME)
        generator = OptimizedLLMGenerator(LLM_MODEL)

        logger.info("="*70)
        logger.info("✅ SCRIPTBEES RAG SERVER READY")
        logger.info(f"   Target response: 8-15 seconds (depends on LLM and model)")
        logger.info(f"   Cached response: <0.1 seconds")
        try:
            logger.info(f"   ScriptBees documents: {retriever.index.ntotal}")
        except Exception:
            logger.info("   ScriptBees documents: unknown")
        logger.info(f"   Cache: {MAX_CACHE_SIZE} entries")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        # Re-raise so the host logs the traceback and the process exits
        raise

    try:
        yield
    finally:
        logger.info("Shutting down ScriptBees RAG Chatbot...")

# attach lifespan
app.router.lifespan_context = lifespan

# -------------------------
# Endpoints
# -------------------------
@app.get("/")
async def root():
    return {
        "service": "ScriptBees RAG Chatbot",
        "version": "4.0",
        "status": "online",
        "company": "ScriptBees IT Pvt Ltd",
        "description": "AI-powered chatbot for ScriptBees.com content",
        "endpoints": {"health": "/health", "ask": "/api/ask"}
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "retriever_loaded": retriever is not None,
        "generator_loaded": generator is not None,
        "num_documents": getattr(retriever.index, "ntotal", 0) if retriever else 0,
        "cache_size": len(response_cache),
        "settings": {"max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "threads": N_THREADS, "top_k": TOP_K}
    }

@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest, api_key: str = Depends(verify_api_key)):
    start_time = time.time()
    try:
        question = request.question.strip()
        logger.info(f"🐝 Q: {question[:160]}")

        # cache
        cached = get_cached_response(question)
        if cached:
            elapsed = time.time() - start_time
            logger.info(f"✓ Cache hit ({elapsed:.3f}s)")
            cached['cached'] = True
            cached['response_time_seconds'] = elapsed
            return AskResponse(**cached)

        if not retriever or not generator:
            raise HTTPException(status_code=503, detail="Service not ready - models still loading")

        # retrieval
        retrieved = retriever.retrieve(question, TOP_K)
        if not retrieved:
            resp = {
                'answer': "I couldn't find relevant information in ScriptBees' documentation. Try rephrasing.",
                'sources': [],
                'retrieved': [],
                'response_time_seconds': time.time() - start_time
            }
            return AskResponse(**resp)

        logger.info(f"✓ Retrieved {len(retrieved)} docs")

        # generation
        answer = generator.generate(question, retrieved)

        sources = [d['url'] for d in retrieved]
        retrieved_info = [Source(url=d['url'], title=d['title'], score=d['score']) for d in retrieved]

        elapsed = time.time() - start_time
        response = {
            'answer': answer,
            'sources': sources,
            'retrieved': retrieved_info,
            'response_time_seconds': elapsed
        }

        cache_response(question, response)
        logger.info(f"✓ Answer in {elapsed:.3f}s")
        return AskResponse(**response)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled exception in /api/ask")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Error handlers
# -------------------------
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(status_code=404, content={"error": "Not Found"})

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.exception("Internal error")
    return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

# -------------------------
# Run locally
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print("\n" + "="*70)
    print("🐝 SCRIPTBEES RAG CHATBOT - DEV/LOCAL START")
    print(f"   PORT: {port}")
    print(f"   INDEX_PATH: {INDEX_PATH}")
    print(f"   PAGES_PATH: {PAGES_PATH}")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
