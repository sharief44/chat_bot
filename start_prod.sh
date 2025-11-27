#!/usr/bin/env bash
set -euo pipefail

# Where Render will run from: project root
PERSIST_DIR="backend/embeddings"
mkdir -p "$PERSIST_DIR"

# If an EMBEDDINGS_URL is provided (tar.gz or zip) download & extract to backend/embeddings
if [ -n "${EMBEDDINGS_URL:-}" ]; then
  echo "Downloading embeddings from: $EMBEDDINGS_URL"
  tmpfile="/tmp/embeddings_download.$(date +%s)"
  # prefer curl, fallback to wget
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$tmpfile" "$EMBEDDINGS_URL"
  else
    wget -O "$tmpfile" "$EMBEDDINGS_URL"
  fi

  # try tar, else unzip
  if file "$tmpfile" | grep -q 'gzip\|tar'; then
    tar -xzf "$tmpfile" -C "$PERSIST_DIR" || tar -xzf "$tmpfile" -C .
  else
    # attempt unzip into persist dir
    unzip -o "$tmpfile" -d "$PERSIST_DIR" || true
  fi
  rm -f "$tmpfile"
fi

# If env points directly to single files (FAISS / json), attempt to download each (optional)
# E.g. EMB_PAGES_JSON, EMB_PAGES_FAISS as direct urls
if [ -n "${EMB_PAGES_JSON:-}" ]; then
  echo "Downloading pages.json from EMB_PAGES_JSON"
  curl -L -o "$PERSIST_DIR/pages.json" "$EMB_PAGES_JSON"
fi
if [ -n "${EMB_PAGES_FAISS:-}" ]; then
  echo "Downloading pages.faiss from EMB_PAGES_FAISS"
  curl -L -o "$PERSIST_DIR/pages.faiss" "$EMB_PAGES_FAISS"
fi
if [ -n "${EMB_PAGES_META:-}" ]; then
  echo "Downloading pages_meta.json from EMB_PAGES_META"
  curl -L -o "$PERSIST_DIR/pages_meta.json" "$EMB_PAGES_META"
fi

# Print listing
echo "Embedding dir contents:"
ls -la "$PERSIST_DIR" || true

# Start gunicorn (match your main:app)
# Use the PORT that Render provides
exec gunicorn -k uvicorn.workers.UvicornWorker --workers 1 --timeout 120 --bind 0.0.0.0:${PORT:-10000} backend.main:app
