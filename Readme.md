# Local (dev) workflow

1. Scrape site:
   python scrape/scraper.py --start-url https://scriptbees.com --max-pages 50 --output-dir content

2. Generate embeddings (run on your machine with sentence-transformers installed):
   python embeddings/embedder.py --content-dir ./content --persist-dir ./backend/embeddings --model all-MiniLM-L6-v2

   This will write:
     backend/embeddings/pages.faiss
     backend/embeddings/pages.json
     backend/embeddings/pages_meta.json

3. Commit the *small* embeddings files if you want Render to serve them:
   git add backend/embeddings/pages.faiss backend/embeddings/pages.json backend/embeddings/pages_meta.json
   git commit -m "Add built FAISS/embeddings"
   git push

4. Deploy to Render:
   - Create new Web Service linking to the repo
   - Set start command: `./start_prod.sh` (or leave empty – Render runs `gunicorn` based on Procfile)
   - Add environment variables (if needed):
       RAG_API_KEY=<your_api_key>
       OPENAI_API_KEY=<if you want OpenAI fallback>
       FRONTEND_ORIGIN=*
   - Use the small `requirements.txt` included

Notes:
- Do NOT install `sentence-transformers` or large LLM packages on Render (keeps deploy small).
- If you prefer not to commit embeddings to git, you can upload `pages.*` via Render dashboard or store in S3 and set FAISS_INDEX/PAGES_PATH env vars.
