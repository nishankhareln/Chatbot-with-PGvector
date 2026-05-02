# How to Run

## Prerequisites
- **Docker Desktop** running
- **Python 3.11+** (only needed if running outside Docker)
- A `.env` file in the project root:
  ```
  GEMINI_API_KEY=your_google_gemini_key
  GOOGLE_API_KEY=your_google_gemini_key
  DATABASE_URL=postgresql://postgres:2060@postgres:5432/rag_database

  # Local-only (used by check_database.py)
  DB_HOST=localhost
  DB_PORT=5433
  DB_NAME=rag_database
  DB_USER=postgres
  DB_PASSWORD=2060
  ```

---

## First-Time Setup

```powershell
cd D:\Projects\SimpleChatbot

# Build the backend image (CPU-only torch, ~5 min first time)
docker build -t simplechatbot-backend:patched .

# Start Postgres + Backend
docker compose up -d
```

Wait ~30 seconds for the backend to load the embedder + reranker models, then open:

**http://localhost:8000**

That's the full app. No Streamlit needed.

---

## Daily Use

Containers persist across reboots. After the first setup, you only need:

```powershell
docker compose up -d        # if stopped
```

Then open **http://localhost:8000**.

---

## Optional: Streamlit UI

A second frontend exists at `frontend/app.py` (same backend, different look):

```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run frontend\app.py
```

Opens at **http://localhost:8501**.

---

## Stopping

```powershell
docker compose down          # stop, keep DB data
docker compose down -v       # stop and wipe DB volume (deletes uploads)
```

---

## After Code Changes

| Changed | Required action |
|---|---|
| Python files in `backend/` | Nothing — uvicorn auto-reloads (`./backend` is volume-mounted) |
| `requirements.txt` or `Dockerfile` | `docker build -t simplechatbot-backend:patched .` then `docker compose up -d --force-recreate backend` |
| `init.sql` | `docker exec -i rag_postgres psql -U postgres -d rag_database < init.sql` (idempotent) |
| `templates/app.html` | Just refresh the browser — read on every request |

---

## Health & Debug

```powershell
docker ps                                       # both should be (healthy)
curl http://localhost:8000/health               # JSON status
docker logs rag_backend --tail 50               # backend logs
docker logs rag_postgres --tail 20              # db logs
docker exec -it rag_postgres psql -U postgres -d rag_database
```

The `/health` response should look like:
```json
{
  "status": "healthy",
  "database": "connected",
  "reranker": true,
  "contextual_retrieval": true
}
```

---

## Common Issues

| Symptom | Fix |
|---|---|
| `Backend Offline` in UI | `docker compose up -d`; wait ~30 s for model load |
| `connection refused` on port 5433 | Postgres container not started; `docker compose up -d postgres` |
| `Could not extract text from PDF` | Scanned PDF — OCR is enabled but slow; try a text-based PDF |
| Old chunks still appear with low scores | Re-upload — old chunks lack the new `tsv` / `parent_section` columns |
| Build downloads ~2 GB CUDA libs | Make sure Dockerfile has the CPU-only torch line (already in repo) |
