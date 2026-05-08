# NanoChat (Frontend + Backend)

This repo now runs as a split app:
- Backend API: FastAPI on port `8000`
- Frontend: Vite on port `5173`

## Backend entrypoint and routes
- Entrypoint: `backend_server.py`
- Start command:
  - `uvicorn backend_server:app --host 0.0.0.0 --port 8000 --reload`
- Routes:
  - `GET /health`
  - `GET /api/health`
  - `POST /chat`
  - `POST /api/chat`

`/api/*` routes are included for easy frontend proxying.

## Frontend API strategy
- Development:
  - Frontend calls relative `/api/*`
  - Vite proxy forwards `/api` to `http://127.0.0.1:8000`
  - No hardcoded localhost in frontend code
- Production:
  - Frontend uses `VITE_API_URL` if set
  - Safe default is `/api`

Create `frontend/.env` from `frontend/.env.example` for production builds if needed:

```bash
cp frontend/.env.example frontend/.env
# then set VITE_API_URL to your deployed backend base path, e.g.:
# VITE_API_URL=https://api.yourdomain.com/api
```

## Run locally
From repo root:

1. Install backend dependencies (one-time):

```bash
pip install -r requirements-backend.txt
```

2. Start backend:

```bash
uvicorn backend_server:app --host 0.0.0.0 --port 8000 --reload
```

3. In a second terminal, start frontend:

```bash
cd frontend
npm install
npm run dev
```

4. Open:
- `http://127.0.0.1:5173`

## Verify end-to-end

### Health check
Hit backend directly:

```bash
curl http://127.0.0.1:8000/api/health
```

Expected shape:

```json
{"status":"ok","upstream":"..."}
```

### Chat request
Hit backend directly:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello from curl","temperature":0.7,"max_tokens":80}'
```

Expected shape:

```json
{"response":"..."}
```

You can also verify through Vite proxy while frontend dev server is running:

```bash
curl http://127.0.0.1:5173/api/health
curl -X POST http://127.0.0.1:5173/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello through proxy","temperature":0.7,"max_tokens":80}'
```
