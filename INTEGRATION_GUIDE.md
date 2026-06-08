# Agent Module — Integration Guide

> **This is NOT a microservice.** Copy the `agents/` folder into your backend project and mount the routes directly on your existing server.

---

## What You're Receiving

```
agents/
├── main.py                  ← FastAPI routes (mount these on your server)
├── requirements.txt         ← Python dependencies to merge
├── .env.example             ← Required environment variables
│
├── orchestrator/            ← Brain — routes queries to the right sub-agent
│   ├── agent.py
│   └── __init__.py
│
├── runner/                  ← Generates k6 scripts & triggers test execution
│   ├── agent.py
│   ├── prompts.py
│   ├── tools.py
│   └── __init__.py
│
├── db_analyst/              ← Queries PostgreSQL for test results & generates PDF reports
│   ├── agent.py
│   ├── config.py
│   ├── database.py
│   ├── prompts.py
│   ├── report_generator.py
│   ├── tools.py
│   └── __init__.py
│
├── rag/                     ← Retrieval-Augmented Generation for k6 documentation Q&A
│   ├── agent.py
│   ├── config.py
│   ├── ingestion_code.py
│   ├── prompts.py
│   ├── retriever.py
│   ├── tools.py
│   ├── utils.py
│   └── __init__.py
│
├── Dockerfile               ← (optional, ignore if embedding directly)
└── test_ui.py               ← (test-only, ignore)
```

---

## How to Integrate (3 Steps)

### Step 1 — Copy the folder

Copy the entire `agents/` folder into your backend project. For example:

```
your-backend/
├── src/
│   ├── auth/
│   ├── users/
│   ├── applications/
│   └── agents/          ← paste here
│       ├── orchestrator/
│       ├── runner/
│       ├── db_analyst/
│       ├── rag/
│       └── main.py
├── server.py            ← your main server
└── requirements.txt
```

### Step 2 — Mount the routes

Open `main.py` inside the agents folder. It defines a FastAPI `app` with all the routes. **Don't run it separately.** Instead, mount it as a sub-application on your main server:

```python
# your server.py
from fastapi import FastAPI

app = FastAPI(title="Your Main Backend")

# ... your auth, users, applications routes ...

# Mount the agent routes under /agent prefix
from agents.main import app as agent_app
app.mount("/agent", agent_app)
```

This gives you these routes automatically:

| Your Server Route | What It Does |
|---|---|
| `POST /agent/api/query` | Send a chat message → AI responds |
| `POST /agent/api/execute-test` | Run a k6 load test (SSE stream) |
| `POST /agent/api/stop-test` | Stop a running test |
| `GET /agent/api/test-results/{run_id}` | Get test metrics after completion |
| `GET /agent/reports/{filename}` | Download PDF report |
| `GET /agent/health` | Health check |

### Step 3 — Install dependencies

Merge the agent's dependencies into your project:

```bash
pip install -r agents/requirements.txt
```

Key packages: `langchain`, `langchain-groq`, `langchain-huggingface`, `langchain-chroma`, `chromadb`, `sentence-transformers`, `psycopg2-binary`, `httpx`, `reportlab`

---

## Environment Variables

Add these to your server's environment:

```env
# Required — LLM API Key
GROQ_API_KEY=gsk_xxxxxxxxxxxxx

# PostgreSQL (same DB you're already using)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=loadtest
DB_USER=postgres
DB_PASSWORD=postgres

# Go Runner Service
RUNNER_TRIGGER_URL=http://localhost:8081/run-test
RUNNER_STOP_URL=http://localhost:8081/stop
```

---

## API Contract

### `POST /agent/api/query` — Main Chat Endpoint

**Your server calls this when a user sends a message in the chatbot.**

**Request:**
```json
{
  "message": "generate a load test for https://api.example.com/users with 50 VUs",
  "history": [
    { "role": "user", "content": "hello" },
    { "role": "assistant", "content": "Hi! How can I help?" }
  ]
}
```

- `message` (string, required) — the user's current query
- `history` (array, optional) — previous messages loaded from YOUR database

**Response:**
```json
{
  "role": "assistant",
  "content": "Here is your k6 script:\n```javascript\n...\n```",
  "route": "runner_agent",
  "reason": "User wants to generate a load test",
  "pdf_url": null,
  "sources": [],
  "script": "import http from 'k6/http';\n...",
  "max_vus": 50
}
```

| Field | Type | Description |
|---|---|---|
| `content` | string | AI response (markdown) — save this in your chat history DB |
| `route` | string | `runner_agent` / `db_agent` / `rag_agent` / `direct` |
| `reason` | string | Why this route was chosen |
| `script` | string or null | Generated k6 test script (if applicable) |
| `max_vus` | int or null | Max VUs extracted from the script |
| `pdf_url` | string or null | Path to download PDF report |
| `sources` | string[] | Reference URLs from k6 documentation |

---

### `POST /agent/api/execute-test` — Run a Test (SSE)

**Call this when the user clicks "Run Test" in the frontend.**

**Request:**
```json
{
  "script": "import http from 'k6/http';\n...",
  "vus": 50,
  "run_id": "your-uuid-here"
}
```

- `script` (string, required) — from the `script` field in `/api/query` response
- `vus` (int, required) — from the `max_vus` field in `/api/query` response
- `run_id` (string, optional) — unique ID for tracking

**Response:** Server-Sent Events (SSE) stream

```
data: {"status": "initializing", "run_id": "abc-123"}
data: Workflow started with ID: load-test-abc-123
data: Workflow status: Running
data: Workflow status: Completed
data: BACKEND COMPLETED
```

> When you see `BACKEND COMPLETED`, the test is done. Remove the "Stop" button and call `/api/test-results/{run_id}` to fetch metrics.

---

### `POST /agent/api/stop-test`

```json
{ "run_id": "abc-123" }
```

---

### `GET /agent/api/test-results/{run_id}` — Post-Completion Metrics

**Call this after the test stream shows `BACKEND COMPLETED`.**

**Response:**
```json
{
  "run_id": "abc-123",
  "duration": "35.2s",
  "peak_vus": 50,
  "total_requests": 1847,
  "requests_per_sec": 52.45,
  "error_rate": "0.00%",
  "response_times": {
    "avg": "245.67 ms",
    "min": "120.34 ms",
    "max": "890.12 ms",
    "med": "230.45 ms",
    "p90": "456.78 ms",
    "p95": "567.89 ms"
  },
  "data_received": "4.23 MB",
  "data_sent": "0.89 MB",
  "checks": [
    { "name": "status is 200", "passes": 1847, "fails": 0, "pass_rate": "100.00%" }
  ]
}
```

---

## Your Responsibility vs Agent's Responsibility

| **You Handle** | **Agent Handles** |
|---|---|
| User auth (login/signup) | ❌ |
| App & endpoint registration | ❌ |
| Store chat history in your DB | ❌ |
| Load history → pass to `/api/query` | ✅ Receives & processes it |
| Save AI response to your DB | ❌ |
| User clicks "Run Test" → call `/api/execute-test` | ✅ Runs test, streams logs |
| User clicks "Stop" → call `/api/stop-test` | ✅ Stops test |
| Show metrics → call `/api/test-results/{run_id}` | ✅ Returns formatted metrics |

---

## Important Notes

1. **The agent is stateless.** It does NOT persist anything. You must send the full `history` array with every `/api/query` call.
2. **The agent uses YOUR PostgreSQL.** It reads test metrics from the `test_summaries` table — same database your Go runner writes to.
3. **Files to ignore:** `test_ui.py` (test-only), `Dockerfile` (not needed if embedding directly).
4. **First call is slow** (~10-15s) because it loads ML models into memory. Subsequent calls are fast.
