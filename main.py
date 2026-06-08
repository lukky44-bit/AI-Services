import os
import sys
import uuid
import json
import logging
import httpx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Ensure the agents directory is in the path to resolve imports correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from orchestrator.agent import OrchestratorAgent
from runner.tools import extract_vus_from_script
from langchain_core.messages import HumanMessage, AIMessage

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator_api")

app = FastAPI(
    title="Orchestrator Agent API",
    description="Backend API services for AI Driven Performance Testing",
    version="1.0.0"
)

# Enable CORS for Next.js frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the global Orchestrator agent
orchestrator_agent = OrchestratorAgent()

# Serve reports static directory
reports_dir = os.path.join(current_dir, "db_analyst", "reports")
os.makedirs(reports_dir, exist_ok=True)
app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")

class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

class ExecuteRequest(BaseModel):
    script: str
    vus: int
    run_id: Optional[str] = None

class StopRequest(BaseModel):
    run_id: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/query")
async def handle_query(request: QueryRequest):
    try:
        # 1. Convert historical messages to LangChain message models
        langchain_messages = []
        for msg in request.history:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))
        
        # 2. Append the current message
        langchain_messages.append(HumanMessage(content=request.message))
        
        # 3. Invoke the Orchestrator Agent
        state = {"messages": langchain_messages}
        
        # Run in a threadpool executor to avoid blocking the event loop for LangGraph/LangChain execution
        from fastapi.concurrency import run_in_threadpool
        response = await run_in_threadpool(orchestrator_agent.invoke, state)
        
        # 4. Process final response and output fields
        ans = response["messages"][0].content
        route = response.get("route", "direct")
        reasoning = response.get("reason", "")
        pdf_path = response.get("pdf_path", "")
        sources = response.get("sources", [])
        
        # Convert absolute PDF file path to static route URL
        pdf_url = None
        if pdf_path:
            pdf_filename = os.path.basename(pdf_path)
            pdf_url = f"/reports/{pdf_filename}"
            
        # Detect if a script was generated during intermediate steps
        generated_script = None
        max_vus = None
        intermediate_steps = response.get("intermediate_steps", [])
        for action, observation in intermediate_steps:
            tool_name = getattr(action, "tool", None)
            if not tool_name and isinstance(action, dict):
                tool_name = action.get("tool")
                
            if tool_name == "generate_k6_script":
                generated_script = observation
                max_vus = extract_vus_from_script(observation)
                
        return {
            "role": "assistant",
            "content": ans,
            "route": route,
            "reason": reasoning,
            "pdf_url": pdf_url,
            "sources": sources,
            "script": generated_script,
            "max_vus": max_vus
        }
    except Exception as e:
        logger.error(f"Error handling query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/execute-test")
async def execute_test(request: ExecuteRequest):
    run_id = request.run_id or str(uuid.uuid4())
    
    async def event_generator():
        runner_url = os.getenv("RUNNER_TRIGGER_URL", "http://localhost:8081/run-test").strip()
        payload = {
            "script": request.script,
            "vus": request.vus or 1,
            "run_id": run_id
        }
        
        # Send initial metadata chunk
        yield f"data: {json.dumps({'status': 'initializing', 'run_id': run_id})}\n\n"
        
        try:
            async with httpx.AsyncClient(timeout=3600.0) as client:
                async with client.stream("POST", runner_url, json=payload) as r:
                    if r.status_code >= 400:
                        error_body = await r.aread()
                        yield f"data: {json.dumps({'status': 'error', 'message': f'Runner returned status code {r.status_code}: {error_body.decode()}'})}\n\n"
                        return
                    
                    async for line in r.aiter_lines():
                        if line:
                            yield f"{line}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/stop-test")
async def stop_test(request: StopRequest):
    runner_stop_url = os.getenv("RUNNER_STOP_URL", "http://localhost:8081/stop").strip()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(runner_stop_url, json={"run_id": request.run_id})
            if res.status_code >= 400:
                raise HTTPException(status_code=res.status_code, detail=res.text)
            return res.json()
    except Exception as e:
        logger.error(f"Error stopping test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test-results/{run_id}")
async def get_test_results(run_id: str):
    """Fetches test summary metrics from the database after a test run completes."""
    from fastapi.concurrency import run_in_threadpool
    
    def _fetch_metrics():
        from db_analyst.database import DatabaseManager
        db = DatabaseManager()
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT metrics FROM test_summaries WHERE run_id = %s;", (run_id,))
        result = cursor.fetchone()
        conn.close()
        return result
    
    try:
        result = await run_in_threadpool(_fetch_metrics)
        
        if not result or not result[0]:
            raise HTTPException(status_code=404, detail=f"No test summary found for run_id: {run_id}")
        
        metrics = result[0]
        if isinstance(metrics, str):
            metrics = json.loads(metrics)
        
        # Extract KPIs from the metrics
        is_new_format = isinstance(metrics, dict) and "metrics" in metrics
        
        if is_new_format:
            raw = metrics["metrics"]
            state = metrics.get("state", {})
            
            # Duration
            duration_ms = state.get("testRunDurationMs", 0)
            duration_str = f"{duration_ms / 1000:.1f}s" if duration_ms else "Unknown"
            
            # VUs
            vus_max = "-"
            if "vus_max" in raw:
                vus_max = raw["vus_max"].get("values", {}).get("max", "-")
            elif "vus" in raw:
                vus_max = raw["vus"].get("values", {}).get("max", "-")
            
            # Requests
            total_reqs = "-"
            reqs_rate = "-"
            if "http_reqs" in raw:
                rv = raw["http_reqs"].get("values", {})
                total_reqs = rv.get("count", "-")
                reqs_rate = rv.get("rate", "-")
            
            # Error rate
            fail_rate = "-"
            if "http_req_failed" in raw:
                fv = raw["http_req_failed"].get("values", {})
                rate_val = fv.get("rate", 0)
                fail_rate = f"{rate_val * 100:.2f}%" if isinstance(rate_val, (int, float)) else str(rate_val)
            
            # Duration stats
            duration_stats = {}
            if "http_req_duration" in raw:
                dv = raw["http_req_duration"].get("values", {})
                duration_stats = {
                    "avg": f"{dv.get('avg', 0):.2f} ms" if isinstance(dv.get('avg'), (int, float)) else "-",
                    "min": f"{dv.get('min', 0):.2f} ms" if isinstance(dv.get('min'), (int, float)) else "-",
                    "max": f"{dv.get('max', 0):.2f} ms" if isinstance(dv.get('max'), (int, float)) else "-",
                    "p90": f"{dv.get('p(90)', 0):.2f} ms" if isinstance(dv.get('p(90)'), (int, float)) else "-",
                    "p95": f"{dv.get('p(95)', 0):.2f} ms" if isinstance(dv.get('p(95)'), (int, float)) else "-",
                    "med": f"{dv.get('med', 0):.2f} ms" if isinstance(dv.get('med'), (int, float)) else "-",
                }
            
            # Data transfer
            data_received = "-"
            data_sent = "-"
            if "data_received" in raw:
                dr = raw["data_received"].get("values", {}).get("count", 0)
                if isinstance(dr, (int, float)):
                    data_received = f"{dr / (1024*1024):.2f} MB" if dr >= 1024*1024 else f"{dr / 1024:.2f} kB"
            if "data_sent" in raw:
                ds = raw["data_sent"].get("values", {}).get("count", 0)
                if isinstance(ds, (int, float)):
                    data_sent = f"{ds / (1024*1024):.2f} MB" if ds >= 1024*1024 else f"{ds / 1024:.2f} kB"
            
            # Checks
            checks = []
            root_group = metrics.get("root_group", {})
            if root_group:
                for check in root_group.get("checks", []):
                    name = check.get("name", "check")
                    passes = check.get("passes", 0)
                    fails = check.get("fails", 0)
                    total = passes + fails
                    pct = (passes / total * 100) if total > 0 else 0.0
                    checks.append({"name": name, "passes": passes, "fails": fails, "pass_rate": f"{pct:.2f}%"})
            
            return {
                "run_id": run_id,
                "duration": duration_str,
                "peak_vus": vus_max,
                "total_requests": total_reqs,
                "requests_per_sec": round(reqs_rate, 2) if isinstance(reqs_rate, (int, float)) else reqs_rate,
                "error_rate": fail_rate,
                "response_times": duration_stats,
                "data_received": data_received,
                "data_sent": data_sent,
                "checks": checks,
                "raw_metrics": metrics
            }
        else:
            # Old flat format fallback
            return {
                "run_id": run_id,
                "raw_metrics": metrics
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching test results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

