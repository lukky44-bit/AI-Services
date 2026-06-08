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
    vus: Optional[int] = None
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
        intermediate_steps = response.get("intermediate_steps", [])
        for action, observation in intermediate_steps:
            tool_name = getattr(action, "tool", None)
            if not tool_name and isinstance(action, dict):
                tool_name = action.get("tool")
                
            if tool_name == "generate_k6_script":
                generated_script = observation
                
        return {
            "role": "assistant",
            "content": ans,
            "route": route,
            "reason": reasoning,
            "pdf_url": pdf_url,
            "sources": sources,
            "script": generated_script
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
