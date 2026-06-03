import os
import sys
import json
import logging
import asyncio
import warnings
from typing import TypedDict, List, Any, Annotated

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="langchain")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Ensure parent directory is in sys.path to resolve subagent modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from db_analyst_agent.agent import DBAnalystAgent
from runner_agent.agent import RunnerAgent
from rag_agent.agent import K6ExpertAgent

load_dotenv()

class GraphState(TypedDict):
    """
    State definition for the orchestrator pipeline.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    route: str
    reason: str
    intermediate_steps: List[Any]
    pdf_path: str
    sources: List[str]

class OrchestratorAgent:
    """
    Core orchestrator agent that receives conversation history, analyzes intent,
    routes to DB Analyst, Runner, or K6 Expert (RAG) Agent, or handles direct conversation.
    Uses LangGraph for stateful routing.
    """
    
    def __init__(self):
        # Initialize specialized sub-agents
        self.db_agent = DBAnalystAgent()
        self.runner_agent = RunnerAgent()
        self.rag_agent = K6ExpertAgent()
        
        # Load environment-specific LLM parameters
        model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        
        # Router LLM
        self.llm = ChatGroq(model=model_name, temperature=temperature)
        
        # Build and compile the workflow graph
        self.workflow = self._build_graph()
        
    def _build_graph(self):
        builder = StateGraph(GraphState)
        
        # Add functional nodes
        builder.add_node("router", self.router_node)
        builder.add_node("runner", self.runner_node)
        builder.add_node("db_analyst", self.db_analyst_node)
        builder.add_node("k6_rag", self.k6_rag_node)
        builder.add_node("direct_handler", self.direct_handler_node)
        
        # Set entry-point
        builder.add_edge(START, "router")
        
        # Define routing logic
        builder.add_conditional_edges(
            "router",
            self.route_decision,
            {
                "runner_agent": "runner",
                "db_agent": "db_analyst",
                "rag_agent": "k6_rag",
                "direct": "direct_handler"
            }
        )
        
        # Connect terminal nodes to END
        builder.add_edge("runner", END)
        builder.add_edge("db_analyst", END)
        builder.add_edge("k6_rag", END)
        builder.add_edge("direct_handler", END)
        
        return builder.compile()
        
    def route_decision(self, state: GraphState) -> str:
        """Determines which node to execute based on the classified route in state."""
        return state.get("route", "direct")
        
    def router_node(self, state: GraphState) -> dict:
        """Classifies the user intent using ChatGroq and updates route and reason."""
        messages = state.get("messages", [])
        if not messages:
            return {
                "route": "direct",
                "reason": "No input messages detected."
            }
            
        system_prompt = (
            "You are the Router/Orchestrator for an AI-driven performance testing platform.\n"
            "Your job is to analyze the user's latest query along with the conversation history and classify the intent into one of four routes:\n"
            "1. 'runner_agent': If the user wants to write, generate, modify, execute, stream, monitor, or stop a k6 load test script.\n"
            "2. 'db_agent': If the user wants to query test runs, view database tables/metrics, inspect completed tests, compare run results, or generate PDF/summary reports of runs.\n"
            "3. 'rag_agent': If the user has a technical or syntax question about k6 documentation, APIs, options, best practices, or custom configuration (e.g. how to write checks or configure threshold metrics).\n"
            "4. 'direct': If the user's query is a general greeting (e.g. hello, hi, how are you), a request for help/guidance on using the platform, or general chit-chat.\n\n"
            "Return a JSON object exactly matching this schema:\n"
            "{\n"
            "  \"route\": \"runner_agent\" | \"db_agent\" | \"rag_agent\" | \"direct\",\n"
            "  \"reason\": \"A brief 1-sentence reason explaining why you routed the query this way.\"\n"
            "}\n\n"
            "Output ONLY valid raw JSON. Do not include markdown code block formatting (like ```json)."
        )
        
        prompt_messages = [SystemMessage(content=system_prompt)]
        
        context_window = messages[-6:] if len(messages) > 6 else messages
        for msg in context_window:
            prompt_messages.append(msg)
            
        try:
            response = self.llm.invoke(prompt_messages)
            content = response.content.strip()
            
            # Sanitize response content
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            data = json.loads(content)
            
            if data.get("route") not in ["runner_agent", "db_agent", "rag_agent", "direct"]:
                data["route"] = "direct"
                data["reason"] = "Classification returned invalid route. Defaulting to orchestrator direct."
            return data
        except Exception as e:
            return {
                "route": "direct",
                "reason": f"Routing failed with exception: {e}. Falling back to orchestrator direct."
            }
            
    def runner_node(self, state: GraphState) -> dict:
        """Invokes the RunnerAgent to compile or execute k6 scripts."""
        messages = state["messages"]
        res = self.runner_agent.invoke({"messages": messages})
        return {
            "messages": res.get("messages", []),
            "intermediate_steps": res.get("intermediate_steps", [])
        }
        
    def db_analyst_node(self, state: GraphState) -> dict:
        """Invokes the DBAnalystAgent and extracts any generated PDF path."""
        messages = state["messages"]
        res = self.db_agent.invoke({"messages": messages})
        
        pdf_path = ""
        intermediate_steps = res.get("intermediate_steps", [])
        for action, observation in intermediate_steps:
            tool_name = getattr(action, "tool", None)
            if not tool_name and isinstance(action, dict):
                tool_name = action.get("tool")
                
            if tool_name == "generate_pdf_report":
                obs_str = str(observation)
                if "at: " in obs_str:
                    pdf_path = obs_str.split("at: ")[1].strip()
                    
        return {
            "messages": res.get("messages", []),
            "intermediate_steps": intermediate_steps,
            "pdf_path": pdf_path
        }
        
    def k6_rag_node(self, state: GraphState) -> dict:
        """Invokes the K6ExpertAgent using defensive event loop handling."""
        messages = state["messages"]
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(self.rag_agent.invoke({"messages": messages})))
                    res = future.result()
            else:
                res = loop.run_until_complete(self.rag_agent.invoke({"messages": messages}))
        except RuntimeError:
            res = asyncio.run(self.rag_agent.invoke({"messages": messages}))
            
        res_msg = res.get("messages", [])[0] if res.get("messages") else None
        sources = []
        if res_msg and hasattr(res_msg, "additional_kwargs") and res_msg.additional_kwargs:
            sources = res_msg.additional_kwargs.get("sources", [])
            
        return {
            "messages": res.get("messages", []),
            "sources": sources
        }
        
    def direct_handler_node(self, state: GraphState) -> dict:
        """Fallback node to reply directly to general inquiries or chit-chat."""
        messages = state["messages"]
        direct_system = (
            "You are the Orchestrator for the AI performance testing platform.\n"
            "You are talking to the user. Explain system capabilities if asked. "
            "The system has three specialized sub-agents:\n"
            "1. Database Analyst Agent (db_agent): Can query test summaries, run customized SQL queries on metrics, and generate PDF summary reports.\n"
            "2. Runner Agent (runner_agent): Can generate k6 load testing scripts, execute load tests in real-time, stream performance logs, and stop active executions.\n"
            "3. K6 Expert Agent (rag_agent): Can answer technical questions about k6 scripting, options, API, best practices, and documentation queries using a semantic search engine.\n\n"
            "Introduce yourself briefly and orient the user. Be helpful, professional, and friendly."
        )
        chat_messages = [SystemMessage(content=direct_system)] + messages
        response = self.llm.invoke(chat_messages)
        
        return {
            "messages": [AIMessage(content=response.content, name="Orchestrator")],
            "intermediate_steps": []
        }
        
    def invoke(self, state: dict) -> dict:
        """
        Processes conversation state and invokes sub-agents or provides direct reply.
        Expects state: {"messages": [list of HumanMessage/AIMessage]}
        """
        messages = state.get("messages", [])
        if not messages:
            return {
                "messages": [],
                "route": "direct",
                "reason": "No input messages detected."
            }
            
        initial_state = {
            "messages": messages,
            "route": "direct",
            "reason": "",
            "intermediate_steps": [],
            "pdf_path": "",
            "sources": []
        }
        
        output_state = self.workflow.invoke(initial_state)
        
        return {
            "messages": [output_state["messages"][-1]], # The latest AI message
            "intermediate_steps": output_state.get("intermediate_steps", []),
            "route": output_state.get("route", "direct"),
            "reason": output_state.get("reason", "")
        }
