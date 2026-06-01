import os
import sys
import json
import logging
import asyncio
import warnings

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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from db_analyst_agent.agent import DBAnalystAgent
from runner_agent.agent import RunnerAgent
from rag_agent.agent import K6ExpertAgent

load_dotenv()

class OrchestratorAgent:
    """
    Core orchestrator agent that receives conversation history, analyzes intent,
    routes to DB Analyst, Runner, or K6 Expert (RAG) Agent, or handles direct conversation.
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
        
    def classify_intent(self, messages: list) -> dict:
        """
        Examines the conversation logs to identify whether the request relates to:
        - db_agent (Database Analyst)
        - runner_agent (Test Runner)
        - rag_agent (K6 Expert RAG)
        - direct (Greetings / General assistance)
        """
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
        
        # Prepare list of messages for routing context
        prompt_messages = [SystemMessage(content=system_prompt)]
        
        # Keep routing context concise by appending up to the last 6 messages
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
            
        # 1. Route classification
        classification = self.classify_intent(messages)
        route = classification["route"]
        reason = classification["reason"]
        
        # 2. Delegate execution based on route
        if route == "db_agent":
            # Invoke DB agent
            res = self.db_agent.invoke({"messages": messages})
            res["route"] = "db_agent"
            res["reason"] = reason
            return res
            
        elif route == "runner_agent":
            # Invoke Runner agent
            res = self.runner_agent.invoke({"messages": messages})
            res["route"] = "runner_agent"
            res["reason"] = reason
            return res
            
        elif route == "rag_agent":
            # Invoke RAG agent asynchronously in a new event loop
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                res = loop.run_until_complete(self.rag_agent.invoke({"messages": messages}))
                loop.close()
            except Exception:
                res = asyncio.run(self.rag_agent.invoke({"messages": messages}))
                
            res["route"] = "rag_agent"
            res["reason"] = reason
            return res
            
        else:
            # Handle directly
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
                "intermediate_steps": [],
                "route": "direct",
                "reason": reason
            }
