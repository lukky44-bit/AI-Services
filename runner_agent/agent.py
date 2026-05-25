import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage

try:
    from .tools import RunnerTools
    from .prompts import RunnerAgentPrompts
except ImportError:
    from tools import RunnerTools
    from prompts import RunnerAgentPrompts

# Load environment variables
load_dotenv()

class RunnerAgent:
    """Orchestrates the LLM, Tools, and Prompts for the Runner Agent."""
    
    def __init__(self):
        # Initialize components
        self.tools_handler = RunnerTools()
        self.tools = self.tools_handler.get_tools()
        self.prompt = RunnerAgentPrompts.get_prompt_template()
        
        # Initialize LLM
        model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        self.llm = ChatGroq(model=model_name, temperature=temperature)
        
        # Create Agent and Executor
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True, 
            return_intermediate_steps=True
        )

    def invoke(self, state: dict) -> dict:
        """Processes the state and returns the response for the orchestrator."""
        result = self.executor.invoke({"messages": state["messages"]})
        
        return {
            "messages": [
                AIMessage(
                    content=result["output"], 
                    name="Runner_Agent"
                )
            ],
            "intermediate_steps": result.get("intermediate_steps", [])
        }

# Create a global instance to avoid re-initializing the LLM and connections
_global_agent = RunnerAgent()

def runner_agent_node(state: dict) -> dict:
    """
    This function acts as the node in a LangGraph Supervisor setup.
    It takes the current state, runs the Runner agent, and appends the summarized response.
    
    Expected State Schema: {"messages": [list of messages]}
    """
    return _global_agent.invoke(state)
