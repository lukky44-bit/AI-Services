from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage

from .database import DatabaseManager
from .tools import DBTools
from .prompts import DBAnalystPrompts
from .config import Config

class DBAnalystAgent:
    """Orchestrates the LLM, Tools, and Prompts for the Database Analyst."""
    
    def __init__(self):
        # Initialize components
        self.db_manager = DatabaseManager()
        self.tools_handler = DBTools(self.db_manager)
        self.tools = self.tools_handler.get_tools()
        self.prompt = DBAnalystPrompts.get_prompt_template()
        
        # Initialize LLM
        self.llm = ChatGroq(model=Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE)
        
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
                    name="Database_Analyst"
                )
            ]
        }

# Create a global instance to avoid re-initializing the LLM and connections
_global_agent = DBAnalystAgent()

def db_agent_node(state: dict) -> dict:
    """
    This function acts as the node in a LangGraph Supervisor setup.
    It takes the current state, runs the DB agent, and appends the summarized response.
    
    Expected State Schema: {"messages": [list of messages]}
    """
    return _global_agent.invoke(state)
