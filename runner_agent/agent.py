import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationSummaryBufferMemory

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
        
        # Initialize Memory
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1500,
            memory_key="messages",
            return_messages=True
        )

    def _sync_memory(self, messages: list):
        """Syncs the ConversationSummaryBufferMemory with incoming UI messages."""
        self.memory.clear()
        i = 0
        while i < len(messages) - 1:
            msg = messages[i]
            if isinstance(msg, HumanMessage) and i + 1 < len(messages):
                next_msg = messages[i+1]
                if isinstance(next_msg, AIMessage):
                    self.memory.save_context(
                        {"input": msg.content},
                        {"output": next_msg.content}
                    )
                    i += 2
                    continue
            i += 1

    def invoke(self, state: dict) -> dict:
        """Processes the state and returns the response for the orchestrator."""
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}
            
        # 1. Sync memory with UI chat history (except the new query)
        self._sync_memory(messages[:-1])
        
        # 2. Load the summarized/buffered history
        memory_vars = self.memory.load_memory_variables({})
        history = memory_vars.get("messages", [])
        
        # 3. Add the latest user query
        latest_query = messages[-1]
        history.append(latest_query)
        
        # 4. Invoke AgentExecutor with pruned history
        result = self.executor.invoke({"messages": history})
        
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
