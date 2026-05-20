import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage


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

def main():
    import sys
    
    # Check if there is an argument passed
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\n\033[1;34m[Agent Query]\033[0m: {query}")
        try:
            agent = DBAnalystAgent()
            state = {"messages": [HumanMessage(content=query)]}
            response = agent.invoke(state)
            ans = response["messages"][0].content
            print(f"\n\033[1;32m[Agent Response]\033[0m:\n{ans}\n")
        except Exception as e:
            print(f"\n\033[1;31m[Error]\033[0m: {e}\n", file=sys.stderr)
    else:
        # Interactive mode
        print("\033[1;36m="*60)
        print("   Database Analyst Agent - Interactive CLI Mode")
        print("="*60 + "\033[0m")
        print("Type your query and press Enter. Type 'exit' or 'quit' to end.\n")
        
        try:
            agent = DBAnalystAgent()
        except Exception as e:
            print(f"\033[1;31mFailed to initialize agent: {e}\033[0m", file=sys.stderr)
            return

        while True:
            try:
                query = input("\033[1;35muser>\033[0m ").strip()
                if not query:
                    continue
                if query.lower() in ("exit", "quit"):
                    print("\033[1;33mGoodbye!\033[0m")
                    break
                
                state = {"messages": [HumanMessage(content=query)]}
                response = agent.invoke(state)
                ans = response["messages"][0].content
                print(f"\n\033[1;32m[Agent Response]\033[0m:\n{ans}\n")
            except KeyboardInterrupt:
                print("\n\033[1;33mGoodbye!\033[0m")
                break
            except Exception as e:
                print(f"\n\033[1;31m[Error]\033[0m: {e}\n", file=sys.stderr)

if __name__ == "__main__":
    main()

