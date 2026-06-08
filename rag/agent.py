import re
from typing import Dict, List
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationSummaryBufferMemory

from .config import Config
from .prompts import K6ExpertPrompts
from .tools import RAGTools


class K6ExpertAgent:
    """Orchestrates the LLM and RAG for k6 performance testing expertise using tool-calling agent executor."""
    
    def __init__(self):
        # Initialize LLMs
        self.llm = ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE
        )
        
        # Initialize tools
        self.tools_handler = RAGTools(self)
        self.tools = self.tools_handler.get_tools()
        
        # Initialize prompt templates
        self.prompt = K6ExpertPrompts.get_prompt_template()
        self.rephrase_hyde_prompt = K6ExpertPrompts.get_rephrase_hyde_prompt()
        
        # Create Tool-Calling Agent and Executor
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=False,
            return_intermediate_steps=True,
            max_iterations=5,
        )
        
        # Initialize Memory (Single Memory instance, no session ID management)
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1500,
            memory_key="messages",
            return_messages=True
        )
        
        # In-memory store to save formatted sources from retriever tool execution
        self.session_sources: List[dict] = []
        # Pre-computed HyDE doc from the merged rephrase+hyde call
        self.hyde_doc: str = ""

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

    async def _rephrase_and_hyde(self, history: list, question: str) -> tuple[str, str]:
        """Single LLM call that rephrases the query AND generates a HyDE hypothetical doc."""
        # Limit history to the latest 2 conversation turns (up to last 4 messages)
        recent_history = history[-4:] if history else []
        
        # Format chat history
        formatted_history = ""
        if recent_history:
            for msg in recent_history:
                if hasattr(msg, "type"):
                    role_label = "User" if msg.type == "human" else "Assistant" if msg.type == "ai" else "System"
                else:
                    role_label = "System"
                formatted_history += f"{role_label}: {msg.content}\n"
        else:
            formatted_history = "(no prior conversation)"
            
        chain = self.rephrase_hyde_prompt | self.llm
        # print("chat history", formatted_history)
        response = await chain.ainvoke({
            "chat_history": formatted_history,
            "question": question
        })
        
        # Parse the structured response
        text = response.content
        
        # Robustly strip any thinking blocks (like <think>...</think>) from reasoning models
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        
        rephrased = question  # fallback
        hyde_doc = ""
        
        if "REPHRASED:" in text and "HYDE:" in text:
            parts = text.split("HYDE:", 1)
            rephrased = parts[0].replace("REPHRASED:", "").strip()
            hyde_doc = parts[1].strip()
        elif "REPHRASED:" in text:
            rephrased = text.split("REPHRASED:", 1)[1].strip()
        # print("Rephrased query", rephrased )
        return rephrased, hyde_doc

    async def invoke(self, state: dict) -> dict:
        """Processes the state, manages memory, and returns the response."""
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}

        # Reset session sources and hyde doc
        self.session_sources = []
        self.hyde_doc = ""
        
        # 1. Sync memory with UI chat history (except the new query)
        self._sync_memory(messages[:-1])
        
        # 2. Load the summarized/buffered history
        memory_vars = self.memory.load_memory_variables({})
        history = memory_vars.get("messages", [])
        
        # 3. Single merged call: rephrase query + generate HyDE doc
        latest_query = messages[-1]
        search_query, self.hyde_doc = await self._rephrase_and_hyde(history, latest_query.content)
        
        # 4. Construct pruned history with the rephrased query
        pruned_history = list(history)
        pruned_history.append(HumanMessage(content=search_query))
        
        # 5. Execute the agent using AgentExecutor with pruned history
        result = await self.executor.ainvoke({"messages": pruned_history})
        
        output_content = result["output"]
        
        # Extract just URLs for deduplication as unique sources list of string URLs
        unique_sources = list(dict.fromkeys(src["url"] for src in self.session_sources))
        
        return {
            "messages": [
                AIMessage(
                    content=output_content,
                    name="K6_Expert",
                    additional_kwargs={"sources": unique_sources}
                )
            ]
        }


# Global instance for use in LangGraph
_global_agent = K6ExpertAgent()


async def k6_expert_node(state: dict) -> dict:
    """LangGraph node for the K6 Expert agent."""
    return await _global_agent.invoke(state)
