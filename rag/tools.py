from langchain_core.tools import StructuredTool
from .utils import build_context, format_sources

class RAGTools:
    """Encapsulates RAG retriever tools for the agent."""

    def __init__(self, agent_instance):
        from .retriever import K6Retriever
        self.retriever = K6Retriever()
        self.agent_instance = agent_instance

    def retrieve_k6_docs(self, query: str) -> str:
        # Use pre-computed HyDE doc from the merged rephrase+hyde call (avoids a redundant LLM call)
        hyde_doc = getattr(self.agent_instance, 'hyde_doc', '')
        chunks = self.retriever.retrieve(query, hyde_doc=hyde_doc)
        context = build_context(chunks)

        # Save sources to the agent instance for return in the final response
        self.agent_instance.session_sources = format_sources(chunks)

        return context

    def get_tools(self) -> list[StructuredTool]:
        """Returns the list of tools configured for LangChain."""
        return [
            StructuredTool.from_function(
                func=self.retrieve_k6_docs,
                name="retrieve_k6_docs",
                description="Search and retrieve official k6 documentation, functions, APIs, configuration settings, and best practices."
            )
        ]
