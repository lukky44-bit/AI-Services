import os
import sys
import asyncio

# Ensure parent directory of rag_agent is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(current_dir)
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

from langchain_core.messages import HumanMessage
from rag_agent.agent import K6ExpertAgent

async def test_rag():
    print("Initializing K6 RAG Agent...")
    agent = K6ExpertAgent()
    
    print("\n==========================================")
    print("TEST 1: Relevant k6 Query (Documentation Context)")
    print("==========================================")
    query_relevant = "What are the setup and teardown stages in k6?"
    print(f"Querying: '{query_relevant}'")
    
    state_relevant = {"messages": [HumanMessage(content=query_relevant)]}
    response_relevant = await agent.invoke(state_relevant)
    
    ans_relevant = response_relevant["messages"][0].content
    sources_relevant = response_relevant["messages"][0].additional_kwargs.get("sources", [])
    
    print(f"\nSources Retrieved ({len(sources_relevant)}):")
    for src in sources_relevant:
        print(f" - {src}")
        
    print(f"\nResponse:\n{ans_relevant}\n")
    
    print("==========================================")
    print("TEST 2: Irrelevant Query (Bypass External Intelligence)")
    print("==========================================")
    query_irrelevant = "How do you bake a chocolate cake in detail?"
    print(f"Querying: '{query_irrelevant}'")
    
    state_irrelevant = {"messages": [HumanMessage(content=query_irrelevant)]}
    response_irrelevant = await agent.invoke(state_irrelevant)
    
    ans_irrelevant = response_irrelevant["messages"][0].content
    sources_irrelevant = response_irrelevant["messages"][0].additional_kwargs.get("sources", [])
    
    print(f"\nSources Retrieved ({len(sources_irrelevant)}): {sources_irrelevant}")
    print(f"Response:\n{ans_irrelevant}\n")
    
    if "I could not find sufficient information" in ans_irrelevant:
        print("RESULT: SUCCESS - The LLM correctly refused to answer outside the retrieved k6 database context!")
    else:
        print("RESULT: FAILURE - The LLM answered using its general intelligence.")

if __name__ == "__main__":
    asyncio.run(test_rag())
