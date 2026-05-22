import streamlit as st
import os
import sys

# Ensure the current directory is in sys.path so db_analyst_agent can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, AIMessage
from db_analyst_agent.agent import DBAnalystAgent

# Set up page config
st.set_page_config(page_title="DB Analyst Agent", layout="wide")

st.title("Database Analyst Agent")

# Initialize session state for chat history and agent
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = DBAnalystAgent()

# Sidebar for downloads
st.sidebar.title("Generated Reports")
if os.path.exists("reports"):
    reports = [f for f in os.listdir("reports") if f.endswith(".pdf")]
    if reports:
        for report in reports:
            file_path = os.path.join("reports", report)
            with open(file_path, "rb") as f:
                st.sidebar.download_button(
                    label=f"Download {report}",
                    data=f,
                    file_name=report,
                    mime="application/pdf"
                )
    else:
        st.sidebar.info("No reports generated yet.")
else:
    st.sidebar.info("No reports generated yet.")

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# React to user input
if prompt := st.chat_input("Ask a query (e.g. generate a summarised pdf for this test with id ...)"):
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare message history for the agent to maintain conversation context
    langchain_messages = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            langchain_messages.append(HumanMessage(content=m["content"]))
        else:
            langchain_messages.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                state = {"messages": langchain_messages}
                response = st.session_state.agent.invoke(state)
                ans = response["messages"][0].content
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
    st.rerun()
