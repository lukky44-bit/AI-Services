import streamlit as st
import os
import sys

# Ensure parent directories are in the path to allow imports from subagents
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Set page config globally in the main app
st.set_page_config(page_title="k6 AI Test Automation Client", layout="wide")

# Import the UIs (this happens after sys.path is updated)
from runner_agent.ui import render_ui as render_runner_ui
from db_analyst_agent.ui import render_ui as render_db_ui

st.markdown("<h1 style='text-align: center; color: #2C3E50;'>k6 AI Test Automation Client</h1>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2 = st.tabs(["Runner Agent", "DB Analyst"])

with tab1:
    render_runner_ui()

with tab2:
    render_db_ui()
