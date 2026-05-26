import streamlit as st
import os
import sys
import uuid
import requests
import threading
import time
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# Ensure the parent directory is in sys.path so we can run this from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from langchain_core.messages import HumanMessage, AIMessage
from runner_agent.agent import RunnerAgent
from runner_agent.tools import RunnerTools

# Set up page config
def render_ui():
    st.title("Runner Agent")

    # Initialize session state
    if "runner_messages" not in st.session_state:
        st.session_state.runner_messages = []

    if "runner_agent" not in st.session_state:
        st.session_state.runner_agent = RunnerAgent()

    if "runner_tools" not in st.session_state:
        st.session_state.runner_tools = RunnerTools()

    if "runner_pending_script" not in st.session_state:
        st.session_state.runner_pending_script = None

    if "runner_pending_vus" not in st.session_state:
        st.session_state.runner_pending_vus = None

    if "runner_test_running" not in st.session_state:
        st.session_state.runner_test_running = False

    if "runner_run_id" not in st.session_state:
        st.session_state.runner_run_id = None

    if "runner_test_status" not in st.session_state:
        st.session_state.runner_test_status = None

    if "runner_is_manual_script" not in st.session_state:
        st.session_state.runner_is_manual_script = False

    if "runner_test_success" not in st.session_state:
        st.session_state.runner_test_success = None

    def format_js(code):
        import subprocess
        try:
            # Run prettier if npx is available
            res = subprocess.run(["npx", "--yes", "prettier", "--stdin-filepath", "script.js"], input=code.encode(), capture_output=True, timeout=5)
            if res.returncode == 0:
                return res.stdout.decode().strip()
        except Exception:
            pass
        
        # Fallback basic formatter if npx prettier is slow/missing
        code = code.replace("{", "{\n").replace("}", "\n}\n").replace(";", ";\n")
        lines = code.split("\n")
        indent = 0
        formatted = []
        for line in lines:
            line = line.strip()
            if not line: continue
            if line.startswith("}") or line.startswith("]"): 
                indent = max(0, indent - 1)
            formatted.append("  " * indent + line)
            if line.endswith("{") or line.endswith("["): 
                indent += 1
        return "\n".join(formatted)

    if "runner_waiting_for_manual_script" not in st.session_state:
        st.session_state.runner_waiting_for_manual_script = False

    def stream_test_execution(script, vus, run_id):
        try:
            vus_val = int(vus) if vus else 1
        except Exception:
            vus_val = 1
            
        payload = {"vus": vus_val, "script": script, "run_id": run_id}
        endpoint = os.getenv("RUNNER_TRIGGER_URL", "http://localhost:8081/run-test").strip()
        
        try:
            accumulated_logs = []
            res = requests.post(endpoint, json=payload, stream=True, timeout=3600)
            for line in res.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    
                    # Filter for lines starting with 'data: '
                    if decoded.startswith("data: "):
                        content = decoded[6:].strip()
                        st.session_state.runner_test_status = content
                        accumulated_logs.append(content)
                        
                        if "Workflow status: Completed" in content:
                            st.session_state.runner_test_running = False
                            
                            # Fetch test summaries from DB
                            try:
                                from db_analyst_agent.database import DatabaseManager
                                import json
                                db = DatabaseManager()
                                conn = db.get_connection()
                                cursor = conn.cursor()
                                cursor.execute("SELECT metrics FROM test_summaries WHERE run_id = %s;", (run_id,))
                                result = cursor.fetchone()
                                conn.close()
                                
                                if result and result[0]:
                                    metrics = result[0]
                                    if isinstance(metrics, str):
                                        metrics = json.loads(metrics)
                                    
                                    # Humanize the metrics into a Markdown table
                                    table_md = "| Metric | Result |\n|---|---|\n"
                                    for key, val in metrics.items():
                                        # Capitalize keys and replace underscores
                                        human_key = key.replace("_", " ").title()
                                        
                                        # Clean up multiple whitespaces in the value
                                        clean_value = " ".join(str(val).split())
                                        
                                        table_md += f"| **{human_key}** | {clean_value} |\n"
                                    
                                    # Format nicely as Markdown
                                    summary_md = f"**Test Execution Completed!** 🎉\n\n### Summary Metrics for run: `{run_id}`\n\n{table_md}"
                                    st.session_state.runner_messages.append({
                                        "role": "assistant", 
                                        "content": summary_md,
                                        "run_id": run_id
                                    })
                                    st.session_state.runner_test_success = True
                                else:
                                    # Extract errors from accumulated logs
                                    error_lines = []
                                    for log in accumulated_logs:
                                        if "level=error" in log.lower() or "[error]" in log.lower() or "exited with error" in log.lower():
                                            clean_log = log
                                            if "[stderr]" in clean_log:
                                                clean_log = clean_log.split("[stderr]")[-1].strip()
                                            if "[STDERR]" in clean_log:
                                                clean_log = clean_log.split("[STDERR]")[-1].strip()
                                            if "[stdout]" in clean_log:
                                                clean_log = clean_log.split("[stdout]")[-1].strip()
                                            error_lines.append(clean_log)
                                            
                                    error_msg_str = "\n".join(error_lines) if error_lines else "Unknown execution or database write error occurred."
                                    fail_md = f"**Test Failed** ❌\n\n**Reason:**\n```text\n{error_msg_str}\n```"
                                    st.session_state.runner_messages.append({
                                        "role": "assistant", 
                                        "content": fail_md
                                    })
                                    st.session_state.runner_test_success = False
                            except Exception as db_e:
                                fail_db_md = f"**Test Failed** ❌\n\n**Error:** {db_e}"
                                st.session_state.runner_messages.append({
                                    "role": "assistant", 
                                    "content": fail_db_md
                                })
                                st.session_state.runner_test_success = False
                            
                            break
        except Exception as e:
            st.session_state.runner_test_status = f"Error: {e}"
            st.session_state.runner_test_running = False


    # Create a scrollable container for the chat history
    chat_container = st.container(height=550)

    with chat_container:
        # Display chat messages from history on app rerun
        for i, msg in enumerate(st.session_state.runner_messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "run_id" in msg:
                    grafana_url = f"http://localhost:3000/d/k6-live-overview/k6-live-overview?orgId=1&from=now-1h&to=now&timezone=browser&var-test_run_id={msg['run_id']}&refresh=5s"
                    href = f'<br><a href="{grafana_url}" target="_blank">*View dashboard*</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
        # --- Interactive Chat Box Controls ---
        if st.session_state.runner_pending_script and not st.session_state.runner_test_running:
            with st.chat_message("assistant"):
                if st.session_state.get("runner_is_manual_script", False):
                    st.info("Please review your uploaded script. Select an action below:")
                else:
                    st.info("A new script has been generated. Select an action below:")
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("Run Test", use_container_width=True, type="primary"):
                        # Generate run_id and execute
                        run_id = str(uuid.uuid4())
                        st.session_state.runner_run_id = run_id
                        st.session_state.runner_test_running = True
                        
                        script = st.session_state.runner_pending_script
                        vus = st.session_state.runner_pending_vus
                        
                        # Clear pending script state
                        st.session_state.runner_pending_script = None
                        st.session_state.runner_is_manual_script = False
                        st.session_state.runner_test_status = "Starting test stream..."
                        
                        st.session_state.runner_messages.append({"role": "user", "content": f"Executing script with run_id: {run_id}"})
                        
                        # Start background thread for streaming
                        ctx = get_script_run_ctx()
                        thread = threading.Thread(target=stream_test_execution, args=(script, vus, run_id), daemon=True)
                        add_script_run_ctx(thread, ctx)
                        thread.start()
                        st.rerun()
                        
                with c2:
                    if st.button("Upload Script", use_container_width=True):
                        st.session_state.runner_pending_script = None
                        st.session_state.runner_waiting_for_manual_script = True
                        st.session_state.runner_messages.append({"role": "assistant", "content": "Please paste your custom k6 JavaScript code in the chat box below and hit Generate."})
                        st.rerun()
                        
                with c3:
                    if st.button("Cancel", use_container_width=True):
                        st.session_state.runner_pending_script = None
                        st.session_state.runner_is_manual_script = False
                        st.session_state.runner_messages.append({"role": "assistant", "content": "Script execution cancelled by user."})
                        st.rerun()

        if st.session_state.runner_test_running:
            with st.chat_message("assistant"):
                st.warning(f"Test is currently running (Run ID: {st.session_state.runner_run_id})")
                
                status = st.session_state.runner_test_status
                if status:
                    st.code(status, language="text")
                    
                # Conditionally show the Stop button if it's running
                if status and "Workflow status: Running" in status:
                    run_id = st.session_state.runner_run_id
                    grafana_url = f"http://localhost:3000/d/k6-live-overview/k6-live-overview?orgId=1&from=now-1h&to=now&timezone=browser&var-test_run_id={run_id}&refresh=5s"
                    href = f'<br><a href="{grafana_url}" target="_blank">*View dashboard*</a><br><br>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    if st.button("Stop Test", use_container_width=True, type="primary"):
                        run_id = st.session_state.runner_run_id
                        st.session_state.runner_messages.append({"role": "user", "content": f"Stopping test with run_id: {run_id}"})
                        
                        with st.spinner("Stopping test..."):
                            try:
                                stop_endpoint = os.getenv("RUNNER_STOP_URL", "http://localhost:8081/stop").strip()
                                res = requests.post(stop_endpoint, json={"run_id": run_id}, timeout=30)
                                raw = res.json() if res.ok else res.text
                                st.session_state.runner_messages.append({
                                    "role": "assistant", 
                                    "content": f"**Stop Response:**\n```json\n{raw}\n```",
                                    "run_id": run_id
                                })
                            except Exception as e:
                                st.session_state.runner_messages.append({
                                    "role": "assistant", 
                                    "content": f"Error stopping test: {e}",
                                    "run_id": run_id
                                })
                        
                        st.session_state.runner_test_running = False
                        st.session_state.runner_run_id = None
                        st.rerun()
                
                # Poll UI to keep checking the thread's updates
                time.sleep(2)
                st.rerun()
                
        elif st.session_state.runner_run_id and st.session_state.runner_test_status and "Workflow status: Completed" in st.session_state.runner_test_status:
            with st.chat_message("assistant"):
                if st.session_state.get("runner_test_success", True):
                    st.success("Test Completed!")
                    st.code(st.session_state.runner_test_status, language="json")
                else:
                    st.error("Test Failed")
                    st.code(st.session_state.runner_test_status, language="text")
                if st.button("Dismiss"):
                    st.session_state.runner_run_id = None
                    st.session_state.runner_test_status = None
                    st.session_state.runner_test_success = None
                    st.rerun()

    st.markdown("---")

    # Layout for the input
    with st.form("runner_input_form", clear_on_submit=True):
        col1, col2 = st.columns([9, 1])
        with col1:
            query_input = st.text_input("Ask a query or generate a test", label_visibility="collapsed", placeholder="E.g., run a load test for https://test.k6.io for 10s with 5 vus")
        with col2:
            gen_btn = st.form_submit_button("Generate", use_container_width=True)



    if gen_btn and query_input:
        if st.session_state.get("runner_waiting_for_manual_script", False):
            st.session_state.runner_waiting_for_manual_script = False
            st.session_state.runner_messages.append({"role": "user", "content": "(Uploaded Custom Script)"})
            
            formatted_script = format_js(query_input)
            
            st.session_state.runner_messages.append({
                "role": "assistant",
                "content": f"**Custom Script Received:**\n```javascript\n{formatted_script}\n```\n\nWould you like to run it?"
            })
            
            st.session_state.runner_pending_script = formatted_script
            st.session_state.runner_pending_vus = 1  # Default fallback
            st.session_state.runner_is_manual_script = True
            st.rerun()
        else:
            # Ensure the prompt explicitly asks for generation if not already stated
            if "generate" not in query_input.lower():
                prompt = f"generate a k6 script for: {query_input}"
            else:
                prompt = query_input
                
            st.session_state.runner_messages.append({"role": "user", "content": prompt})
        
            langchain_messages = []
            for m in st.session_state.runner_messages:
                if m["role"] == "user":
                    langchain_messages.append(HumanMessage(content=m["content"]))
                else:
                    langchain_messages.append(AIMessage(content=m["content"]))
                    
            with st.spinner("Thinking..."):
                try:
                    state = {"messages": langchain_messages}
                    response = st.session_state.runner_agent.invoke(state)
                    ans = response["messages"][0].content
                    
                    # Check intermediate steps to see if a script was generated
                    script_generated = False
                    intermediate_steps = response.get("intermediate_steps", [])
                    for action, observation in intermediate_steps:
                        if action.tool == "generate_k6_script":
                            script_generated = True
                            st.session_state.runner_pending_script = observation
                            st.session_state.runner_is_manual_script = False
                            st.session_state.runner_pending_vus = action.tool_input.get("vus", None)
                            
                    if script_generated:
                        st.session_state.runner_messages.append({
                            "role": "assistant",
                            "content": f"```javascript\n{st.session_state.runner_pending_script}\n```"
                        })
                    else:
                        st.session_state.runner_messages.append({"role": "assistant", "content": ans})
                            
                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.session_state.runner_messages.append({"role": "assistant", "content": error_msg})
                    
            st.rerun()


if __name__ == "__main__":
    st.set_page_config(page_title="Runner Agent", layout="wide")
    render_ui()