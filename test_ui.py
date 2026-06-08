import streamlit as st
import requests
import json
import uuid

# Set up page config
st.set_page_config(page_title="Orchestrator API Test UI", layout="wide", page_icon="🔮")

st.title("🔮 Orchestrator API Test UI")
st.write("A client UI to verify the FastAPI Orchestrator endpoint running on `http://localhost:8000`.")

# Endpoint configuration
API_URL = st.sidebar.text_input("FastAPI Server URL", value="http://localhost:8000")

# Initialize session state keys
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_script" not in st.session_state:
    st.session_state.pending_script = None

if "active_run_id" not in st.session_state:
    st.session_state.active_run_id = str(uuid.uuid4())

# Sidebar controls
if st.sidebar.button("🗑️ Clear History"):
    st.session_state.chat_history = []
    st.session_state.pending_script = None
    st.session_state.active_run_id = str(uuid.uuid4())
    st.rerun()

# Interface Columns
col_chat, col_runner = st.columns([1.8, 1.2])

with col_chat:
    st.subheader("💬 Orchestrator Chat")
    
    # Render chat messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("route"):
                st.caption(f"**Route:** `{msg['route']}` | **Reason:** *{msg['reason']}*")
            if msg.get("pdf_url"):
                pdf_download_url = f"{API_URL.rstrip('/')}{msg['pdf_url']}"
                st.markdown(f"[📥 **Download Summary Report PDF**]({pdf_download_url})")
            if msg.get("sources"):
                st.markdown("**References & Sources:**")
                for s in msg["sources"]:
                    st.markdown(f"- [{s.split('/')[-1]}]({s})")

    # Chat Input Box
    user_query = st.chat_input("Enter your request here...")

    if user_query:
        # Add to local history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.spinner("Connecting to Orchestrator API..."):
            try:
                # Format conversation history
                api_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history[:-1]]
                response = requests.post(
                    f"{API_URL.rstrip('/')}/api/query",
                    json={"message": user_query, "history": api_history},
                    timeout=60
                )
                
                if response.status_code == 200:
                    res_data = response.json()
                    
                    # Capture generated k6 script
                    if res_data.get("script"):
                        st.session_state.pending_script = res_data["script"]
                        
                    # Save assistant response
                    assistant_msg = {
                        "role": "assistant",
                        "content": res_data["content"],
                        "route": res_data.get("route"),
                        "reason": res_data.get("reason"),
                        "pdf_url": res_data.get("pdf_url"),
                        "sources": res_data.get("sources")
                    }
                    st.session_state.chat_history.append(assistant_msg)
                    st.rerun()
                else:
                    st.error(f"Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

with col_runner:
    st.subheader("🏃‍♂️ k6 Execution Hub")
    
    if st.session_state.pending_script:
        st.info("k6 script detected! Adjust options and execute below:")
        st.code(st.session_state.pending_script, language="javascript")
        
        vus = st.number_input("Peak Virtual Users (VUs)", min_value=1, value=5, step=1)
        st.session_state.active_run_id = st.text_input("Run ID", value=st.session_state.active_run_id)
        run_id = st.session_state.active_run_id
        
        c1, c2 = st.columns(2)
        with c1:
            exec_btn = st.button("🚀 Execute Test", use_container_width=True, type="primary")
        with c2:
            stop_btn = st.button("⏹️ Stop Test", use_container_width=True)
            
        if exec_btn:
            st.write("---")
            st.write("**Real-time Execution Stream:**")
            log_container = st.empty()
            accumulated_logs = ""
            
            try:
                # Connect to execution endpoint with stream parsing
                response = requests.post(
                    f"{API_URL.rstrip('/')}/api/execute-test",
                    json={"script": st.session_state.pending_script, "vus": vus, "run_id": run_id},
                    stream=True,
                    timeout=3600
                )
                
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            decoded = line.decode('utf-8')
                            if decoded.startswith("data: "):
                                content = decoded[6:].strip()
                                try:
                                    status_data = json.loads(content)
                                    if status_data.get("status") == "initializing":
                                        accumulated_logs += f"[SYSTEM] Initialized run: {status_data.get('run_id')}\n"
                                    elif status_data.get("status") == "error":
                                        accumulated_logs += f"[ERROR] {status_data.get('message')}\n"
                                    else:
                                        accumulated_logs += f"{content}\n"
                                except json.JSONDecodeError:
                                    accumulated_logs += f"{content}\n"
                                
                                log_container.code(accumulated_logs, language="text")
                else:
                    st.error(f"Execution failed ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Stream interrupted: {e}")
                
        if stop_btn:
            with st.spinner("Broadcasting stop signal..."):
                try:
                    res = requests.post(f"{API_URL.rstrip('/')}/api/stop-test", json={"run_id": run_id})
                    if res.status_code == 200:
                        st.success(f"Stop response: {res.json()}")
                    else:
                        st.error(f"Failed to halt: {res.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
    else:
        st.info("No k6 script generated. Ask the chat orchestrator to write a performance test to populate the runner hub.")
