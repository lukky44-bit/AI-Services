import os
import sys
import uuid
import requests
import threading
import time
import base64
import logging
import warnings
import streamlit as st

# Suppress noisy warnings and INFO logs
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# Ensure parent and root directories are in sys.path to resolve subagent modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from langchain_core.messages import HumanMessage, AIMessage
from orchestrator.agent import OrchestratorAgent
from runner_agent.tools import RunnerTools, extract_vus_from_script


def format_test_metrics(metrics, run_id):
    """
    Formats the raw k6 execution metrics fetched from the database
    into a stunning, premium markdown dashboard.
    """
    is_new_format = isinstance(metrics, dict) and "metrics" in metrics
    
    if not is_new_format:
        table_md = "| Metric | Result |\n|---|---|\n"
        for key, val in metrics.items():
            human_key = key.replace("_", " ").title()
            clean_value = " ".join(str(val).split())
            table_md += f"| **{human_key}** | {clean_value} |\n"
        return f"**Test Execution Completed!** 🎉\n\n### Summary Metrics for run: `{run_id}`\n\n{table_md}"
        
    raw_metrics = metrics["metrics"]
    state = metrics.get("state", {})
    root_group = metrics.get("root_group", {})
    
    # Extract KPIs
    vus_max_val = "-"
    if "vus_max" in raw_metrics:
        vus_max_val = raw_metrics["vus_max"].get("values", {}).get("max", "-")
    elif "vus" in raw_metrics:
        vus_max_val = raw_metrics["vus"].get("values", {}).get("max", "-")
        
    total_reqs = "-"
    reqs_rate = "-"
    if "http_reqs" in raw_metrics:
        reqs_vals = raw_metrics["http_reqs"].get("values", {})
        total_reqs = f"{reqs_vals.get('count', '-'):,}" if isinstance(reqs_vals.get('count'), (int, float)) else str(reqs_vals.get('count', '-'))
        rate_val = reqs_vals.get('rate')
        reqs_rate = f"{rate_val:.2f}" if isinstance(rate_val, (int, float)) else str(rate_val)
        
    fail_rate_pct = "-"
    fails_count = 0
    if "http_req_failed" in raw_metrics:
        fail_vals = raw_metrics["http_req_failed"].get("values", {})
        rate_val = fail_vals.get("rate")
        fail_rate_pct = f"{rate_val * 100:.2f}%" if isinstance(rate_val, (int, float)) else str(rate_val)
        fails_count = fail_vals.get("passes", 0)
        
    avg_duration = "-"
    if "http_req_duration" in raw_metrics:
        duration_vals = raw_metrics["http_req_duration"].get("values", {})
        avg_val = duration_vals.get("avg")
        avg_duration = f"{avg_val:.2f} ms" if isinstance(avg_val, (int, float)) else str(avg_val)

    # Format Response Times Table (Trends)
    trend_keys = ["http_req_duration", "http_req_waiting", "http_req_connecting", "iteration_duration"]
    trend_rows = []
    for tk in trend_keys:
        if tk in raw_metrics:
            vals = raw_metrics[tk].get("values", {})
            contains_time = raw_metrics[tk].get("contains") == "time"
            suffix = " ms" if contains_time else ""
            
            def fmt(v):
                if v is None or v == "-": return "-"
                return f"{v:.2f}{suffix}" if isinstance(v, (int, float)) else f"{v}{suffix}"
                
            trend_rows.append(
                f"| **{tk}** | {fmt(vals.get('avg'))} | {fmt(vals.get('p(90)'))} | {fmt(vals.get('p(95)'))} | {fmt(vals.get('med'))} | {fmt(vals.get('min'))} | {fmt(vals.get('max'))} |"
            )
            
    trend_table = ""
    if trend_rows:
        trend_table = (
            "| Metric | Average | p(90) | p(95) | Median | Min | Max |\n"
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
            + "\n".join(trend_rows)
        )
        
    # Format Throughput
    counter_keys = ["data_received", "data_sent", "http_reqs", "iterations"]
    counter_rows = []
    for ck in counter_keys:
        if ck in raw_metrics:
            vals = raw_metrics[ck].get("values", {})
            contains_data = raw_metrics[ck].get("contains") == "data"
            count = vals.get("count", "-")
            rate = vals.get("rate", "-")
            
            if contains_data and isinstance(count, (int, float)):
                if count >= 1024 * 1024:
                    count_str = f"{count / (1024 * 1024):.2f} MB"
                elif count >= 1024:
                    count_str = f"{count / 1024:.2f} kB"
                else:
                    count_str = f"{count} B"
            else:
                count_str = f"{count:,}" if isinstance(count, (int, float)) else str(count)
                
            if contains_data and isinstance(rate, (int, float)):
                if rate >= 1024 * 1024:
                    rate_str = f"{rate / (1024 * 1024):.2f} MB/s"
                elif rate >= 1024:
                    rate_str = f"{rate / 1024:.2f} kB/s"
                else:
                    rate_str = f"{rate:.2f} B/s"
            else:
                rate_str = f"{rate:.2f}/s" if isinstance(rate, (int, float)) else str(rate)
                
            counter_rows.append(f"*   **{ck.replace('_', ' ').title()}:** {count_str} ({rate_str})")
            
    throughput_section = "\n".join(counter_rows) if counter_rows else "No throughput metrics recorded."

    # Format Checks
    checks_rows = []
    if root_group:
        def process_group(group, prefix=""):
            for check in group.get("checks", []):
                name = check.get("name", "check")
                passes = check.get("passes", 0)
                fails = check.get("fails", 0)
                total = passes + fails
                pct = (passes / total * 100) if total > 0 else 0.0
                status_emoji = "✅" if fails == 0 else "⚠️"
                checks_rows.append(f"*   {status_emoji} **{prefix}{name}:** `{pct:.2f}% passed` ({passes} passed, {fails} failed)")
            for sub_group in group.get("groups", []):
                sub_name = sub_group.get("name", "")
                process_group(sub_group, prefix=f"{prefix}{sub_name} > ")
                
        process_group(root_group)
            
    checks_section = "\n".join(checks_rows) if checks_rows else "*None defined in test*"

    thresholds_rows = []
    for metric_name, metric_data in raw_metrics.items():
        if isinstance(metric_data, dict) and "thresholds" in metric_data:
            for cond, status_dict in metric_data["thresholds"].items():
                ok = status_dict.get("ok", True)
                status_str = "✅ Pass" if ok else "❌ Fail"
                thresholds_rows.append(f"*   **{metric_name}** ({cond}): `{status_str}`")
                
    thresholds_section = ""
    if thresholds_rows:
        thresholds_section = "\n\n#### **🛡️ Thresholds**\n" + "\n".join(thresholds_rows)

    duration_sec = state.get("testRunDurationMs", 0) / 1000.0 if "testRunDurationMs" in state else 0.0
    duration_str = f"{duration_sec:.1f}s" if duration_sec > 0 else "Unknown"

    summary_md = f"""### **Test Execution Completed!** 🎉
**Run ID:** `{run_id}` | **Duration:** `{duration_str}`

#### **📊 Key Performance Indicators (KPIs)**
| Peak Users | Total Requests | Error Rate | Avg Response Time |
| :---: | :---: | :---: | :---: |
| **{vus_max_val} VUs** | **{total_reqs}** ({reqs_rate}/s) | **{fail_rate_pct}** ({fails_count} failed) | **{avg_duration}** |

---

#### **⏱️ Response Time Distribution (Trends)**
{trend_table}

---

#### **📥 Throughput & Data Transfer**
{throughput_section}

---

#### **✅ Test Checks**
{checks_section}{thresholds_section}
"""
    return summary_md


def check_db_connection():
    """Checks the live connectivity to the system metrics database."""
    try:
        from db_analyst_agent.database import DatabaseManager
        db = DatabaseManager()
        conn = db.get_connection()
        conn.close()
        return True
    except Exception:
        return False


def format_js(code):
    """Prettifies uploaded Javascript or k6 code blocks."""
    import subprocess
    try:
        res = subprocess.run(["npx", "--yes", "prettier", "--stdin-filepath", "script.js"], input=code.encode(), capture_output=True, timeout=5)
        if res.returncode == 0:
            return res.stdout.decode().strip()
    except Exception:
        pass
    
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


def stream_test_execution(script, vus, run_id):
    """
    Triggered in a daemon thread to send the script to the runner execution server,
    streams logs in real-time, and fetches resulting metrics from postgres database.
    """
    try:
        vus_val = int(vus) if vus else extract_vus_from_script(script)
    except Exception:
        vus_val = extract_vus_from_script(script)
        
    payload = {"vus": vus_val, "script": script, "run_id": run_id}
    endpoint = os.getenv("RUNNER_TRIGGER_URL", "http://localhost:8081/run-test").strip()
    
    try:
        accumulated_logs = []
        res = requests.post(endpoint, json=payload, stream=True, timeout=3600)
        for line in res.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith("data: "):
                    content = decoded[6:].strip()
                    st.session_state.runner_test_status = content
                    accumulated_logs.append(content)
                    
                    if "Workflow status: Completed" in content:
                        st.session_state.runner_test_running = False
                        
                        # Query postgres test_summaries table for the result
                        try:
                            from db_analyst_agent.database import DatabaseManager
                            import json
                            db = DatabaseManager()
                            conn = db.get_connection()
                            cursor = conn.cursor()
                            cursor.execute("SELECT metrics FROM test_summaries WHERE run_id = %s;", (run_id,))
                            db_res = cursor.fetchone()
                            conn.close()
                            
                            if db_res and db_res[0]:
                                metrics = db_res[0]
                                if isinstance(metrics, str):
                                    metrics = json.loads(metrics)
                                    
                                summary_md = format_test_metrics(metrics, run_id)
                                st.session_state.orchestrator_messages.append({
                                    "role": "assistant", 
                                    "content": summary_md,
                                    "route": "runner_agent",
                                    "reason": "Test finished successfully. Rerouted to parse performance metrics.",
                                    "run_id": run_id
                                })
                                st.session_state.runner_test_success = True
                            else:
                                error_lines = []
                                for log in accumulated_logs:
                                    if "level=error" in log.lower() or "[error]" in log.lower() or "exited with error" in log.lower():
                                        clean_log = log
                                        if "[stderr]" in clean_log: clean_log = clean_log.split("[stderr]")[-1].strip()
                                        if "[STDERR]" in clean_log: clean_log = clean_log.split("[STDERR]")[-1].strip()
                                        if "[stdout]" in clean_log: clean_log = clean_log.split("[stdout]")[-1].strip()
                                        error_lines.append(clean_log)
                                        
                                error_msg_str = "\n".join(error_lines) if error_lines else "Unknown execution or database write error occurred."
                                st.session_state.runner_test_status = error_msg_str
                                st.session_state.runner_test_success = False
                        except Exception as db_e:
                            st.session_state.runner_test_status = f"Database Error: {db_e}"
                            st.session_state.runner_test_success = False
                        break
    except Exception as e:
        st.session_state.runner_test_status = f"Error: {e}"
        st.session_state.runner_test_running = False


# --- STREAMLIT PAGE SETUP ---
st.set_page_config(page_title="AI Driven Performance Testing", layout="wide", page_icon="🔮")

# Inject Custom High-Fidelity CSS styling
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
        
        /* Set base background theme */
        .stApp {
            background-color: #0b0f19 !important;
            color: #f8fafc !important;
            font-family: 'Inter', sans-serif;
        }
        
        /* Modern Header Banner */
        .header-container {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #1e1b4b 0%, #311042 100%);
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            border: 1px solid #4c1d95;
        }
        
        /* Custom Routing Badges */
        .routing-badge-db {
            background: linear-gradient(135deg, #1e3a8a 0%, #581c87 100%);
            border: 1px solid #7c3aed;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 12px;
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.25);
            font-family: 'Outfit', sans-serif;
            color: #ffffff;
        }
        
        .routing-badge-runner {
            background: linear-gradient(135deg, #064e3b 0%, #0891b2 100%);
            border: 1px solid #06b6d4;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 12px;
            box-shadow: 0 4px 12px rgba(6, 182, 212, 0.25);
            font-family: 'Outfit', sans-serif;
            color: #ffffff;
        }
        
        .routing-badge-rag {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            border: 1px solid #6366f1;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 12px;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
            font-family: 'Outfit', sans-serif;
            color: #ffffff;
        }
        
        .routing-badge-direct {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border: 1px solid #475569;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 12px;
            box-shadow: 0 4px 12px rgba(71, 85, 105, 0.2);
            font-family: 'Outfit', sans-serif;
            color: #ffffff;
        }
        
        .badge-title {
            font-weight: 700;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .badge-reason {
            font-size: 0.85rem;
            opacity: 0.9;
            margin-top: 4px;
            font-weight: 400;
        }
        
        /* Glowing Console for Log Streaming */
        .glowing-terminal {
            background-color: #020617;
            border: 1px solid #22c55e;
            box-shadow: 0 0 18px rgba(34, 197, 94, 0.3);
            border-radius: 8px;
            padding: 15px;
            font-family: 'JetBrains Mono', monospace;
            color: #22c55e;
            font-size: 0.9rem;
            line-height: 1.5;
            margin-top: 12px;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .terminal-header {
            border-bottom: 1px solid #15803d;
            padding-bottom: 6px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #86efac;
            font-weight: 600;
        }
        
        /* Streamlit Chat Element styling overrides */
        div[data-testid="stChatMessage"] {
            background-color: #111827 !important;
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }
        
        /* Text input form custom design */
        div[data-testid="stForm"] {
            border: 1px solid #1f2937 !important;
            border-radius: 12px !important;
            background-color: #0f172a !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
        }
        
        /* Custom Premium Hoverable Buttons */
        div.stButton > button {
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.25s ease !important;
        }
        
        div.stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Banner
st.markdown(
    """
    <div class="header-container">
        <h1 style="color: #f8fafc; font-family: 'Outfit', sans-serif; font-size: 2.2rem; margin: 0; font-weight: 800;">🔮 AI Driven Performance Testing</h1>
        <p style="color: #cbd5e1; font-family: 'Inter', sans-serif; font-size: 0.95rem; margin-top: 6px; margin-bottom: 0; font-weight: 300;">
            A unified interface that coordinates automated k6 execution and advanced postgres performance metrics analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- INLINE UTILITY TOOLBAR ---
col_space, col_action = st.columns([8.2, 1.8])
with col_action:
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.orchestrator_messages = []
        st.session_state.runner_pending_script = None
        st.session_state.runner_test_running = False
        st.session_state.runner_run_id = None
        st.session_state.runner_test_status = None
        st.rerun()

# --- SESSION STATE INITIALIZATION ---
if "orchestrator_messages" not in st.session_state:
    st.session_state.orchestrator_messages = []

if "orchestrator_agent" not in st.session_state:
    st.session_state.orchestrator_agent = OrchestratorAgent()

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

if "runner_waiting_for_manual_script" not in st.session_state:
    st.session_state.runner_waiting_for_manual_script = False

# --- MAIN CONVERSATION INTERFACE ---
chat_container = st.container(height=520)

with chat_container:
    # Display the unified conversation history
    for i, msg in enumerate(st.session_state.orchestrator_messages):
        with st.chat_message(msg["role"]):
            # Display Routing Badge for AI responses (if available)
            if msg["role"] == "assistant" and msg.get("route"):
                route_type = msg["route"]
                reasoning = msg.get("reason", "No classification reason provided.")
                
                if route_type == "db_agent":
                    badge_html = f"""
                    <div class="routing-badge-db">
                        <div class="badge-title">📊 Routed to Database Analyst Agent</div>
                        <div class="badge-reason">{reasoning}</div>
                    </div>
                    """
                elif route_type == "runner_agent":
                    badge_html = f"""
                    <div class="routing-badge-runner">
                        <div class="badge-title">🏃‍♂️ Routed to Load Test Runner Agent</div>
                        <div class="badge-reason">{reasoning}</div>
                    </div>
                    """
                elif route_type == "rag_agent":
                    badge_html = f"""
                    <div class="routing-badge-rag">
                        <div class="badge-title">📚 Routed to K6 Expert (RAG) Agent</div>
                        <div class="badge-reason">{reasoning}</div>
                    </div>
                    """
                else:
                    badge_html = f"""
                    <div class="routing-badge-direct">
                        <div class="badge-title">💬 Handled by Orchestrator Router</div>
                        <div class="badge-reason">{reasoning}</div>
                    </div>
                    """
                st.markdown(badge_html, unsafe_allow_html=True)
                
            # Render message markdown content
            st.markdown(msg["content"])
            
            # Render references and sources if available
            if "sources" in msg and msg["sources"]:
                st.markdown("<div style='margin-top: 12px; font-size: 0.8rem; color: #94a3b8; font-weight: 600;'>📚 References & Sources:</div>", unsafe_allow_html=True)
                sources_html = ""
                for src in msg["sources"]:
                    basename = src.split("/")[-1] if "/" in src else src
                    if not basename.strip():
                        basename = src
                    sources_html += f'<a href="{src}" target="_blank" style="text-decoration:none;"><span style="background-color: #1e293b; border: 1px solid #334155; padding: 4px 10px; border-radius: 6px; color: #38bdf8; display: inline-block; margin-right: 8px; margin-bottom: 6px; font-size: 0.75rem; font-weight: 500; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">📖 {basename} ↗</span></a>'
                st.markdown(f"<div style='margin-top: 6px;'>{sources_html}</div>", unsafe_allow_html=True)
                
            # Sub-agent specialized inline content (Grafana Dashboards & PDF Download)
            if "run_id" in msg:
                grafana_url = f"http://localhost:3000/d/k6-live-overview/k6-live-overview?orgId=1&from=now-1h&to=now&timezone=browser&var-test_run_id={msg['run_id']}&refresh=5s"
                btn_html = f"""
                <a href="{grafana_url}" target="_blank" style="text-decoration:none;">
                    <div style="background: linear-gradient(135deg, #1f4068 0%, #162447 100%); padding: 10px 18px; border-radius: 8px; color: #00fff0; display: inline-block; font-weight: bold; margin-top: 12px; border: 1px solid #00fff0; font-size: 0.85rem; box-shadow: 0 4px 10px rgba(0, 255, 240, 0.15);">
                        📊 Open Grafana Live Dashboard ↗
                    </div>
                </a>
                """
                st.markdown(btn_html, unsafe_allow_html=True)
                
            if "pdf_path" in msg and os.path.exists(msg["pdf_path"]):
                try:
                    with open(msg["pdf_path"], "rb") as f:
                        b64_data = base64.b64encode(f.read()).decode()
                        pdf_basename = os.path.basename(msg["pdf_path"])
                        dl_html = f"""
                        <a href="data:application/pdf;base64,{b64_data}" download="{pdf_basename}" style="text-decoration:none;">
                            <div style="background: linear-gradient(135deg, #ea580c 0%, #dc2626 100%); padding: 10px 18px; border-radius: 8px; color: white; display: inline-block; font-weight: bold; margin-top: 12px; font-size: 0.85rem; box-shadow: 0 4px 10px rgba(220, 38, 38, 0.3);">
                                📥 Download Summary Report (PDF)
                            </div>
                        </a>
                        """
                        st.markdown(dl_html, unsafe_allow_html=True)
                except Exception as ex:
                    st.error(f"Failed to serve PDF for download: {ex}")

    # --- INLINE INTERACTIVE SUB-AGENT flows ---
    
    # 1. Runner Agent: Generated Script waiting to run or cancel
    if st.session_state.runner_pending_script and not st.session_state.runner_test_running:
        with st.chat_message("assistant"):
            if st.session_state.get("runner_is_manual_script", False):
                st.info("Please review your custom uploaded k6 script. Pick an action below:")
            else:
                st.info("A new k6 performance test script has been generated! Pick an action below:")
                
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("🚀 Run Test", use_container_width=True, type="primary"):
                    run_id = str(uuid.uuid4())
                    st.session_state.runner_run_id = run_id
                    st.session_state.runner_test_running = True
                    
                    script = st.session_state.runner_pending_script
                    vus = st.session_state.runner_pending_vus
                    
                    # Clear pending state
                    st.session_state.runner_pending_script = None
                    st.session_state.runner_is_manual_script = False
                    st.session_state.runner_test_status = "Starting execution engine stream..."
                    
                    st.session_state.orchestrator_messages.append({
                        "role": "user", 
                        "content": f"Triggering load test run with Run ID: `{run_id}`"
                    })
                    
                    # Start streaming thread with script execution contexts
                    ctx = get_script_run_ctx()
                    thread = threading.Thread(target=stream_test_execution, args=(script, vus, run_id), daemon=True)
                    add_script_run_ctx(thread, ctx)
                    thread.start()
                    st.rerun()
                    
            with c2:
                if st.button("✏️ Upload Custom Script", use_container_width=True):
                    st.session_state.runner_pending_script = None
                    st.session_state.runner_waiting_for_manual_script = True
                    st.session_state.orchestrator_messages.append({
                        "role": "assistant",
                        "content": "Please paste your custom k6 JavaScript script into the chat form and click Send."
                    })
                    st.rerun()
                    
            with c3:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.runner_pending_script = None
                    st.session_state.runner_is_manual_script = False
                    st.session_state.orchestrator_messages.append({
                        "role": "assistant", 
                        "content": "Script execution canceled by the operator."
                    })
                    st.rerun()

    # 2. Runner Agent: Real-time Streaming console
    if st.session_state.runner_test_running:
        with st.chat_message("assistant"):
            st.warning(f"Test Execution in Progress (Run ID: `{st.session_state.runner_run_id}`)")
            
            # Show live streaming log container
            status = st.session_state.runner_test_status
            if status:
                log_html = f"""
                <div class="glowing-terminal">
                    <div class="terminal-header">
                        <span>📟 live performance metrics console</span>
                        <span style="color: #22c55e; font-weight: bold; animation: pulse 1.5s infinite;">● STREAMING</span>
                    </div>
                    {status}
                </div>
                """
                st.markdown(log_html, unsafe_allow_html=True)
                
            # Offer active stop functionality if the test runner is actively broadcasting
            if status and "Workflow status: Running" in status:
                run_id = st.session_state.runner_run_id
                grafana_url = f"http://localhost:3000/d/k6-live-overview/k6-live-overview?orgId=1&from=now-1h&to=now&timezone=browser&var-test_run_id={run_id}&refresh=5s"
                btn_html = f"""
                <a href="{grafana_url}" target="_blank" style="text-decoration:none;">
                    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 10px 18px; border-radius: 8px; color: #00fff0; display: inline-block; font-weight: bold; margin-bottom: 15px; border: 1px solid #00fff0; font-size: 0.85rem; box-shadow: 0 4px 10px rgba(0, 255, 240, 0.1);">
                        📊 View Active Grafana Dashboard ↗
                    </div>
                </a>
                """
                st.markdown(btn_html, unsafe_allow_html=True)
                
                if st.button("⏹️ Stop Active Test", use_container_width=True, type="primary"):
                    st.session_state.orchestrator_messages.append({
                        "role": "user", 
                        "content": f"Stopping active test with run ID: `{run_id}`"
                    })
                    
                    with st.spinner("Broadcasting interrupt signal to execution cluster..."):
                        try:
                            stop_endpoint = os.getenv("RUNNER_STOP_URL", "http://localhost:8081/stop").strip()
                            res = requests.post(stop_endpoint, json={"run_id": run_id}, timeout=30)
                            raw = res.json() if res.ok else res.text
                            st.session_state.orchestrator_messages.append({
                                "role": "assistant",
                                "content": f"**Stop Command Response:**\n```json\n{raw}\n```",
                                "route": "runner_agent",
                                "reason": "Operator halted load testing run manually.",
                                "run_id": run_id
                            })
                        except Exception as stop_e:
                            st.session_state.orchestrator_messages.append({
                                "role": "assistant",
                                "content": f"Failed to interrupt runner server: {stop_e}",
                                "route": "runner_agent",
                                "reason": "Stop operation raised connection exception."
                            })
                    st.session_state.runner_test_running = False
                    st.session_state.runner_run_id = None
                    st.rerun()
            
            # Polling delay & refresh loop to draw streaming lines
            time.sleep(2)
            st.rerun()

    # 3. Dismiss Completed Execution Dialog
    elif st.session_state.runner_run_id and st.session_state.runner_test_status and "Workflow status: Completed" in st.session_state.runner_test_status:
        if st.session_state.get("runner_test_success", True):
            # Silently auto-clear states on success
            st.session_state.runner_run_id = None
            st.session_state.runner_test_status = None
            st.session_state.runner_test_success = None
            st.rerun()
        else:
            with st.chat_message("assistant"):
                st.error("Test Run Execution Interrupted/Failed!")
                st.code(st.session_state.runner_test_status, language="text")
                
                if st.button("Clear Notification"):
                    st.session_state.runner_run_id = None
                    st.session_state.runner_test_status = None
                    st.session_state.runner_test_success = None
                    st.rerun()

# --- INPUT AND FORMS CONTROL ---
st.markdown("---")

with st.form("orchestrator_input_form", clear_on_submit=True):
    col1, col2 = st.columns([8.5, 1.5])
    with col1:
        query_input = st.text_input(
            "Enter message", 
            label_visibility="collapsed", 
            placeholder="Ask a query, generate a k6 load test, query DB summaries, or write scripts..."
        )
    with col2:
        send_btn = st.form_submit_button("Send", use_container_width=True)

# Handle message submission
if send_btn and query_input:
    # 1. Custom pasted script case
    if st.session_state.get("runner_waiting_for_manual_script", False):
        st.session_state.runner_waiting_for_manual_script = False
        st.session_state.orchestrator_messages.append({
            "role": "user", 
            "content": "(Uploaded Custom Script)"
        })
        
        formatted_script = format_js(query_input)
        st.session_state.orchestrator_messages.append({
            "role": "assistant",
            "content": f"**Pasted Custom k6 Script Received:**\n```javascript\n{formatted_script}\n```\n\nWould you like to execute this test run?",
            "route": "runner_agent",
            "reason": "Operator submitted script manually."
        })
        
        st.session_state.runner_pending_script = formatted_script
        st.session_state.runner_pending_vus = extract_vus_from_script(formatted_script)
        st.session_state.runner_is_manual_script = True
        st.rerun()
        
    # 2. General LLM orchestration query routing
    else:
        st.session_state.orchestrator_messages.append({
            "role": "user", 
            "content": query_input
        })
        
        # Assemble message histories to supply context
        langchain_hist = []
        for m in st.session_state.orchestrator_messages:
            if m["role"] == "user":
                langchain_hist.append(HumanMessage(content=m["content"]))
            else:
                langchain_hist.append(AIMessage(content=m["content"]))
                
        with st.spinner("AI Router is organizing task workflow..."):
            try:
                # Invoke the Orchestrator Router agent
                state = {"messages": langchain_hist}
                response = st.session_state.orchestrator_agent.invoke(state)
                
                ans = response["messages"][0].content
                route = response.get("route", "direct")
                reason = response.get("reason", "Handled directly.")
                
                # Check for generated k6 scripts
                script_generated = False
                intermediate_steps = response.get("intermediate_steps", [])
                for action, observation in intermediate_steps:
                    if action.tool == "generate_k6_script":
                        script_generated = True
                        st.session_state.runner_pending_script = observation
                        st.session_state.runner_is_manual_script = False
                        st.session_state.runner_pending_vus = extract_vus_from_script(observation)
                
                # Check for generated DB summary PDF reports
                pdf_path = None
                for action, observation in intermediate_steps:
                    if action.tool == "generate_pdf_report":
                        if "at: " in str(observation):
                            pdf_path = str(observation).split("at: ")[1].strip()
                            
                # Check for sources in additional_kwargs
                sources = None
                first_msg = response["messages"][0]
                if hasattr(first_msg, "additional_kwargs") and first_msg.additional_kwargs and "sources" in first_msg.additional_kwargs:
                    sources = first_msg.additional_kwargs["sources"]
                    
                # Package response
                msg_payload = {
                    "role": "assistant",
                    "content": ans,
                    "route": route,
                    "reason": reason
                }
                
                if script_generated:
                    msg_payload["content"] = f"### Generated k6 Test Script:\n```javascript\n{st.session_state.runner_pending_script}\n```"
                    
                if pdf_path:
                    msg_payload["pdf_path"] = pdf_path
                    
                if sources:
                    msg_payload["sources"] = sources
                    
                st.session_state.orchestrator_messages.append(msg_payload)
            except Exception as router_err:
                st.session_state.orchestrator_messages.append({
                    "role": "assistant",
                    "content": f"Orchestration engine encountered a failure: {router_err}",
                    "route": "direct",
                    "reason": "Router execution raised exception."
                })
        st.rerun()
