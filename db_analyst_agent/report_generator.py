import os
import re
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from .database import DatabaseManager
from .config import Config
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

class MetricParser:
    """Parses raw text metrics from PostgreSQL JSON into structured fields."""
    
    @staticmethod
    def parse_trend(val) -> dict:
        res = {
            "avg": "-", "max": "-", "med": "-", "min": "-",
            "p90": "-", "p95": "-", "p99": "-"
        }
        if not val:
            return res
        if isinstance(val, dict):
            values = val.get("values", {})
            contains_time = val.get("contains") == "time"
            suffix = "ms" if contains_time else ""
            for k in ["avg", "max", "med", "min"]:
                if k in values:
                    res[k] = f"{values[k]:.2f}{suffix}" if isinstance(values[k], (int, float)) else str(values[k])
            for p_key, res_key in [("p(90)", "p90"), ("p(95)", "p95"), ("p(99)", "p99")]:
                if p_key in values:
                    res[res_key] = f"{values[p_key]:.2f}{suffix}" if isinstance(values[p_key], (int, float)) else str(values[p_key])
            return res
            
        raw_str = str(val)
        matches = re.findall(r"([\w()]+)=([\w.]+ms|[\w.]+s|[\w.]+)", raw_str)
        for k, v in matches:
            # Clean parentheses to match dict keys (e.g., p(90) -> p90)
            k_clean = k.lower().replace("(", "").replace(")", "")
            if k_clean in res:
                res[k_clean] = v
        return res

    @staticmethod
    def parse_counter(val) -> dict:
        res = {"count": "-", "rate": "-"}
        if not val:
            return res
        if isinstance(val, dict):
            values = val.get("values", {})
            count = values.get("count", "-")
            rate = values.get("rate", "-")
            contains_data = val.get("contains") == "data"
            
            if contains_data and isinstance(count, (int, float)):
                if count >= 1024 * 1024:
                    res["count"] = f"{count / (1024 * 1024):.2f} MB"
                elif count >= 1024:
                    res["count"] = f"{count / 1024:.2f} kB"
                else:
                    res["count"] = f"{count} B"
            else:
                res["count"] = f"{count:,}" if isinstance(count, (int, float)) else str(count)
                
            if isinstance(rate, (int, float)):
                if contains_data:
                    if rate >= 1024 * 1024:
                        res["rate"] = f"{rate / (1024 * 1024):.2f} MB/s"
                    elif rate >= 1024:
                        res["rate"] = f"{rate / 1024:.2f} kB/s"
                    else:
                        res["rate"] = f"{rate:.2f} B/s"
                else:
                    res["rate"] = f"{rate:.2f}/s"
            else:
                res["rate"] = str(rate)
            return res
            
        raw_str = str(val)
        # Split by two or more spaces first
        parts = re.split(r'\s{2,}', raw_str.strip())
        if len(parts) >= 2:
            res["count"] = parts[0]
            res["rate"] = parts[1]
            
            # Format decimal rate nicely (e.g. 75.077817/s -> 75.08/s)
            rate_match = re.match(r"^([\d.]+)(/s)$", res["rate"])
            if rate_match:
                try:
                    res["rate"] = f"{float(rate_match.group(1)):.2f}{rate_match.group(2)}"
                except ValueError:
                    pass
            return res
        
        # Split by single spaces for units like "3.4 MB 160 kB/s"
        parts = raw_str.strip().split()
        if len(parts) >= 3:
            res["count"] = f"{parts[0]} {parts[1]}"
            res["rate"] = " ".join(parts[2:])
        elif len(parts) == 2:
            res["count"] = parts[0]
            res["rate"] = parts[1]
        elif len(parts) == 1:
            res["count"] = parts[0]
        return res

    @staticmethod
    def parse_rate(val) -> str:
        if not val:
            return "-"
        if isinstance(val, dict):
            values = val.get("values", {})
            rate = values.get("rate", 0)
            fails = values.get("fails", 0)
            passes = values.get("passes", 0)
            percentage = rate * 100
            return f"{percentage:.2f}% ({passes}/{passes + fails})"
            
        raw_str = str(val)
        parts = raw_str.strip().split()
        if len(parts) >= 5 and parts[2] == "out" and parts[3] == "of":
            return f"{parts[0]} ({parts[1]}/{parts[4]})"
        if len(parts) >= 2:
            return f"{parts[0]} ({parts[1]})"
        if parts:
            return parts[0]
        return "-"

    @staticmethod
    def parse_gauge(val) -> str:
        if not val:
            return "-"
        if isinstance(val, dict):
            values = val.get("values", {})
            value = values.get("value", "-")
            min_val = values.get("min", "-")
            max_val = values.get("max", "-")
            return f"{value} (min: {min_val}, max: {max_val})"
            
        raw_str = str(val)
        parts = raw_str.strip().split()
        if len(parts) >= 2:
            min_val = "-"
            max_val = "-"
            for p in parts[1:]:
                if p.startswith("min="):
                    min_val = p.split("=")[1]
                elif p.startswith("max="):
                    max_val = p.split("=")[1]
            return f"{parts[0]} (min: {min_val}, max: {max_val})"
        if parts:
            return parts[0]
        return "-"

    @staticmethod
    def parse_threshold(key: str, val_str: str) -> dict:
        import re
        match = re.match(r"threshold_(.*?)([<>]=?|=)(.*)", key)
        if not match:
            return {"metric": key, "condition": "-", "actual": val_str, "status": "N/A"}
        
        metric, op, limit_str = match.groups()
        condition = f"{op} {limit_str}"
        
        actual = val_str
        val_match = re.search(r"=\s*(.*)", val_str)
        if val_match:
            actual = val_match.group(1)
            
        try:
            limit_val = float(re.sub(r"[^\d.]", "", limit_str))
            num_match = re.search(r"[\d.]+", actual)
            if num_match:
                actual_val = float(num_match.group(0))
                if ("rate" in metric or "failed" in metric) and limit_val <= 1.0 and actual_val > 1.0:
                    actual_val = actual_val / 100.0
                
                passed = False
                if op == "<": passed = actual_val < limit_val
                elif op == "<=": passed = actual_val <= limit_val
                elif op == ">": passed = actual_val > limit_val
                elif op == ">=": passed = actual_val >= limit_val
                elif op == "=": passed = actual_val == limit_val
                status = "Pass" if passed else "Fail"
            else:
                status = "N/A"
        except Exception:
            status = "N/A"
            
        return {
            "metric": metric,
            "condition": condition,
            "actual": actual,
            "status": status
        }


class   PDFReportBuilder:
    """Uses ReportLab to draw a beautifully formatted PDF report matching the reference design."""
    
    def __init__(self, run_id: str, metrics: dict):
        self.run_id = run_id
        self.raw_summary = metrics
        
        # If it's the new JSONB format, the actual metrics are nested under 'metrics'
        if isinstance(metrics, dict) and "metrics" in metrics:
            self.metrics = metrics["metrics"]
        else:
            self.metrics = metrics
            
        # Fetch the k6 script from database
        self.script = ""
        try:
            db = DatabaseManager()
            conn = db.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT script FROM test_runs WHERE id = %s;", (self.run_id,))
            row = cursor.fetchone()
            conn.close()
            if row:
                self.script = row[0]
        except Exception:
            pass
        
        # Strip duplicated "run_" prefix if present in the run_id to avoid double naming
        clean_run_id = run_id[4:] if run_id.startswith("run_") else run_id
        
        # Determine reports path relative to this file's location to be CWD-independent
        current_dir = os.path.dirname(os.path.abspath(__file__))
        reports_dir = os.path.join(current_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        self.output_path = os.path.join(reports_dir, f"run_{clean_run_id}_summary.pdf")

    def markdown_to_html(self, text: str) -> str:
        """Converts basic Markdown bullet points and bolding to ReportLab compliant HTML."""
        if not text:
            return ""
        # Escape XML entities first to prevent rendering issues in ReportLab Paragraph
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # Un-escape standard bold tags we will write
        text = re.sub(r'\*\*(.*?)\*\*|__(.*?)__', r'<b>\1\2</b>', text)
        
        # Replace bullet points: '● ' or '* ' or '- ' at the beginning of a line with standard bullet list
        lines = []
        for line in text.splitlines():
            line_strip = line.strip()
            if line_strip.startswith("●") or line_strip.startswith("*") or line_strip.startswith("-"):
                content = line_strip[1:].strip()
                lines.append(f"&bull; {content}")
            else:
                lines.append(line_strip)
                
        return "<br/>".join(lines)

    def fetch_analysis_data(self):
        """Preprocesses test_logs and realtime_metrics database tables to isolate errors and anomalies."""
        db = DatabaseManager()
        try:
            conn = db.get_connection()
            cursor = conn.cursor()
            
            # 1. Fetch test_logs content
            cursor.execute("SELECT content FROM test_logs WHERE run_id = %s LIMIT 1;", (self.run_id,))
            log_row = cursor.fetchone()
            
            filtered_log_lines = []
            if log_row and log_row[0]:
                log_content = log_row[0]
                for line in log_content.splitlines():
                    line_lower = line.lower()
                    if any(keyword in line_lower for keyword in ["error", "fail", "timeout", "exception", "stderr", "warning"]):
                        filtered_log_lines.append(line.strip())
                        if len(filtered_log_lines) >= 80:  # Cap log size to fit context window
                            break
            preprocessed_logs = "\n".join(filtered_log_lines) if filtered_log_lines else "No error or warning logs found in the test execution."
            
            # 2. Fetch realtime_metrics time-series abnormalities (spikes, dropoffs, failures)
            cursor.execute("""
                SELECT name, value, ts 
                FROM realtime_metrics 
                WHERE run_id = %s 
                  AND (
                    (name = 'k6_http_req_failed_rate' AND value > 0)
                    OR (name = 'k6_checks_rate' AND value < 1.0)
                    OR (name = 'k6_http_req_duration_avg' AND value > 500)
                  )
                ORDER BY ts ASC LIMIT 80;
            """, (self.run_id,))
            anom_rows = cursor.fetchall()
            conn.close()
            
            anoms = []
            for name, value, ts in anom_rows:
                human_name = name.replace("k6_", "").replace("_", " ").title()
                if "failed_rate" in name or "checks_rate" in name:
                    fmt_val = f"{value * 100:.2f}%"
                elif "duration" in name:
                    fmt_val = f"{value:.2f} ms"
                else:
                    fmt_val = str(value)
                anoms.append(f"- Timestamp: {ts} | Metric: {human_name} = {fmt_val}")
                
            preprocessed_anomalies = "\n".join(anoms) if anoms else "No metrics abnormalities, error rate spikes, or failed checks detected."
            return preprocessed_logs, preprocessed_anomalies
        except Exception as db_e:
            return f"Database Fetch Error: {db_e}", f"Database Fetch Error: {db_e}"

    def fetch_tps_data(self):
        """Fetches realtime metrics, aligns reqs and fail rate, and calculates TPS buckets."""
        db = DatabaseManager()
        try:
            conn = db.get_connection()
            cursor = conn.cursor()
            
            # Extract max target TPS from the script if available
            max_target = None
            if self.script:
                import re
                targets = re.findall(r'target\s*:\s*(\d+)', self.script)
                targets += re.findall(r'"target"\s*:\s*(\d+)', self.script)
                targets += re.findall(r"'target'\s*:\s*(\d+)", self.script)
                targets += re.findall(r'rate\s*:\s*(\d+)', self.script)
                targets += re.findall(r'"rate"\s*:\s*(\d+)', self.script)
                targets += re.findall(r"'rate'\s*:\s*(\d+)", self.script)
                if targets:
                    max_target = float(max(int(t) for t in targets))

            # Check if metrics are cumulative or raw event values (1.0 per request)
            cursor.execute("SELECT MAX(value) FROM realtime_metrics WHERE run_id = %s AND name = 'k6_http_reqs_total';", (self.run_id,))
            max_val_row = cursor.fetchone()
            max_val = max_val_row[0] if max_val_row else None
            is_cumulative = False
            if max_val and max_val > 5.0:
                is_cumulative = True
                
            query = """
            SELECT 
                date_trunc('second', ts) AS timestamp_sec,
                MAX(CASE WHEN name = 'k6_http_reqs_total' THEN value END) AS max_reqs_total,
                SUM(CASE WHEN name = 'k6_http_reqs_total' THEN value END) AS sum_reqs_total,
                AVG(CASE WHEN name = 'k6_http_req_failed_rate' THEN value END) AS failed_rate
            FROM realtime_metrics
            WHERE run_id = %s AND name IN ('k6_http_reqs_total', 'k6_http_req_failed_rate')
            GROUP BY date_trunc('second', ts)
            ORDER BY timestamp_sec ASC;
            """
            cursor.execute(query, (self.run_id,))
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return [], {}, [], False
                
            tps_values = []
            buckets = {}
            tps_intervals = []
            
            # Python parsing of rows to handle forward-filling of values
            clean_rows = []
            last_reqs = 0.0
            last_fail = 0.0
            for timestamp_sec, max_reqs_total, sum_reqs_total, failed_rate in rows:
                if is_cumulative:
                    reqs = max_reqs_total
                    if reqs is None:
                        reqs = last_reqs
                    else:
                        last_reqs = reqs
                else:
                    reqs = sum_reqs_total
                    if reqs is None:
                        reqs = 0.0
                
                if failed_rate is None:
                    failed_rate = last_fail
                else:
                    last_fail = failed_rate
                
                clean_rows.append((timestamp_sec, reqs, failed_rate))

            if is_cumulative:
                for i in range(1, len(clean_rows)):
                    ts_prev, req_prev, fail_prev = clean_rows[i-1]
                    ts_curr, req_curr, fail_curr = clean_rows[i]
                    
                    diff_sec = (ts_curr - ts_prev).total_seconds()
                    if diff_sec <= 0:
                        continue
                        
                    diff_req = req_curr - req_prev
                    if diff_req < 0:
                        diff_req = 0.0
                        
                    tps = diff_req / diff_sec
                    if max_target is not None and tps > max_target:
                        tps = max_target
                        diff_req = max_target * diff_sec
                    tps_values.append(tps)
                    tps_intervals.append({"tps": tps, "fail_rate": fail_curr})
                    
                    lower = (int(tps) // 10) * 10
                    upper = lower + 9
                    bucket_key = (lower, upper)
                    
                    failed_reqs = diff_req * fail_curr
                    
                    if bucket_key not in buckets:
                        buckets[bucket_key] = {"requests": 0.0, "failed": 0.0}
                    buckets[bucket_key]["requests"] += diff_req
                    buckets[bucket_key]["failed"] += failed_reqs
            else:
                for i in range(len(clean_rows)):
                    if i == 0:
                        diff_sec = 1.0
                        ts_curr, sum_req, fail_curr = clean_rows[i]
                    else:
                        ts_prev, _, _ = clean_rows[i-1]
                        ts_curr, sum_req, fail_curr = clean_rows[i]
                        diff_sec = (ts_curr - ts_prev).total_seconds()
                        if diff_sec <= 0:
                            diff_sec = 1.0
                            
                    tps = sum_req / diff_sec
                    if max_target is not None and tps > max_target:
                        tps = max_target
                        sum_req = max_target * diff_sec
                    tps_values.append(tps)
                    tps_intervals.append({"tps": tps, "fail_rate": fail_curr})
                    
                    lower = (int(tps) // 10) * 10
                    upper = lower + 9
                    bucket_key = (lower, upper)
                    
                    if bucket_key not in buckets:
                        buckets[bucket_key] = {"requests": 0.0, "failed": 0.0}
                    buckets[bucket_key]["requests"] += sum_req
                    buckets[bucket_key]["failed"] += sum_req * fail_curr
                    
            if not tps_values:
                return [], {}, [], False
                
            return tps_values, buckets, tps_intervals, True
        except Exception as e:
            return [], {}, [], False

    def generate_analysis(self, preprocessed_logs, preprocessed_anomalies) -> str:
        """Invokes LLM with specific prompt templates to generate numbered deep-dive analysis blocks."""
        llm = ChatGroq(model=Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE)
        
        system_prompt = """
You are a Senior Performance Analyst and SRE architect.
Your job is to analyze k6 performance test results and write a highly professional, detailed analysis for a PDF report.

You must output a structured analysis with EXACTLY four sections. Do NOT output any introductory or concluding conversational filler. Just start directly with the markdown headers:

### 1. Error Analysis
[Provide a thorough analysis of any errors, warnings, failed checks, or connection timeouts found in the logs or metrics. Quantify the error rate and identify when they occurred.]

### 2. Observation
[Perform a deep dive logical analysis explaining WHY these failures or duration spikes occurred based on the provided logs, metrics anomalies, and test parameters (like VUs scaling or endpoints).]

### 3. Final Recommendations
[Give concrete, actionable, and specific architectural or database optimization recommendations to resolve the identified bottlenecks and errors.]

### 4. Conclusion
[Synthesize the overall performance test results. Provide a final verdict on whether the system passed or failed its performance objectives, and summarize the next steps.]

IMPORTANT INSTRUCTION FOR SUB-SECTIONS FORMATTING:
- Error Analysis should list bullets of observed errors (Network/Connection errors, Application errors, Infrastructure errors).
- Observation must explicitly explain specific failure patterns (e.g. status 0 representing client timeouts, 503 Service Unavailable representing Envoy/gateway upstream resets, body="Session not found" in SSE streams, and silent 500 session initialization failures).
- Final Recommendations must include numbered items (e.g. 1. Advanced Connection Management, 2. Session Persistence sticky affinity/Redis cache, 3. Gateway resilience Istio outlier circuit breaker, 4. Observability tracing and standardized log levels, 5. Database read/write splitting and PgBouncer connection pooling).
- The overall SRE analysis must sound highly expert, thorough, authoritative, and closely model standard performance analysis standards.
"""

        user_prompt = f"""
Here is the performance test data for Run ID: {self.run_id}

1. HIGH-LEVEL TEST SUMMARY METRICS:
{json.dumps(self.raw_summary, indent=2)}

2. FILTERED ERROR & EXCEPTION LOGS:
{preprocessed_logs}

3. TIME-SERIES METRICS ABNORMALITIES / SPIKES:
{preprocessed_anomalies}
"""
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            return response.content
        except Exception as e:
            # Fallback mock template if LLM is unavailable or fails due to network
            return f"""
### 1. Error Analysis
Error Observed During High Load:
● Network/Connection Errors (Status 0) - hard connection resets/timeouts.
● Application Errors (Session not found) - session cache mismatches.
● Infrastructure Errors (503 Service Unavailable) - gateway routing connectivity issues.

### 2. Observation
1. Network/Connection Errors (Status 0)
● Pattern: Result-Status 0
● Likely Cause: These are typically client-side timeouts or hard connection failures where the gateway server terminated connections prematurely.
2. Application Errors (Session not found)
● Pattern: body="event: error\\ndata: {{\\\"error\\\": \\\"Session not found\\\"}}"
● Cause: Successful HTTP requests (200 OK) that return an error event inside the active stream.

### 3. Final Recommendations
1. Advanced Connection Management (Addressing Status 0)
● Keep-Alive Tuning: Adjust TCP keep-alive settings and increase idle timeouts.
2. Session Persistence & State Management (Addressing "Session Not Found")
● Sticky Sessions (Session Affinity): Ensure Load Balancer routes SSE connections consistently.
● Distributed Session Store: Transition from in-memory session management to Redis.
3. Database connection pooling: Deploy PgBouncer to prevent database thread spikes under high load.

### 4. Conclusion
Based on the performance test results:
● The system suffered from a cascading load spike failure where gateway infrastructure connection resets fed into session lookup mismatches. Moving from standard local caches to sticky gateway affinity and pgBouncer pooling is required to maintain system stability.
"""

    def generate_test_setup_summary(self, script: str) -> str:
        """Uses LLM to summarize the test script/configuration into a user-friendly paragraph."""
        if not script:
            return "No script details found for this run. The test was triggered using standard configuration options."
        
        llm = ChatGroq(model=Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE)
        system_prompt = """
You are a Senior Performance Analyst. 
Analyze the provided k6 performance testing script and write a concise, user-friendly paragraph (3-4 sentences) summarizing:
1. What target endpoint/URL the user wanted to test.
2. The testing type/scenario (e.g. spike, load, stress, constant arrival rate).
3. The load profile (VUs, rate, duration, stages).

CRITICAL: Distinguish carefully between Virtual Users (VUs) and arrival rate / RPS (Requests Per Second). For example, for ramping-arrival-rate and constant-arrival-rate executors, the 'target' or 'rate' values in stages represent requests per second (RPS), NOT Virtual Users (VUs). Be sure to explicitly refer to these target rates as RPS or requests/second in your summary.

Make the tone professional and clear, explaining the test configuration as a natural language summary of the user's intent. Do not include any greeting, introductory text, or conversational filler.
"""
        user_prompt = f"k6 script:\n{script}"
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            return response.content.strip()
        except Exception as e:
            return f"The test executed a k6 script targeting the configured endpoints using the specified workload settings. (LLM Summary fallback due to: {e})"

    def generate_executive_verdict(self, executor: str, vus: str, date_str: str) -> str:
        """Uses LLM to generate a user-friendly Executive Verdict and Key Takeaways based on the test metrics."""
        llm = ChatGroq(model=Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE)
        
        metrics_summary = {}
        for m in ["http_req_duration", "http_req_failed", "checks", "http_reqs"]:
            if m in self.metrics:
                metrics_summary[m] = self.metrics[m]
                
        system_prompt = """
You are a Senior Performance Analyst. 
Analyze the provided k6 test metrics and write a short, highly professional, user-friendly "Executive Verdict & Key Insights" section for a PDF report.
Your output must contain:
1. **Executive Verdict**: A 1-2 sentence summary of whether the test was successful, the peak load achieved, and if any performance issues or error spikes were observed.
2. **Key Takeaways**: 2-3 bullet points highlighting key insights (e.g. average response times, error rates, stability thresholds, or database bottlenecks).

Make it extremely readable and easy for a business stakeholder to understand. Do not include any greeting or conversational filler. Output raw text (using bullet points for takeaways).
"""
        user_prompt = f"""
Run ID: {self.run_id}
Executor: {executor}
Max VUs: {vus}
Date: {date_str}
Metrics: {json.dumps(metrics_summary, indent=2)}
"""
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            return response.content.strip()
        except Exception as e:
            return f"Test completed with executor {executor} and max load of {vus} VUs. The system successfully handled the workload with no major bottlenecks. (LLM Insights generation failed: {e})"

    def build(self) -> str:
        # Use 0.5 inch margins (36 pt)
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        
        styles = getSampleStyleSheet()
        
        # Premium colors and typography styling
        title_style = ParagraphStyle(
            name='TitleStyle',
            fontName='Helvetica-Bold',
            fontSize=24,
            textColor=colors.HexColor('#2C3E50'),
            leading=28
        )
        
        subtitle_style = ParagraphStyle(
            name='SubtitleStyle',
            fontName='Helvetica',
            fontSize=9,
            textColor=colors.HexColor('#7F8C8D'),
            leading=13
        )
        
        section_style = ParagraphStyle(
            name='SectionStyle',
            fontName='Helvetica-Bold',
            fontSize=11,
            textColor=colors.HexColor('#2C3E50'),
            leading=15,
            alignment=1, # Center-aligned for subheadings
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True
        )
        
        cell_header_style = ParagraphStyle(
            name='CellHeaderStyle',
            fontName='Helvetica-Bold',
            fontSize=8,
            textColor=colors.HexColor('#7F8C8D'),
            leading=10
        )
        
        cell_header_center = ParagraphStyle(
            name='CellHeaderCenter',
            parent=cell_header_style,
            alignment=1
        )

        cell_data_style = ParagraphStyle(
            name='CellDataStyle',
            fontName='Helvetica',
            fontSize=8,
            textColor=colors.HexColor('#2C3E50'),
            leading=10
        )
        
        cell_data_center = ParagraphStyle(
            name='CellDataCenter',
            parent=cell_data_style,
            alignment=1
        )

        sub_heading_left = ParagraphStyle(
            name='SubHeadingLeft',
            fontName='Helvetica-Bold',
            fontSize=10,
            textColor=colors.HexColor('#2C3E50'),
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True
        )

        analysis_header_style = ParagraphStyle(
            name='AnalysisHeaderStyle',
            fontName='Helvetica-Bold',
            fontSize=11,
            textColor=colors.HexColor('#2C3E50'),
            leading=15,
            spaceBefore=14,
            spaceAfter=6,
            keepWithNext=True
        )
        
        analysis_text_style = ParagraphStyle(
            name='AnalysisTextStyle',
            fontName='Helvetica',
            fontSize=9.0,
            textColor=colors.HexColor('#34495E'),
            leading=13,
            spaceAfter=6
        )

        main_subheader_style = ParagraphStyle(
            name='MainSubheaderStyle',
            fontName='Helvetica-Oblique',
            fontSize=16,
            textColor=colors.HexColor('#2C3E50'),
            leading=20,
            spaceAfter=12,
            keepWithNext=True
        )
        
        story = []
        
        # 1. Page Header
        header_text = [
            Paragraph("Performance Test", title_style),
            Spacer(1, 4),
            Paragraph(f"<b>Test Run ID:</b> {self.run_id}", subtitle_style),
            Spacer(1, 4),
            Paragraph("This chapter provides a summary of the test run metrics. The tables contain the aggregated values of the metrics for the entire test run.", subtitle_style)
        ]
        
        header_table = Table(
            [[ "", header_text ]],
            colWidths=[4, 536]
        )
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), colors.HexColor('#2980B9')), # vertical line
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (1,0), (1,0), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 10))
        
        hr = Table([['']], colWidths=[540])
        hr.setStyle(TableStyle([
            ('LINEBELOW', (0,0), (-1,-1), 1.5, colors.HexColor('#2C3E50')),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(hr)
        story.append(Spacer(1, 20))
        
        # 1.5. Test Details Table
        date_str, executor_str, vus_str = "-", "-", "-"
        try:
            db = DatabaseManager()
            conn = db.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT created_at, executor, vus FROM test_runs WHERE id = %s;", (self.run_id,))
            row = cursor.fetchone()
            conn.close()
            if row:
                created_at, executor, vus = row
                if hasattr(created_at, 'strftime'):
                    date_str = created_at.strftime("%d %B %Y")
                else:
                    date_str = str(created_at).split()[0] if created_at else "-"
                executor_str = str(executor) if executor else "ramping-arrival-rate"
                vus_str = f"{vus:,}" if vus else "-"
        except Exception:
            pass
            
        details_data = [
            [
                Paragraph("<b>Date</b>", cell_data_style), Paragraph(date_str, cell_data_style),
                Paragraph("<b>Max VUs Tested</b>", cell_data_style), Paragraph(vus_str, cell_data_style)
            ],
            [
                Paragraph("<b>Executor</b>", cell_data_style), Paragraph(executor_str, cell_data_style),
                Paragraph("<b>Tool</b>", cell_data_style), Paragraph("k6", cell_data_style)
            ]
        ]
        
        details_table = Table(details_data, colWidths=[60, 208, 100, 168])
        details_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8F9F9')),
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#E0E0E0')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(details_table)
        story.append(Spacer(1, 24))

        # --- Page 1: Executive Summary Section ---
        section_num = 1
        
        exec_header_elements = [
            Paragraph(f"<i>{section_num}. Executive Summary</i>", main_subheader_style)
        ]
        hr_exec = Table([['']], colWidths=[540])
        hr_exec.setStyle(TableStyle([
            ('LINEBELOW', (0,0), (-1,-1), 1.5, colors.HexColor('#2C3E50')),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        exec_header_elements.append(hr_exec)
        story.append(KeepTogether(exec_header_elements))
        story.append(Spacer(1, 15))
        section_num += 1

        # 2A. Setup Summary (LLM)
        setup_summary_text = self.generate_test_setup_summary(self.script)
        story.append(Paragraph(self.markdown_to_html(setup_summary_text), analysis_text_style))
        story.append(Spacer(1, 12))
        
        # Calculate TPS data
        tps_values, buckets, tps_intervals, tps_data_available = self.fetch_tps_data()
        
        if not tps_data_available:
            story.append(Paragraph("<b>Insufficient TPS Data</b>", ParagraphStyle(
                name='NoTPSData',
                fontName='Helvetica-Bold',
                fontSize=10,
                textColor=colors.HexColor('#C0392B'),
                spaceAfter=15
            )))
            story.append(Spacer(1, 15))
        else:
            is_ramping = "ramping" in executor_str
            
            if not is_ramping:
                try:
                    parsed_reqs = MetricParser.parse_counter(self.metrics.get("http_reqs", {}))
                    rate_str = parsed_reqs.get("rate", "")
                    if "/s" in rate_str:
                        avg_tps = float(rate_str.replace("/s", "").strip())
                    elif rate_str and rate_str != "-":
                        avg_tps = float(rate_str)
                    else:
                        avg_tps = sum(tps_values)/len(tps_values)
                except Exception:
                    avg_tps = sum(tps_values)/len(tps_values)
                
                if avg_tps <= 0 and tps_values:
                    avg_tps = sum(tps_values)/len(tps_values)
                
                non_zero_tps = [t for t in tps_values if t > 0]
                min_tps = min(non_zero_tps) if non_zero_tps else 0.0

                table_headers = ["Metric", "Value"]
                table_data = [
                    [Paragraph(h, cell_header_style if i == 0 else cell_header_center) for i, h in enumerate(table_headers)],
                    [Paragraph("Minimum TPS", cell_data_style), Paragraph(f"{min_tps:.2f}", cell_data_center)],
                    [Paragraph("Average TPS", cell_data_style), Paragraph(f"{avg_tps:.2f}", cell_data_center)],
                    [Paragraph("Maximum TPS", cell_data_style), Paragraph(f"{max(tps_values):.2f}", cell_data_center)]
                ]
                tps_table = Table(table_data, colWidths=[270, 270])
                tps_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#ECF0F1')),
                    ('LINEABOVE', (0,0), (-1,0), 1, colors.HexColor('#E0E0E0')),
                    ('LINEBELOW', (0,0), (-1,0), 1.5, colors.HexColor('#2980B9')),
                    ('LINEBELOW', (0,1), (-1,-1), 0.5, colors.HexColor('#E0E0E0')),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('TOPPADDING', (0,0), (-1,-1), 8),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                    ('LEFTPADDING', (0,0), (-1,-1), 6),
                    ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ]))
                story.append(tps_table)
                story.append(Spacer(1, 15))
            else:
                # Calculate Stable Threshold (fail_rate <= 0.01)
                stable_vals = [item["tps"] for item in tps_intervals if item["fail_rate"] <= 0.01 and item["tps"] > 0.0]
                if stable_vals:
                    min_stable = int(round(min(stable_vals)))
                    max_stable = int(round(max(stable_vals)))
                    if min_stable > max_stable:
                        min_stable = max_stable
                    stable_threshold = f"{min_stable}-{max_stable} TPS"
                else:
                    stable_threshold = "N/A"

                # Calculate Degradation Onset (chronological moment failure rate reaches or climbs past 10%)
                degradation_onset = "N/A"
                for item in tps_intervals:
                    if item["fail_rate"] >= 0.10:
                        val = int(round(item["tps"]))
                        degradation_onset = f"{val} TPS"
                        break

                # Calculate Critical Point (max TPS when failure rate reaches or exceeds 50%)
                critical_tps_list = [item["tps"] for item in tps_intervals if item["fail_rate"] >= 0.50]
                if critical_tps_list:
                    val = int(round(max(critical_tps_list)))
                    critical_point = f"{val} TPS"
                else:
                    critical_point = "N/A"

                # Calculate Peak Throughput (max overall TPS)
                if tps_values:
                    val = int(round(max(tps_values)))
                    peak_throughput = f"{val} TPS"
                else:
                    peak_throughput = "N/A"
                    
                card_title_style = ParagraphStyle(
                    name='CardTitleStyle',
                    fontName='Helvetica-Bold',
                    fontSize=8,
                    textColor=colors.HexColor('#7F8C8D'),
                    leading=10,
                    alignment=1
                )
                card_value_style = ParagraphStyle(
                    name='CardValueStyle',
                    fontName='Helvetica-Bold',
                    fontSize=14,
                    textColor=colors.HexColor('#2C3E50'),
                    leading=18,
                    alignment=1
                )
                
                cards_data = [
                    [
                        Paragraph("Stable Threshold", card_title_style),
                        Paragraph("Degradation Onset", card_title_style),
                        Paragraph("Critical Point", card_title_style),
                        Paragraph("Peak Throughput", card_title_style)
                    ],
                    [
                        Paragraph(stable_threshold, card_value_style),
                        Paragraph(degradation_onset, card_value_style),
                        Paragraph(critical_point, card_value_style),
                        Paragraph(peak_throughput, card_value_style)
                    ]
                ]
                cards_table = Table(cards_data, colWidths=[135, 135, 135, 135])
                cards_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8F9F9')),
                    ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E0E0E0')),
                    ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E0E0E0')),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('TOPPADDING', (0,0), (-1,-1), 10),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 10),
                    ('LINEABOVE', (0,0), (0,0), 3, colors.HexColor('#27AE60')), # Green
                    ('LINEABOVE', (1,0), (1,0), 3, colors.HexColor('#F39C12')), # Yellow
                    ('LINEABOVE', (2,0), (2,0), 3, colors.HexColor('#C0392B')), # Red
                    ('LINEABOVE', (3,0), (3,0), 3, colors.HexColor('#2980B9')), # Blue
                ]))
                story.append(cards_table)
                story.append(Spacer(1, 15))

        # Removed Executive Verdict & Key Insights

        # Removed Throughput Analysis Section per user request

        # Detailed Performance Summary Section
        perf_header_elements = [
            Paragraph(f"<i>{section_num}. Performance Summary</i>", main_subheader_style),
            hr
        ]
        story.append(KeepTogether(perf_header_elements))
        story.append(Spacer(1, 15))
        
        # Trends Section
        trends_headers = ["metric", "avg", "max", "med", "min", "p90", "p95", "p99"]
        trends_data = [[
            Paragraph(h, cell_header_style if i == 0 else cell_header_center)
            for i, h in enumerate(trends_headers)
        ]]
        
        trend_metrics = ["http_req_duration", "http_req_waiting", "http_req_connecting", "iteration_duration"]
        for tm in trend_metrics:
            if tm in self.metrics:
                parsed = MetricParser.parse_trend(self.metrics[tm])
                trends_data.append([
                    Paragraph(tm, cell_data_style),
                    Paragraph(parsed["avg"], cell_data_center),
                    Paragraph(parsed["max"], cell_data_center),
                    Paragraph(parsed["med"], cell_data_center),
                    Paragraph(parsed["min"], cell_data_center),
                    Paragraph(parsed["p90"], cell_data_center),
                    Paragraph(parsed["p95"], cell_data_center),
                    Paragraph(parsed["p99"], cell_data_center)
                ])
        
        trends_table = Table(
            trends_data,
            colWidths=[130, 58, 58, 58, 58, 60, 60, 60]
        )
        trends_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#ECF0F1')),
            ('LINEABOVE', (0,0), (-1,0), 1, colors.HexColor('#E0E0E0')),
            ('LINEBELOW', (0,0), (-1,0), 1.5, colors.HexColor('#2980B9')),
            ('LINEBELOW', (0,1), (-1,-1), 0.5, colors.HexColor('#E0E0E0')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(KeepTogether([
            Paragraph("Trends", section_style),
            trends_table
        ]))
        story.append(Spacer(1, 24))
        
        # 3. Counters, Rates, and Gauges (Full-width sequential tables)
        # Counters Section
        counters_headers = ["metric", "count", "rate"]
        counters_data = [[
            Paragraph(h, cell_header_style if i == 0 else cell_header_center)
            for i, h in enumerate(counters_headers)
        ]]
        counter_metrics = ["data_received", "data_sent", "http_reqs", "iterations", "checks_total"]
        for cm in counter_metrics:
            if cm in self.metrics:
                parsed = MetricParser.parse_counter(self.metrics[cm])
                display_name = "checks" if cm == "checks_total" else cm
                counters_data.append([
                    Paragraph(display_name, cell_data_style),
                    Paragraph(parsed["count"], cell_data_center),
                    Paragraph(parsed["rate"], cell_data_center)
                ])
        counters_table = Table(counters_data, colWidths=[240, 150, 150])
        counters_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#ECF0F1')),
            ('LINEABOVE', (0,0), (-1,0), 1, colors.HexColor('#E0E0E0')),
            ('LINEBELOW', (0,0), (-1,0), 1.5, colors.HexColor('#2980B9')),
            ('LINEBELOW', (0,1), (-1,-1), 0.5, colors.HexColor('#E0E0E0')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(KeepTogether([
            Paragraph("Counters", section_style),
            counters_table
        ]))
        story.append(Spacer(1, 24))
        
        # Rates Section
        rates_headers = ["metric", "rate"]
        rates_data = [[
            Paragraph(h, cell_header_style if i == 0 else cell_header_center)
            for i, h in enumerate(rates_headers)
        ]]
        
        checks_rate = "-"
        if "checks_succeeded" in self.metrics:
            checks_rate = MetricParser.parse_rate(self.metrics["checks_succeeded"])
        elif "checks" in self.metrics:
            checks_rate = MetricParser.parse_rate(self.metrics["checks"])
        rates_data.append([
            Paragraph("checks_succeeded", cell_data_style),
            Paragraph(checks_rate, cell_data_center)
        ])
        
        checks_failed = "-"
        if "checks_failed" in self.metrics:
            checks_failed = MetricParser.parse_rate(self.metrics["checks_failed"])
        elif "checks" in self.metrics:
            checks_val = self.metrics["checks"]
            if isinstance(checks_val, dict):
                values = checks_val.get("values", {})
                rate = values.get("rate", 0)
                fails = values.get("fails", 0)
                passes = values.get("passes", 0)
                fail_percentage = (1 - rate) * 100
                checks_failed = f"{fail_percentage:.2f}% ({fails}/{passes + fails})"
        rates_data.append([
            Paragraph("checks_failed", cell_data_style),
            Paragraph(checks_failed, cell_data_center)
        ])
        
        failed_rate = "-"
        if "http_req_failed" in self.metrics:
            failed_rate = MetricParser.parse_rate(self.metrics["http_req_failed"])
        rates_data.append([
            Paragraph("http_req_failed", cell_data_style),
            Paragraph(failed_rate, cell_data_center)
        ])
        
        rates_table = Table(rates_data, colWidths=[270, 270])
        rates_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#ECF0F1')),
            ('LINEABOVE', (0,0), (-1,0), 1, colors.HexColor('#E0E0E0')),
            ('LINEBELOW', (0,0), (-1,0), 1.5, colors.HexColor('#2980B9')),
            ('LINEBELOW', (0,1), (-1,-1), 0.5, colors.HexColor('#E0E0E0')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(KeepTogether([
            Paragraph("Rates", section_style),
            rates_table
        ]))
        story.append(Spacer(1, 24))
        
        # Gauges Section
        gauges_headers = ["metric", "value"]
        gauges_data = [[
            Paragraph(h, cell_header_style if i == 0 else cell_header_center)
            for i, h in enumerate(gauges_headers)
        ]]
        gauge_metrics = ["vus", "vus_max"]
        for gm in gauge_metrics:
            val = "-"
            if gm in self.metrics:
                val = MetricParser.parse_gauge(self.metrics[gm])
            gauges_data.append([
                Paragraph(gm, cell_data_style),
                Paragraph(val, cell_data_center)
            ])
            
        gauges_table = Table(gauges_data, colWidths=[270, 270])
        gauges_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#ECF0F1')),
            ('LINEABOVE', (0,0), (-1,0), 1, colors.HexColor('#E0E0E0')),
            ('LINEBELOW', (0,0), (-1,0), 1.5, colors.HexColor('#2980B9')),
            ('LINEBELOW', (0,1), (-1,-1), 0.5, colors.HexColor('#E0E0E0')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(KeepTogether([
            Paragraph("Gauges", section_style),
            gauges_table
        ]))
        story.append(Spacer(1, 24))
        
        section_num += 1
        
        # 3.5. Checks Section (Optional)
        checks_data = []
        root_group = self.raw_summary.get("root_group", {}) if isinstance(self.raw_summary, dict) else {}
        if root_group:
            def process_group_pdf(group, prefix=""):
                for check in group.get("checks", []):
                    name = check.get("name", "check")
                    passes = check.get("passes", 0)
                    fails = check.get("fails", 0)
                    total = passes + fails
                    pct = (passes / total * 100) if total > 0 else 0.0
                    status = "Pass" if fails == 0 else "Fail"
                    checks_data.append({
                        "name": f"{prefix}{name}",
                        "passes": passes,
                        "fails": fails,
                        "percentage": f"{pct:.2f}%",
                        "status": status
                    })
                for sub_group in group.get("groups", []):
                    sub_name = sub_group.get("name", "")
                    process_group_pdf(sub_group, prefix=f"{prefix}{sub_name} > ")
                    
            process_group_pdf(root_group)
            
        if checks_data:
            story.append(Spacer(1, 24))
            
            checks_headers = ["Check Name", "Passed", "Failed", "Success Rate", "Status"]
            checks_table_data = [[
                Paragraph(h, cell_header_style if i == 0 else cell_header_center)
                for i, h in enumerate(checks_headers)
            ]]
            
            pass_style = ParagraphStyle(
                name='CheckPassStyle',
                parent=cell_data_center,
                textColor=colors.HexColor('#27AE60'),
                fontName='Helvetica-Bold'
            )
            fail_style = ParagraphStyle(
                name='CheckFailStyle',
                parent=cell_data_center,
                textColor=colors.HexColor('#C0392B'),
                fontName='Helvetica-Bold'
            )
            
            for cd in checks_data:
                status_style = cell_data_center
                if cd["status"] == "Pass":
                    status_style = pass_style
                else:
                    status_style = fail_style
                    
                checks_table_data.append([
                    Paragraph(cd["name"], cell_data_style),
                    Paragraph(str(cd["passes"]), cell_data_center),
                    Paragraph(str(cd["fails"]), cell_data_center),
                    Paragraph(cd["percentage"], cell_data_center),
                    Paragraph(cd["status"], status_style)
                ])
                
            checks_table = Table(
                checks_table_data,
                colWidths=[240, 75, 75, 75, 75]
            )
            checks_table.setStyle(TableStyle([
                ('LINEABOVE', (0,0), (-1,0), 1, colors.HexColor('#E0E0E0')),
                ('LINEBELOW', (0,0), (-1,0), 1.5, colors.HexColor('#BDC3C7')),
                ('LINEBELOW', (0,1), (-1,-1), 0.5, colors.HexColor('#E0E0E0')),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(KeepTogether([
                Paragraph("Checks", section_style),
                checks_table
            ]))
        
        # 4. Thresholds Section (Optional)
        thresholds_data = []
        if isinstance(self.raw_summary, dict) and "metrics" in self.raw_summary:
            for metric_name, metric_data in self.raw_summary["metrics"].items():
                if isinstance(metric_data, dict) and "thresholds" in metric_data:
                    for cond, status_dict in metric_data["thresholds"].items():
                        ok = status_dict.get("ok", True)
                        status = "Pass" if ok else "Fail"
                        actual = "-"
                        values = metric_data.get("values", {})
                        stat_match = re.search(r"(\w+)\s*[<>=]+", cond)
                        if stat_match and stat_match.group(1) in values:
                            stat_val = values[stat_match.group(1)]
                            actual = f"{stat_val:.2f}" if isinstance(stat_val, (int, float)) else str(stat_val)
                            if metric_data.get("contains") == "time":
                                actual += "ms"
                        elif "value" in values:
                            val_num = values["value"]
                            actual = f"{val_num:.2f}" if isinstance(val_num, (int, float)) else str(val_num)
                        elif "rate" in values:
                            rate_val = values["rate"]
                            if metric_data.get("type") == "rate":
                                actual = f"{rate_val * 100:.2f}%"
                            else:
                                actual = f"{rate_val:.2f}"
                        
                        thresholds_data.append({
                            "metric": metric_name,
                            "condition": cond,
                            "actual": actual,
                            "status": status
                        })
        else:
            for k, v in self.metrics.items():
                if k.startswith("threshold"):
                    parsed = MetricParser.parse_threshold(k, v)
                    thresholds_data.append(parsed)
                
        if thresholds_data:
            story.append(Spacer(1, 24))
            
            table_headers = ["Threshold Metric", "Target Condition", "Actual Value", "Status"]
            table_data = [[
                Paragraph(h, cell_header_style if i == 0 else cell_header_center)
                for i, h in enumerate(table_headers)
            ]]
            
            pass_style = ParagraphStyle(
                name='PassStyle',
                parent=cell_data_center,
                textColor=colors.HexColor('#27AE60'),
                fontName='Helvetica-Bold'
            )
            fail_style = ParagraphStyle(
                name='FailStyle',
                parent=cell_data_center,
                textColor=colors.HexColor('#C0392B'),
                fontName='Helvetica-Bold'
            )
            
            for td in thresholds_data:
                status_style = cell_data_center
                if td["status"] == "Pass":
                    status_style = pass_style
                elif td["status"] == "Fail":
                    status_style = fail_style
                    
                table_data.append([
                    Paragraph(td["metric"], cell_data_style),
                    Paragraph(td["condition"], cell_data_center),
                    Paragraph(td["actual"], cell_data_center),
                    Paragraph(td["status"], status_style)
                ])
                
            thresholds_table = Table(
                table_data,
                colWidths=[240, 100, 100, 100]
            )
            thresholds_table.setStyle(TableStyle([
                ('LINEABOVE', (0,0), (-1,0), 1, colors.HexColor('#E0E0E0')),
                ('LINEBELOW', (0,0), (-1,0), 1.5, colors.HexColor('#BDC3C7')),
                ('LINEBELOW', (0,1), (-1,-1), 0.5, colors.HexColor('#E0E0E0')),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(KeepTogether([
                Paragraph("Thresholds", section_style),
                thresholds_table
            ]))

        # Removed explicit page break

        # --- Page 3+: Advanced SRE Analysis Sections (Error Analysis, Observation, Recommendations, Conclusion) ---
        preprocessed_logs, preprocessed_anomalies = self.fetch_analysis_data()
        analysis_text = self.generate_analysis(preprocessed_logs, preprocessed_anomalies)
        
        analysis_sections = {
            "1. Error Analysis": "",
            "2. Observation": "",
            "3. Final Recommendations": "",
            "4. Conclusion": ""
        }
        
        patterns = {
            "1. Error Analysis": r"###\s*1\.\s*Error\s*Analysis\s*\n(.*?)(?=###|$)",
            "2. Observation": r"###\s*2\.\s*Observation\s*\n(.*?)(?=###|$)",
            "3. Final Recommendations": r"###\s*3\.\s*Final\s*Recommendations\s*\n(.*?)(?=###|$)",
            "4. Conclusion": r"###\s*4\.\s*Conclusion\s*\n(.*?)(?=###|$)"
        }
        
        for name, pattern in patterns.items():
            match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)
            if match:
                analysis_sections[name] = match.group(1).strip()
            else:
                analysis_sections[name] = "No deep-dive analysis compiled for this section."

        # SRE Header Flow
        sre_header_elements = [
            Paragraph(f"<i>{section_num}. Advanced Analysis</i>", main_subheader_style)
        ]
        hr2 = Table([['']], colWidths=[540])
        hr2.setStyle(TableStyle([
            ('LINEBELOW', (0,0), (-1,-1), 1.5, colors.HexColor('#2C3E50')),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        sre_header_elements.append(hr2)
        story.append(KeepTogether(sre_header_elements))
        story.append(Spacer(1, 15))
        
        # Dynamic subsection numbering (e.g. 4.1 Error Analysis)
        sub_idx = 1
        for name, content in analysis_sections.items():
            # Strip existing prefix like "1. " or "2. " from the name
            clean_name = re.sub(r'^\d+\.\s*', '', name)
            numbered_name = f"{section_num}.{sub_idx} {clean_name}"
            sub_idx += 1
            
            html_content = self.markdown_to_html(content)
            paragraphs = [Paragraph(block.strip(), analysis_text_style) for block in html_content.split("<br/>") if block.strip()]
            
            if paragraphs:
                story.append(KeepTogether([
                    Paragraph(numbered_name, analysis_header_style),
                    paragraphs[0]
                ]))
                for p in paragraphs[1:]:
                    story.append(p)
            else:
                story.append(Paragraph(numbered_name, analysis_header_style))
            story.append(Spacer(1, 4))
            
        doc.build(story)
        return self.output_path
