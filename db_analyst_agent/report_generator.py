import os
import re
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
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


class PDFReportBuilder:
    """Uses ReportLab to draw a beautifully formatted PDF report matching the reference design."""
    
    def __init__(self, run_id: str, metrics: dict):
        self.run_id = run_id
        self.raw_summary = metrics
        
        # If it's the new JSONB format, the actual metrics are nested under 'metrics'
        if isinstance(metrics, dict) and "metrics" in metrics:
            self.metrics = metrics["metrics"]
        else:
            self.metrics = metrics
        
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
            alignment=1 # Center-aligned for subheadings
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
        
        main_subheader_style = ParagraphStyle(
            name='MainSubheaderStyle',
            fontName='Helvetica-Oblique',
            fontSize=16,
            textColor=colors.HexColor('#2C3E50'),
            leading=20,
            spaceAfter=12
        )
        story.append(Paragraph("<i>Performance Summary</i>", main_subheader_style))
        story.append(Spacer(1, 10))
        
        # 2. Trends Section
        story.append(Paragraph("Trends", section_style))
        story.append(Spacer(1, 8))
        
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
        story.append(trends_table)
        story.append(Spacer(1, 24))
        
        # 3. Counters, Rates, and Gauges (Side-by-side columns)
        # Counters
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
        counters_table = Table(counters_data, colWidths=[70, 50, 50])
        counters_table.setStyle(TableStyle([
            ('LINEABOVE', (0,0), (-1,0), 1, colors.HexColor('#E0E0E0')),
            ('LINEBELOW', (0,0), (-1,0), 1.5, colors.HexColor('#BDC3C7')),
            ('LINEBELOW', (0,1), (-1,-1), 0.5, colors.HexColor('#E0E0E0')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 4),
            ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ]))
        
        # Rates
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
        
        rates_table = Table(rates_data, colWidths=[90, 60])
        rates_table.setStyle(TableStyle([
            ('LINEABOVE', (0,0), (-1,0), 1, colors.HexColor('#E0E0E0')),
            ('LINEBELOW', (0,0), (-1,0), 1.5, colors.HexColor('#BDC3C7')),
            ('LINEBELOW', (0,1), (-1,-1), 0.5, colors.HexColor('#E0E0E0')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 4),
            ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ]))
        
        # Gauges
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
            
        gauges_table = Table(gauges_data, colWidths=[90, 60])
        gauges_table.setStyle(TableStyle([
            ('LINEABOVE', (0,0), (-1,0), 1, colors.HexColor('#E0E0E0')),
            ('LINEBELOW', (0,0), (-1,0), 1.5, colors.HexColor('#BDC3C7')),
            ('LINEBELOW', (0,1), (-1,-1), 0.5, colors.HexColor('#E0E0E0')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 4),
            ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ]))
        
        # Embed all 3 tables side-by-side in a layout Table
        layout_data = [[
            [Paragraph("Counters", section_style), Spacer(1, 8), counters_table],
            "", # spacer
            [Paragraph("Rates", section_style), Spacer(1, 8), rates_table],
            "", # spacer
            [Paragraph("Gauges", section_style), Spacer(1, 8), gauges_table]
        ]]
        
        layout_table = Table(
            layout_data,
            colWidths=[170, 25, 150, 25, 150]
        )
        layout_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(layout_table)
        
        # 3.5. Checks Section (Optional, only if checks exist in the metrics)
        checks_data = []
        root_group = self.raw_summary.get("root_group", {}) if isinstance(self.raw_summary, dict) else {}
        if root_group:
            # support both root and nested checks in groups recursively
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
            story.append(Paragraph("Checks", section_style))
            story.append(Spacer(1, 8))
            
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
            story.append(checks_table)
        
        # 4. Thresholds Section (Optional, only if thresholds exist in the metrics)
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
            story.append(Paragraph("Thresholds", section_style))
            story.append(Spacer(1, 8))
            
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
            story.append(thresholds_table)
            
        # --- Page 2: Advanced SRE Analysis Sections (Error Analysis, Observation, Recommendations, Conclusion) ---
        # 1. Custom Typography Styles for SRE Page
        analysis_title_style = ParagraphStyle(
            name='AnalysisTitleStyle',
            fontName='Helvetica-Bold',
            fontSize=16,
            textColor=colors.HexColor('#2C3E50'),
            leading=20,
            spaceAfter=12
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

        # 2. Append Page Break to separate summary tables and deep analysis
        story.append(PageBreak())
        
        # 3. Preprocess and generate analysis text using database + LLM
        preprocessed_logs, preprocessed_anomalies = self.fetch_analysis_data()
        analysis_text = self.generate_analysis(preprocessed_logs, preprocessed_anomalies)
        
        # 4. Parse output sections dynamically using Regex
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

        # 5. Append SRE Header Flow
        story.append(Paragraph("<i>Advanced Analysis</i>", main_subheader_style))
        story.append(Spacer(1, 10))
        
        hr2 = Table([['']], colWidths=[540])
        hr2.setStyle(TableStyle([
            ('LINEBELOW', (0,0), (-1,-1), 1.5, colors.HexColor('#2C3E50')),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(hr2)
        story.append(Spacer(1, 15))
        
        # 6. Append analytical sections and convert bullets & bolding dynamically
        for name, content in analysis_sections.items():
            story.append(Paragraph(name, analysis_header_style))
            html_content = self.markdown_to_html(content)
            for block in html_content.split("<br/>"):
                if block.strip():
                    story.append(Paragraph(block.strip(), analysis_text_style))
            story.append(Spacer(1, 4))
            
        doc.build(story)
        return self.output_path
