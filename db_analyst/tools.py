from typing import Literal
from langchain_core.tools import StructuredTool
from .database import DatabaseManager
from .report_generator import PDFReportBuilder

MetricType = str


class DBTools:
    """Encapsulates database querying tools for the agent."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def _get_realtime_metrics(self, run_id: str, metric_name: MetricType, limit: int = 100) -> str:
        """
        Fetches specific time-series metrics for a k6 test run.
        Use this to find how a metric (like http_req_duration or vus) changed over time.
        Available metrics: k6_vus, k6_vus_max, k6_http_req_duration_avg, k6_http_req_duration_min, 
        k6_http_req_duration_max, k6_http_req_duration_p90, k6_http_req_duration_p95, k6_http_req_duration_p99, 
        k6_http_reqs_total, k6_http_req_failed_rate, k6_iteration_duration_avg, k6_iteration_duration_min, 
        k6_iteration_duration_max, k6_iteration_duration_p90, k6_iteration_duration_p95, k6_iteration_duration_p99, 
        k6_iterations_total, k6_data_received_total, k6_data_sent_total, k6_checks_rate
        """
        valid_metrics = [
            "k6_vus", "k6_vus_max", "k6_http_req_duration_avg", "k6_http_req_duration_min",
            "k6_http_req_duration_max", "k6_http_req_duration_p90", "k6_http_req_duration_p95",
            "k6_http_req_duration_p99", "k6_http_reqs_total", "k6_http_req_failed_rate",
            "k6_iteration_duration_avg", "k6_iteration_duration_min", "k6_iteration_duration_max",
            "k6_iteration_duration_p90", "k6_iteration_duration_p95", "k6_iteration_duration_p99",
            "k6_iterations_total", "k6_data_received_total", "k6_data_sent_total", "k6_checks_rate",
            "k6_http_req_waiting_avg", "k6_http_req_waiting_min", "k6_http_req_waiting_max",
            "k6_http_req_waiting_p90", "k6_http_req_waiting_p95", "k6_http_req_waiting_p99",
            "k6_http_req_receiving_avg", "k6_http_req_receiving_min", "k6_http_req_receiving_max",
            "k6_http_req_receiving_p90", "k6_http_req_receiving_p95", "k6_http_req_receiving_p99",
            "k6_http_req_sending_avg", "k6_http_req_sending_min", "k6_http_req_sending_max",
            "k6_http_req_sending_p90", "k6_http_req_sending_p95", "k6_http_req_sending_p99"
        ]
        
        # Clean the input metric name
        clean_name = metric_name.lower().strip()
        if not clean_name.startswith("k6_"):
            clean_name = f"k6_{clean_name}"
            
        if clean_name not in valid_metrics:
            # Try to find close matches or fallback to returning all valid ones
            close_matches = [m for m in valid_metrics if clean_name in m or m in clean_name]
            match_str = f" Did you mean: {', '.join(close_matches)}?" if close_matches else ""
            return f"Metric '{metric_name}' is not recognized.{match_str} Available metrics: {', '.join(valid_metrics)}"

        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        query = """
            SELECT ts, value 
            FROM realtime_metrics 
            WHERE run_id = %s AND name = %s 
            ORDER BY ts ASC LIMIT %s;
        """
        cursor.execute(query, (run_id, clean_name, limit))
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return f"No metric data found for '{clean_name}'."
        
        output = str(results)
        if len(output) > 5000:
            return output[:5000] + "... [Output Truncated]"
        return output

    def _get_test_logs(self, run_id: str, keyword: str = "") -> str:
        """
        Fetches raw stdout/stderr logs for a k6 test run. 
        Provide a keyword to filter for specific errors or events.
        To find progress at a specific time, use a keyword like 12.0s or 05.0s.
        """
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        query = "SELECT content FROM test_logs WHERE run_id = %s AND content ILIKE %s LIMIT 20;"
        cursor.execute(query, (run_id, f"%{keyword}%"))
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return "No logs found matching the criteria."
        
        output_lines = []
        for r in results:
            content = r[0]
            if keyword:
                # Split content into lines and filter for keyword
                lines = content.split('\n')
                matching_lines = [line for line in lines if keyword.lower() in line.lower()]
                output_lines.extend(matching_lines)
            else:
                output_lines.append(content)
                
        output = "\n".join(output_lines)
        if len(output) > 5000:
            return output[:5000] + "... [Output Truncated]"
        return output

    def _get_test_summaries(self, run_id: str) -> str:
        """
        Fetches the final aggregated end-of-test results (total requests, averages, thresholds).
        """
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT metrics FROM test_summaries WHERE run_id = %s;", (run_id,))
        result = cursor.fetchone()
        conn.close()
        return str(result) if result else "Summary not found."

    def _get_test_metadata(self, run_id: str) -> str:
        """
        Fetches the configuration metadata for a run, such as the number of VUs and the k6 script used.
        """
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT vus, script, status, created_at FROM test_runs WHERE id = %s;", (run_id,))
        result = cursor.fetchone()
        conn.close()
        return str(result) if result else "Run metadata not found."

    def _generate_pdf_report(self, run_id: str) -> str:
        """
        Generates a beautifully formatted PDF report of the test summaries for the given run_id.
        The PDF includes Trends, Counters, Rates, and Gauges.
        Returns the file path of the generated PDF file.
        """
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT metrics FROM test_summaries WHERE run_id = %s;", (run_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return f"Summary not found for run_id: {run_id}. Unable to generate PDF."
        
        metrics = result[0]
        try:
            builder = PDFReportBuilder(run_id, metrics)
            pdf_path = builder.build()
            return f"PDF report successfully generated at: {pdf_path}"
        except Exception as e:
            return f"Error generating PDF report: {e}"

    def get_tools(self) -> list[StructuredTool]:
        """Returns the list of tools configured for LangChain."""
        return [
            StructuredTool.from_function(
                func=self._get_realtime_metrics,
                name="get_realtime_metrics",
                description="Fetches specific time-series metrics for a k6 test run. Use this to find how a metric like http_req_duration or vus changed over time."
            ),
            StructuredTool.from_function(
                func=self._get_test_logs,
                name="get_test_logs",
                description="Fetches raw stdout/stderr logs for a k6 test run. Provide a keyword to filter for specific errors or events. To find progress at a specific time, use a keyword like 12.0s or 05.0s."
            ),
            StructuredTool.from_function(
                func=self._get_test_summaries,
                name="get_test_summaries",
                description="Fetches the final aggregated end-of-test results including total requests, averages, and thresholds."
            ),
            StructuredTool.from_function(
                func=self._get_test_metadata,
                name="get_test_metadata",
                description="Fetches the configuration metadata for a run, such as the number of VUs and the k6 script used."
            ),
            StructuredTool.from_function(
                func=self._generate_pdf_report,
                name="generate_pdf_report",
                description="Generates a beautifully formatted PDF report of the test summaries including Trends, Counters, Rates, and Gauges for a given run_id. Returns the path to the saved PDF file."
            )
        ]
