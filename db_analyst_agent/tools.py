from typing import Literal
from langchain_core.tools import StructuredTool
from .database import DatabaseManager

MetricType = Literal[
    "k6_vus", 
    "k6_vus_max", 
    "k6_http_req_duration_p99", 
    "k6_http_reqs_total", 
    "k6_http_req_failed_rate",
    "k6_iteration_duration_p99",
    "k6_iterations_total",
    "k6_data_received_total",
    "k6_data_sent_total",
    "k6_checks_rate"
]

class DBTools:
    """Encapsulates database querying tools for the agent."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def _get_realtime_metrics(self, run_id: str, metric_name: MetricType, limit: int = 100) -> str:
        """
        Fetches specific time-series metrics for a k6 test run.
        Use this to find how a metric (like http_req_duration or vus) changed over time.
        """
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        query = """
            SELECT ts, value 
            FROM realtime_metrics 
            WHERE run_id = %s AND name = %s 
            ORDER BY ts ASC LIMIT %s;
        """
        cursor.execute(query, (run_id, metric_name, limit))
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return "No metric data found."
        
        output = str(results)
        if len(output) > 5000:
            return output[:5000] + "... [Output Truncated]"
        return output

    def _get_test_logs(self, run_id: str, keyword: str = "") -> str:
        """
        Fetches raw stdout/stderr logs for a k6 test run. 
        Provide a keyword to filter for specific errors or events.
        To find progress at a specific time, use a keyword like '(12.0s)' or '(05.0s)'.
        """
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        query = "SELECT content FROM test_logs WHERE run_id = %s AND content ILIKE %s LIMIT 20;"
        cursor.execute(query, (run_id, f"%{keyword}%"))
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return "No logs found matching the criteria."
        
        output = "\\n".join([r[0] for r in results])
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

    def get_tools(self) -> list[StructuredTool]:
        """Returns the list of tools configured for LangChain."""
        return [
            StructuredTool.from_function(
                func=self._get_realtime_metrics,
                name="get_realtime_metrics",
                description="Fetches specific time-series metrics for a k6 test run. Use this to find how a metric (like http_req_duration or vus) changed over time."
            ),
            StructuredTool.from_function(
                func=self._get_test_logs,
                name="get_test_logs",
                description="Fetches raw stdout/stderr logs for a k6 test run. Provide a keyword to filter for specific errors or events. To find progress at a specific time, use a keyword like '(12.0s)' or '(05.0s)'."
            ),
            StructuredTool.from_function(
                func=self._get_test_summaries,
                name="get_test_summaries",
                description="Fetches the final aggregated end-of-test results (total requests, averages, thresholds)."
            ),
            StructuredTool.from_function(
                func=self._get_test_metadata,
                name="get_test_metadata",
                description="Fetches the configuration metadata for a run, such as the number of VUs and the k6 script used."
            )
        ]
