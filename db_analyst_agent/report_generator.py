import os
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

class MetricParser:
    """Parses raw text metrics from PostgreSQL JSON into structured fields."""
    
    @staticmethod
    def parse_trend(raw_str: str) -> dict:
        res = {
            "avg": "-", "max": "-", "med": "-", "min": "-",
            "p90": "-", "p95": "-", "p99": "-"
        }
        if not raw_str:
            return res
        matches = re.findall(r"([\w()]+)=([\w.]+ms|[\w.]+s|[\w.]+)", raw_str)
        for k, v in matches:
            # Clean parentheses to match dict keys (e.g., p(90) -> p90)
            k_clean = k.lower().replace("(", "").replace(")", "")
            if k_clean in res:
                res[k_clean] = v
        return res

    @staticmethod
    def parse_counter(raw_str: str) -> dict:
        res = {"count": "-", "rate": "-"}
        if not raw_str:
            return res
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
    def parse_rate(raw_str: str) -> str:
        if not raw_str:
            return "-"
        parts = raw_str.strip().split()
        if len(parts) >= 5 and parts[2] == "out" and parts[3] == "of":
            return f"{parts[0]} ({parts[1]}/{parts[4]})"
        if len(parts) >= 2:
            return f"{parts[0]} ({parts[1]})"
        if parts:
            return parts[0]
        return "-"

    @staticmethod
    def parse_gauge(raw_str: str) -> str:
        if not raw_str:
            return "-"
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
        self.metrics = metrics
        
        # Strip duplicated "run_" prefix if present in the run_id to avoid double naming
        clean_run_id = run_id[4:] if run_id.startswith("run_") else run_id
        
        # Use workspace-relative paths
        os.makedirs("reports", exist_ok=True)
        self.output_path = f"reports/run_{clean_run_id}_summary.pdf"

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
        
        # 1. Page Header (Title + Left Gray Vertical Bar)
        header_text = [
            Paragraph("Summary", title_style),
            Spacer(1, 4),
            Paragraph("This chapter provides a summary of the test run metrics. The tables contain the aggregated values of the metrics for the entire test run.", subtitle_style)
        ]
        
        header_table = Table(
            [[ "", header_text ]],
            colWidths=[4, 536]
        )
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), colors.HexColor('#BDC3C7')), # Vertical line
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (1,0), (1,0), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 24))
        
        # 2. Trends Section
        story.append(Paragraph("Trends", section_style))
        story.append(Spacer(1, 8))
        
        trends_headers = ["metric", "avg", "max", "med", "min", "p90", "p95", "p99"]
        trends_data = [[
            Paragraph(h, cell_header_style if i == 0 else cell_header_center)
            for i, h in enumerate(trends_headers)
        ]]
        
        trend_metrics = ["http_req_duration", "iteration_duration"]
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
            ('LINEABOVE', (0,0), (-1,0), 1, colors.HexColor('#E0E0E0')),
            ('LINEBELOW', (0,0), (-1,0), 1.5, colors.HexColor('#BDC3C7')),
            ('LINEBELOW', (0,1), (-1,-1), 0.5, colors.HexColor('#E0E0E0')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
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
        rates_data.append([
            Paragraph("checks_succeeded", cell_data_style),
            Paragraph(checks_rate, cell_data_center)
        ])
        
        checks_failed = "-"
        if "checks_failed" in self.metrics:
            checks_failed = MetricParser.parse_rate(self.metrics["checks_failed"])
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
        
        # 4. Thresholds Section (Optional, only if thresholds exist in the metrics)
        thresholds_data = []
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
            
        doc.build(story)
        return self.output_path
