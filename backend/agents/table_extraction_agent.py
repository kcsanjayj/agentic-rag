"""
Table Extraction Agent - Extracts structured data from tables in documents
Handles table detection, extraction, and formatting
"""

from typing import Dict, Any, List, Optional
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class TableExtractionAgent:
    """Extracts and processes table data from documents"""
    
    def __init__(self):
        self.use_mock = True  # Will use mock unless table extraction model is available
        logger.info("Table Extraction Agent initialized (mock mode)")
    
    def extract_tables(self, document_text: str, format_type: str = "markdown") -> List[Dict[str, Any]]:
        """
        Extract tables from document text
        
        Args:
            document_text: The document content
            format_type: Output format (markdown, json, csv)
            
        Returns:
            List of extracted tables with data
        """
        try:
            if self.use_mock:
                return self._mock_extract(document_text, format_type)
            else:
                return self._model_extract(document_text, format_type)
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
            return []
    
    def _mock_extract(self, document_text: str, format_type: str) -> List[Dict[str, Any]]:
        """Mock table extraction - returns placeholder table"""
        return [
            {
                "table_id": 1,
                "headers": ["Column 1", "Column 2", "Column 3"],
                "rows": [
                    ["Data 1", "Data 2", "Data 3"],
                    ["Data 4", "Data 5", "Data 6"]
                ],
                "format": format_type,
                "confidence": 0.8
            }
        ]
    
    def _model_extract(self, document_text: str, format_type: str) -> List[Dict[str, Any]]:
        """Use actual table extraction model"""
        try:
            from transformers import pipeline
            
            # Use table extraction model
            extractor = pipeline("table-question-answering", model="google/tapas-base-finetuned-wikisql-supervised")
            
            # For now, return mock as table extraction requires specific format
            logger.warning("Table extraction model requires specific table structure, using mock")
            self.use_mock = True
            return self._mock_extract(document_text, format_type)
            
        except Exception as e:
            logger.warning(f"Table extraction model not available, using mock: {str(e)}")
            self.use_mock = True
            return self._mock_extract(document_text, format_type)
    
    def parse_csv_table(self, csv_text: str) -> Dict[str, Any]:
        """
        Parse CSV table data
        
        Args:
            csv_text: CSV formatted text
            
        Returns:
            Parsed table data
        """
        try:
            import io
            import csv
            
            reader = csv.reader(io.StringIO(csv_text))
            headers = next(reader, [])
            rows = list(reader)
            
            return {
                "headers": headers,
                "rows": rows,
                "row_count": len(rows),
                "column_count": len(headers)
            }
        except Exception as e:
            logger.error(f"Error parsing CSV: {str(e)}")
            return {"headers": [], "rows": [], "row_count": 0, "column_count": 0}
    
    def convert_table_format(self, table_data: Dict[str, Any], target_format: str) -> str:
        """
        Convert table data to different formats
        
        Args:
            table_data: Table data with headers and rows
            target_format: Target format (markdown, csv, json, html)
            
        Returns:
            Formatted table string
        """
        try:
            headers = table_data.get("headers", [])
            rows = table_data.get("rows", [])
            
            if target_format == "markdown":
                return self._to_markdown(headers, rows)
            elif target_format == "csv":
                return self._to_csv(headers, rows)
            elif target_format == "json":
                return self._to_json(headers, rows)
            elif target_format == "html":
                return self._to_html(headers, rows)
            else:
                return str(table_data)
        except Exception as e:
            logger.error(f"Error converting table format: {str(e)}")
            return str(table_data)
    
    def _to_markdown(self, headers: List[str], rows: List[List[str]]) -> str:
        """Convert to markdown table format"""
        md = "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            md += "| " + " | ".join(str(cell) for cell in row) + " |\n"
        return md
    
    def _to_csv(self, headers: List[str], rows: List[List[str]]) -> str:
        """Convert to CSV format"""
        import io
        import csv
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)
        return output.getvalue()
    
    def _to_json(self, headers: List[str], rows: List[List[str]]) -> str:
        """Convert to JSON format"""
        import json
        
        data = [dict(zip(headers, row)) for row in rows]
        return json.dumps(data, indent=2)
    
    def _to_html(self, headers: List[str], rows: List[List[str]]) -> str:
        """Convert to HTML table format"""
        html = "<table>\n<thead>\n<tr>"
        for header in headers:
            html += f"<th>{header}</th>"
        html += "</tr>\n</thead>\n<tbody>\n"
        for row in rows:
            html += "<tr>"
            for cell in row:
                html += f"<td>{cell}</td>"
            html += "</tr>\n"
        html += "</tbody>\n</table>"
        return html
    
    def is_available(self) -> bool:
        """Check if table extraction model is available"""
        return not self.use_mock
