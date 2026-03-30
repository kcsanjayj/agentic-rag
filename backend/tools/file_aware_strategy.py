"""
File-Aware Strategy Module - Determines processing strategy based on file type
"""

from typing import Dict, Any, List
from pathlib import Path


class FileAwareStrategy:
    """Determines processing strategy based on file type"""
    
    STRATEGIES = {
        "pdf": {
            "strategy": "text_summary",
            "description": "Compress + classify + insight extraction",
            "chunk_size": 300,
            "chunk_overlap": 75,
            "priority_fields": ["title", "author", "summary", "key_points"]
        },
        "docx": {
            "strategy": "text_summary",
            "description": "Compress + classify + insight extraction",
            "chunk_size": 300,
            "chunk_overlap": 75,
            "priority_fields": ["title", "headings", "summary"]
        },
        "txt": {
            "strategy": "text_summary",
            "description": "General text analysis",
            "chunk_size": 300,
            "chunk_overlap": 75,
            "priority_fields": ["content", "structure"]
        },
        "md": {
            "strategy": "text_summary",
            "description": "Markdown-aware text analysis",
            "chunk_size": 300,
            "chunk_overlap": 75,
            "priority_fields": ["headers", "content", "structure"]
        },
        "csv": {
            "strategy": "table_analysis",
            "description": "Table extraction and analysis",
            "chunk_size": 200,
            "chunk_overlap": 50,
            "priority_fields": ["headers", "rows", "statistics", "patterns"]
        },
        "xlsx": {
            "strategy": "table_analysis",
            "description": "Spreadsheet extraction and analysis",
            "chunk_size": 200,
            "chunk_overlap": 50,
            "priority_fields": ["sheets", "tables", "formulas", "data"]
        },
        "xls": {
            "strategy": "table_analysis",
            "description": "Spreadsheet extraction and analysis",
            "chunk_size": 200,
            "chunk_overlap": 50,
            "priority_fields": ["sheets", "tables", "formulas", "data"]
        },
        "json": {
            "strategy": "structured_parsing",
            "description": "JSON structure analysis",
            "chunk_size": 250,
            "chunk_overlap": 60,
            "priority_fields": ["keys", "values", "structure", "nested_data"]
        },
        "xml": {
            "strategy": "structured_parsing",
            "description": "XML structure analysis",
            "chunk_size": 250,
            "chunk_overlap": 60,
            "priority_fields": ["tags", "attributes", "structure"]
        },
        "yaml": {
            "strategy": "structured_parsing",
            "description": "YAML structure analysis",
            "chunk_size": 250,
            "chunk_overlap": 60,
            "priority_fields": ["keys", "values", "structure"]
        },
        "py": {
            "strategy": "code_explanation",
            "description": "Python code analysis",
            "chunk_size": 200,
            "chunk_overlap": 50,
            "priority_fields": ["functions", "classes", "logic", "comments"]
        },
        "js": {
            "strategy": "code_explanation",
            "description": "JavaScript code analysis",
            "chunk_size": 200,
            "chunk_overlap": 50,
            "priority_fields": ["functions", "classes", "logic", "comments"]
        },
        "java": {
            "strategy": "code_explanation",
            "description": "Java code analysis",
            "chunk_size": 200,
            "chunk_overlap": 50,
            "priority_fields": ["classes", "methods", "logic", "comments"]
        },
        "ts": {
            "strategy": "code_explanation",
            "description": "TypeScript code analysis",
            "chunk_size": 200,
            "chunk_overlap": 50,
            "priority_fields": ["types", "interfaces", "functions", "logic"]
        },
        "html": {
            "strategy": "code_explanation",
            "description": "HTML structure analysis",
            "chunk_size": 200,
            "chunk_overlap": 50,
            "priority_fields": ["tags", "structure", "content"]
        },
        "css": {
            "strategy": "code_explanation",
            "description": "CSS style analysis",
            "chunk_size": 200,
            "chunk_overlap": 50,
            "priority_fields": ["selectors", "properties", "structure"]
        },
        "png": {
            "strategy": "ocr_analysis",
            "description": "OCR text extraction + analysis",
            "chunk_size": 150,
            "chunk_overlap": 40,
            "priority_fields": ["text", "labels", "structure"]
        },
        "jpg": {
            "strategy": "ocr_analysis",
            "description": "OCR text extraction + analysis",
            "chunk_size": 150,
            "chunk_overlap": 40,
            "priority_fields": ["text", "labels", "structure"]
        },
        "jpeg": {
            "strategy": "ocr_analysis",
            "description": "OCR text extraction + analysis",
            "chunk_size": 150,
            "chunk_overlap": 40,
            "priority_fields": ["text", "labels", "structure"]
        },
        "default": {
            "strategy": "general_analysis",
            "description": "General content analysis",
            "chunk_size": 300,
            "chunk_overlap": 75,
            "priority_fields": ["content", "structure"]
        }
    }
    
    @classmethod
    def get_strategy(cls, file_ext: str) -> Dict[str, Any]:
        """Get processing strategy for file type"""
        file_ext = file_ext.lower().lstrip('.')
        return cls.STRATEGIES.get(file_ext, cls.STRATEGIES["default"])
    
    @classmethod
    def get_strategy_from_filename(cls, filename: str) -> Dict[str, Any]:
        """Get processing strategy from filename"""
        ext = Path(filename).suffix.lstrip('.').lower()
        return cls.get_strategy(ext)
    
    @classmethod
    def classify_document(cls, content: str, metadata: Dict[str, Any]) -> str:
        """Classify document type based on content and metadata"""
        content_lower = content.lower()
        
        # Check for sample/test documents
        if len(content) < 1000 and "sample" in content_lower and "test" in content_lower:
            return "Low-information Document"
        
        # Check for technical documentation
        if any(word in content_lower for word in ["documentation", "guide", "manual", "api", "reference"]):
            return "Technical Documentation"
        
        # Check for code
        if any(marker in content_lower for marker in ["def ", "class ", "function", "import ", "# ", "// "]):
            return "Code Document"
        
        # Check for data/tables
        if any(marker in content for marker in [",", "|", "\t"]):
            lines = content.split('\n')
            if len(lines) > 2:
                # Check for consistent separators (CSV-like)
                first_line_commas = lines[0].count(',')
                if first_line_commas > 2 and all(l.count(',') == first_line_commas for l in lines[1:min(5, len(lines))]):
                    return "Data Document"
        
        return "General Document"
    
    @classmethod
    def get_document_value(cls, doc_type: str, content_length: int) -> str:
        """Assess document value for knowledge extraction"""
        if doc_type == "Low-information Document":
            return "low_value"
        elif doc_type == "Technical Documentation" and content_length > 2000:
            return "high_value"
        elif content_length < 500:
            return "low_value"
        else:
            return "medium_value"


class MultiFileReasoning:
    """Handles reasoning across multiple files"""
    
    @classmethod
    def merge_insights(cls, documents: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Merge insights from multiple documents
        
        Returns:
            Dict with combined insights and source tracking
        """
        if not documents:
            return {"merged_content": "", "sources": [], "cross_references": []}
        
        if len(documents) == 1:
            return {
                "merged_content": documents[0].get("content", ""),
                "sources": [documents[0].get("metadata", {}).get("filename", "unknown")],
                "cross_references": [],
                "single_file": True
            }
        
        # Multi-file synthesis
        merged_parts = []
        sources = []
        cross_refs = []
        
        for i, doc in enumerate(documents[:5]):  # Limit to top 5 files
            filename = doc.get("metadata", {}).get("filename", f"File {i+1}")
            content = doc.get("content", "")[:2000]  # Limit content per file
            strategy = doc.get("metadata", {}).get("strategy", "general_analysis")
            
            header = f"\n{'='*50}\nFILE {i+1}: {filename} [{strategy}]\n{'='*50}\n"
            merged_parts.append(header + content)
            sources.append(filename)
            
            # Track cross-references
            if i > 0:
                cross_refs.append({
                    "from": sources[i-1],
                    "to": filename,
                    "relationship": "sequential"
                })
        
        return {
            "merged_content": "\n\n".join(merged_parts),
            "sources": sources,
            "cross_references": cross_refs,
            "single_file": False,
            "file_count": len(documents)
        }
    
    @classmethod
    def detect_conflicts(cls, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect potential conflicts between documents"""
        conflicts = []
        
        if len(documents) < 2:
            return conflicts
        
        # Simple conflict detection based on key terms
        for i, doc1 in enumerate(documents):
            for doc2 in documents[i+1:]:
                content1 = set(doc1.get("content", "").lower().split())
                content2 = set(doc2.get("content", "").lower().split())
                
                # Check for overlapping but different information
                overlap = content1.intersection(content2)
                unique1 = content1 - content2
                unique2 = content2 - content1
                
                if len(overlap) > 10 and (len(unique1) > 50 or len(unique2) > 50):
                    conflicts.append({
                        "files": [
                            doc1.get("metadata", {}).get("filename", "unknown"),
                            doc2.get("metadata", {}).get("filename", "unknown")
                        ],
                        "type": "content_divergence",
                        "severity": "low"
                    })
        
        return conflicts
