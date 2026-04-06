"""
Planner Agent - Decides what type of analysis to perform
"""

from typing import Dict, Any
import re
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class PlannerAgent:
    """Brain of the agentic system - decides the task based on document and query"""
    
    def __init__(self):
        self.task_patterns = {
            "resume_analysis": [
                r"\bresume\b", r"\bcv\b", r"\bcurriculum\b", r"\bexperience\b",
                r"\bskills\b", r"\beducation\b", r"\bqualification\b", r"\bjob\b"
            ],
            "research_summary": [
                r"\bresearch\b", r"\bpaper\b", r"\bstudy\b", r"\babstract\b",
                r"\bmethodology\b", r"\bresults\b", r"\bconclusion\b", r"\bf findings\b"
            ],
            "invoice_analysis": [
                r"\binvoice\b", r"\bbill\b", r"\breceipt\b", r"\bpayment\b",
                r"\bamount\b", r"\bdue\b", r"\bcharge\b", r"\btotal\b"
            ],
            "legal_document": [
                r"\bagreement\b", r"\bcontract\b", r"\blegal\b", r"\blaw\b",
                r"\bterms\b", r"\bconditions\b", r"\bclause\b", r"\bsection\b"
            ]
        }
    
    def plan(self, query: str, doc_preview: str) -> Dict[str, Any]:
        """
        Plan the task based on query and document preview
        
        Args:
            query: User's query
            doc_preview: Preview of document content
            
        Returns:
            Task plan with type and strategy
        """
        try:
            # 🔥 PLANNER: Intelligent document type detection
            doc_type = self._detect_document_type(doc_preview)
            
            # Determine task type based on document type and query
            if doc_type == "resume" or "resume" in query.lower():
                task_type = "resume_analysis"
                strategy = "extract_skills_experience"
                reasoning = "Resume detected - focusing on skills, experience, and qualifications"
            elif doc_type == "research" or "research" in query.lower() or "paper" in query.lower():
                task_type = "research_summary"
                strategy = "abstract_key_findings"
                reasoning = "Research paper detected - focusing on abstract, methodology, and key findings"
            elif doc_type == "invoice" or "invoice" in query.lower() or "bill" in query.lower():
                task_type = "invoice_analysis"
                strategy = "extract_amounts_dates"
                reasoning = "Invoice detected - focusing on amounts, dates, and line items"
            elif doc_type == "legal" or "legal" in query.lower() or "contract" in query.lower():
                task_type = "legal_document"
                strategy = "extract_clauses_obligations"
                reasoning = "Legal document detected - focusing on clauses, obligations, and key terms"
            elif "summarize" in query.lower() or "summary" in query.lower():
                task_type = "general_summary"
                strategy = "comprehensive_overview"
                reasoning = "Summary request - providing comprehensive overview"
            else:
                task_type = "general_qa"
                strategy = "contextual_answer"
                reasoning = "General query - providing contextual answer"
            
            # Add confidence based on clarity
            confidence = 0.8 if doc_type != "unknown" else 0.6
            
            logger.info(f"📋 Plan: {task_type} | Strategy: {strategy} | Doc Type: {doc_type}")
            
            return {
                "task_type": task_type,
                "strategy": strategy,
                "reasoning": reasoning,
                "confidence": confidence,
                "document_type": doc_type,
                "query_type": self._detect_query_type(query)
            }
            
        except Exception as e:
            logger.error(f"Error in planning: {str(e)}")
            return {
                "task_type": "general_qa",
                "strategy": "fallback",
                "reasoning": "Planning failed - using fallback strategy",
                "confidence": 0.3,
                "document_type": "unknown",
                "query_type": "unknown"
            }
    
    def _detect_document_type(self, doc_preview: str) -> str:
        """Detect document type from preview"""
        if not doc_preview:
            return "unknown"
        
        preview_lower = doc_preview.lower()
        
        # Resume indicators
        resume_keywords = ["experience", "education", "skills", "resume", "cv", "curriculum vitae", "employment", "work history"]
        if sum(1 for kw in resume_keywords if kw in preview_lower) >= 2:
            return "resume"
        
        # Research paper indicators
        research_keywords = ["abstract", "methodology", "results", "conclusion", "references", "doi", "journal", "conference"]
        if sum(1 for kw in research_keywords if kw in preview_lower) >= 2:
            return "research"
        
        # Invoice indicators
        invoice_keywords = ["invoice", "bill", "amount", "due", "payment", "total", "item", "quantity", "price"]
        if sum(1 for kw in invoice_keywords if kw in preview_lower) >= 2:
            return "invoice"
        
        # Legal document indicators
        legal_keywords = ["agreement", "contract", "clause", "party", "obligation", "liability", "terms", "conditions"]
        if sum(1 for kw in legal_keywords if kw in preview_lower) >= 2:
            return "legal"
        
        return "unknown"
    
    def _detect_query_type(self, query: str) -> str:
        """Detect query type"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "who", "where", "when", "how"]):
            return "question"
        elif any(word in query_lower for word in ["summarize", "summary", "overview"]):
            return "summary"
        elif any(word in query_lower for word in ["list", "extract", "find"]):
            return "extraction"
        else:
            return "general"
    
    def _get_strategy(self, task_type: str) -> str:
        """Get the strategy for each task type"""
        strategies = {
            "resume_analysis": "extract_sections",
            "research_summary": "extract_findings",
            "invoice_analysis": "extract_financial",
            "legal_document": "extract_clauses",
            "general_summary": "comprehensive_summary"
        }
        return strategies.get(task_type, "comprehensive_summary")
