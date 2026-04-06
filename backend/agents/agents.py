"""
Agents - Consolidated Agent Components
All agent classes in one professional file
"""

from typing import Dict, Any, List, Optional
import re
import time
from backend.utils.logger import setup_logger
from backend.core.llm import LLMClient

logger = setup_logger(__name__)


# =============================================================================
# 1. PLANNER AGENT - Decision Maker
# =============================================================================

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
                r"\bmethodology\b", r"\bresults\b", r"\bconclusion\b", r"\bfindings\b"
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
        """Plan the task based on query and document preview"""
        try:
            doc_type = self._detect_document_type(doc_preview)
            
            if doc_type == "resume" or "resume" in query.lower():
                task_type = "resume_analysis"
                strategy = "extract_skills_experience"
            elif doc_type == "research":
                task_type = "research_summary"
                strategy = "abstract_key_findings"
            elif doc_type == "invoice":
                task_type = "invoice_analysis"
                strategy = "extract_amounts_dates"
            elif doc_type == "legal":
                task_type = "legal_document"
                strategy = "extract_clauses_obligations"
            elif "summarize" in query.lower():
                task_type = "general_summary"
                strategy = "comprehensive_overview"
            else:
                task_type = "general_qa"
                strategy = "contextual_answer"
            
            confidence = 0.8 if doc_type != "unknown" else 0.6
            
            logger.info(f"📋 Plan: {task_type} | Strategy: {strategy}")
            
            return {
                "task_type": task_type,
                "strategy": strategy,
                "confidence": confidence,
                "document_type": doc_type
            }
            
        except Exception as e:
            logger.error(f"Planning error: {e}")
            return {
                "task_type": "general_qa",
                "strategy": "fallback",
                "confidence": 0.3,
                "document_type": "unknown"
            }
    
    def _detect_document_type(self, doc_preview: str) -> str:
        """Detect document type from preview"""
        if not doc_preview:
            return "unknown"
        
        preview_lower = doc_preview.lower()
        
        resume_keywords = ["experience", "education", "skills", "resume", "cv"]
        if sum(1 for kw in resume_keywords if kw in preview_lower) >= 2:
            return "resume"
        
        research_keywords = ["abstract", "methodology", "results", "conclusion"]
        if sum(1 for kw in research_keywords if kw in preview_lower) >= 2:
            return "research"
        
        invoice_keywords = ["invoice", "bill", "amount", "payment", "total"]
        if sum(1 for kw in invoice_keywords if kw in preview_lower) >= 2:
            return "invoice"
        
        legal_keywords = ["agreement", "contract", "clause", "terms", "conditions"]
        if sum(1 for kw in legal_keywords if kw in preview_lower) >= 2:
            return "legal"
        
        return "unknown"


# =============================================================================
# 2. REASONING AGENT - LLM Core
# =============================================================================

class ReasoningAgent:
    """Core LLM agent that generates responses based on context and task"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.task_prompts = {
            "resume_analysis": """You are analyzing a resume/CV. Extract and present:
1. Professional Summary (2-3 sentences)
2. Key Points (4-6 bullets): Experience, skills, education, achievements, expertise, leadership

Use ONLY the resume content provided.""",
            
            "research_summary": """You are analyzing a research paper. Extract and present:
1. Research Summary (2-3 sentences)
2. Key Points (4-6 bullets): Objective, methodology, findings, conclusions, implications, limitations

Use ONLY the research paper content provided.""",
            
            "invoice_analysis": """You are analyzing an invoice/bill. Extract and present:
1. Invoice Summary (2-3 sentences)
2. Key Points (4-6 bullets): Invoice number, vendor, items, total amount, payment terms, due date

Use ONLY the invoice content provided.""",
            
            "legal_document": """You are analyzing a legal document. Extract and present:
1. Document Summary (2-3 sentences)
2. Key Points (4-6 bullets): Document type, parties, obligations, terms, duration, clauses

Use ONLY the legal document content provided.""",
            
            "general_summary": """You are analyzing a document. Extract and present:
1. Document Summary (2-3 sentences)
2. Key Points (4-6 bullets): Purpose, important info, key details, features, context, implications

Use ONLY the document content provided."""
        }
    
    async def reason(self, context: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response based on context and task"""
        try:
            start_time = time.time()
            
            task_type = task.get("task_type", "general_summary")
            base_prompt = self.task_prompts.get(task_type, self.task_prompts["general_summary"])
            
            prompt = f"""{base_prompt}

DOCUMENT CONTENT:
{context[:4000]}

Return your response in this format:

📄 Summary:
<2-3 sentence summary>

📌 Key Points:
• <Point 1>
• <Point 2>
• <Point 3>
• <Point 4>
• <Point 5>
• <Point 6>"""
            
            logger.info(f"🧠 Generating response for: {task_type}")
            response = await self.llm_client.generate_response(
                prompt=prompt, temperature=0.3, max_tokens=1500
            )
            
            return {
                "answer": response.strip(),
                "task_type": task_type,
                "confidence": task.get("confidence", 0.5),
                "processing_time": time.time() - start_time,
                "agent_trace": {"agent": "reasoning", "task": task_type}
            }
            
        except Exception as e:
            logger.error(f"Reasoning error: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "task_type": task.get("task_type", "unknown"),
                "confidence": 0.0,
                "error": str(e)
            }


# =============================================================================
# 3. CRITIC AGENT - Self-Evaluation
# =============================================================================

class CriticAgent:
    """Self-checks responses for quality - Forces regeneration for weak responses"""
    
    def __init__(self):
        self.generic_phrases = [
            "i don't have", "no information", "not mentioned", "not specified",
            "cannot determine", "unable to", "no specific", "not provided",
            "no details", "generic response", "template"
        ]
        self.min_response_length = 120
    
    def critique(self, answer: str, context: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Critique and validate the generated response"""
        try:
            score = 0.0
            issues = []
            
            # Immediate check: Generic or short
            if "generic" in answer.lower() or len(answer) < 120:
                return {
                    "is_valid": False,
                    "score": 0.0,
                    "issues": ["Generic/short response"],
                    "force_regenerate": True
                }
            
            # Check for generic phrases
            generic_count = 0
            for phrase in self.generic_phrases:
                if phrase.lower() in answer.lower():
                    generic_count += 1
                    issues.append(f"Generic: '{phrase}'")
            
            if generic_count == 0:
                score += 0.3
            
            # Check length
            if len(answer) >= self.min_response_length:
                score += 0.2
            else:
                issues.append(f"Too short: {len(answer)} chars")
            
            # Check structure
            has_summary = "Summary:" in answer or "📄" in answer
            has_points = "Key Points:" in answer or "•" in answer or "📌" in answer
            
            if has_summary and has_points:
                score += 0.2
            
            # Check grounding
            context_words = set(context.lower().split()[:200])
            answer_words = set(answer.lower().split())
            overlap = len(context_words.intersection(answer_words))
            
            if overlap > 10:
                score += 0.1
            
            is_valid = score >= 0.6
            
            if not is_valid:
                logger.warning(f"🔍 Critic: {issues}")
            else:
                logger.info("🔍 Critic: Validated ✓")
            
            return {
                "is_valid": is_valid,
                "score": score,
                "issues": issues,
                "force_regenerate": not is_valid
            }
            
        except Exception as e:
            logger.error(f"Critic error: {e}")
            return {"is_valid": False, "score": 0.0, "issues": [str(e)], "force_regenerate": True}


# =============================================================================
# 4. RETRY AGENT - Autonomy
# =============================================================================

class RetryAgent:
    """Handles intelligent retries with different strategies"""
    
    def __init__(self):
        self.reasoning_agent = ReasoningAgent()
        self.retry_strategies = ["broader_query", "full_document", "different_prompt", "minimal_summary"]
        self.max_retries = 2
    
    async def retry_if_needed(
        self, 
        is_valid: bool, 
        original_query: str,
        context: str, 
        task: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]],
        full_text: Optional[str] = None,
        force_regenerate: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Retry with different strategies if response is invalid"""
        
        if is_valid and not force_regenerate:
            return None
        
        logger.info("🔄 Starting retry...")
        
        # Check for weak context
        if self._is_weak_context(context, retrieved_docs) and full_text:
            context = full_text[:4000]
        
        for attempt in range(self.max_retries):
            strategy = self.retry_strategies[attempt]
            logger.info(f"🔄 Retry {attempt + 1}: {strategy}")
            
            try:
                if strategy == "broader_query":
                    new_context = self._retry_with_broader_query(retrieved_docs)
                elif strategy == "full_document" and full_text:
                    new_context = full_text[:4000]
                else:
                    new_context = context[:2000]
                    task = {"task_type": "general_summary", "strategy": "minimal"}
                
                response = await self.reasoning_agent.reason(new_context, task)
                response["agent_trace"]["retry_attempt"] = attempt + 1
                response["agent_trace"]["retry_strategy"] = strategy
                
                return response
                
            except Exception as e:
                logger.error(f"🔄 Retry {attempt + 1} failed: {e}")
                continue
        
        logger.error("🔄 All retries failed")
        return None
    
    def _is_weak_context(self, context: str, retrieved_docs: List[Dict[str, Any]]) -> bool:
        """Check if context is weak"""
        if len(context) < 500 or len(retrieved_docs) < 3:
            return True
        return False
    
    def _retry_with_broader_query(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Use more documents"""
        more_docs = retrieved_docs[:10] if len(retrieved_docs) > 6 else retrieved_docs
        return "\n\n".join([doc.get('content', '') for doc in more_docs])


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PlannerAgent",
    "ReasoningAgent", 
    "CriticAgent",
    "RetryAgent"
]
