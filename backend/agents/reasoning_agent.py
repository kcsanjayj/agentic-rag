"""
Reasoning Agent - Core LLM generation with task-aware prompting
"""

from typing import Dict, Any, List
import time
from backend.utils.logger import setup_logger
from backend.core.llm import LLMClient
from backend.config import settings

logger = setup_logger(__name__)


class ReasoningAgent:
    """Core LLM agent that generates responses based on context and task"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.task_prompts = {
            "resume_analysis": """
You are analyzing a resume/CV. Extract and present:
1. Professional Summary (2-3 sentences)
2. Key Points (4-6 bullets):
   - Experience and skills
   - Education background
   - Notable achievements
   - Career progression
   - Technical expertise
   - Leadership roles

Use ONLY the resume content provided.
""",
            "research_summary": """
You are analyzing a research paper. Extract and present:
1. Research Summary (2-3 sentences)
2. Key Points (4-6 bullets):
   - Research objective/question
   - Methodology used
   - Main findings
   - Conclusions
   - Implications
   - Limitations

Use ONLY the research paper content provided.
""",
            "invoice_analysis": """
You are analyzing an invoice/bill. Extract and present:
1. Invoice Summary (2-3 sentences)
2. Key Points (4-6 bullets):
   - Invoice number and date
   - Vendor/seller information
   - Items or services charged
   - Total amount due
   - Payment terms
   - Due date

Use ONLY the invoice content provided.
""",
            "legal_document": """
You are analyzing a legal document. Extract and present:
1. Document Summary (2-3 sentences)
2. Key Points (4-6 bullets):
   - Document type and parties
   - Key obligations
   - Important terms
   - Duration/termination
   - Key dates
   - Notable clauses

Use ONLY the legal document content provided.
""",
            "general_summary": """
You are analyzing a document. Extract and present:
1. Document Summary (2-3 sentences)
2. Key Points (4-6 bullets):
   - Main purpose/topic
   - Important information
   - Key details
   - Notable features
   - Context/background
   - Implications

Use ONLY the document content provided.
"""
        }
    
    async def reason(self, context: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate response based on context and task
        
        Args:
            context: Retrieved document content
            task: Task information from planner
            
        Returns:
            Generated response with metadata
        """
        try:
            start_time = time.time()
            
            # Get task-specific prompt
            task_type = task.get("task_type", "general_summary")
            base_prompt = self.task_prompts.get(task_type, self.task_prompts["general_summary"])
            
            # Build complete prompt
            prompt = f"""You are an intelligent document assistant.

{base_prompt}

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
            
            # Generate response
            logger.info(f"🧠 Reasoning agent generating response for task: {task_type}")
            response = await self.llm_client.generate_response(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1500
            )
            
            processing_time = time.time() - start_time
            
            return {
                "answer": response.strip(),
                "task_type": task_type,
                "confidence": task.get("confidence", 0.5),
                "processing_time": processing_time,
                "context_length": len(context),
                "agent_trace": {
                    "agent": "reasoning",
                    "task": task_type,
                    "strategy": task.get("strategy", "unknown"),
                    "reasoning": task.get("reasoning", "")
                }
            }
            
        except Exception as e:
            logger.error(f"Error in reasoning agent: {str(e)}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "task_type": task.get("task_type", "unknown"),
                "confidence": 0.0,
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0,
                "error": str(e)
            }
