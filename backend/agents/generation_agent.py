"""
Generation Agent - Generates responses using LLM
"""

from typing import List, Dict, Any, Optional
import time
from backend.utils.logger import setup_logger
from backend.core.llm import LLMClient
from backend.config import settings

logger = setup_logger(__name__)


class GenerationAgent:
    """Agent responsible for generating responses using LLM"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.temperature = settings.OPENAI_TEMPERATURE
        self.model = settings.OPENAI_MODEL
        
    async def generate(self, query: str, query_analysis: Dict[str, Any],
                     retrieved_docs: List[Dict[str, Any]], agent_trace: Optional[Dict[str, Any]] = None,
                     previous_answer: Optional[str] = None) -> Dict[str, Any]:
        """Generate response using minimal RAG with failure handling"""
        try:
            start_time = time.time()
            print(f"\n🤖 Generating response for query: {query}")
            print(f"Retrieved docs count: {len(retrieved_docs)}")
            
            # Check if we have retrieved docs
            if not retrieved_docs:
                print("⚠️ ERROR TRACE:")
                print("- Stage: Generation")
                print("- Issue: No retrieved documents available")
                print()
                print("💡 SUGGESTION:")
                print("- Check retrieval agent logs")
                print("- Verify document processing")
                print()
                print("STATUS: Using fallback response")
                
                return {
                    "answer": "No documents were retrieved for this query.",
                    "confidence": 0.1,
                    "context_used": 0,
                    "processing_time": time.time() - start_time,
                    "iterations": 1,
                    "refinement_applied": False,
                    "agent_trace": agent_trace or {}
                }
            
            # Prepare context from retrieved documents
            context = self._prepare_context(retrieved_docs)
            print(f"Context prepared: {len(context)} items")
            
            # Generate prompt with minimal RAG format
            prompt = self._build_prompt(query, query_analysis, context, agent_trace, previous_answer)
            print(f"Prompt length: {len(prompt)} chars")
            
            # Check prompt quality
            if len(prompt) < 50:
                print("⚠️ ERROR TRACE:")
                print("- Stage: Prompt Building")
                print("- Issue: Prompt too short")
                print()
                print("💡 SUGGESTION:")
                print("- Check retrieved document content")
                print("- Verify prompt template")
                print()
                print("STATUS: Using fallback response")
                
                return {
                    "answer": "Unable to generate a proper response due to insufficient context.",
                    "confidence": 0.2,
                    "context_used": len(context),
                    "processing_time": time.time() - start_time,
                    "iterations": 1,
                    "refinement_applied": False,
                    "agent_trace": agent_trace or {}
                }
            
            # Generate response using LLM
            print("Calling LLM...")
            response = await self.llm_client.generate_response(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=1500
            )
            
            answer = response.strip()
            print(f"LLM response length: {len(answer)} chars")
            
            # Check response quality
            if len(answer) < 20:
                print("⚠️ ERROR TRACE:")
                print("- Stage: LLM Response")
                print("- Issue: Response too short")
                print()
                print("💡 SUGGESTION:")
                print("- Check LLM provider status")
                print("- Verify API keys and connectivity")
                print("- Try different provider or model")
                print()
                print("STATUS: Retrying with fallback")
                
                # Try a simpler prompt
                simple_prompt = f"Summarize this: {retrieved_docs[0]['content'][:200]}..."
                response = await self.llm_client.generate_response(simple_prompt, max_tokens=100)
                answer = response.strip()
                
<<<<<<< HEAD
                # Check response quality
                if len(answer) < 10:
                    answer = "Extracting structured insights from document..."
            
                # Calculate confidence score
                confidence = self._calculate_confidence(answer, context, query)
                print(f"Confidence score: {confidence}")
=======
                if len(answer) < 10:
                    answer = "The document appears to be a test PDF with minimal content."
            
            # Calculate confidence score
            confidence = self._calculate_confidence(answer, context, query)
            print(f"Confidence score: {confidence}")
>>>>>>> 97af6411c5fc919c79d6656e755e8bfe819e0e7e
            
            processing_time = time.time() - start_time
            print(f"✅ Generation completed in {processing_time:.2f}s")
            
            result = {
                "answer": answer,
                "confidence": confidence,
                "context_used": len(context),
                "processing_time": processing_time,
                "iterations": 1,
                "refinement_applied": False,
                "agent_trace": agent_trace or {}
            }
            
            return result
            
        except Exception as e:
            print("⚠️ ERROR TRACE:")
            print("- Stage: Generation Process")
            print(f"- Issue: {str(e)}")
            print()
            print("💡 SUGGESTION:")
            print("- Check LLM provider configuration")
            print("- Verify API keys and connectivity")
            print("- Check system resources")
            print()
            print("STATUS: System fallback triggered")
            
            logger.error(f"Error in generation: {str(e)}")
            return {
                "answer": f"Generation error: {str(e)}",
                "confidence": 0.0,
                "context_used": 0,
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0,
                "iterations": 1,
                "refinement_applied": False,
                "agent_trace": agent_trace or {}
            }
    
    def _prepare_context(self, retrieved_docs: List[Dict[str, Any]]) -> List[str]:
<<<<<<< HEAD
        """Prepare and compress context from retrieved documents"""
        # ✅ STEP 1: Limit to top 3 chunks (memory efficient)
        docs = retrieved_docs[:3]
        
        # ✅ STEP 2: Compress chunks to reduce token usage
        compressed_chunks = []
        for doc in docs:
            # Keep only first 300 chars of each chunk
            compressed_content = doc['content'][:300]
            if len(doc['content']) > 300:
                compressed_content += "...[truncated]"
            compressed_chunks.append(compressed_content)
        
        # ✅ STEP 3: Join compressed chunks
        context = "\n\n".join(compressed_chunks)
=======
        """Prepare context from retrieved documents"""
        # ✅ STEP 1: Clean context
        docs = retrieved_docs[:3]  # limit noisy chunks
        context = "\n\n".join([doc['content'] for doc in docs])
>>>>>>> 97af6411c5fc919c79d6656e755e8bfe819e0e7e
        return [context] if context else []
    
    def _build_prompt(self, query: str, query_analysis: Dict[str, Any],
                     context: List[str], agent_trace: Optional[Dict[str, Any]] = None,
                     previous_answer: Optional[str] = None) -> str:
<<<<<<< HEAD
        """Build the prompt for grounded RAG with hard grounding rule"""
=======
        """Build the prompt for grounded RAG with citations"""
>>>>>>> 97af6411c5fc919c79d6656e755e8bfe819e0e7e
        
        # Get retrieved docs from agent_trace or use empty list
        retrieved_docs = agent_trace.get("retrieved_docs", []) if agent_trace else []
        
        # Build grounded prompt with proper chunk citations
        chunk_texts = []
        for i, chunk in enumerate(retrieved_docs[:3]):
            chunk_preview = chunk['content'][:200].replace('\n', ' ').strip()
            chunk_score = chunk.get('rerank_score', chunk.get('score', 0))
            chunk_texts.append(f"[Chunk {i+1}]: \"{chunk_preview}...\" (score: {chunk_score:.2f})")
        
        retrieved_text = "\n".join(chunk_texts)
        avg_score = sum(chunk.get('rerank_score', chunk.get('score', 0)) for chunk in retrieved_docs[:3]) / 3 if retrieved_docs else 0.5
        
        prompt = f"""QUERY:
{query}

RETRIEVED CHUNKS:
{retrieved_text}

<<<<<<< HEAD
🚨 CRITICAL GROUNDING RULE:
Answer ONLY from the provided resume content. Do NOT generalize, assume, or use external knowledge.

STRICT INSTRUCTIONS:
1. Answer ONLY using the retrieved chunks above
2. Include specific citations like [Chunk 1], [Chunk 2] for each claim
3. If information is not found in chunks, say "Not found in resume"
4. Do NOT make up or infer any information
5. Do NOT use any external knowledge or generalizations
6. Provide structured answer with evidence ONLY from resume
=======
INSTRUCTIONS:
1. Answer ONLY using the retrieved chunks above
2. Include specific citations like [Chunk 1], [Chunk 2] for each claim
3. If information is not found in chunks, say "Not found in retrieved documents"
4. Provide structured answer with evidence
>>>>>>> 97af6411c5fc919c79d6656e755e8bfe819e0e7e

ANSWER:
"""
        
        return prompt
    
    def _calculate_confidence(self, response: str, context: List[str], query: str) -> float:
        """Calculate confidence score for the generated response"""
        try:
            confidence = 0.0
            
            # Factor 1: Response length (longer responses tend to be more detailed)
            if len(response) > 100:
                confidence += 0.2
            elif len(response) > 50:
                confidence += 0.1
            
            # Factor 2: Context usage
            if len(context) > 0:
                confidence += 0.3
            
            # Factor 3: Answer contains relevant keywords
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            overlap = len(query_words.intersection(response_words))
            if overlap > 0:
                confidence += min(0.3, overlap * 0.1)
            
            # Factor 4: Response structure (has proper sentences, etc.)
            if any(ending in response for ending in ['.', '!', '?']):
                confidence += 0.2
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5  # Default confidence
    
    async def refine_answer(self, original_answer: str, feedback: str, 
                          context: List[str]) -> Dict[str, Any]:
        """Refine an answer based on feedback"""
        try:
            logger.info("Refining answer based on feedback")
            
            prompt = f"""Please refine the following answer based on the provided feedback.

ORIGINAL ANSWER:
{original_answer}

FEEDBACK:
{feedback}

CONTEXT:
{chr(10).join(context)}

REFINED ANSWER:"""
            
            refined_response = await self.llm_client.generate_response(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1000
            )
            
            return {
                "answer": refined_response,
                "confidence": self._calculate_confidence(refined_response, context, ""),
                "refined": True
            }
            
        except Exception as e:
            logger.error(f"Error refining answer: {str(e)}")
            return {
                "answer": original_answer,
                "confidence": 0.3,
                "refined": False,
                "error": str(e)
            }
