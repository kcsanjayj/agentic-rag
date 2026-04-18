"""
PRO Agentic Loop - Multi-step reasoning with tools and memory
This is where the system becomes truly agentic (not just RAG).
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from openai import OpenAI
from backend.agents.tools import (
    AVAILABLE_TOOLS, get_tool_by_name, create_tool_selection_prompt,
    parse_tool_selection
)
from backend.agents.memory import ConversationMemory
from backend.core.embeddings import EmbeddingGenerator
from backend.core.vector_store import get_vector_store
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class AgentStep:
    """Single step in agent loop"""
    step_number: int
    action: str  # "tool_call" or "retrieve" or "answer"
    input_data: str
    output_data: str
    latency_ms: float = 0


@dataclass
class AgentResult:
    """Final result from agentic loop"""
    answer: str
    steps: List[AgentStep]
    tool_calls: List[Dict[str, Any]]
    retrieved_docs: int
    confidence: float


class AgenticRAG:
    """
    PRO Agentic RAG with:
    - Multi-step reasoning loop
    - Tool usage (calculator, search, etc.)
    - Memory injection
    - Adaptive retrieval
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.embedding_gen = EmbeddingGenerator(api_key)
        self.vector_store = get_vector_store()
        self.max_steps = 3
        
        logger.info("AgenticRAG initialized")
    
    async def run(
        self,
        query: str,
        memory: Optional[ConversationMemory] = None,
        document_id: Optional[str] = None,
        enable_tools: bool = True
    ) -> AgentResult:
        """
        Run agentic loop with tools, memory, and retrieval.
        
        Args:
            query: User query
            memory: Conversation memory (optional)
            document_id: Active document ID for retrieval
            enable_tools: Whether to use tools
            
        Returns:
            AgentResult with answer and metadata
        """
        import time
        start_time = time.time()
        
        steps: List[AgentStep] = []
        tool_calls: List[Dict[str, Any]] = []
        
        # Build context from memory
        memory_context = ""
        if memory:
            memory_context = memory.get_context_for_prompt()
        
        current_query = query
        
        # === AGENT LOOP ===
        for step_num in range(self.max_steps):
            step_start = time.time()
            
            # 1. PRO: Decide tool and prepare retrieval in parallel (if safe)
            tool_decision = None
            retrieval_task = None
            
            if enable_tools:
                tool_decision = await self._select_tool(current_query)
            
            # 2. PRO: Execute tool and retrieve in parallel when possible
            if tool_decision and tool_decision["use_tool"]:
                tool_name = tool_decision["tool_name"]
                tool_input = tool_decision["tool_input"]
                
                # Execute tool
                tool_result = await self._execute_tool(tool_name, tool_input)
                tool_calls.append({
                    "step": step_num + 1,
                    "tool": tool_name,
                    "input": tool_input,
                    "result": tool_result
                })
                
                # Update query with tool result
                current_query = f"""Original question: {query}

Tool used: {tool_name}
Tool result: {tool_result}

Based on this, answer the original question."""
                
                steps.append(AgentStep(
                    step_number=step_num + 1,
                    action=f"tool:{tool_name}",
                    input_data=tool_input,
                    output_data=tool_result[:200],
                    latency_ms=(time.time() - step_start) * 1000
                ))
                # Continue loop - might need another tool
                continue
            
            # No tool needed - proceed to retrieval
            break
        
        # 2. Retrieve relevant documents
        retrieve_start = time.time()
        
        if document_id:
            # Generate embedding for current query
            query_embedding = await self.embedding_gen.generate_query_embedding(current_query)
            
            # Retrieve from vector store
            docs = await self.vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=6,
                threshold=0.0,
                filter_dict={"document_id": document_id}
            )
        else:
            docs = []
        
        context = "\n\n".join([d.get('content', '') for d in docs[:4]])
        
        steps.append(AgentStep(
            step_number=len(steps) + 1,
            action="retrieve",
            input_data=current_query[:100],
            output_data=f"Retrieved {len(docs)} documents",
            latency_ms=(time.time() - retrieve_start) * 1000
        ))
        
        # 3. Generate final answer with LLM
        final_prompt = self._create_final_prompt(
            query=query,
            current_query=current_query,
            context=context,
            memory_context=memory_context,
            tool_calls=tool_calls
        )
        
        llm_start = time.time()
        answer = await self._generate_with_llm(final_prompt)
        
        steps.append(AgentStep(
            step_number=len(steps) + 1,
            action="llm_generate",
            input_data=final_prompt[:200],
            output_data=answer[:200],
            latency_ms=(time.time() - llm_start) * 1000
        ))
        
        # Calculate confidence based on tool usage and docs
        confidence = self._calculate_confidence(len(docs), tool_calls, answer)
        
        total_time = time.time() - start_time
        logger.info(f"Agentic loop completed in {total_time:.2f}s ({len(steps)} steps, {len(docs)} docs)")
        
        return AgentResult(
            answer=answer,
            steps=steps,
            tool_calls=tool_calls,
            retrieved_docs=len(docs),
            confidence=confidence
        )
    
    async def _select_tool(self, query: str) -> Dict[str, Any]:
        """Use LLM to decide if a tool should be used"""
        try:
            prompt = create_tool_selection_prompt(query)
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=150
                )
            )
            
            response_text = response.choices[0].message.content
            decision = parse_tool_selection(response_text)
            
            logger.debug(f"Tool decision: {decision}")
            return decision
            
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            return {"use_tool": False, "tool_name": "NONE", "tool_input": ""}
    
    async def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool by name"""
        tool = get_tool_by_name(tool_name)
        
        if not tool:
            return f"Error: Tool '{tool_name}' not found"
        
        # Run tool in executor to keep it async
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, tool.execute, tool_input)
        
        logger.info(f"Tool '{tool_name}' executed: {result[:100]}...")
        return result
    
    def _create_final_prompt(
        self,
        query: str,
        current_query: str,
        context: str,
        memory_context: str,
        tool_calls: List[Dict]
    ) -> str:
        """Create final prompt for LLM"""
        
        tool_section = ""
        if tool_calls:
            tool_section = "Tools used:\n"
            for tc in tool_calls:
                tool_section += f"- {tc['tool']}: {tc['result'][:150]}\n"
            tool_section += "\n"
        
        memory_section = f"{memory_context}\n\n" if memory_context else ""
        
        return f"""You are a helpful AI assistant with access to documents and tools.

{memory_section}Document context:
{context}

{tool_section}Original question:
{query}

Current query (may include tool results):
{current_query}

Instructions:
1. Answer based primarily on the document context provided
2. Use tool results to supplement your answer if helpful
3. Be specific and cite information from the documents
4. If the answer isn't in the documents, say so clearly
5. Format your response with clear sections

Answer:"""
    
    async def _generate_with_llm(self, prompt: str) -> str:
        """Generate response using LLM"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1500
                )
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"I apologize, but I encountered an error generating the response: {str(e)}"
    
    def _calculate_confidence(
        self,
        num_docs: int,
        tool_calls: List[Dict],
        answer: str
    ) -> float:
        """Calculate confidence score (0-1)"""
        score = 0.5  # Base score
        
        # More docs = higher confidence (up to a point)
        if num_docs >= 3:
            score += 0.2
        elif num_docs >= 1:
            score += 0.1
        
        # Tool usage adds confidence
        if tool_calls:
            score += min(0.15, len(tool_calls) * 0.05)
        
        # Answer length as weak signal
        if len(answer) > 200:
            score += 0.1
        
        # Generic phrases reduce confidence
        generic_phrases = ["i don't know", "not found", "no information", "unclear"]
        if any(phrase in answer.lower() for phrase in generic_phrases):
            score -= 0.2
        
        return max(0.1, min(1.0, score))


__all__ = [
    "AgenticRAG",
    "AgentResult",
    "AgentStep",
]
