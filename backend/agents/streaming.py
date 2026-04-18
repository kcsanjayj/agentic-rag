"""
PRO Streaming Response Handler (Server-Sent Events)
ChatGPT-like streaming for real-time UX
"""

import json
from typing import AsyncGenerator, Dict, Any
from openai import OpenAI
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


async def stream_chat_response(
    client: OpenAI,
    messages: list,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 1500
) -> AsyncGenerator[str, None]:
    """
    Stream LLM response as Server-Sent Events.
    
    Yields:
        SSE formatted strings: "data: {...}\n\n"
    """
    try:
        # Send start event
        yield "data: {\"type\": \"start\"}\n\n"
        
        # Create streaming completion
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        full_content = ""
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_content += content
                
                # Send content chunk
                event_data = json.dumps({"type": "content", "content": content})
                yield f"data: {event_data}\n\n"
        
        # Send completion event with full text
        done_data = json.dumps({
            "type": "done",
            "full_text": full_content,
            "token_count": len(full_content.split())  # Approximate
        })
        yield f"data: {done_data}\n\n"
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_data = json.dumps({"type": "error", "message": str(e)})
        yield f"data: {error_data}\n\n"


async def stream_agentic_response(
    api_key: str,
    query: str,
    context: str,
    tool_results: list = None,
    memory_context: str = ""
) -> AsyncGenerator[str, None]:
    """
    PRO: Full agentic pipeline with streaming.
    Streams tokens while showing tool usage.
    """
    client = OpenAI(api_key=api_key)
    
    # Build messages
    messages = []
    if memory_context:
        messages.append({"role": "system", "content": f"Previous context:\n{memory_context}"})
    
    if tool_results:
        tool_section = "Tools used:\n" + "\n".join([
            f"- {r['tool']}: {r['result'][:100]}" 
            for r in tool_results
        ])
    else:
        tool_section = ""
    
    messages.append({
        "role": "user",
        "content": f"""{tool_section}

Document context:
{context}

Question: {query}

Answer:"""
    })
    
    # Stream the response
    async for event in stream_chat_response(client, messages):
        yield event


def create_sse_headers() -> Dict[str, str]:
    """Create headers for SSE response"""
    return {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"  # Disable nginx buffering
    }


__all__ = [
    "stream_chat_response",
    "stream_agentic_response",
    "create_sse_headers"
]
