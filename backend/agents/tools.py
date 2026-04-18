"""
PRO Agentic Tools - Calculator, Search, Web, etc.
Tools that the agent can use to enhance responses.
"""

import json
import re
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass
from datetime import datetime
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Tool:
    """Tool definition for agentic system"""
    name: str
    description: str
    func: Callable[[str], str]
    parameters: Dict[str, Any] = None

    def execute(self, input_str: str) -> str:
        """Execute the tool with given input"""
        try:
            result = self.func(input_str)
            return str(result)
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {e}")
            return f"Error using {self.name}: {str(e)}"


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

def calculator_tool(expression: str) -> str:
    """
    Safe calculator for math operations.
    Only allows basic math operations.
    """
    # Security: Only allow safe characters
    allowed_chars = set('0123456789+-*/.() ')
    if not all(c in allowed_chars for c in expression):
        return "Error: Invalid characters in expression. Only numbers and + - * / . () allowed."
    
    try:
        # Use eval with restricted globals/locals for safety
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


def date_time_tool(_: str = "") -> str:
    """Get current date and time"""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def word_count_tool(text: str) -> str:
    """Count words in text"""
    words = len(text.split())
    chars = len(text)
    return f"Word count: {words}, Character count: {chars}"


def summarize_text_tool(text: str) -> str:
    """Simple extractive summary (first and last sentences)"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 2:
        return text
    
    # Return first and last sentences as summary
    summary = f"{sentences[0]}. {sentences[-1]}."
    return f"Quick summary: {summary}"


# Placeholder for web search (user can implement with SerpAPI, etc.)
def web_search_tool(query: str) -> str:
    """
    Web search tool - placeholder.
    To implement: Add SerpAPI, Bing API, or similar.
    """
    return f"[Web search not configured. Query was: {query}]"


# =============================================================================
# AVAILABLE TOOLS REGISTRY
# =============================================================================

AVAILABLE_TOOLS: List[Tool] = [
    Tool(
        name="calculator",
        description="Performs mathematical calculations. Input: math expression like '2 + 2' or '10 * 5'",
        func=calculator_tool
    ),
    Tool(
        name="date_time",
        description="Gets current date and time. Input: any text or empty string",
        func=date_time_tool
    ),
    Tool(
        name="word_count",
        description="Counts words and characters in text. Input: text to count",
        func=word_count_tool
    ),
    Tool(
        name="summarize",
        description="Creates a quick summary of text. Input: long text to summarize",
        func=summarize_text_tool
    ),
    Tool(
        name="web_search",
        description="Searches the web for information. Input: search query",
        func=web_search_tool
    ),
]


def get_tool_by_name(name: str) -> Optional[Tool]:
    """Get a tool by its name"""
    for tool in AVAILABLE_TOOLS:
        if tool.name.lower() == name.lower():
            return tool
    return None


def get_tools_description() -> str:
    """Get formatted description of all tools for LLM prompt"""
    lines = []
    for tool in AVAILABLE_TOOLS:
        lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines)


# =============================================================================
# TOOL SELECTION (LLM-Powered)
# =============================================================================

def create_tool_selection_prompt(query: str) -> str:
    """Create prompt for LLM to decide which tool to use"""
    return f"""You are an AI assistant with access to tools.

Available tools:
{get_tools_description()}

User query: "{query}"

Analyze the query and decide:
1. Should a tool be used? (YES or NO)
2. Which tool? (tool name or NONE)
3. What input to give the tool?

Respond ONLY with JSON in this format:
{{"use_tool": true/false, "tool_name": "tool_name_or_NONE", "tool_input": "input_for_tool"}}

Rules:
- Use calculator for math expressions (2+2, 10*5, etc.)
- Use date_time for current time questions
- Use word_count when asked about document length
- Use summarize when asked to condense text
- Use web_search for current events or general knowledge NOT in documents
- Use NONE if the query can be answered from document context alone"""


def parse_tool_selection(response_text: str) -> Dict[str, Any]:
    """Parse LLM tool selection response safely"""
    try:
        # Extract JSON from response (handle markdown code blocks)
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(text)
        
        # Validate structure
        return {
            "use_tool": result.get("use_tool", False),
            "tool_name": result.get("tool_name", "NONE"),
            "tool_input": result.get("tool_input", "")
        }
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse tool selection JSON: {e}")
        return {"use_tool": False, "tool_name": "NONE", "tool_input": ""}
    except Exception as e:
        logger.error(f"Unexpected error parsing tool selection: {e}")
        return {"use_tool": False, "tool_name": "NONE", "tool_input": ""}


__all__ = [
    "Tool",
    "AVAILABLE_TOOLS",
    "get_tool_by_name",
    "get_tools_description",
    "create_tool_selection_prompt",
    "parse_tool_selection",
    # Individual tools
    "calculator_tool",
    "date_time_tool",
    "word_count_tool",
    "summarize_text_tool",
    "web_search_tool",
]
