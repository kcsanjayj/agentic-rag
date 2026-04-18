"""
PRO Tool Calling with OpenAI Function Calling API
Structured tool execution - no more JSON parsing headaches
"""

import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from openai import OpenAI
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ToolDefinition:
    """OpenAI function calling compatible tool definition"""
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[[Dict], str]


def create_calculator_tool() -> ToolDefinition:
    """Calculator tool with structured parameters"""
    
    def calculator_func(params: Dict) -> str:
        expression = params.get("expression", "")
        # Security: only allow safe chars
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters"
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    return ToolDefinition(
        name="calculator",
        description="Perform mathematical calculations safely",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression like '2 + 2' or '10 * 5'"
                }
            },
            "required": ["expression"]
        },
        func=calculator_func
    )


def create_datetime_tool() -> ToolDefinition:
    """Date/time tool"""
    from datetime import datetime
    
    def datetime_func(params: Dict) -> str:
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")
    
    return ToolDefinition(
        name="get_datetime",
        description="Get current date and time",
        parameters={
            "type": "object",
            "properties": {},
            "required": []
        },
        func=datetime_func
    )


def create_wordcount_tool() -> ToolDefinition:
    """Word count tool"""
    
    def wordcount_func(params: Dict) -> str:
        text = params.get("text", "")
        words = len(text.split())
        chars = len(text)
        return f"Words: {words}, Characters: {chars}"
    
    return ToolDefinition(
        name="word_count",
        description="Count words and characters in text",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to count"
                }
            },
            "required": ["text"]
        },
        func=wordcount_func
    )


def create_websearch_tool(api_key: Optional[str] = None) -> ToolDefinition:
    """Web search tool (SerpAPI)"""
    
    def websearch_func(params: Dict) -> str:
        query = params.get("query", "")
        
        # If no API key, return placeholder
        if not api_key:
            return f"[Web search not configured. Query: {query}]"
        
        try:
            import requests
            url = "https://serpapi.com/search"
            params_req = {
                "q": query,
                "api_key": api_key,
                "engine": "google",
                "num": 3
            }
            res = requests.get(url, params=params_req, timeout=10).json()
            
            results = res.get("organic_results", [])[:3]
            if not results:
                return "No search results found"
            
            output = []
            for r in results:
                title = r.get("title", "No title")
                snippet = r.get("snippet", "No snippet")
                output.append(f"- {title}: {snippet}")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Search error: {str(e)}"
    
    return ToolDefinition(
        name="web_search",
        description="Search the web for current information",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        },
        func=websearch_func
    )


class ToolCaller:
    """
    PRO Tool Caller using OpenAI Function Calling API
    
    No more JSON parsing - OpenAI returns structured tool_calls
    """
    
    def __init__(self, openai_api_key: str, serpapi_key: Optional[str] = None):
        self.client = OpenAI(api_key=openai_api_key)
        
        # Register available tools
        self.tools: Dict[str, ToolDefinition] = {
            "calculator": create_calculator_tool(),
            "get_datetime": create_datetime_tool(),
            "word_count": create_wordcount_tool(),
            "web_search": create_websearch_tool(serpapi_key)
        }
    
    def get_openai_tools(self) -> List[Dict]:
        """Convert tools to OpenAI format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in self.tools.values()
        ]
    
    def execute_tool(self, tool_name: str, arguments: str) -> str:
        """Execute a tool by name with JSON arguments"""
        tool = self.tools.get(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            params = json.loads(arguments)
            result = tool.func(params)
            logger.info(f"Tool '{tool_name}' executed: {result[:100]}...")
            return result
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON arguments - {str(e)}"
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    
    def run_with_tools(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        max_steps: int = 3
    ) -> Dict[str, Any]:
        """
        Run conversation with automatic tool calling.
        
        Returns:
            Dict with "content", "tool_calls", "steps"
        """
        all_tool_calls = []
        steps = 0
        
        while steps < max_steps:
            steps += 1
            
            # Make request with tools
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=self.get_openai_tools(),
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            # If no tool calls, we're done
            if not message.tool_calls:
                return {
                    "content": message.content,
                    "tool_calls": all_tool_calls,
                    "steps": steps
                }
            
            # Execute tool calls
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })
            
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                arguments = tool_call.function.arguments
                
                # Execute tool
                result = self.execute_tool(tool_name, arguments)
                
                all_tool_calls.append({
                    "step": steps,
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result
                })
                
                # Add tool response to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        
        # Max steps reached - get final response
        response = self.client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        return {
            "content": response.choices[0].message.content,
            "tool_calls": all_tool_calls,
            "steps": steps,
            "max_steps_reached": True
        }


__all__ = [
    "ToolDefinition",
    "ToolCaller",
    "create_calculator_tool",
    "create_datetime_tool",
    "create_wordcount_tool",
    "create_websearch_tool"
]
