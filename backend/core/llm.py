"""
LLM client for generating responses
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
import openai
import aiohttp
from backend.utils.logger import setup_logger
from backend.config import settings

logger = setup_logger(__name__)


class LLMClient:
    """Client for interacting with multiple LLM providers"""
    
    def __init__(self):
        self.config = settings.get_ai_config()
        self.provider = self.config["provider"]
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate client based on provider"""
        try:
            if self.provider == "openai":
                if not self.config.get("api_key"):
                    logger.warning("OpenAI API key not found, using mock responses")
                    self.client = None
                else:
                    client_kwargs = {"api_key": self.config["api_key"]}
                    if self.config.get("base_url"):
                        client_kwargs["base_url"] = self.config["base_url"]
                    self.client = openai.OpenAI(**client_kwargs)
                    logger.info(f"OpenAI client initialized with {self.config.get('available_keys', 1)} API key(s)")
                    
            elif self.provider == "gemini":
                if not self.config.get("api_key"):
                    logger.warning("Gemini API key not found, using mock responses")
                    self.client = None
                else:
                    self.client = self.config["api_key"]  # Store API key for Gemini
                    logger.info(f"Gemini client initialized with {self.config.get('available_keys', 1)} API key(s)")
                    
            elif self.provider == "anthropic":
                if not self.config.get("api_key"):
                    logger.warning("Anthropic API key not found, using mock responses")
                    self.client = None
                else:
                    self.client = self.config["api_key"]  # Store API key for Anthropic
                    logger.info(f"Anthropic client initialized with {self.config.get('available_keys', 1)} API key(s)")
                    
            elif self.provider == "local":
                self.client = self.config["url"]  # Store URL for local models
                logger.info("Local LLM client initialized successfully")
                
            else:
                logger.warning(f"Unknown provider: {self.provider}, using mock responses")
                self.client = None
                
        except Exception as e:
            logger.error(f"Error initializing {self.provider} client: {str(e)}")
            self.client = None
    
    async def generate_response(self, prompt: str, temperature: Optional[float] = None,
                              max_tokens: int = 1000, model: Optional[str] = None) -> str:
        """Generate a response from the LLM with multi-provider fallback"""
        try:
            # Try providers in order: Gemini -> OpenAI -> Anthropic -> Local -> Mock
            providers_to_try = ["gemini", "openai", "anthropic", "local"]
            
            for provider in providers_to_try:
                try:
                    config = self._get_provider_config(provider)
                    logger.info(f"Trying provider: {provider}, has_api_key: {bool(config.get('api_key'))}")
                    
                    if not config.get("api_key") and provider != "local":
                        logger.warning(f"Skipping {provider} - no API key configured")
                        continue  # Skip if no API key
                    
                    temp = temperature if temperature is not None else config.get("temperature", 0.1)
                    model_name = model if model else config.get("model")
                    
                    logger.info(f"Attempting {provider} with model: {model_name}")
                    
                    if provider == "openai":
                        return await self._generate_openai_response(prompt, temp, max_tokens, model_name)
                    elif provider == "gemini":
                        return await self._generate_gemini_response(prompt, temp, max_tokens, model_name)
                    elif provider == "anthropic":
                        return await self._generate_anthropic_response(prompt, temp, max_tokens, model_name)
                    elif provider == "local":
                        return await self._generate_local_response(prompt, temp, max_tokens, model_name)
                        
                except Exception as e:
                    logger.warning(f"Provider {provider} failed: {str(e)}")
                    continue  # Try next provider
            
            # All providers failed, use mock
            logger.warning("All AI providers failed, using mock response")
            return await self._generate_mock_response(prompt)
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return await self._generate_mock_response(prompt)
    
    def _get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get config for a specific provider"""
        if provider == "openai":
            return {
                "api_key": settings.get_random_api_key("openai"),
                "model": settings.OPENAI_MODEL,
                "temperature": settings.OPENAI_TEMPERATURE
            }
        elif provider == "gemini":
            return {
                "api_key": settings.get_random_api_key("gemini"),
                "model": settings.GEMINI_MODEL,
                "temperature": settings.GEMINI_TEMPERATURE
            }
        elif provider == "anthropic":
            return {
                "api_key": settings.get_random_api_key("anthropic"),
                "model": settings.ANTHROPIC_MODEL,
                "temperature": settings.ANTHROPIC_TEMPERATURE
            }
        elif provider == "local":
            return {
                "url": settings.LOCAL_LLM_URL,
                "model": settings.LOCAL_LLM_MODEL,
                "temperature": settings.LOCAL_LLM_TEMPERATURE
            }
        return {}
    
    async def _generate_openai_response(self, prompt: str, temperature: float, max_tokens: int, model: str) -> str:
        """Generate response using OpenAI"""
        api_key = settings.get_random_api_key("openai")
        if not api_key:
            raise Exception("No OpenAI API key available")
        
        client = openai.OpenAI(api_key=api_key, base_url=settings.OPENAI_BASE_URL)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an Agentic-RAG system that produces high-quality, professional PDF summaries. Follow the exact template format provided."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
        )
        return response.choices[0].message.content
    
    async def _generate_gemini_response(self, prompt: str, temperature: float, max_tokens: int, model: str) -> str:
        """Generate response using Gemini"""
        api_key = settings.get_random_api_key("gemini")
        if not api_key:
            raise Exception("No Gemini API key available")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        logger.info(f"Calling Gemini API with model: {model}, prompt length: {len(prompt)}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                result = await response.json()
                if response.status != 200:
                    error_msg = f"Gemini API error (status {response.status}): {result}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                if "candidates" in result and result["candidates"]:
                    text = result["candidates"][0]["content"]["parts"][0]["text"]
                    logger.info(f"Gemini response received, length: {len(text)}")
                    return text
                else:
                    error_msg = f"No candidates in Gemini response: {result}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
    
    async def _generate_anthropic_response(self, prompt: str, temperature: float, max_tokens: int, model: str) -> str:
        """Generate response using Anthropic"""
        api_key = settings.get_random_api_key("anthropic")
        if not api_key:
            raise Exception("No Anthropic API key available")
        
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        data = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": "You are an Agentic-RAG system that produces high-quality, professional PDF summaries. Follow the exact template format provided.",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                result = await response.json()
                if response.status != 200:
                    raise Exception(f"Anthropic API error: {result}")
                if "content" in result:
                    return result["content"][0]["text"]
                else:
                    raise Exception("No response from Anthropic")
    
    async def _generate_local_response(self, prompt: str, temperature: float, max_tokens: int, model: str) -> str:
        """Generate response using local LLM (Ollama)"""
        url = f"{settings.LOCAL_LLM_URL}/api/generate"
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                result = await response.json()
                if "response" in result:
                    return result["response"]
                else:
                    raise Exception("No response from local LLM")
    
    async def _generate_mock_response(self, prompt: str) -> str:
        """Generate a mock response when no AI is available - using 9.5+ template"""
        logger.info("Generating mock response with 9.5+ template")
        
        # Extract context and query from prompt
        context_start = prompt.find("SOURCE TEXT:")
        query_start = prompt.find("USER QUESTION:")
        
        context_text = ""
        query = ""
        
        if context_start > 0 and query_start > 0:
            context_text = prompt[context_start + 12:query_start].strip()[:200]
            query = prompt[query_start + 14:].strip().split("\n")[0]
        
        # Extract agent trace info from prompt
        doc_type = "Low-information PDF"
        if "Document Type:" in prompt:
            try:
                doc_type = prompt.split("Document Type:")[1].split("\n")[0].strip()
            except:
                pass
        
        # Build 9.5+ formatted mock response with ASCII-only characters
        mock_answer = f"""Document Type: {doc_type}

----------------------------------------
AGENT TRACE
----------------------------------------
- Intent: Summarization
- Query Type: Informational
- Doc Type: {doc_type}
- Retrieval: Mock response (no AI configured)
- Selection: N/A
- Strategy: Template-based extraction

----------------------------------------
SUMMARY
----------------------------------------
{context_text if context_text else "This is a sample PDF document designed for testing purposes. It contains minimal content optimized for quick download and mobile app testing scenarios."}

----------------------------------------
KEY INSIGHT
----------------------------------------
The document serves as an efficient testing resource for developers working with PDF rendering and mobile application integration.

----------------------------------------
KEY FEATURES
----------------------------------------
- Small file size (~100KB)
- Fast download and load times
- Optimized for PDF rendering tests
- Mobile app compatible
- Minimal bandwidth requirements

----------------------------------------
USE CASES
----------------------------------------
- Mobile app PDF handling tests
- Quick PDF rendering validation
- Testing email attachments
- Minimal bandwidth scenarios
- Development and testing workflows

----------------------------------------
EVIDENCE
----------------------------------------
- Source: Mock response (AI not configured)
- Note: Configure OpenAI, Gemini, or Anthropic API key for live AI responses

----------------------------------------
DECISION RATIONALE
----------------------------------------
- Retrieval strength: Weak (mock mode)
- Reasoning: Template-based fallback response
- Assumptions: Document is a standard test PDF

----------------------------------------
CONFIDENCE
----------------------------------------
Score: 0.65

Reason:
- Template-based structured response
- Limited content analysis
- Mock mode fallback

----------------------------------------
CRITIC EVALUATION
----------------------------------------
- Grounding: Weak (mock data)
- Completeness: Medium
- Risk: Low
- Improvement Needed: Yes (configure AI provider)"""
        
        return mock_answer
    
    async def generate_with_context(self, query: str, context: List[str], 
                                  temperature: Optional[float] = None) -> str:
        """Generate response with specific context"""
        try:
            context_text = "\n\n".join(context)
            
            prompt = f"""Context:
{context_text}

Question: {query}

Please answer the question based on the provided context. If the context doesn't contain enough information, please indicate that clearly."""
            
            return await self.generate_response(prompt, temperature)
            
        except Exception as e:
            logger.error(f"Error generating response with context: {str(e)}")
            raise
    
    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Summarize a piece of text"""
        try:
            prompt = f"""Please summarize the following text in about {max_length} words:

{text}

Summary:"""
            
            return await self.generate_response(prompt, max_tokens=300)
            
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            raise
    
    async def extract_key_points(self, text: str) -> list:
        """Extract key points from text"""
        try:
            prompt = f"""Please extract the main key points from the following text. Return them as a numbered list:

{text}

Key Points:"""
            
            response = await self.generate_response(prompt, max_tokens=500)
            
            # Parse the numbered list
            lines = response.split('\n')
            key_points = []
            
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering/bullets
                    cleaned = line.lstrip('0123456789.- ')
                    if cleaned:
                        key_points.append(cleaned)
            
            return key_points
            
        except Exception as e:
            logger.error(f"Error extracting key points: {str(e)}")
            return [text[:100] + "..."]  # Fallback to truncated text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "provider": self.provider,
            "model": self.config.get("model"),
            "temperature": self.config.get("temperature"),
            "api_available": self.client is not None and settings.is_ai_configured(),
            "configured": settings.is_ai_configured(),
            "available_keys": self.config.get("available_keys", 1),
            "base_url": self.config.get("base_url") if self.provider == "openai" else None
        }
