"""
Pydantic schemas for API request/response models
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Query request schema"""
    query: str = Field(..., description="The user query")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    top_k: Optional[int] = Field(6, description="Number of documents to retrieve")
    use_agents: Optional[bool] = Field(True, description="Whether to use agentic processing")


class DocumentInfo(BaseModel):
    """Document information schema"""
    id: str
    filename: str
    file_type: str
    size: int
    upload_date: datetime
    chunk_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Query response schema with execution metadata"""
    query: str
    answer: str
    sources: List[DocumentInfo]
    agent_steps: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time: float
    confidence_score: float
    evaluation_score: float = Field(default=0.0, description="Final evaluation score (0-10)")
    iterations: int = Field(default=1, description="Number of refinement iterations")
    retrieved_docs: int = Field(default=0, description="Number of documents retrieved")
    retry_reason: Optional[Dict[str, Any]] = Field(default=None, description="Why retry was triggered")
    agent_latencies: Dict[str, float] = Field(default_factory=dict, description="Latency per agent")
    conversation_id: str


class DocumentUploadResponse(BaseModel):
    """Document upload response schema"""
    success: bool
    document_id: Optional[str] = None
    message: str
    filename: str
    chunks_created: int


class ChatMessage(BaseModel):
    """Chat message schema"""
    role: str = Field(..., description="User or assistant")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Agent response schema"""
    agent_name: str
    action: str
    result: Dict[str, Any]
    confidence: float
    processing_time: float
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    components: Dict[str, str] = Field(default_factory=dict)
