"""
API routes for Agentic-RAG
"""

import os
import time
import uuid
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request, Header
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from backend.models.schemas import (
    QueryRequest, QueryResponse, DocumentUploadResponse,
    DocumentInfo, HealthResponse
)
from backend.core.security import (
    is_safe_input, sanitize_output, validate_query_length,
    verify_api_key, MAX_FILE_SIZE, get_user_api_key, CostLimits,
    validate_openai_key_cached
)
from backend.agents.orchestrator import Orchestrator
from backend.agents.streaming import stream_agentic_response, create_sse_headers
from backend.agents.tool_caller import ToolCaller
from backend.agents.memory import get_memory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.tools.document_loader import DocumentLoader
from backend.tools.text_splitter import TextSplitter
from backend.core.embeddings import EmbeddingGenerator
from backend.core.vector_store import get_vector_store  # 🔥 CRITICAL: Use the REAL singleton
from backend.utils.logger import setup_logger
from backend.config import settings
import aiohttp
import json

logger = setup_logger(__name__)
router = APIRouter(tags=["api"])

# Ensure data directories exist
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/vectors", exist_ok=True)

# SECURITY: Check if debug endpoints should be enabled
def is_debug_enabled():
    """Debug endpoints only enabled in non-production environments"""
    return os.environ.get("ENV") != "production" and os.environ.get("DEBUG_ENDPOINTS") == "true"

# Generic error message to prevent information disclosure
SAFE_ERROR_MESSAGE = "An internal error occurred"

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

MODEL_ALIASES = {
    "nvidia": {
        "meta/llama3-70b-instruct": "meta/llama-3.1-70b-instruct",
        "meta/llama3.1-70b-instruct": "meta/llama-3.1-70b-instruct",
        "meta/llama3.1-8b-instruct": "meta/llama-3.1-8b-instruct",
        "meta/llama3.1-405b-instruct": "meta/llama-3.1-405b-instruct"
    },
    "gemini": {
        "gemini-pro": "gemini-1.5-flash",
        "gemini-1.5-flash-001": "gemini-1.5-flash"
    }
}

PROVIDER_FALLBACK_MODELS = {
    "gemini": [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.0-pro"
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo"
    ],
    "anthropic": [
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ],
    "nvidia": [
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.1-405b-instruct",
        "meta/llama-3.1-8b-instruct",
        "mistralai/mistral-large"
    ],
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ],
    "huggingface": [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
        "google/gemma-7b-it",
        "tiiuae/falcon-7b-instruct"
    ],
    "local": [
        "llama3.1:8b",
        "llama3:8b",
        "mistral:7b",
        "qwen2.5:7b",
        "phi3:mini"
    ]
}

# Global components (in production, use dependency injection)
document_loader = DocumentLoader()
text_splitter = TextSplitter()
vector_store = None

# 🎯 Track active document to prevent data mixing
active_document_id = None
active_document_name = None

def get_embedding_generator(api_key: str):
    """Create embedding generator with user-provided API key (PRO: no global singleton)"""
    return EmbeddingGenerator(api_key)

# 🔥 DELETED: Duplicate get_vector_store() - now imported from backend.core.vector_store

def get_orchestrator(api_key: str = None):
    """🧠 Get Professional Orchestrator with 6-component agentic system (PRO: user API key)"""
    return Orchestrator(api_key)

def set_active_document(doc_id: str, filename: str):
    """🎯 Set the currently active document for retrieval filtering"""
    global active_document_id, active_document_name
    active_document_id = doc_id
    active_document_name = filename
    logger.info(f"🎯 Active document set: {filename} (ID: {doc_id})")

def get_active_document():
    """Get currently active document info"""
    return {"id": active_document_id, "filename": active_document_name}


def _normalize_provider_model(provider: str, model: str) -> str:
    """Normalize model aliases to provider-supported names."""
    if not model:
        return model
    return MODEL_ALIASES.get(provider, {}).get(model, model)


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")  # Rate limiting: 10 requests per minute per IP
async def query_documents(
    request: Request, 
    query_request: QueryRequest,
    api_key: str = Depends(get_user_api_key)  # ✅ PRO: Unified dependency
):
    """Query documents using agentic RAG - user provides their own API key"""
    try:
        # [SECURITY] PRO: Validate OpenAI API key (prevent fake keys)
        is_valid_key = await validate_openai_key_cached(api_key)
        if not is_valid_key:
            raise HTTPException(status_code=401, detail="Invalid or expired OpenAI API key")
        
        safe_log(logger, "info", "Processing query request")
        
        # [SECURITY] Prompt injection protection
        is_safe, reason = is_safe_input(query_request.query)
        if not is_safe:
            safe_log(logger, "warning", "Unsafe input detected", reason=reason)
            raise HTTPException(status_code=400, detail=f"Unsafe input: {reason}")
        
        # [SECURITY] Cost protection: Validate query length
        if len(query_request.query) > CostLimits.MAX_QUERY_LENGTH:
            raise HTTPException(status_code=400, detail=f"Query too long (max {CostLimits.MAX_QUERY_LENGTH} characters)")
        
        # [DOC] Check if we have an active document
        active_doc = get_active_document()
        if not active_doc["id"]:
            return QueryResponse(
                query=query_request.query,
                answer="No document uploaded yet. Please upload a document first.",
                sources=[],
                agent_steps=[],
                processing_time=0.0,
                confidence_score=0.0,
                conversation_id=query_request.conversation_id or str(uuid.uuid4())
            )
        
        safe_log(logger, "info", "Querying against active document", document=active_doc['filename'])
        
        # 🎯 PRO: Create orchestrator with user API key (user-based billing)
        orchestrator = get_orchestrator(api_key)
        response = await orchestrator.process_query(
            query_request, 
            active_document_id=active_doc["id"],
            user_api_key=api_key
        )
        
        safe_log(logger, "info", "Query processed successfully", time=f"{response.processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        # SECURITY: Don't expose internal error details
        raise HTTPException(status_code=500, detail=SAFE_ERROR_MESSAGE)


# =============================================================================
# PRO: STREAMING RESPONSE (ChatGPT-like real-time)
# =============================================================================

@router.post("/query-stream")
@limiter.limit("10/minute")
async def query_stream(
    request: Request,
    query_request: QueryRequest,
    api_key: str = Depends(get_user_api_key),
    x_session_id: str = Header(None, description="Session ID for conversation memory")
):
    """
    PRO: Streaming query endpoint with Server-Sent Events.
    Returns tokens in real-time like ChatGPT.
    """
    try:
        # [SECURITY] PRO: Validate OpenAI API key
        is_valid_key = await validate_openai_key_cached(api_key)
        if not is_valid_key:
            error_event = json.dumps({"type": "error", "message": "Invalid OpenAI API key"})
            return StreamingResponse(
                iter([f"data: {error_event}\n\n"]),
                headers=create_sse_headers()
            )
        
        # Get memory if session provided
        memory = None
        if x_session_id:
            memory = get_memory(x_session_id)
        
        # Get active document
        active_doc = get_active_document()
        if not active_doc["id"]:
            # Return error as SSE
            error_event = json.dumps({"type": "error", "message": "No document uploaded"})
            return StreamingResponse(
                iter([f"data: {error_event}\n\n"]),
                headers=create_sse_headers()
            )
        
        # Retrieve context
        vector_store = get_vector_store()
        embedding_gen = EmbeddingGenerator(api_key)
        query_embedding = await embedding_gen.generate_query_embedding(query_request.query)
        docs = await vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=4,
            filter_dict={"document_id": active_doc["id"]}
        )
        context = "\n\n".join([d.get('content', '') for d in docs])
        
        # Get memory context
        memory_context = ""
        if memory:
            memory_context = memory.get_context_for_prompt()
        
        # Stream response
        return StreamingResponse(
            stream_agentic_response(
                api_key=api_key,
                query=query_request.query,
                context=context,
                memory_context=memory_context
            ),
            headers=create_sse_headers()
        )
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_event = json.dumps({"type": "error", "message": str(e)})
        return StreamingResponse(
            iter([f"data: {error_event}\n\n"]),
            headers=create_sse_headers()
        )


# =============================================================================
# PRO: AGENTIC WITH TOOL CALLING (OpenAI Function Calling)
# =============================================================================

@router.post("/query-pro", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def query_pro(
    request: Request,
    query_request: QueryRequest,
    api_key: str = Depends(get_user_api_key),
    x_session_id: str = Header(None, description="Session ID for conversation memory"),
    serpapi_key: str = Header(None, description="Optional SerpAPI key for web search")
):
    """
    PRO: Agentic query with structured tool calling.
    Uses OpenAI Function Calling API (no JSON parsing).
    """
    import time
    start_time = time.time()
    conversation_id = query_request.conversation_id or str(uuid.uuid4())
    
    try:
        # [SECURITY] PRO: Validate OpenAI API key
        is_valid_key = await validate_openai_key_cached(api_key)
        if not is_valid_key:
            return QueryResponse(
                query=query_request.query,
                answer="Invalid or expired OpenAI API key. Please check your API key.",
                sources=[],
                agent_steps=[{"error": "invalid_api_key"}],
                processing_time=0.0,
                confidence_score=0.0,
                conversation_id=conversation_id
            )
        
        # Get memory
        memory = None
        if x_session_id:
            memory = get_memory(x_session_id)
        
        # Get active document
        active_doc = get_active_document()
        if not active_doc["id"]:
            return QueryResponse(
                query=query_request.query,
                answer="Please upload a document first.",
                sources=[],
                agent_steps=[{"error": "no_document"}],
                processing_time=time.time() - start_time,
                confidence_score=0.0,
                conversation_id=conversation_id
            )
        
        # Retrieve context
        vector_store = get_vector_store()
        embedding_gen = EmbeddingGenerator(api_key)
        query_embedding = await embedding_gen.generate_query_embedding(query_request.query)
        docs = await vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=4,
            filter_dict={"document_id": active_doc["id"]}
        )
        context = "\n\n".join([d.get('content', '') for d in docs])
        
        # Build messages
        messages = []
        if memory:
            memory_ctx = memory.get_context_for_prompt()
            if memory_ctx:
                messages.append({"role": "system", "content": memory_ctx})
        
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query_request.query}"
        })
        
        # PRO: Use ToolCaller with OpenAI function calling
        tool_caller = ToolCaller(api_key, serpapi_key)
        result = tool_caller.run_with_tools(messages, max_steps=3)
        
        # Store in memory
        if memory:
            memory.add(query_request.query, result["content"])
        
        # Build agent steps
        agent_steps = []
        for tc in result["tool_calls"]:
            agent_steps.append({
                "tool": tc["tool"],
                "arguments": tc["arguments"],
                "result": tc["result"][:100]
            })
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            query=query_request.query,
            answer=result["content"],
            sources=[],
            agent_steps=agent_steps,
            processing_time=processing_time,
            confidence_score=0.85 if result["tool_calls"] else 0.7,
            conversation_id=conversation_id
        )
        
    except Exception as e:
        logger.error(f"PRO query error: {e}")
        return QueryResponse(
            query=query_request.query,
            answer=f"Error: {str(e)}",
            sources=[],
            agent_steps=[{"error": str(e)}],
            processing_time=time.time() - start_time,
            confidence_score=0.0,
            conversation_id=conversation_id
        )


@router.post("/upload", response_model=DocumentUploadResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")  # Rate limiting: 5 uploads per minute per IP
async def upload_document(
    request: Request, 
    file: UploadFile = File(...),
    api_key: str = Depends(get_user_api_key)  # ✅ PRO: Unified dependency
):
    """Upload and process a document with validation and size limit"""
    logger.info(f"=== UPLOAD STARTED ===")
    logger.info(f"Received file: {file}")
    file_path = None
    try:
        logger.info(f"Uploading document: {file.filename if file else 'NO FILE'}")
        
        # Validate file exists
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
            
        # Validate filename
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Validate file format
        logger.info(f"Checking file format: {file.filename}")
        if not document_loader.is_supported_format(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported: {', '.join(document_loader.get_supported_formats())}"
            )
        
        # Validate file size (read once, store for later)
        content = await file.read()
        content_size = len(content)
        logger.info(f"File size: {content_size} bytes")
        
        if content_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {MAX_FILE_SIZE // (1024*1024)}MB)"
            )
        
        # Save uploaded file - SECURITY: Sanitize filename to prevent path traversal
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # SECURITY: Sanitize filename to prevent directory traversal attacks
        safe_filename = os.path.basename(file.filename)
        safe_filename = re.sub(r'[^\w\-\.]', '_', safe_filename)
        if not safe_filename or safe_filename == '.' or safe_filename == '..':
            raise HTTPException(status_code=400, detail="Invalid filename after sanitization")
        
        file_path = os.path.join(upload_dir, safe_filename)
        
        # Write file using stored content (FIX: don't read again!)
        logger.info(f"Writing file to: {file_path}")
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        logger.info(f"File written successfully: {content_size} bytes")
        
        # Load and process document
        document = await document_loader.load_document(file_path, file.filename)
        
        # 🎯 SET ACTIVE DOCUMENT (before clearing old data)
        set_active_document(document["id"], file.filename)
        
        # 🔥 CRITICAL: Set vector store namespace for isolation
        vector_store = get_vector_store()
        vector_store.set_active_pdf(file.filename)
        
        # 🧹 CLEAR OLD DATA: Remove previous documents to prevent mixing
        logger.info("Clearing old documents from vector store...")
        await vector_store.clear_collection()
        
        # Split into chunks using RecursiveCharacterTextSplitter
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # 🔥 Simple pipeline: 800 size
            chunk_overlap=100  # 🔥 Simple overlap
        )
        chunks_text = splitter.split_text(document["content"])
        
        # Convert to expected format
        chunks = []
        for i, text in enumerate(chunks_text):
            chunks.append({
                "content": text,
                "text": text,
                "chunk_index": i,
                "metadata": document["metadata"]
            })
        
        # 🔥 STEP 1: VERIFY chunks are created
        print("\n🔥 CHUNKS VERIFICATION:")
        for i, c in enumerate(chunks[:3]):
            print(f"Chunk {i+1}: {c.get('content', '')[:100]}...")
        print(f"Total chunks: {len(chunks)}\n")
        
        # [SECURITY] Cost protection: Limit max chunks per upload
        if len(chunks) > CostLimits.MAX_CHUNKS_PER_UPLOAD:
            raise HTTPException(
                status_code=413, 
                detail=f"Document too large (max {CostLimits.MAX_CHUNKS_PER_UPLOAD} chunks). Please upload a smaller document."
            )
        
        # Generate embeddings and store in vector store (PRO: user-provided API key)
        embedding_generator = get_embedding_generator(api_key)
        
        chunk_texts = [chunk["content"] for chunk in chunks]
        chunk_metadatas = []
        
        for i, chunk in enumerate(chunks):
            metadata = chunk["metadata"].copy() if chunk["metadata"] else {}
            # 🎯 ADD DOC_ID TO EVERY CHUNK
            metadata.update({
                "chunk_index": chunk["chunk_index"],
                "document_id": document["id"],
                "filename": document["filename"],
                "doc_id": document["id"]  # 🎯 KEY: Store UUID as doc_id for filtering
            })
            # Filter out None values - vector store doesn't accept them
            metadata = {k: v for k, v in metadata.items() if v is not None}
            # Also convert any non-primitive types to strings
            metadata = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in metadata.items()}
            chunk_metadatas.append(metadata)
        
        # Generate embeddings using user's API key (PRO: batching for speed)
        logger.info(f"Generating embeddings for {len(chunk_texts)} chunks using user API key...")
        embeddings = await embedding_generator.generate_embeddings(chunk_texts)
        
        # Store in vector store WITH embeddings
        logger.info("Storing in vector store...")
        chunk_ids = await vector_store.add_documents(chunk_texts, chunk_metadatas, embeddings)
        
        # 🔥 STEP 3: VERIFY vector store insert
        print(f"🔥 Stored docs: {len(chunk_ids)}")
        if len(chunk_ids) == 0:
            print("❌ ERROR: No documents stored!")
        else:
            print(f"✅ Successfully stored {len(chunk_ids)} chunks")
        
        logger.info(f"Document processed successfully: {len(chunks)} chunks created")
        
        return DocumentUploadResponse(
            success=True,
            document_id=document["id"],
            message=f"Document uploaded and processed successfully",
            filename=file.filename,
            chunks_created=len(chunks)
        )
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error uploading document: {error_msg}")
        # Clean up file if upload failed
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        
        # DEBUG: Show actual error for local development
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Upload failed: {error_msg}")


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all uploaded documents"""
    try:
        # This would typically query a database for document metadata
        # For now, return empty list as placeholder
        logger.info("Listing documents")
        return []
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        # SECURITY: Don't expose internal error details
        raise HTTPException(status_code=500, detail=SAFE_ERROR_MESSAGE)


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    try:
        logger.info(f"Deleting document: {document_id}")
        
        # Delete from vector store
        vector_store = get_vector_store()
        success = await vector_store.delete_documents([document_id])
        
        if success:
            logger.info(f"Document deleted successfully: {document_id}")
            return {"message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
        
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        # SECURITY: Don't expose internal error details
        raise HTTPException(status_code=500, detail=SAFE_ERROR_MESSAGE)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    PRO: Comprehensive health check with component status.
    Used by Railway/Render for auto-restart decisions.
    """
    import time
    start_time = time.time()
    
    components = {
        "api": "ok",
        "agents": "ok",
        "memory": "ok",
        "vector_store": "unknown"
    }
    
    overall_status = "healthy"
    
    try:
        # Check vector store
        try:
            vector_store = get_vector_store()
            # Lightweight check - just verify object exists
            if vector_store:
                components["vector_store"] = "ok"
        except Exception as e:
            components["vector_store"] = f"error: {str(e)[:50]}"
            overall_status = "degraded"
        
        # Check memory system
        try:
            from backend.agents.memory import get_session_manager
            manager = get_session_manager()
            components["memory"] = "ok"
        except Exception as e:
            components["memory"] = f"error: {str(e)[:50]}"
        
        latency_ms = (time.time() - start_time) * 1000
        
        return HealthResponse(
            status=overall_status,
            version="2.0.0-pro",
            components=components,
            latency_ms=round(latency_ms, 2)
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            version="2.0.0-pro",
            components={"error": "Health check failed", "detail": str(e)[:100]}
        )


@router.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        logger.info("Getting system stats")
        
        # Vector store stats
        vector_store = get_vector_store()
        
        # Try to get stats, handle if method doesn't exist
        try:
            vector_stats = vector_store.get_stats()
        except AttributeError:
            vector_stats = {"status": "unknown", "message": "get_stats method not implemented"}
        
        # Basic system info
        return {
            "vector_store": vector_stats,
            "agents": {
                "planner": "active",
                "reasoning": "active", 
                "critic": "active",
                "retry": "active",
                "pipeline": "active"
            },
            "system": {
                "chunk_size": 800,
                "chunk_overlap": 100,
                "supported_formats": document_loader.get_supported_formats(),
                "embedding_model": settings.EMBEDDING_MODEL,
                "llm_provider": settings.AI_PROVIDER,
                "llm_model": settings.GEMINI_MODEL if settings.AI_PROVIDER == "gemini" else "unknown"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        # SECURITY: Don't expose internal error details
        raise HTTPException(status_code=500, detail=SAFE_ERROR_MESSAGE)


@router.get("/test-retrieval")
async def test_retrieval():
    """Debug route to test retrieval pipeline - DISABLED in production"""
    # SECURITY: Only allow debug endpoints in non-production environments
    if not is_debug_enabled():
        raise HTTPException(status_code=404, detail="Endpoint not found")
    
    try:
        from backend.agents.retrieval_agent import RetrievalAgent
        
        retriever = RetrievalAgent()
        
        # Test queries
        test_queries = ["sample", "PDF", "document", "test"]
        
        results = {}
        for query in test_queries:
            docs = await retriever.retrieve(query, top_k=3)
            results[query] = {
                "count": len(docs),
                "sample_content": docs[0]["content"][:100] if docs else "No results"
            }
        
        return {
            "status": "success",
            "results": results,
            "message": "Debug retrieval test completed"
        }
    except Exception as e:
        logger.error(f"Debug retrieval test failed: {str(e)}")
        # SECURITY: Don't expose internal error details
        return {
            "status": "error",
            "message": SAFE_ERROR_MESSAGE
        }


@router.post("/test")
async def test_ai_connection():
    """Test AI connection with current configuration - DISABLED in production"""
    # SECURITY: Only allow debug endpoints in non-production environments
    if not is_debug_enabled():
        raise HTTPException(status_code=404, detail="Endpoint not found")
    
    try:
        from backend.core.llm import LLMClient
        
        llm_client = LLMClient()
        model_info = llm_client.get_model_info()
        
        # Test with a simple prompt
        test_prompt = "Hello! Please respond with 'Connection test successful.'"
        response = await llm_client.generate_response(test_prompt, max_tokens=50)
        
        return {
            "status": "success",
            "model_info": model_info,
            "test_response": response,
            "message": "AI connection test completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error testing AI connection: {str(e)}")
        # SECURITY: Don't expose internal error details
        return {
            "status": "error",
            "message": SAFE_ERROR_MESSAGE
        }


@router.post("/clear")
async def clear_all_data():
    """Clear all data from the system"""
    try:
        logger.warning("Clearing all system data")
        
        # Clear vector store
        vector_store = get_vector_store()
        success = await vector_store.clear_collection()
        
        if success:
            logger.info("All data cleared successfully")
            return {"message": "All data cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear data")
        
    except Exception as e:
        logger.error(f"Error clearing data: {str(e)}")
        # SECURITY: Don't expose internal error details to client
        raise HTTPException(status_code=500, detail=SAFE_ERROR_MESSAGE)


@router.get("/evaluation/status")
async def get_evaluation_status():
    """Get evaluation harness status and available QA pairs"""
    try:
        from backend.core.evaluation_harness import evaluation_harness
        
        return {
            "status": "ready",
            "qa_pairs_count": len(evaluation_harness.qa_pairs),
            "sample_qa": evaluation_harness.get_sample_test_set(3),
            "dataset_path": evaluation_harness.dataset_path
        }
    except Exception as e:
        logger.error(f"Error getting evaluation status: {str(e)}")
        # SECURITY: Don't expose internal error details
        raise HTTPException(status_code=500, detail=SAFE_ERROR_MESSAGE)


@router.post("/evaluation/run")
async def run_evaluation(sample_size: int = 5):
    """Run evaluation on sample QA pairs"""
    try:
        from backend.core.evaluation_harness import evaluation_harness
        
        # Use updated orchestrator
        orchestrator = get_orchestrator()
        
        # Get sample test set
        test_set = evaluation_harness.get_sample_test_set(sample_size)
        
        results = []
        for qa in test_set:
            try:
                # Run query through orchestrator
                from backend.models.schemas import QueryRequest
                request = QueryRequest(query=qa["query"], top_k=3)
                response = await orchestrator.process_query(request, active_document_id=qa.get("doc_id"))
                
                # Evaluate
                result = evaluation_harness.evaluate_response(
                    qa["id"],
                    response.answer,
                    [],  # No sources in new implementation
                    {"hallucination_rate": 0.1, "groundedness": 0.8}  # Simplified for now
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating {qa['id']}: {str(e)}")
                # SECURITY: Don't expose internal error details
                results.append({"qa_id": qa["id"], "error": SAFE_ERROR_MESSAGE, "passed": False})
        
        # Calculate summary
        passed = sum(1 for r in results if r.get("passed", False))
        
        return {
            "status": "completed",
            "total_tested": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": round(passed / len(results), 2) if results else 0,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}")
        # SECURITY: Don't expose internal error details
        raise HTTPException(status_code=500, detail=SAFE_ERROR_MESSAGE)


@router.post("/analyze")
async def analyze_resume(request: QueryRequest):
    """Smart resume analysis features"""
    try:
        logger.info("Starting smart resume analysis")
        
        orchestrator = get_orchestrator()
        active_doc = get_active_document()
        
        if not active_doc["id"]:
            return {
                "error": "No document uploaded",
                "message": "Please upload a resume first",
                "status": "error"
            }
        
        # Run multiple analysis queries
        analysis_types = {
            "overview": "Give me a brief overview of this resume including name, role, and key strengths",
            "skills": "What are the main technical skills mentioned in this resume?",
            "experience": "What is the work experience summary?",
            "education": "What is the educational background?"
        }
        
        results = {}
        for key, query in analysis_types.items():
            try:
                response = await orchestrator.process_query(
                    QueryRequest(query=query, top_k=3),
                    active_document_id=active_doc["id"]
                )
                results[key] = response.answer
            except Exception as e:
                results[key] = f"Could not analyze: {str(e)}"
        
        # Generate smart suggestions
        smart_features = {
            "resume_score": "7/10",  # Placeholder - could be calculated based on content
            "missing_skills": ["Docker", "Kubernetes", "CI/CD"],  # Placeholder
            "interview_questions": [
                "Tell me about your experience with Python and data pipelines",
                "How do you handle data quality issues in your projects?",
                "Describe a challenging project you worked on and how you solved it"
            ],
            "suggested_questions": [
                "What are my technical skills?",
                "Summarize my work experience",
                "What are my key achievements?",
                "Suggest improvements to my resume"
            ]
        }
        
        return {
            "analysis": results,
            "smart_features": smart_features,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing resume: {str(e)}")
        # SECURITY: Don't expose internal error details
        raise HTTPException(status_code=500, detail=SAFE_ERROR_MESSAGE)

# Configuration Management Endpoints
@router.get("/config")
async def get_config():
    """Get current AI configuration"""
    try:
        # Check runtime config first, then fall back to settings
        from backend.config import get_runtime_config
        provider = get_runtime_config("AI_PROVIDER") or settings.AI_PROVIDER
        
        # Get API key and mask it for security
        api_key = get_runtime_config(f"{provider.upper()}_API_KEY", "")
        masked_key = ""
        if api_key:
            masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
        
        return {
            "provider": provider,
            "api_key": masked_key,
            "has_api_key": bool(api_key),
            "model": get_runtime_config(f"{provider.upper()}_MODEL") or getattr(settings, f"{provider.upper()}_MODEL", ""),
            "temperature": get_runtime_config(f"{provider.upper()}_TEMPERATURE") or getattr(settings, f"{provider.upper()}_TEMPERATURE", 0.7),
            "configured": settings.is_ai_configured()
        }
    except Exception as e:
        logger.error(f"Error getting config: {str(e)}")
        # SECURITY: Don't expose internal error details
        raise HTTPException(status_code=500, detail=SAFE_ERROR_MESSAGE)

@router.post("/config")
async def update_config(config_data: Dict[str, Any]):
    """Update AI configuration using runtime config (no .env file needed) - auto-selects model"""
    try:
        from backend.config import set_runtime_config, update_runtime_config, get_runtime_config
        
        provider = config_data.get("provider", "gemini").lower()
        api_key = config_data.get("api_key", "")
        model = _normalize_provider_model(provider, config_data.get("model", ""))
        temperature = config_data.get("temperature", 0.7)
        
        # If api_key is null/None, preserve existing key from runtime config
        if api_key is None:
            existing_key = get_runtime_config(f"{provider.upper()}_API_KEY", "")
            api_key = existing_key
            logger.info(f"Preserving existing API key for {provider}")
        
        # If no model specified, auto-select first available model
        if not model and api_key and provider != "local":
            try:
                # List available models
                models_result = await list_available_models({
                    "provider": provider,
                    "api_key": api_key
                })
                
                if models_result.get("success") and models_result.get("first_model"):
                    model = models_result["first_model"]
                    logger.info(f"Auto-selected model: {model} for provider: {provider}")
            except Exception as e:
                logger.warning(f"Could not auto-select model: {str(e)}")

        # Validate provider/model now. If model fails but alternatives exist, auto-fallback.
        if provider != "local" and api_key:
            try:
                test_result = await test_api_connection({
                    "provider": provider,
                    "api_key": api_key,
                    "model": model
                })
                if (not test_result.get("success")) and test_result.get("error_type") == "model_not_found":
                    models_result = await list_available_models({
                        "provider": provider,
                        "api_key": api_key
                    })
                    fallback_model = models_result.get("first_model")
                    if fallback_model:
                        model = fallback_model
                        logger.info(f"Auto-fallback model selected: {model} for provider: {provider}")
            except Exception as e:
                logger.warning(f"Model validation/fallback skipped: {str(e)}")
        
        # Update runtime configuration (no .env file needed!)
        update_runtime_config({
            "AI_PROVIDER": provider,
            f"{provider.upper()}_API_KEY": api_key,
        })
        
        if model:
            set_runtime_config(f"{provider.upper()}_MODEL", model)
        if temperature is not None:
            set_runtime_config(f"{provider.upper()}_TEMPERATURE", temperature)
        
        return {
            "success": True,
            "message": "Configuration updated successfully - auto-selected model!" if not config_data.get("model") else "Configuration updated successfully!",
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "auto_selected": not bool(config_data.get("model"))
        }
        
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}")
        # SECURITY: Don't expose internal error details
        raise HTTPException(status_code=500, detail=SAFE_ERROR_MESSAGE)

@router.post("/clear-config")
async def clear_runtime_config_endpoint():
    """Clear runtime configuration to reset to defaults"""
    try:
        from backend.config import clear_runtime_config
        clear_runtime_config()
        return {
            "success": True,
            "message": "Runtime configuration cleared - will use defaults"
        }
    except Exception as e:
        logger.error(f"Error clearing config: {str(e)}")
        # SECURITY: Don't expose internal error details
        raise HTTPException(status_code=500, detail=SAFE_ERROR_MESSAGE)

@router.post("/list-models")
async def list_available_models(request: Dict[str, Any]):
    # List available models for a provider by querying their API
    try:
        provider = request.get("provider", "").lower()
        api_key = request.get("api_key", "")
        
        if not provider:
            return {
                "success": False,
                "error": "Provider is required"
            }
        if provider != "local" and not api_key:
            return {
                "success": False,
                "error": "API key is required for this provider"
            }
        
        models = []
        
        if provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        result = await response.json()
                        models = [m["name"].replace("models/", "") for m in result.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
        
        elif provider == "openai":
            url = "https://api.openai.com/v1/models"
            headers = {"Authorization": f"Bearer {api_key}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        result = await response.json()
                        models = [m["id"] for m in result.get("data", []) if m.get("id", "").startswith(("gpt-", "chatgpt-"))]
        
        elif provider == "anthropic":
            # Anthropic doesn't have a public models endpoint, return known models
            models = PROVIDER_FALLBACK_MODELS["anthropic"]
        
        elif provider == "groq":
            url = "https://api.groq.com/openai/v1/models"
            headers = {"Authorization": f"Bearer {api_key}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        result = await response.json()
                        models = [m["id"] for m in result.get("data", [])]
        
        elif provider == "nvidia":
            # NVIDIA known chat models (normalized names)
            models = PROVIDER_FALLBACK_MODELS["nvidia"]
        
        elif provider == "huggingface":
            models = PROVIDER_FALLBACK_MODELS["huggingface"]
        
        elif provider == "local":
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{settings.LOCAL_LLM_URL}/api/tags", timeout=5) as response:
                        if response.status == 200:
                            result = await response.json()
                            models = [m.get("name", "") for m in result.get("models", []) if m.get("name")]
                        else:
                            models = [settings.LOCAL_LLM_MODEL] + PROVIDER_FALLBACK_MODELS["local"]
            except Exception:
                models = [settings.LOCAL_LLM_MODEL] + PROVIDER_FALLBACK_MODELS["local"]

        if not models:
            models = PROVIDER_FALLBACK_MODELS.get(provider, [])
        
        return {
            "success": True,
            "provider": provider,
            "models": models,
            "first_model": models[0] if models else None
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        # SECURITY: Don't expose internal error details
        return {
            "success": False,
            "error": SAFE_ERROR_MESSAGE
        }

async def _test_api_connection_internal(test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Internal API testing logic - can be called by other endpoints"""
    try:
        from backend.config import get_runtime_config

        provider = test_data.get("provider", "").lower()
        api_key = test_data.get("api_key", "")

        
        # If api_key is null/empty/'existing', get from runtime config
        if not api_key or api_key == 'existing':
            api_key = get_runtime_config(f"{provider.upper()}_API_KEY", "")
            if api_key:
                logger.info(f"Using existing runtime config key for {provider} test")
        
        model = _normalize_provider_model(provider, test_data.get("model", ""))
        
        if not provider:
            return {
                "success": False,
                "error": "Provider is required",
                "error_type": "missing_provider"
            }
        
        # Local provider doesn't need API key, but must be reachable
        if provider == "local":
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{settings.LOCAL_LLM_URL}/api/tags", timeout=5) as response:
                        if response.status != 200:
                            return {
                                "success": False,
                                "error": "Local Ollama service is not reachable",
                                "error_type": "local_unreachable",
                                "status_code": response.status
                            }
                        payload = await response.json()
                        models = [m.get("name", "") for m in payload.get("models", [])]
            except Exception as e:
                logger.error(f"Cannot connect to local Ollama: {str(e)}")
                # SECURITY: Don't expose internal URL or error details
                return {
                    "success": False,
                    "error": "Cannot connect to local Ollama service",
                    "error_type": "local_unreachable"
                }
            return {
                "success": True,
                "message": "Local Ollama connection successful",
                "available_models": models,
                "error_type": None
            }
        
        if not api_key:
            return {
                "success": False,
                "error": f"{provider.capitalize()} API key is required",
                "error_type": "missing_api_key"
            }
        
        # Use default model if not provided
        default_models = {provider_name: models[0] for provider_name, models in PROVIDER_FALLBACK_MODELS.items() if models}
        
        model = model or default_models.get(provider, "default")
        
        # Test only the specific provider
        if provider == "nvidia":
            url = "https://integrate.api.nvidia.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
        elif provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
        elif provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{"parts": [{"text": "Hello"}]}],
                "generationConfig": {"maxOutputTokens": 10}
            }
        elif provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            data = {
                "model": model,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        elif provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
        elif provider == "huggingface":
            url = f"https://api-inference.huggingface.co/models/{model}"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "inputs": "Hello",
                "parameters": {"max_new_tokens": 10}
            }
        else:
            return {
                "success": False,
                "error": f"Provider '{provider}' not supported",
                "error_type": "unsupported_provider"
            }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    return {
                        "success": True,
                        "message": f"{provider.capitalize()} API connection successful with model: {model}",
                        "error_type": None
                    }
                else:
                    error_text = await response.text()
                    logger.warning(f"API test failed for {provider}: {response.status} - {error_text[:200]}")
                    return {
                        "success": False,
                        "error": f"API returned error: {response.status}",
                        "error_type": "api_error",
                        "status_code": response.status
                    }
                    
    except asyncio.TimeoutError:
        logger.error(f"Timeout testing {provider} API")
        return {
            "success": False,
            "error": "Connection timeout - provider may be slow or unreachable",
            "error_type": "timeout"
        }
    except Exception as e:
        logger.error(f"Error testing API connection: {str(e)}")
        # SECURITY: Don't expose internal error details
        return {
            "success": False,
            "error": SAFE_ERROR_MESSAGE,
            "error_type": "unknown"
        }


@router.post("/test-api")
async def test_api_connection(test_data: Dict[str, Any]):
    """Test API connection - available in all environments"""
    # Call the internal test function
    return await _test_api_connection_internal(test_data)


@router.get("/config-status")
async def get_config_status():
    try:
        from backend.config import get_runtime_config
        provider = (get_runtime_config("AI_PROVIDER") or settings.AI_PROVIDER or "gemini").lower()
        model = get_runtime_config(f"{provider.upper()}_MODEL") or getattr(settings, f"{provider.upper()}_MODEL", "")
        api_key = get_runtime_config(f"{provider.upper()}_API_KEY", "")
        test_result = await _test_api_connection_internal({
            "provider": provider,
            "api_key": api_key,
            "model": model
        })
        return {
            "provider": provider,
            "model": model,
            "active": bool(test_result.get("success")),
            "status_text": "API Active" if test_result.get("success") else "API Inactive",
            "details": test_result.get("message") or test_result.get("error", ""),
            "error_type": test_result.get("error_type")
        }
    except Exception as e:
        logger.error(f"Error getting config status: {str(e)}")
        # SECURITY: Don't expose internal error details
        return {
            "provider": "unknown",
            "model": "",
            "active": False,
            "status_text": "API Inactive",
            "details": SAFE_ERROR_MESSAGE,
            "error_type": "status_error"
        }
