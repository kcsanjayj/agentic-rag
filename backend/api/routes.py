"""
API routes for Agentic-RAG
"""

import os
import time
import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from backend.models.schemas import (
    QueryRequest, QueryResponse, DocumentUploadResponse,
    DocumentInfo, HealthResponse
)
from backend.agents.orchestrator import Orchestrator
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

# Global components (in production, use dependency injection)
orchestrator = None
document_loader = DocumentLoader()
text_splitter = TextSplitter()
embedding_generator = None
vector_store = None

# 🎯 Track active document to prevent data mixing
active_document_id = None
active_document_name = None

def get_embedding_generator():
    """Get or create embedding generator singleton"""
    global embedding_generator
    if embedding_generator is None:
        embedding_generator = EmbeddingGenerator()
    return embedding_generator

# 🔥 DELETED: Duplicate get_vector_store() - now imported from backend.core.vector_store

def get_orchestrator():
    """🧠 Get Professional Orchestrator with 6-component agentic system"""
    return Orchestrator()

def set_active_document(doc_id: str, filename: str):
    """🎯 Set the currently active document for retrieval filtering"""
    global active_document_id, active_document_name
    active_document_id = doc_id
    active_document_name = filename
    logger.info(f"🎯 Active document set: {filename} (ID: {doc_id})")

def get_active_document():
    """Get currently active document info"""
    return {"id": active_document_id, "filename": active_document_name}


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using agentic RAG - filtered by active document"""
    try:
        logger.info(f"Processing query: {request.query}")
        
        # 🎯 Check if we have an active document
        active_doc = get_active_document()
        if not active_doc["id"]:
            return QueryResponse(
                query=request.query,
                answer="❌ No document uploaded yet. Please upload a resume first.",
                sources=[],
                agent_steps=[],
                processing_time=0.0,
                confidence_score=0.0,
                conversation_id=request.conversation_id or str(uuid.uuid4())
            )
        
        logger.info(f"🎯 Querying against active document: {active_doc['filename']}")
        
        orchestrator = get_orchestrator()
        response = await orchestrator.process_query(request, active_document_id=active_doc["id"])
        
        logger.info(f"Query processed successfully in {response.processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
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
        
        # Save uploaded file
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        
        # Write file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
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
        from langchain.text_splitter import RecursiveCharacterTextSplitter
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
        
        # Generate embeddings and store in vector store
        embedding_generator = get_embedding_generator()
        
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
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
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
        logger.error(f"Error uploading document: {str(e)}")
        # Clean up file if upload failed
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Basic health check
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            components={"status": "operational"}
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            components={"error": str(e)}
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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_config():
    """Get current AI configuration"""
    try:
        return {
            "ai_provider": settings.AI_PROVIDER,
            "ai_configured": settings.is_ai_configured(),
            "config": settings.get_ai_config(),
            "supported_providers": ["openai", "gemini", "anthropic", "local"],
            "embedding_model": settings.EMBEDDING_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP
        }
        
    except Exception as e:
        logger.error(f"Error getting config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test-retrieval")
async def test_retrieval():
    """Debug route to test retrieval pipeline"""
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
        return {
            "status": "error",
            "message": str(e)
        }


@router.post("/test")
async def test_ai_connection():
    """Test AI connection with current configuration"""
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
        return {
            "status": "error",
            "error": str(e),
            "message": "AI connection test failed"
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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
                results.append({"qa_id": qa["id"], "error": str(e), "passed": False})
        
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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))

# Configuration Management Endpoints
@router.get("/config")
async def get_config():
    """Get current AI configuration"""
    try:
        # Check runtime config first, then fall back to settings
        from backend.config import get_runtime_config
        provider = get_runtime_config("AI_PROVIDER") or settings.AI_PROVIDER
        
        return {
            "provider": provider,
            "model": get_runtime_config(f"{provider.upper()}_MODEL") or getattr(settings, f"{provider.upper()}_MODEL", ""),
            "temperature": get_runtime_config(f"{provider.upper()}_TEMPERATURE") or getattr(settings, f"{provider.upper()}_TEMPERATURE", 0.7),
            "configured": settings.is_ai_configured()
        }
    except Exception as e:
        logger.error(f"Error getting config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/config")
async def update_config(config_data: Dict[str, Any]):
    """Update AI configuration using runtime config (no .env file needed)"""
    try:
        from backend.config import set_runtime_config, update_runtime_config
        
        provider = config_data.get("provider", "gemini")
        api_key = config_data.get("api_key", "")
        model = config_data.get("model", "")
        temperature = config_data.get("temperature", 0.7)
        
        # Update runtime configuration (no .env file needed!)
        update_runtime_config({
            "AI_PROVIDER": provider,
            f"{provider.upper()}_API_KEY": api_key,
        })
        
        if model:
            set_runtime_config(f"{provider.upper()}_MODEL", model)
        if temperature:
            set_runtime_config(f"{provider.upper()}_TEMPERATURE", temperature)
        
        return {
            "success": True,
            "message": "Configuration updated successfully - no .env file needed!",
            "provider": provider,
            "model": model,
            "temperature": temperature
        }
        
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test-api")
async def test_api_connection(test_data: Dict[str, Any]):
    """Test API connection for a provider"""
    try:
        provider = test_data.get("provider", "")
        api_key = test_data.get("api_key", "")
        
        if not provider or not api_key:
            return {
                "success": False,
                "error": "Provider and API key are required"
            }
        
        # Test different providers
        if provider == "nvidia":
            url = "https://integrate.api.nvidia.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "meta/llama3-70b-instruct",
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
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
        elif provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{"parts": [{"text": "Hello"}]}],
                "generationConfig": {"maxOutputTokens": 10}
            }
        else:
            return {
                "success": False,
                "error": f"Provider {provider} not supported for testing"
            }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data, timeout=10) as response:
                if response.status == 200:
                    return {
                        "success": True,
                        "message": f"{provider} API connection successful"
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"{provider} API error: {response.status} - {error_text[:200]}"
                    }
        
    except Exception as e:
        logger.error(f"Error testing API: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
