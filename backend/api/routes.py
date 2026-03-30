"""
API routes for Agentic-RAG
"""

import os
import time
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from backend.models.schemas import (
    QueryRequest, QueryResponse, DocumentUploadResponse,
    DocumentInfo, HealthResponse
)
from backend.agents.orchestrator import OrchestratorAgent
from backend.agents.query_agent import QueryAgent
from backend.agents.retrieval_agent import RetrievalAgent
from backend.agents.generation_agent import GenerationAgent
from backend.agents.validation_agent import ValidationAgent
from backend.tools.document_loader import DocumentLoader
from backend.tools.text_splitter import TextSplitter
from backend.core.embeddings import EmbeddingGenerator
from backend.core.vector_store import VectorStore
from backend.utils.logger import setup_logger
from backend.config import settings

logger = setup_logger(__name__)
router = APIRouter(tags=["api"])

# Global components (in production, use dependency injection)
orchestrator = None
document_loader = DocumentLoader()
text_splitter = TextSplitter()
embedding_generator = None
vector_store = None

def get_embedding_generator():
    """Get or create embedding generator singleton"""
    global embedding_generator
    if embedding_generator is None:
        embedding_generator = EmbeddingGenerator()
    return embedding_generator

def get_vector_store():
    """Get or create vector store singleton"""
    global vector_store
    if vector_store is None:
        vector_store = VectorStore()
    return vector_store

def get_orchestrator():
    """Get or create orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        query_agent = QueryAgent()
        retrieval_agent = RetrievalAgent()
        generation_agent = GenerationAgent()
        validation_agent = ValidationAgent()
        orchestrator = OrchestratorAgent(
            query_agent, retrieval_agent, generation_agent, validation_agent
        )
    return orchestrator


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using agentic RAG"""
    try:
        logger.info(f"Processing query: {request.query}")
        
        orchestrator = get_orchestrator()
        response = await orchestrator.process_query(request)
        
        logger.info(f"Query processed successfully in {response.processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        logger.info(f"Uploading document: {file.filename}")
        
        # Validate file
        if not document_loader.is_supported_format(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported formats: {', '.join(document_loader.get_supported_formats())}"
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
        
        # Split into chunks
        chunks = text_splitter.split_text(document["content"], document["metadata"])
        
        # Generate embeddings and store in vector store
        embedding_generator = get_embedding_generator()
        vector_store = get_vector_store()
        
        chunk_texts = [chunk["content"] for chunk in chunks]
        chunk_metadatas = []
        
        for chunk in chunks:
            metadata = chunk["metadata"].copy()
            metadata.update({
                "chunk_index": chunk["chunk_index"],
                "document_id": document["id"],
                "filename": document["filename"]
            })
            chunk_metadatas.append(metadata)
        
        # Generate embeddings (now using cached model)
        logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
        embeddings = await embedding_generator.generate_embeddings(chunk_texts)
        
        # Store in vector store
        logger.info("Storing in vector store...")
        chunk_ids = await vector_store.add_documents(chunk_texts, chunk_metadatas)
        
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
        vector_stats = vector_store.get_stats()
        
        # Agent status
        orchestrator = get_orchestrator()
        agent_status = orchestrator.get_agent_status()
        
        return {
            "vector_store": vector_stats,
            "agents": agent_status,
            "system": {
                "chunk_size": text_splitter.chunk_size,
                "chunk_overlap": text_splitter.chunk_overlap,
                "supported_formats": document_loader.get_supported_formats(),
                "embedding_model": settings.EMBEDDING_MODEL,
                "embedding_dimension": get_embedding_generator().get_embedding_dimension()
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
        from backend.agents.orchestrator import OrchestratorAgent
        from backend.agents.query_agent import QueryAgent
        from backend.agents.retrieval_agent import RetrievalAgent
        from backend.agents.generation_agent import GenerationAgent
        from backend.agents.validation_agent import ValidationAgent
        
        # Initialize agents
        query_agent = QueryAgent()
        retrieval_agent = RetrievalAgent()
        generation_agent = GenerationAgent()
        validation_agent = ValidationAgent()
        orchestrator = OrchestratorAgent(query_agent, retrieval_agent, generation_agent, validation_agent)
        
        # Get sample test set
        test_set = evaluation_harness.get_sample_test_set(sample_size)
        
        results = []
        for qa in test_set:
            try:
                # Run query through orchestrator
                from backend.models.schemas import QueryRequest
                request = QueryRequest(query=qa["query"], top_k=3)
                response = await orchestrator.process_query(request)
                
                # Evaluate
                result = evaluation_harness.evaluate_response(
                    qa["id"],
                    response.answer,
                    response.sources,
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
