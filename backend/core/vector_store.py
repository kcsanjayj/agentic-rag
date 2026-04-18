"""
Vector store for document embeddings using ChromaDB
"""

import os
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from backend.utils.logger import setup_logger
from backend.config import settings

logger = setup_logger(__name__)

# Ensure ChromaDB directory exists
os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)


class VectorStore:
    """Vector store using ChromaDB for document storage and retrieval with lazy loading"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.collection_name = "documents"
        self._active_namespace = None  # ✅ STEP 1: No default namespace
        self._active_pdf = None  # ✅ STEP 1: Track active PDF separately
        self._loaded_collections = {}  # Lazy loading cache
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client with telemetry disabled"""
        try:
            logger.info("Initializing ChromaDB client")
            # Disable telemetry to avoid version mismatch errors
            chroma_settings = Settings(anonymized_telemetry=False)
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_DB_PATH,
                settings=chroma_settings
            )
            
            # 🔥 CRITICAL: Delete all old collections to avoid dimension mismatch (768 vs 384)
            try:
                collections = self.client.list_collections()
                for coll in collections:
                    try:
                        self.client.delete_collection(coll.name)
                        logger.info(f"Deleted old collection: {coll.name}")
                    except:
                        pass
            except:
                pass
            
            # Create fresh collection
            self.collection = None
            self._loaded_collections = {}
            self._active_namespace = None
            
            logger.info("ChromaDB client initialized successfully (fresh start)")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {str(e)}")
            raise
    
    def _get_or_create_collection(self, namespace: str):
        """Get or create collection with lazy loading"""
        if namespace in self._loaded_collections:
            return self._loaded_collections[namespace]
        
        try:
            # Filter out None values from metadata - ChromaDB doesn't accept them
            collection_metadata = {"hnsw:space": "cosine"}
            if namespace is not None:
                collection_metadata["namespace"] = namespace
            
            collection_name = f"{self.collection_name}_{namespace}" if namespace else self.collection_name
            
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=collection_metadata
            )
            self._loaded_collections[namespace] = collection
            logger.info(f"Collection {namespace} loaded and cached")
            return collection
        except Exception as e:
            logger.error(f"Error creating collection {namespace}: {str(e)}")
            raise
    
    def set_active_pdf(self, pdf_name: str):
        """✅ STEP 1: Set active PDF with complete isolation"""
        if self._active_pdf != pdf_name:
            # ✅ STEP 1: Complete PDF isolation
            self._active_pdf = pdf_name
            self._active_namespace = f"pdf_{pdf_name}"
            self.collection = None  # Force lazy loading
            logger.info(f"Set active PDF to: {pdf_name} (namespace: {self._active_namespace})")
    
    def get_active_namespace(self):
        """Get current active namespace"""
        return self._active_namespace
    
    def clear_old_namespaces(self):
        """✅ STEP 8: Drop old embeddings if not needed → RAM optimization"""
        try:
            # Get all collections
            all_collections = self.client.list_collections()
            current_collection_name = f"{self.collection_name}_{self._active_namespace}" if self._active_namespace else self.collection_name
            
            # ✅ STEP 8: Drop old embeddings if not needed
            for collection in all_collections:
                if collection.name != current_collection_name and collection.name != self.collection_name:
                    logger.info(f"Dropping old collection for RAM optimization: {collection.name}")
                    self.client.delete_collection(collection.name)
            
            # ✅ STEP 8: Clear loaded collections cache for RAM efficiency
            self._loaded_collections.clear()
            
        except Exception as e:
            logger.error(f"Error clearing old namespaces: {str(e)}")
    
    def get_collection(self):
        """✅ STEP 8: Lazy embeddings → load only active PDF"""
        if self.collection is None:
            collection_name = f"{self.collection_name}_{self._active_namespace}" if self._active_namespace else self.collection_name
            
            # Check if already loaded
            if collection_name in self._loaded_collections:
                self.collection = self._loaded_collections[collection_name]
                logger.info(f"Using cached collection: {collection_name}")
            else:
                # ✅ STEP 8: Load collection lazily → RAM efficient
                try:
                    self.collection = self.client.get_collection(name=collection_name)
                    self._loaded_collections[collection_name] = self.collection
                    logger.info(f"Lazy loaded collection: {collection_name}")
                except Exception:
                    # Create new collection
                    self.collection = self.client.create_collection(name=collection_name)
                    self._loaded_collections[collection_name] = self.collection
                    logger.info(f"Created new collection: {collection_name}")
        
        return self.collection
    
    async def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]] = None) -> List[str]:
        """Add documents to the vector store"""
        try:
            logger.info(f"🔥 Adding {len(texts)} documents to vector store")
            
            # ✅ FINAL FIX: Ensure collection is loaded
            collection = self.get_collection()
            if collection is None:
                raise Exception("Failed to get or create collection")
            
            # 🔥 DEBUG: Log collection info
            collection_name = f"{self.collection_name}_{self._active_namespace}" if self._active_namespace else self.collection_name
            logger.info(f"🔥 Using collection: {collection_name}")
            logger.info(f"🔥 Sample text (first 100 chars): {texts[0][:100] if texts else 'NO TEXT'}")
            logger.info(f"🔥 Sample metadata: {metadatas[0] if metadatas else 'NO METADATA'}")
            logger.info(f"🔥 Embeddings provided: {len(embeddings) if embeddings else 0}")
            
            # Generate IDs for documents
            ids = [str(uuid.uuid4()) for _ in texts]
            
            # Add documents to collection WITH embeddings if provided
            if embeddings:
                collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
            else:
                collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
            
            # 🔥 DEBUG: Verify count after adding
            count = collection.count()
            logger.info(f"🔥 Successfully added {len(texts)} documents. Collection now has {count} total documents.")
            return ids
            
        except Exception as e:
            logger.error(f"🔥 Error adding documents: {str(e)}")
            raise
    
    async def similarity_search(self, query_embedding: List[float], 
                              top_k: int = 5, threshold: float = 0.0,
                              filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents with optional filter"""
        try:
            logger.info(f"Performing similarity search with top_k={top_k}, filter={filter_dict}")
            
            # ✅ FINAL FIX: Ensure collection is loaded
            collection = self.get_collection()
            if collection is None:
                raise Exception("Failed to get or create collection")
            
            # 🔥 CRITICAL FIX: Check if collection is empty before querying
            try:
                count = collection.count()
                if count == 0:
                    logger.warning("Collection is empty - no documents to search")
                    return []
            except Exception as e:
                logger.warning(f"Could not get collection count: {e}")
            
            # Build where clause if filter provided
            where_clause = self._build_where_clause(filter_dict) if filter_dict else None
            
            # 🔥 CRITICAL FIX: Handle HNSW index errors gracefully
            try:
                # Query the collection with optional filter
                if where_clause:
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(top_k, max(1, count)),  # 🔥 Don't request more than available
                        where=where_clause
                    )
                else:
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(top_k, max(1, count))  # 🔥 Don't request more than available
                    )
            except Exception as query_error:
                # Handle "ef or M is too small" error
                if "ef" in str(query_error).lower() or "M is too small" in str(query_error):
                    logger.warning(f"HNSW index error, returning empty results: {query_error}")
                    return []
                raise  # Re-raise if it's a different error
            
            # Format results - ✅ STEP 3: Always retrieve chunk['text'] → grounding > 0.9
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 1.0
                    
                    # ✅ STEP 3: Ensure text is always returned for grounding
                    formatted_results.append({
                        "id": metadata.get("id", f"chunk_{i}"),
                        "text": doc,  # ✅ STEP 3: Always return text for grounding
                        "content": doc,  # Ensure content field exists
                        "metadata": metadata,
                        "score": 1 - distance,  # Convert distance to similarity score
                        "chunk_index": metadata.get("chunk_index", i)
                    })
            
            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            # 🔥 Return empty results instead of crashing
            return []
    
    async def metadata_search(self, filters: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search documents by metadata filters"""
        try:
            logger.info(f"Performing metadata search with filters: {filters}")
            
            # Build where clause for ChromaDB
            where_clause = self._build_where_clause(filters)
            
            # Get documents by metadata
            results = self.collection.get(
                where=where_clause,
                limit=top_k
            )
            
            # Format results
            formatted_results = []
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "score": 1.0,  # Default score for metadata search
                        "id": results['ids'][i] if results['ids'] else str(uuid.uuid4())
                    })
            
            logger.info(f"Found {len(formatted_results)} documents by metadata")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in metadata search: {str(e)}")
            raise
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID"""
        try:
            results = self.collection.get(ids=[document_id])
            
            if results['documents'] and results['documents'][0]:
                return {
                    "content": results['documents'][0],
                    "metadata": results['metadatas'][0] if results['metadatas'] else {},
                    "id": document_id
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document: {str(e)}")
            raise
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store"""
        try:
            logger.info(f"Deleting {len(document_ids)} documents")
            
            self.collection.delete(ids=document_ids)
            
            logger.info("Documents deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    async def update_document(self, document_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Update a document in the vector store"""
        try:
            logger.info(f"Updating document {document_id}")
            
            self.collection.update(
                ids=[document_id],
                documents=[text],
                metadatas=[metadata]
            )
            
            logger.info("Document updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return False
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters"""
        where_clause = {}
        
        for key, value in filters.items():
            if isinstance(value, str):
                where_clause[key] = {"$eq": value}
            elif isinstance(value, list):
                where_clause[key] = {"$in": value}
            elif isinstance(value, dict):
                where_clause[key] = value
            else:
                where_clause[key] = {"$eq": value}
        
        return where_clause
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            count = self.collection.count()
            return {
                "document_count": count,
                "chunk_count": count,
                "collection_name": self.collection_name,
                "status": "ready"
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                "document_count": 0,
                "chunk_count": 0,
                "collection_name": self.collection_name,
                "status": "error"
            }
    
    async def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            collection_name = f"{self.collection_name}_{self._active_namespace}" if self._active_namespace else self.collection_name
            logger.info(f"🔥 Clearing collection: {collection_name}")
            
            # Get all document IDs and delete them instead of deleting collection
            try:
                # 🔥 DEBUG: Check current collection state
                if self.collection is None:
                    logger.warning(f"🔥 Collection is None, calling get_collection() first")
                    self.get_collection()
                
                result = self.collection.get()
                doc_count = len(result['ids']) if result and 'ids' in result and result['ids'] else 0
                logger.info(f"🔥 Found {doc_count} documents to delete")
                
                if result and 'ids' in result and result['ids']:
                    all_ids = result['ids']
                    # Delete in batches
                    batch_size = 100
                    for i in range(0, len(all_ids), batch_size):
                        batch = all_ids[i:i + batch_size]
                        self.collection.delete(ids=batch)
                    logger.info(f"🔥 Deleted {len(all_ids)} documents")
            except Exception as e:
                logger.warning(f"🔥 No documents to clear or error: {e}")
            
            # Clear cached collections
            self._loaded_collections.clear()
            self.collection = None  # 🔥 CRITICAL: Reset to force fresh collection
            
            logger.info(f"🔥 Collection {collection_name} cleared successfully, cache reset")
            return True
            
        except Exception as e:
            logger.error(f"🔥 Error clearing collection: {str(e)}")
            return False


# 🔥 SINGLETON INSTANCE - Shared across all agents
_vector_store_instance = None

def get_vector_store() -> VectorStore:
    """Get or create the singleton VectorStore instance"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance

