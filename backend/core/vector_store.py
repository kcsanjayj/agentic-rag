"""
Vector store for document embeddings using ChromaDB
"""

import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from backend.utils.logger import setup_logger
from backend.config import settings

logger = setup_logger(__name__)


class VectorStore:
<<<<<<< HEAD
    """Vector store using ChromaDB for document storage and retrieval with lazy loading"""
=======
    """Vector store using ChromaDB for document storage and retrieval"""
>>>>>>> 97af6411c5fc919c79d6656e755e8bfe819e0e7e
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.collection_name = "documents"
<<<<<<< HEAD
        self._active_namespace = None  # ✅ STEP 1: No default namespace
        self._active_pdf = None  # ✅ STEP 1: Track active PDF separately
        self._loaded_collections = {}  # Lazy loading cache
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client with backward compatibility"""
        try:
            logger.info("Initializing ChromaDB client")
            self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
            
            # Try to get the old default collection first (for backward compatibility)
            try:
                old_collection = self.client.get_collection(self.collection_name)
                self.collection = old_collection
                self._active_namespace = "legacy"
                self._loaded_collections["legacy"] = old_collection
                logger.info("Using legacy collection for backward compatibility")
            except:
                # Fall back to new namespace system
                self.collection = self._get_or_create_collection(self._active_namespace)
            
=======
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client"""
        try:
            logger.info("Initializing ChromaDB client")
            self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
>>>>>>> 97af6411c5fc919c79d6656e755e8bfe819e0e7e
            logger.info("ChromaDB client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {str(e)}")
            raise
    
<<<<<<< HEAD
    def _get_or_create_collection(self, namespace: str):
        """Get or create collection with lazy loading"""
        if namespace in self._loaded_collections:
            return self._loaded_collections[namespace]
        
        try:
            collection = self.client.get_or_create_collection(
                name=f"{self.collection_name}_{namespace}",
                metadata={"hnsw:space": "cosine", "namespace": namespace}
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
    
=======
>>>>>>> 97af6411c5fc919c79d6656e755e8bfe819e0e7e
    async def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the vector store"""
        try:
            logger.info(f"Adding {len(texts)} documents to vector store")
            
<<<<<<< HEAD
            # ✅ FINAL FIX: Ensure collection is loaded
            collection = self.get_collection()
            if collection is None:
                raise Exception("Failed to get or create collection")
            
=======
>>>>>>> 97af6411c5fc919c79d6656e755e8bfe819e0e7e
            # Generate IDs for documents
            ids = [str(uuid.uuid4()) for _ in texts]
            
            # Add documents to collection
<<<<<<< HEAD
            collection.add(
=======
            self.collection.add(
>>>>>>> 97af6411c5fc919c79d6656e755e8bfe819e0e7e
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(texts)} documents")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    async def similarity_search(self, query_embedding: List[float], 
                              top_k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            logger.info(f"Performing similarity search with top_k={top_k}")
            
<<<<<<< HEAD
            # ✅ FINAL FIX: Ensure collection is loaded
            collection = self.get_collection()
            if collection is None:
                raise Exception("Failed to get or create collection")
            
            # Query the collection
            results = collection.query(
=======
            # Query the collection
            results = self.collection.query(
>>>>>>> 97af6411c5fc919c79d6656e755e8bfe819e0e7e
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
<<<<<<< HEAD
            # Format results - ✅ STEP 3: Always retrieve chunk['text'] → grounding > 0.9
=======
            # Format results
>>>>>>> 97af6411c5fc919c79d6656e755e8bfe819e0e7e
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 1.0
                    
<<<<<<< HEAD
                    # ✅ STEP 3: Ensure text is always returned for grounding
                    formatted_results.append({
                        "id": metadata.get("id", f"chunk_{i}"),
                        "text": doc,  # ✅ STEP 3: Always return text for grounding
                        "content": doc,  # Ensure content field exists
                        "metadata": metadata,
                        "score": 1 - distance,  # Convert distance to similarity score
                        "chunk_index": metadata.get("chunk_index", i)
                    })
=======
                    # Convert distance to similarity score (cosine distance to similarity)
                    similarity = 1 - distance
                    
                    if similarity >= threshold:
                        formatted_results.append({
                            "content": doc,
                            "metadata": metadata,
                            "score": similarity,
                            "id": results['ids'][0][i] if results['ids'] and results['ids'][0] else str(uuid.uuid4())
                        })
>>>>>>> 97af6411c5fc919c79d6656e755e8bfe819e0e7e
            
            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
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
            logger.info("Clearing collection")
            
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("Collection cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False
