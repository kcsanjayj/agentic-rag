"""
Multimodal Retrieval Agent - Handles retrieval across different modalities (text, images, tables)
Provides unified retrieval for mixed content types
"""

from typing import List, Dict, Any, Optional
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class MultimodalRetrievalAgent:
    """Handles retrieval across different content modalities"""
    
    def __init__(self):
        self.modality_handlers = {
            "text": self._retrieve_text,
            "image": self._retrieve_image,
            "table": self._retrieve_table,
            "document": self._retrieve_document
        }
        logger.info("Multimodal Retrieval Agent initialized")
    
    def retrieve(self, query: str, modality: str, vector_store, top_k: int = 10, 
                 filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents based on query and modality type
        
        Args:
            query: The search query
            modality: Type of content (text, image, table, document)
            vector_store: The vector store instance
            top_k: Number of results to return
            filters: Optional filters for retrieval
            
        Returns:
            List of retrieved documents
        """
        try:
            handler = self.modality_handlers.get(modality.lower(), self._retrieve_text)
            return handler(query, vector_store, top_k, filters)
        except Exception as e:
            logger.error(f"Error in multimodal retrieval: {str(e)}")
            return []
    
    def _retrieve_text(self, query: str, vector_store, top_k: int, 
                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve text documents"""
        try:
            results = vector_store.search(query, top_k=top_k, filter=filters)
            return results
        except Exception as e:
            logger.error(f"Text retrieval error: {str(e)}")
            return []
    
    def _retrieve_image(self, query: str, vector_store, top_k: int,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve image-based documents (using OCR/captions)"""
        try:
            # Add image-specific filter
            image_filters = filters or {}
            image_filters["modality"] = "image"
            
            results = vector_store.search(query, top_k=top_k, filter=image_filters)
            return results
        except Exception as e:
            logger.error(f"Image retrieval error: {str(e)}")
            return []
    
    def _retrieve_table(self, query: str, vector_store, top_k: int,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve table data"""
        try:
            # Add table-specific filter
            table_filters = filters or {}
            table_filters["modality"] = "table"
            
            results = vector_store.search(query, top_k=top_k, filter=table_filters)
            return results
        except Exception as e:
            logger.error(f"Table retrieval error: {str(e)}")
            return []
    
    def _retrieve_document(self, query: str, vector_store, top_k: int,
                         filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve complete documents (all modalities)"""
        try:
            results = vector_store.search(query, top_k=top_k, filter=filters)
            return results
        except Exception as e:
            logger.error(f"Document retrieval error: {str(e)}")
            return []
    
    def hybrid_search(self, query: str, vector_store, top_k: int = 10,
                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search across all modalities
        Combines results from different modalities
        """
        all_results = []
        
        for modality in ["text", "image", "table"]:
            try:
                results = self.retrieve(query, modality, vector_store, top_k, filters)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Hybrid search failed for {modality}: {str(e)}")
        
        # Remove duplicates and return top_k
        unique_results = self._deduplicate(all_results)
        return unique_results[:top_k]
    
    def _deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate documents based on document_id"""
        seen_ids = set()
        unique = []
        
        for result in results:
            doc_id = result.get("document_id") or result.get("doc_id")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique.append(result)
        
        return unique
