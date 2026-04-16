"""
Conflict Resolution Agent - Resolves conflicts and inconsistencies in retrieved information
Handles contradictory information from multiple sources
"""

from typing import Dict, Any, List, Optional
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class ConflictResolutionAgent:
    """Resolves conflicts between different information sources"""
    
    def __init__(self):
        self.use_mock = True  # Will use mock unless conflict resolution model is available
        logger.info("Conflict Resolution Agent initialized (mock mode)")
    
    def resolve_conflicts(self, documents: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Resolve conflicts between documents for a given query
        
        Args:
            documents: List of potentially conflicting documents
            query: The original query
            
        Returns:
            Resolved information with conflict analysis
        """
        try:
            if self.use_mock:
                return self._mock_resolve(documents, query)
            else:
                return self._model_resolve(documents, query)
        except Exception as e:
            logger.error(f"Error resolving conflicts: {str(e)}")
            return {"resolved": False, "conflicts": [], "resolution": ""}
    
    def _mock_resolve(self, documents: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Mock conflict resolution - returns placeholder resolution"""
        return {
            "resolved": True,
            "conflicts": [
                {
                    "type": "factual_disagreement",
                    "documents": ["doc1", "doc2"],
                    "issue": "Different values for the same entity"
                }
            ],
            "resolution": "Conflicting information resolved by selecting most recent source",
            "confidence": 0.75
        }
    
    def _model_resolve(self, documents: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Use actual conflict resolution model"""
        try:
            # This would typically use NLI or reasoning models
            # For now, return mock as conflict resolution requires specific setup
            logger.warning("Conflict resolution model requires specific setup, using mock")
            self.use_mock = True
            return self._mock_resolve(documents, query)
            
        except Exception as e:
            logger.warning(f"Conflict resolution model not available, using mock: {str(e)}")
            self.use_mock = True
            return self._mock_resolve(documents, query)
    
    def detect_conflicts(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect conflicts between documents
        
        Args:
            documents: List of documents to check
            
        Returns:
            List of detected conflicts
        """
        try:
            conflicts = []
            
            # Check for factual disagreements
            for i in range(len(documents)):
                for j in range(i + 1, len(documents)):
                    conflict = self._compare_documents(documents[i], documents[j])
                    if conflict:
                        conflicts.append(conflict)
            
            return conflicts
        except Exception as e:
            logger.error(f"Error detecting conflicts: {str(e)}")
            return []
    
    def _compare_documents(self, doc1: Dict[str, Any], doc2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Compare two documents for conflicts
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            Conflict information if found, None otherwise
        """
        try:
            text1 = doc1.get("text", "").lower()
            text2 = doc2.get("text", "").lower()
            
            # Simple conflict detection based on negation words
            negation_pairs = [
                ("is", "is not"),
                ("has", "does not have"),
                ("will", "will not"),
                ("can", "cannot")
            ]
            
            for positive, negative in negation_pairs:
                if positive in text1 and negative in text2:
                    return {
                        "type": "factual_disagreement",
                        "documents": [doc1.get("document_id"), doc2.get("document_id")],
                        "issue": f"Contradictory information: '{positive}' vs '{negative}'"
                    }
            
            return None
        except Exception as e:
            logger.error(f"Error comparing documents: {str(e)}")
            return None
    
    def resolve_by_source_trust(self, documents: List[Dict[str, Any]], trust_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Resolve conflicts by selecting information from more trusted sources
        
        Args:
            documents: List of documents
            trust_scores: Dictionary mapping document IDs to trust scores
            
        Returns:
            Resolved list of documents
        """
        try:
            # Sort documents by trust score
            sorted_docs = sorted(
                documents,
                key=lambda x: trust_scores.get(x.get("document_id"), 0),
                reverse=True
            )
            
            return sorted_docs
        except Exception as e:
            logger.error(f"Error resolving by source trust: {str(e)}")
            return documents
    
    def resolve_by_recency(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve conflicts by selecting information from most recent sources
        
        Args:
            documents: List of documents
            
        Returns:
            Resolved list of documents
        """
        try:
            # Sort documents by metadata timestamp if available
            sorted_docs = sorted(
                documents,
                key=lambda x: x.get("metadata", {}).get("timestamp", 0),
                reverse=True
            )
            
            return sorted_docs
        except Exception as e:
            logger.error(f"Error resolving by recency: {str(e)}")
            return documents
    
    def merge_conflicting_info(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge information from conflicting documents
        
        Args:
            documents: List of documents with conflicts
            
        Returns:
            Merged information
        """
        try:
            merged = {
                "merged_text": "",
                "sources": [],
                "confidence": 0.0
            }
            
            # Combine texts from all sources
            texts = [doc.get("text", "") for doc in documents]
            merged["merged_text"] = " ".join(texts)
            merged["sources"] = [doc.get("document_id") for doc in documents]
            merged["confidence"] = 0.7  # Average confidence
            
            return merged
        except Exception as e:
            logger.error(f"Error merging conflicting info: {str(e)}")
            return {}
    
    def is_available(self) -> bool:
        """Check if conflict resolution model is available"""
        return not self.use_mock
