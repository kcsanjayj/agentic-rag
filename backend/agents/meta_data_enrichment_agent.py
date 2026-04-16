"""
Meta Data Enrichment Agent - Enriches documents with additional metadata
Extracts and adds contextual metadata to improve retrieval and understanding
"""

from typing import Dict, Any, List, Optional
from backend.utils.logger import setup_logger
import datetime

logger = setup_logger(__name__)


class MetaDataEnrichmentAgent:
    """Enriches documents with additional metadata for better retrieval"""
    
    def __init__(self):
        self.use_mock = True  # Will use mock unless NER/NER model is available
        logger.info("Meta Data Enrichment Agent initialized (mock mode)")
    
    def enrich_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a document with additional metadata
        
        Args:
            document: The document to enrich
            
        Returns:
            Enriched document with additional metadata
        """
        try:
            if self.use_mock:
                return self._mock_enrich(document)
            else:
                return self._model_enrich(document)
        except Exception as e:
            logger.error(f"Error enriching document: {str(e)}")
            return document
    
    def _mock_enrich(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Mock enrichment - returns document with placeholder metadata"""
        enriched = document.copy()
        enriched["metadata"] = enriched.get("metadata", {})
        enriched["metadata"].update({
            "entities": ["Entity1", "Entity2"],
            "topics": ["Topic1", "Topic2"],
            "keywords": ["keyword1", "keyword2"],
            "sentiment": "neutral",
            "language": "en",
            "enrichment_timestamp": datetime.datetime.now().isoformat()
        })
        return enriched
    
    def _model_enrich(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Use actual NER/NER models for enrichment"""
        try:
            # This would typically use spaCy, transformers, etc.
            # For now, return mock as NER requires specific setup
            logger.warning("NER model requires specific setup, using mock")
            self.use_mock = True
            return self._mock_enrich(document)
            
        except Exception as e:
            logger.warning(f"NER model not available, using mock: {str(e)}")
            self.use_mock = True
            return self._mock_enrich(document)
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text
        
        Args:
            text: The text to extract entities from
            
        Returns:
            List of extracted entities with types
        """
        try:
            # Simple entity extraction based on capitalization
            entities = []
            words = text.split()
            
            for i, word in enumerate(words):
                if word[0].isupper() and len(word) > 2:
                    entities.append({
                        "text": word,
                        "type": "UNKNOWN",
                        "start": i,
                        "end": i + 1
                    })
            
            return entities[:20]  # Return top 20 entities
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text
        
        Args:
            text: The text to extract keywords from
            
        Returns:
            List of extracted keywords
        """
        try:
            # Simple keyword extraction based on word frequency
            words = text.lower().split()
            word_freq = {}
            
            for word in words:
                if len(word) > 3:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency and return top 10
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in sorted_words[:10]]
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def detect_topics(self, text: str) -> List[str]:
        """
        Detect topics in text
        
        Args:
            text: The text to detect topics in
            
        Returns:
            List of detected topics
        """
        try:
            # Simple topic detection based on keyword clusters
            keywords = self.extract_keywords(text)
            
            topic_keywords = {
                "technology": ["computer", "software", "system", "digital", "data"],
                "business": ["company", "market", "sales", "customer", "revenue"],
                "science": ["research", "study", "experiment", "analysis", "result"],
                "health": ["health", "medical", "patient", "treatment", "disease"]
            }
            
            detected_topics = []
            for topic, topic_words in topic_keywords.items():
                if any(keyword in keywords for keyword in topic_words):
                    detected_topics.append(topic)
            
            return detected_topics if detected_topics else ["general"]
        except Exception as e:
            logger.error(f"Error detecting topics: {str(e)}")
            return []
    
    def detect_sentiment(self, text: str) -> str:
        """
        Detect sentiment of text
        
        Args:
            text: The text to analyze
            
        Returns:
            Sentiment label (positive, negative, neutral)
        """
        try:
            # Simple sentiment detection based on word lists
            positive_words = ["good", "great", "excellent", "positive", "success", "improve"]
            negative_words = ["bad", "poor", "negative", "failure", "problem", "worse"]
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                return "positive"
            elif negative_count > positive_count:
                return "negative"
            else:
                return "neutral"
        except Exception as e:
            logger.error(f"Error detecting sentiment: {str(e)}")
            return "neutral"
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text
        
        Args:
            text: The text to analyze
            
        Returns:
            Language code (e.g., 'en', 'es', 'fr')
        """
        try:
            # Simple language detection based on common words
            # For now, default to English
            return "en"
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return "en"
    
    def enrich_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich a batch of documents
        
        Args:
            documents: List of documents to enrich
            
        Returns:
            List of enriched documents
        """
        try:
            enriched_docs = []
            for doc in documents:
                enriched = self.enrich_document(doc)
                enriched_docs.append(enriched)
            return enriched_docs
        except Exception as e:
            logger.error(f"Error enriching batch: {str(e)}")
            return documents
    
    def is_available(self) -> bool:
        """Check if NER/NER model is available"""
        return not self.use_mock
