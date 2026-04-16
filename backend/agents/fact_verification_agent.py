"""
Fact Verification Agent - Verifies factual accuracy of generated content
Cross-references claims against retrieved documents to ensure accuracy
"""

from typing import Dict, Any, List, Optional
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class FactVerificationAgent:
    """Verifies factual accuracy of generated content against source documents"""
    
    def __init__(self):
        self.use_mock = True  # Will use mock unless fact-checking model is available
        logger.info("Fact Verification Agent initialized (mock mode)")
    
    def verify_facts(self, generated_text: str, source_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify facts in generated text against source documents
        
        Args:
            generated_text: The text to verify
            source_documents: List of source documents for cross-reference
            
        Returns:
            Verification results with confidence scores
        """
        try:
            if self.use_mock:
                return self._mock_verify(generated_text, source_documents)
            else:
                return self._model_verify(generated_text, source_documents)
        except Exception as e:
            logger.error(f"Error verifying facts: {str(e)}")
            return {"verified": False, "confidence": 0.0, "issues": []}
    
    def _mock_verify(self, generated_text: str, source_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mock verification - returns placeholder verification"""
        return {
            "verified": True,
            "confidence": 0.85,
            "issues": [],
            "fact_checks": [
                {
                    "claim": "Sample claim from generated text",
                    "supported": True,
                    "source": "source_documents[0]",
                    "confidence": 0.9
                }
            ]
        }
    
    def _model_verify(self, generated_text: str, source_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use actual fact-checking model"""
        try:
            # This would typically use NLI or fact-checking models
            # For now, return mock as fact-checking requires specific setup
            logger.warning("Fact-checking model requires specific setup, using mock")
            self.use_mock = True
            return self._mock_verify(generated_text, source_documents)
            
        except Exception as e:
            logger.warning(f"Fact-checking model not available, using mock: {str(e)}")
            self.use_mock = True
            return self._mock_verify(generated_text, source_documents)
    
    def extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text
        
        Args:
            text: The text to extract claims from
            
        Returns:
            List of extracted claims
        """
        try:
            # Simple claim extraction based on sentence structure
            sentences = text.split(". ")
            claims = []
            
            for sentence in sentences:
                # Simple heuristic: sentences with numbers, dates, or specific entities
                if any(char.isdigit() for char in sentence):
                    claims.append(sentence.strip())
            
            return claims[:10]  # Return top 10 claims
        except Exception as e:
            logger.error(f"Error extracting claims: {str(e)}")
            return []
    
    def verify_claim(self, claim: str, source_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify a single claim against source documents
        
        Args:
            claim: The claim to verify
            source_documents: Source documents for verification
            
        Returns:
            Verification result for the claim
        """
        try:
            claim_lower = claim.lower()
            
            # Check if claim is supported by source documents
            supported = False
            supporting_docs = []
            
            for doc in source_documents:
                doc_text = doc.get("text", "").lower()
                # Simple keyword overlap check
                claim_words = set(claim_lower.split())
                doc_words = set(doc_text.split())
                overlap = len(claim_words & doc_words)
                
                if overlap / max(len(claim_words), 1) > 0.5:
                    supported = True
                    supporting_docs.append(doc.get("document_id"))
            
            return {
                "claim": claim,
                "supported": supported,
                "supporting_documents": supporting_docs,
                "confidence": 0.8 if supported else 0.3
            }
        except Exception as e:
            logger.error(f"Error verifying claim: {str(e)}")
            return {
                "claim": claim,
                "supported": False,
                "supporting_documents": [],
                "confidence": 0.0
            }
    
    def identify_hallucinations(self, generated_text: str, source_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify potential hallucinations in generated text
        
        Args:
            generated_text: The text to check for hallucinations
            source_documents: Source documents for verification
            
        Returns:
            List of potential hallucinations
        """
        try:
            claims = self.extract_claims(generated_text)
            hallucinations = []
            
            for claim in claims:
                verification = self.verify_claim(claim, source_documents)
                if not verification["supported"] and verification["confidence"] < 0.5:
                    hallucinations.append({
                        "claim": claim,
                        "reason": "Not supported by source documents",
                        "confidence": 1 - verification["confidence"]
                    })
            
            return hallucinations
        except Exception as e:
            logger.error(f"Error identifying hallucinations: {str(e)}")
            return []
    
    def is_available(self) -> bool:
        """Check if fact-checking model is available"""
        return not self.use_mock
