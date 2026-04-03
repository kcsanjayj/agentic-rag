"""
Smart Document Agent - Adaptive intelligence for document analysis
"""

from typing import Dict, Any, List
import re
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class SmartDocumentAgent:
    """Agent for intelligent document analysis and adaptive responses"""
    
    def __init__(self):
        self.quality_indicators = {
            "high_info": ["methodology", "analysis", "results", "conclusion", "data", "research"],
            "medium_info": ["summary", "overview", "introduction", "background", "description"],
            "low_info": ["test", "sample", "example", "demo", "placeholder", "template"]
        }
        
        self.document_types = {
            "research_paper": ["abstract", "methodology", "results", "discussion", "references"],
            "technical_doc": ["installation", "configuration", "api", "parameters", "examples"],
            "business_doc": ["executive", "financial", "strategy", "market", "revenue"],
            "legal_doc": ["terms", "conditions", "liability", "agreement", "clause"],
            "test_doc": ["sample", "test", "demo", "placeholder", "example"]
        }
    
    async def analyze_document_quality(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
<<<<<<< HEAD
        """✅ FINAL FIX: Dynamic extraction mode based on size and content"""
=======
        """Analyze document quality with 9.8 semantic density detection"""
>>>>>>> 97af6411c5fc919c79d6656e755e8bfe819e0e7e
        if not documents:
            return {
                "quality_score": 0.0,
                "semantic_depth": "none",
                "document_type": "unknown",
                "recommendations": ["upload_documents"],
                "avg_chunk_length": 0,
                "semantic_density": 0.0,
<<<<<<< HEAD
                "is_high_density": False,
                "extraction_mode": "unknown"
=======
                "is_high_density": False
>>>>>>> 97af6411c5fc919c79d6656e755e8bfe819e0e7e
            }
        
        # Combine all document content
        all_content = " ".join([doc["content"] for doc in documents])
        content_lower = all_content.lower()
        
<<<<<<< HEAD
        # ✅ STEP 2: Auto mode detection based on PDF type & size
        file_size_bytes = len(all_content.encode('utf-8'))
        
        if file_size_bytes < 200 * 1024:  # small PDFs < 200KB
            extraction_mode = 'simple_extraction'
        elif any(word in content_lower for word in ['experience', 'project']):
            extraction_mode = 'resume_mode'
        else:
            extraction_mode = 'full_rag_mode'
        
        # ✅ FINAL FIX: Keyword-based quality check
        keywords = ['project', 'experience', 'pipeline', 'skills', 'dataset', 'model']
        keyword_score = sum(1 for k in keywords if k in content_lower)
        
        # Set document type based on keywords and mode
        if extraction_mode == 'resume_mode':
            doc_type = "resume"
            semantic_depth = "resume_content"
        elif keyword_score >= 2:
            doc_type = "structured_doc"
            semantic_depth = "structured"
        else:
            doc_type = "sparse"
            semantic_depth = "sparse"
        
        # Basic metrics
        word_count = len(content_lower.split())
        avg_chunk_length = len(all_content) / len(documents) if documents else 0
        
        return {
            "quality_score": keyword_score / len(keywords),  # ✅ FINAL FIX: Normalized keyword score
            "semantic_depth": semantic_depth,
            "document_type": doc_type,
            "recommendations": [],
            "avg_chunk_length": round(avg_chunk_length, 1),
            "word_count": word_count,
            "semantic_density": keyword_score / len(keywords),
            "is_technical": doc_type in ["structured_doc", "resume"],
            "is_high_density": doc_type == "structured_doc",
            "high_density_keywords_found": keyword_score,
            "content_preview": all_content[:200] + "..." if len(all_content) > 200 else all_content,
            "analysis_summary": f"Mode: {extraction_mode}, Keywords: {keyword_score}",
            "mini_score": keyword_score / len(keywords),
            "extraction_mode": extraction_mode  # ✅ FINAL FIX: Add extraction mode
=======
        # Calculate quality metrics
        quality_score = 0.0
        semantic_depth = "low"
        document_type = "general"
        
        # 1. Content length analysis
        word_count = len(content_lower.split())
        avg_chunk_length = len(all_content) / len(documents) if documents else 0
        
        # 2. Semantic density calculation (9.5+ feature)
        semantic_indicators = [
            "methodology", "analysis", "research", "results", "conclusion", "data",
            "process", "algorithm", "implementation", "architecture", "design",
            "specification", "requirements", "functionality", "performance"
        ]
        
        semantic_density = sum(1 for indicator in semantic_indicators if indicator in content_lower) / len(semantic_indicators)
        
        # 10/10 FEATURE: Technical/Dense Detection (improved)
        # Check for technical patterns even in short content
        high_density_keywords = [
            "threshold", "classification", "model", "rate", "analysis",
            "fraction", "percentage", "binary", "converted", "composite",
            "flood", "remote sensing", "mapping", "detection",
            # 10/10: Add missing technical terms
            "estimate", "range", "bound", "combined", "conservative",
            "aggregation", "maximum", "minimum", "uncertainty", "statistical"
        ]
        
        # Count high-density technical terms
        high_density_count = sum(1 for keyword in high_density_keywords if keyword in content_lower)
        
        # 10/10: More sensitive detection - even 1 strong keyword in short text is enough
        is_technical_short = high_density_count >= 1 and word_count < 50
        is_high_density = high_density_count >= 2 and word_count < 100
        is_technical = is_technical_short or is_high_density
        
        # 10/10 RULE: Short + Technical/Dense = SPARSE_REASONING (not fallback)
        if is_technical:
            # 10/10: Technical short content - SPARSE_REASONING mode
            document_type = "high_density_technical"
            quality_score = 0.6  # Medium-high for dense technical content
            semantic_depth = "sparse_reasoning"
        elif (
            avg_chunk_length < 100 or  # Short chunks
            semantic_density < 0.1 or  # Low semantic density
            word_count < 200 or  # Very short overall
            any(test_word in content_lower for test_word in ["test", "sample", "demo", "example", "placeholder"])
        ):
            document_type = "low_information"
            quality_score = min(quality_score + 0.1, 0.3)  # Cap at 0.3 for low-info docs
            semantic_depth = "minimal"
        else:
            # Standard quality calculation for normal documents
            if word_count > 1000:
                quality_score += 0.3
                semantic_depth = "high"
            elif word_count > 300:
                quality_score += 0.2
                semantic_depth = "medium"
            else:
                quality_score += 0.1
                semantic_depth = "low"
            
            # Semantic richness
            high_info_count = sum(1 for indicator in self.quality_indicators["high_info"] 
                                  if indicator in content_lower)
            if high_info_count >= 3:
                quality_score += 0.4
            elif high_info_count >= 1:
                quality_score += 0.2
            
            # Document type detection
            type_scores = {}
            for doc_type, indicators in self.document_types.items():
                score = sum(1 for indicator in indicators if indicator in content_lower)
                type_scores[doc_type] = score
            
            if type_scores:
                best_type = max(type_scores, key=type_scores.get)
                if type_scores[best_type] >= 2:
                    document_type = best_type
                    quality_score += 0.3
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            quality_score, semantic_depth, document_type, word_count
        )
        
        return {
            "quality_score": round(quality_score, 2),
            "semantic_depth": semantic_depth,
            "document_type": document_type,
            "recommendations": recommendations,
            "avg_chunk_length": round(avg_chunk_length, 1),
            "word_count": word_count,
            "semantic_density": round(semantic_density, 2),
            "is_technical": is_technical,  # 10/10: Export for routing
            "is_high_density": is_high_density,
            "high_density_keywords_found": high_density_count if is_technical else 0,
            "content_preview": all_content[:200] + "..." if len(all_content) > 200 else all_content,
            "analysis_summary": self._generate_summary(quality_score, semantic_depth, document_type),
            "mini_score": self._calculate_mini_score(quality_score, semantic_depth, avg_chunk_length, word_count)
>>>>>>> 97af6411c5fc919c79d6656e755e8bfe819e0e7e
        }
    
    def _generate_recommendations(self, quality_score: float, semantic_depth: str,
                                 document_type: str, word_count: int) -> List[str]:
        """Generate intelligent recommendations based on document analysis"""
        recommendations = []
        
        if quality_score < 0.3:
            recommendations.extend([
                "Document contains insufficient semantic depth",
                "Consider uploading more comprehensive documents",
                "Switching to metadata-based summary"
            ])
        elif quality_score < 0.6:
            recommendations.extend([
                "Document has moderate information content",
                "Answers will be based on limited available context"
            ])
        
        if semantic_depth == "low":
            recommendations.append("Use general knowledge to supplement document content")
        
        # Type-specific recommendations
        if document_type == "test_doc":
            recommendations.append("This appears to be a test document - expect limited answers")
        elif document_type == "research_paper":
            recommendations.append("Focus on methodology and results sections")
        
        return recommendations
    
    def _generate_summary(self, quality_score: float, semantic_depth: str, 
                          document_type: str) -> str:
        """Generate analysis summary"""
        if quality_score < 0.3:
            return f"Low-quality {document_type} with {semantic_depth} semantic depth"
        elif quality_score < 0.7:
            return f"Medium-quality {document_type} with {semantic_depth} information content"
        else:
            return f"High-quality {document_type} with rich semantic content"
    
    def _calculate_mini_score(self, quality_score: float, semantic_depth: str, 
                           avg_chunk_length: float, word_count: int) -> Dict[str, Any]:
        """9.7+ FEATURE: Calculate mini score for document quality"""
        
        # Content density score (0-10)
        density_score = min(quality_score * 10, 10)
        
        # Structure score (0-10)
        structure_score = 0
        if semantic_depth == "high":
            structure_score = 8
        elif semantic_depth == "medium":
            structure_score = 5
        elif semantic_depth == "low":
            structure_score = 3
        else:
            structure_score = 1
        
        # Extractability score (0-10)
        extractability_score = 0
        if avg_chunk_length > 200:
            extractability_score = 8
        elif avg_chunk_length > 100:
            extractability_score = 5
        elif avg_chunk_length > 50:
            extractability_score = 3
        else:
            extractability_score = 1
        
        # Overall mini score (0-10, weighted)
        overall_score = (density_score * 0.4 + structure_score * 0.3 + extractability_score * 0.3)
        
        return {
            "overall": round(overall_score, 1),
            "content_density": round(density_score, 1),
            "structure": round(structure_score, 1),
            "extractability": round(extractability_score, 1),
            "grade": self._get_grade(overall_score)
        }
    
    def _get_grade(self, score: float) -> str:
        """Get letter grade from score"""
        if score >= 8:
            return "A"
        elif score >= 6:
            return "B"
        elif score >= 4:
            return "C"
        elif score >= 2:
            return "D"
        else:
            return "F"
    
    async def adapt_response_strategy(self, doc_analysis: Dict[str, Any], 
                                     query: str) -> Dict[str, Any]:
        """Adapt response strategy based on document quality"""
        quality_score = doc_analysis["quality_score"]
        semantic_depth = doc_analysis["semantic_depth"]
        
        strategy = {
            "retrieval_strategy": "standard",
            "generation_approach": "grounded",
            "confidence_adjustment": 0.0,
            "response_guidelines": []
        }
        
        if quality_score < 0.3:
            # Low-quality documents
            strategy.update({
                "retrieval_strategy": "expanded",
                "generation_approach": "supplemented",
                "confidence_adjustment": -0.2,
                "response_guidelines": [
                    "Acknowledge limited document content",
                    "Use general knowledge to supplement",
                    "Provide caveats about answer limitations"
                ]
            })
        elif quality_score < 0.7:
            # Medium-quality documents
            strategy.update({
                "retrieval_strategy": "standard",
                "generation_approach": "balanced",
                "confidence_adjustment": -0.1,
                "response_guidelines": [
                    "Focus on available document content",
                    "Note any information gaps"
                ]
            })
        else:
            # High-quality documents
            strategy.update({
                "retrieval_strategy": "focused",
                "generation_approach": "strictly_grounded",
                "confidence_adjustment": 0.1,
                "response_guidelines": [
                    "Strictly use document content",
                    "Provide detailed evidence-based answers"
                ]
            })
        
        return strategy
    
    async def detect_content_gaps(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect gaps between query intent and available content"""
        if not retrieved_docs:
            return {
                "has_gaps": True,
                "gap_type": "no_documents",
                "suggestions": ["upload_relevant_documents", "broaden_query"]
            }
        
        query_terms = set(query.lower().split())
        available_terms = set()
        
        for doc in retrieved_docs:
            available_terms.update(doc["content"].lower().split())
        
        missing_terms = query_terms - available_terms
        coverage_ratio = len(query_terms - missing_terms) / len(query_terms) if query_terms else 0
        
        has_gaps = coverage_ratio < 0.5
        gap_type = "partial_coverage" if 0.2 < coverage_ratio < 0.5 else "poor_coverage"
        
        suggestions = []
        if has_gaps:
            if coverage_ratio < 0.2:
                suggestions.extend(["upload_relevant_documents", "rephrase_query"])
            else:
                suggestions.extend(["expand_query", "use_synonyms"])
        
        return {
            "has_gaps": has_gaps,
            "gap_type": gap_type,
            "coverage_ratio": round(coverage_ratio, 2),
            "missing_terms": list(missing_terms)[:5],  # Top 5 missing terms
            "suggestions": suggestions
        }
