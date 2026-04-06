"""
Text splitting utilities for chunking documents
"""

import re
from typing import List, Dict, Any
from backend.utils.logger import setup_logger
from backend.config import settings

logger = setup_logger(__name__)


class TextSplitter:
    """Recursive Character Text Splitter - preserves sentence boundaries"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        # 🔥 Order matters: try to keep paragraphs, then sentences, then words
        self.separators = ['\n\n', '\n', '. ', '! ', '? ', ' ', '']
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text using recursive character splitter approach"""
        try:
            logger.info(f"🔥 Splitting text (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
            
            # Use recursive splitting
            chunks = self._recursive_split(text, self.separators)
            
            chunked_documents = []
            pdf_name = metadata.get('filename', 'doc') if metadata else 'doc'
            pdf_name = pdf_name.replace('.pdf', '').replace('.txt', '').replace('.docx', '')
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                chunk_doc = {
                    "id": f"{pdf_name}_{i}",
                    "text": chunk.strip(),
                    "content": chunk.strip(),
                    "chunk_index": i,
                    "metadata": metadata or {}
                }
                chunked_documents.append(chunk_doc)
            
            logger.info(f"🔥 Created {len(chunked_documents)} chunks")
            return chunked_documents
            
        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            # 🔥 Fallback: return as single chunk
            return [{"id": "doc_0", "text": text.strip(), "content": text.strip(), "chunk_index": 0, "metadata": metadata or {}}]
    
    def _smart_chunk(self, text: str) -> List[str]:
        """🔥 EXACT FIX: Recursive character splitting for small PDFs"""
        try:
            # 🔥 Use recursive splitting
            return self._recursive_split(text, self.separators)
            
        except Exception as e:
            logger.error(f"Error in smart chunking: {str(e)}")
            return self._fallback_split(text)
    
    def _fallback_split(self, text: str) -> List[str]:
        """Fallback splitting method"""
        try:
            if len(text) <= self.chunk_size:
                return [text]
            
            # Simple character-based split as fallback
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk = text[start:end]
                if chunk.strip():
                    chunks.append(chunk.strip())
                start = end
            
            return chunks
        except Exception as e:
            logger.error(f"Error in fallback split: {str(e)}")
            return [text]
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text - fix broken words, extra spaces, duplicates"""
        try:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Fix broken words (e.g., "del iv" → "deliv", but be careful with actual spaces)
            # Pattern: lowercase followed by space followed by lowercase (likely broken word)
            text = re.sub(r'([a-z])\s+([a-z])', r'\1\2', text)
            
            # Remove control characters
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
            
            # Normalize line endings
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            
            # Remove duplicate lines while preserving order
            lines = text.split('\n')
            seen = set()
            unique_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped and stripped not in seen:
                    seen.add(stripped)
                    unique_lines.append(line)
            text = '\n'.join(unique_lines)
            
            # Fix common OCR/extraction errors
            text = text.replace(' | ', ' ')  # Remove table separators
            text = re.sub(r'\.{3,}', '...', text)  # Normalize ellipsis
            text = re.sub(r'-\s*\n\s*', '', text)  # Fix hyphenation at line breaks
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """🔥 Recursive splitting - tries to keep meaningful units"""
        # If text is short enough, return as is
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # Try each separator in order
        for i, separator in enumerate(separators):
            if separator and separator in text:
                # Split by this separator
                parts = text.split(separator)
                
                # Merge parts into chunks
                chunks = []
                current_chunk = ""
                
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    
                    # Check if adding this part would exceed chunk size
                    test_chunk = current_chunk + separator + part if current_chunk else part
                    
                    if len(test_chunk) <= self.chunk_size:
                        current_chunk = test_chunk
                    else:
                        # Save current chunk if it has content
                        if current_chunk:
                            chunks.append(current_chunk)
                        
                        # If part itself is too big, split recursively with next separator
                        if len(part) > self.chunk_size:
                            sub_chunks = self._recursive_split(part, separators[i+1:])
                            chunks.extend(sub_chunks)
                            current_chunk = ""
                        else:
                            current_chunk = part
                
                # Add the last chunk
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Apply overlap between chunks
                if self.chunk_overlap > 0 and len(chunks) > 1:
                    chunks = self._apply_overlap(chunks)
                
                return chunks
        
        # No separators worked, split by character
        return self._split_by_character(text)
    
    def _split_by_character(self, text: str) -> List[str]:
        """Split text by character count"""
        try:
            chunks = []
            start = 0
            
            while start < len(text):
                end = start + self.chunk_size
                chunk = text[start:end]
                chunks.append(chunk)
                start = end - self.chunk_overlap if self.chunk_overlap > 0 else end
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting by character: {str(e)}")
            return [text]
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Apply overlap between chunks"""
        try:
            if len(chunks) <= 1 or self.chunk_overlap <= 0:
                return chunks
            
            overlapped_chunks = [chunks[0]]
            
            for i in range(1, len(chunks)):
                prev_chunk = chunks[i-1]
                current_chunk = chunks[i]
                
                # Find overlap point
                overlap_start = max(0, len(prev_chunk) - self.chunk_overlap)
                overlap_text = prev_chunk[overlap_start:]
                
                # Combine overlap with current chunk
                combined_chunk = overlap_text + current_chunk
                
                overlapped_chunks.append(combined_chunk)
            
            return overlapped_chunks
            
        except Exception as e:
            logger.error(f"Error applying overlap: {str(e)}")
            return chunks
    
    def split_by_paragraphs(self, text: str, max_paragraphs_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """Split text by paragraphs"""
        try:
            paragraphs = text.split('\n\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            chunks = []
            current_chunk = ""
            current_paragraphs = 0
            
            for paragraph in paragraphs:
                if current_paragraphs >= max_paragraphs_per_chunk:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append({
                            "content": current_chunk.strip(),
                            "paragraph_count": current_paragraphs
                        })
                    current_chunk = paragraph
                    current_paragraphs = 1
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                    current_paragraphs += 1
            
            # Add the last chunk
            if current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "paragraph_count": current_paragraphs
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting by paragraphs: {str(e)}")
            return [{"content": text, "paragraph_count": 1}]
    
    def split_by_sentences(self, text: str, max_sentences_per_chunk: int = 5) -> List[Dict[str, Any]]:
        """Split text by sentences"""
        try:
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            chunks = []
            current_chunk = ""
            current_sentences = 0
            
            for sentence in sentences:
                if current_sentences >= max_sentences_per_chunk:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append({
                            "content": current_chunk.strip() + ".",
                            "sentence_count": current_sentences
                        })
                    current_chunk = sentence
                    current_sentences = 1
                else:
                    if current_chunk:
                        current_chunk += ". " + sentence
                    else:
                        current_chunk = sentence
                    current_sentences += 1
            
            # Add the last chunk
            if current_chunk:
                chunks.append({
                    "content": current_chunk.strip() + ".",
                    "sentence_count": current_sentences
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting by sentences: {str(e)}")
            return [{"content": text, "sentence_count": 1}]
    
    def get_chunk_info(self, text: str) -> Dict[str, Any]:
        """Get information about how text would be chunked"""
        try:
            chunks = self.split_text(text)
            
            return {
                "total_chunks": len(chunks),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "average_chunk_length": sum(len(chunk["content"]) for chunk in chunks) / len(chunks) if chunks else 0,
                "min_chunk_length": min(len(chunk["content"]) for chunk in chunks) if chunks else 0,
                "max_chunk_length": max(len(chunk["content"]) for chunk in chunks) if chunks else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting chunk info: {str(e)}")
            return {
                "total_chunks": 0,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "average_chunk_length": 0,
                "min_chunk_length": 0,
                "max_chunk_length": 0
            }
