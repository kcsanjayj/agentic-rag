"""
Document loading utilities for various file formats
"""

import os
import uuid
from typing import List, Dict, Any, Optional
import aiofiles
from pathlib import Path
from backend.utils.logger import setup_logger
from backend.config import settings

logger = setup_logger(__name__)


class DocumentLoader:
    """Loads documents from various file formats"""
    
    def __init__(self):
        self.supported_formats = {
            '.txt', '.md', '.pdf', '.docx', '.doc', 
            '.html', '.htm', '.rtf', '.csv'
        }
        self.max_file_size = settings.MAX_DOCUMENT_SIZE
    
    async def load_document(self, file_path: str, filename: str = None) -> Dict[str, Any]:
        """Load a single document"""
        try:
            logger.info(f"Loading document: {file_path}")
            
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
            
            # Get file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Use filename from parameter or actual filename
            actual_filename = filename or os.path.basename(file_path)
            
            # Load content based on file type
            content = await self._load_content(file_path, file_ext)
            
            # Create document metadata
            document = {
                "id": str(uuid.uuid4()),
                "filename": actual_filename,
                "file_type": file_ext,
                "file_size": file_size,
                "content": content,
                "upload_date": None,  # Will be set by caller
                "metadata": {
                    "source": file_path,
                    "format": file_ext,
                    "size": file_size
                }
            }
            
            logger.info(f"Document loaded successfully: {actual_filename}")
            return document
            
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}")
            raise
    
    async def load_multiple_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load multiple documents"""
        try:
            logger.info(f"Loading {len(file_paths)} documents")
            
            documents = []
            for file_path in file_paths:
                try:
                    document = await self.load_document(file_path)
                    documents.append(document)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
                    continue
            
            logger.info(f"Successfully loaded {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading multiple documents: {str(e)}")
            raise
    
    async def _load_content(self, file_path: str, file_ext: str) -> str:
        """Load content based on file type"""
        try:
            if file_ext in ['.txt', '.md']:
                return await self._load_text_file(file_path)
            elif file_ext in ['.html', '.htm']:
                return await self._load_html_file(file_path)
            elif file_ext == '.csv':
                return await self._load_csv_file(file_path)
            elif file_ext == '.pdf':
                return await self._load_pdf_file(file_path)
            elif file_ext in ['.docx', '.doc']:
                return await self._load_docx_file(file_path)
            elif file_ext == '.rtf':
                return await self._load_rtf_file(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error loading content from {file_path}: {str(e)}")
            raise
    
    async def _load_text_file(self, file_path: str) -> str:
        """Load text file content"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                content = await file.read()
            return content
        except UnicodeDecodeError:
            # Try with different encoding
            async with aiofiles.open(file_path, 'r', encoding='latin-1') as file:
                content = await file.read()
            return content
    
    async def _load_html_file(self, file_path: str) -> str:
        """Load HTML file and extract text"""
        try:
            from bs4 import BeautifulSoup
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                html_content = await file.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            return soup.get_text(separator=' ', strip=True)
            
        except ImportError:
            # Fallback to plain text if BeautifulSoup not available
            logger.warning("BeautifulSoup not available, loading as plain text")
            return await self._load_text_file(file_path)
    
    async def _load_csv_file(self, file_path: str) -> str:
        """Load CSV file content"""
        try:
            import csv
            
            content_parts = []
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                content = await file.read()
            
            # Parse CSV and convert to readable format
            reader = csv.DictReader(content.splitlines())
            for row in reader:
                content_parts.append(" | ".join([f"{k}: {v}" for k, v in row.items()]))
            
            return "\n".join(content_parts)
            
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            # Fallback to plain text
            return await self._load_text_file(file_path)
    
    async def _load_pdf_file(self, file_path: str) -> str:
        """Load PDF file content"""
        try:
            import PyPDF2
            
            content_parts = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    content_parts.append(page.extract_text())
            
            return "\n".join(content_parts)
            
        except ImportError:
            logger.error("PyPDF2 not available for PDF processing")
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
    
    async def _load_docx_file(self, file_path: str) -> str:
        """Load DOCX file content"""
        try:
            import docx
            
            doc = docx.Document(file_path)
            content_parts = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    content_parts.append(paragraph.text)
            
            return "\n".join(content_parts)
            
        except ImportError:
            logger.error("python-docx not available for DOCX processing")
            raise ImportError("python-docx is required for DOCX processing. Install with: pip install python-docx")
    
    async def _load_rtf_file(self, file_path: str) -> str:
        """Load RTF file content"""
        try:
            # Basic RTF parsing - removes RTF tags
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                content = await file.read()
            
            # Simple RTF tag removal
            import re
            # Remove RTF control words and braces
            content = re.sub(r'\\[a-zA-Z]+\d*', '', content)
            content = re.sub(r'[{}]', '', content)
            content = re.sub(r'\s+', ' ', content).strip()
            
            return content
            
        except Exception as e:
            logger.error(f"Error loading RTF file: {str(e)}")
            raise
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return list(self.supported_formats)
    
    def is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported"""
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_formats
    
    async def validate_document(self, file_path: str) -> Dict[str, Any]:
        """Validate document before loading"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                validation_result["valid"] = False
                validation_result["errors"].append("File does not exist")
                return validation_result
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                validation_result["valid"] = False
                validation_result["errors"].append(f"File too large: {file_size} bytes")
            
            # Check file format
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Unsupported format: {file_ext}")
            
            # Check if file is readable
            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                    await file.read(1)  # Try to read first character
            except Exception as e:
                validation_result["valid"] = False
                validation_result["errors"].append(f"File not readable: {str(e)}")
            
            return validation_result
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
            return validation_result
