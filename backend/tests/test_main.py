"""Tests for main FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_document_storage():
    """Mock document storage."""
    return {
        "test_doc": {
            "filename": "sample.pdf",
            "content": "This is a sample PDF file for testing purposes. It is lightweight and quick to download.",
            "size": 102400,
            "type": "application/pdf"
        }
    }


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestConfigEndpoint:
    """Test configuration endpoint."""

    def test_config_endpoint(self, client):
        """Test config endpoint returns configuration."""
        response = client.get("/api/v1/config")
        assert response.status_code == 200
        data = response.json()
        assert "app_name" in data
        assert "model_info" in data
        assert data["model_info"]["provider"] == "gemini"


class TestQueryEndpoint:
    """Test query endpoint."""

    def test_query_with_empty_storage(self, client):
        """Test query with no documents."""
        with patch('backend.main.document_storage', {}):
            response = client.post(
                "/api/v1/query",
                json={"query": "What is this about?", "max_context": 5, "similarity_threshold": 0.1}
            )
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "confidence_score" in data
            assert data["confidence_score"] == 0.1

    def test_query_with_documents(self, client, mock_document_storage):
        """Test query with documents available."""
        with patch('backend.main.document_storage', mock_document_storage):
            response = client.post(
                "/api/v1/query",
                json={"query": "What is this PDF about?", "max_context": 5, "similarity_threshold": 0.1}
            )
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "confidence_score" in data
            assert "📄 Summary:" in data["answer"]
            assert "⚡ Key Features:" in data["answer"]
            assert "📌 Use Cases:" in data["answer"]
            assert "📊 Confidence:" in data["answer"]

    def test_query_invalid_request(self, client):
        """Test query with invalid request."""
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422  # Validation error


class TestDocumentsEndpoint:
    """Test documents endpoint."""

    def test_get_documents(self, client, mock_document_storage):
        """Test getting documents list."""
        with patch('backend.main.document_storage', mock_document_storage):
            response = client.get("/api/v1/documents")
            assert response.status_code == 200
            data = response.json()
            assert "documents" in data
            assert len(data["documents"]) == 1
            assert data["documents"][0]["filename"] == "sample.pdf"


class TestUploadEndpoint:
    """Test upload endpoint."""

    def test_upload_file(self, client):
        """Test file upload."""
        file_content = b"Test file content"
        files = {"file": ("test.txt", file_content, "text/plain")}
        
        response = client.post("/api/v1/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "filename" in data
        assert "size" in data
        assert "file_type" in data


class TestQualityScoring:
    """Test quality scoring system."""

    def test_score_response_high_quality(self):
        """Test scoring of high-quality response."""
        from backend.main import score_response
        
        response = """📄 Summary:
Sample PDF file for testing.

⚡ Key Features:
- Lightweight (100KB)
- Fast processing speed

📌 Use Cases:
- Email testing
- PDF validation

📊 Confidence:
High"""
        
        score = score_response(response)
        assert score >= 9.5

    def test_score_response_low_quality(self):
        """Test scoring of low-quality response."""
        from backend.main import score_response
        
        response = "This is a generic response without proper structure."
        
        score = score_response(response)
        assert score < 9.5

    def test_validate_response_success(self):
        """Test validation of proper response."""
        from backend.main import validate_response
        
        response = """📄 Summary:
Sample PDF file for testing.

⚡ Key Features:
- Lightweight (100KB)
- Fast processing speed

📌 Use Cases:
- Email testing
- PDF validation

📊 Confidence:
High"""
        
        assert validate_response(response) is True

    def test_validate_response_failure(self):
        """Test validation of improper response."""
        from backend.main import validate_response
        
        response = "This response lacks proper structure."
        
        assert validate_response(response) is False


class TestIntentDetection:
    """Test intent detection."""

    @pytest.mark.parametrize("query,expected_intent", [
        ("What is this PDF about?", "explanation"),
        ("Summarize this document", "summary"),
        ("What are the use cases?", "use_cases"),
        ("What are the key features?", "features"),
        ("What are the benefits?", "benefits"),
        ("How large is this file?", "size"),
        ("How fast does it download?", "performance"),
        ("Tell me about it", "general"),
    ])
    def test_detect_intent(self, query, expected_intent):
        """Test intent detection for various queries."""
        from backend.main import detect_intent
        
        intent = detect_intent(query)
        assert intent == expected_intent


class TestDeduplication:
    """Test deduplication functionality."""

    def test_deduplicate_chunks(self):
        """Test chunk deduplication."""
        from backend.main import deduplicate_chunks
        
        chunks = [
            "This is a test document.",
            "This is a test document with more words.",  # High similarity
            "Completely different content about something else.",
            "Another unique piece of information."
        ]
        
        unique_chunks = deduplicate_chunks(chunks, threshold=0.85)
        assert len(unique_chunks) < len(chunks)
        assert len(unique_chunks) >= 2  # Should keep at least 2 unique chunks

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        from backend.main import calculate_cosine_similarity
        
        text1 = "This is about testing PDF files"
        text2 = "This concerns testing PDF documents"
        text3 = "Completely unrelated content about cooking"
        
        sim1 = calculate_cosine_similarity(text1, text2)
        sim2 = calculate_cosine_similarity(text1, text3)
        
        assert sim1 > sim2  # Similar texts should have higher similarity
        assert 0 <= sim1 <= 1
        assert 0 <= sim2 <= 1


class TestConfidenceCalculation:
    """Test confidence calculation."""

    def test_get_confidence_high(self):
        """Test high confidence calculation."""
        from backend.main import get_confidence, all_chunks_consistent
        
        consistent_chunks = [
            "This PDF is for testing email attachments.",
            "Testing email attachments with this PDF file.",
            "Email attachment testing using this PDF document."
        ]
        
        with patch('backend.main.all_chunks_consistent', return_value=True):
            confidence = get_confidence(consistent_chunks)
            assert confidence == "High"

    def test_get_confidence_medium(self):
        """Test medium confidence calculation."""
        from backend.main import get_confidence, all_chunks_consistent, partial_overlap
        
        chunks = ["Some content"]
        
        with patch('backend.main.all_chunks_consistent', return_value=False), \
             patch('backend.main.partial_overlap', return_value=True):
            confidence = get_confidence(chunks)
            assert confidence == "Medium"

    def test_get_confidence_low(self):
        """Test low confidence calculation."""
        from backend.main import get_confidence, all_chunks_consistent, partial_overlap
        
        chunks = ["Some content"]
        
        with patch('backend.main.all_chunks_consistent', return_value=False), \
             patch('backend.main.partial_overlap', return_value=False):
            confidence = get_confidence(chunks)
            assert confidence == "Low"
