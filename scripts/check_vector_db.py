import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.vector_store import get_vector_store
from backend.core.embeddings import EmbeddingGenerator

async def check_vector_db():
    try:
        vector_store = get_vector_store()
        embedding_gen = EmbeddingGenerator()
        
        # Generate embedding for query
        query = "sample document content"
        query_embedding = await embedding_gen.generate_embeddings([query])
        
        # Test retrieval
        results = await vector_store.similarity_search(
            query_embedding=query_embedding[0],
            top_k=3,
            threshold=0.0
        )
        print(f'Found {len(results)} documents')
        if results:
            print(f'First doc content preview: {results[0].get("content", "")[:200]}...')
            print(f'Document ID: {results[0].get("metadata", {}).get("document_id", "unknown")}')
        return results
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    results = asyncio.run(check_vector_db())
