import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.vector_store import get_vector_store

async def check_collections():
    try:
        vector_store = get_vector_store()
        collection = vector_store.get_collection()
        
        if collection:
            count = collection.count()
            print(f'Collection name: {collection.name}')
            print(f'Document count in collection: {count}')
            
            # Get a sample of documents
            if count > 0:
                results = collection.get(limit=3)
                print(f'Sample document IDs: {results["ids"][:3]}')
                print(f'Sample metadatas: {results["metadatas"][:3]}')
        else:
            print('No collection found')
            
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_collections())
