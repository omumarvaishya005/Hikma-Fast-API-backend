# from qdrant_client import QdrantClient
# from qdrant_client.models import PointStruct

# import os

# QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
# QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
# COLLECTION_NAME = "saudi_labor_law"

# # Connect to Qdrant
# qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# # Create collection if it doesn't exist
# if COLLECTION_NAME not in [c.name for c in qdrant_client.get_collections().collections]:
#     qdrant_client.recreate_collection(
#         collection_name=COLLECTION_NAME,
#         vectors={"size": 1536, "distance": "Cosine"},  # embedding size depends on model
#     )
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import os

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "saudi_labor_law"

# Connect to Qdrant
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def create_collection(collection_name=COLLECTION_NAME, vector_size=384):
    """Create collection if it doesn't exist"""
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Created collection: {collection_name}")
        else:
            print(f"Collection {collection_name} already exists")
    except Exception as e:
        print(f"Error creating collection: {e}")

def get_client():
    """Return the Qdrant client"""
    return qdrant_client