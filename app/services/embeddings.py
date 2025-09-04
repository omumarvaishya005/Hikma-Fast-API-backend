from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def get_embeddings():
    """Initialize and return HuggingFace embeddings model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        # Note: HuggingFaceEmbeddings doesn't need API token for sentence-transformers models
    )

def get_embedding_dimension():
    """Get the dimension of the embedding model"""
    # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    return 384