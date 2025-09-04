"""
RAG (Retrieval-Augmented Generation) system for Saudi Labor Law Chatbot
"""

from typing import List, Dict, Any
# from embeddings import get_embeddings
# from qdrant import get_client, COLLECTION_NAME

from app.services.embeddings import get_embeddings
from app.services.qdrant import get_client, COLLECTION_NAME
from app.services.pdf_loader import load_pdf_to_chunks

class RAGSystem:
    def __init__(self, collection_name: str = COLLECTION_NAME, top_k: int = 5):
        """
        Initialize RAG system
        
        Args:
            collection_name: Name of Qdrant collection
            top_k: Number of top similar chunks to retrieve
        """
        self.collection_name = collection_name
        self.top_k = top_k
        self.client = get_client()
        self.embeddings = get_embeddings()
        
    def retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a given query
        
        Args:
            query: User's question
            
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=self.top_k,
                with_payload=True,
                with_vectors=False  # We don't need vectors in response
            )
            
            # Format results
            context_chunks = []
            for result in search_results:
                chunk = {
                    "text": result.payload.get("text", ""),
                    "source_file": result.payload.get("source_file", "unknown"),
                    "page": result.payload.get("page", 0),
                    "score": result.score,
                    "chunk_id": result.payload.get("chunk_id", result.id)
                }
                context_chunks.append(chunk)
            
            return context_chunks
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []
    
    def format_context_for_llm(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved context into a string for LLM prompt
        
        Args:
            context_chunks: List of relevant chunks
            
        Returns:
            Formatted context string
        """
        if not context_chunks:
            return "No relevant information found in the Saudi Labor Law documents."
        
        context_parts = []
        context_parts.append("=== RELEVANT SAUDI LABOR LAW INFORMATION ===\n")
        
        for i, chunk in enumerate(context_chunks, 1):
            source_info = f"[Source: {chunk['source_file']}, Page: {chunk['page']}, Relevance: {chunk['score']:.3f}]"
            context_parts.append(f"Context {i}: {source_info}")
            context_parts.append(chunk['text'])
            context_parts.append("---")
        
        context_parts.append("=== END CONTEXT ===")
        
        return "\n".join(context_parts)
    
    def generate_rag_prompt(self, query: str, context: str) -> str:
        """
        Generate a comprehensive prompt for the LLM with context
        
        Args:
            query: User's question
            context: Retrieved context
            
        Returns:
            Complete prompt for LLM
        """
        prompt = f"""You are an expert assistant specializing in Saudi Labor Law. You have access to relevant sections of the Saudi Labor Law documents to answer questions accurately.

INSTRUCTIONS:
1. Answer the question based primarily on the provided context from Saudi Labor Law documents
2. If the context doesn't contain enough information, clearly state this
3. Be specific and cite relevant articles, sections, or provisions when possible
4. Provide practical, actionable advice when appropriate
5. If you're unsure about any legal interpretation, recommend consulting with a legal professional
6. Answer in a clear, professional manner suitable for someone seeking legal guidance

{context}

QUESTION: {query}

ANSWER: Based on the Saudi Labor Law documents provided above, """

        return prompt
    
    def get_augmented_response(self, query: str) -> Dict[str, Any]:
        """
        Get complete RAG response including context and metadata
        
        Args:
            query: User's question
            
        Returns:
            Dictionary with answer, context, and metadata
        """
        # Step 1: Retrieve relevant context
        context_chunks = self.retrieve_context(query)
        
        # Step 2: Format context for LLM
        formatted_context = self.format_context_for_llm(context_chunks)
        
        # Step 3: Generate complete prompt
        rag_prompt = self.generate_rag_prompt(query, formatted_context)
        
        # Step 4: Prepare response data
        response_data = {
            "query": query,
            "rag_prompt": rag_prompt,
            "context_chunks": context_chunks,
            "num_context_chunks": len(context_chunks),
            "formatted_context": formatted_context
        }
        
        return response_data

# Utility functions for easy integration
def get_rag_system() -> RAGSystem:
    """Get initialized RAG system instance"""
    return RAGSystem()

def quick_rag_query(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Quick function to get RAG response for a query
    
    Args:
        query: User's question
        top_k: Number of context chunks to retrieve
        
    Returns:
        RAG response data
    """
    rag = RAGSystem(top_k=top_k)
    return rag.get_augmented_response(query)

def get_context_only(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Get only the context chunks for a query (useful for debugging)
    
    Args:
        query: User's question
        top_k: Number of context chunks to retrieve
        
    Returns:
        List of context chunks
    """
    rag = RAGSystem(top_k=top_k)
    return rag.retrieve_context(query)

# Test function
if __name__ == "__main__":
    # Test the RAG system
    test_query = "What are the working hours in Saudi Arabia?"
    
    print("Testing RAG System...")
    print(f"Query: {test_query}")
    print("-" * 50)
    
    rag = get_rag_system()
    response = rag.get_augmented_response(test_query)
    
    print(f"Found {response['num_context_chunks']} relevant chunks")
    print("\nContext chunks:")
    for chunk in response['context_chunks']:
        print(f"- {chunk['source_file']} (Page {chunk['page']}) - Score: {chunk['score']:.3f}")
    
    print(f"\nGenerated prompt length: {len(response['rag_prompt'])} characters")
    print("\nRAG Prompt Preview:")
    # print(response['rag_prompt'][:500] + "..." if len(response['rag_prompt']) > 500 else response['rag_prompt'])
    print(response['formatted_context'])  # or try response["answer"]

    # "query": query,
    # "rag_prompt": rag_prompt,
    # "context_chunks": context_chunks,
    # "num_context_chunks": len(context_chunks),
    # "formatted_context": formatted_context