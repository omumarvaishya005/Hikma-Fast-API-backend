from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import time

# Import your existing model and new RAG system
from app.models.generated import model
from app.services.rag import get_rag_system, RAGSystem

# Initialize FastAPI app
app = FastAPI(
    title="Saudi Labor Law Chatbot with RAG", 
    description="AI-powered chatbot for Saudi Labor Law queries using Retrieval-Augmented Generation",
    version="1.0.0"
)

# Initialize RAG system (do this once at startup)
rag_system = get_rag_system()

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    include_context: Optional[bool] = False  # Whether to return context in response
    max_context_chunks: Optional[int] = 5    # Number of context chunks to retrieve

class ContextChunk(BaseModel):
    text: str
    source_file: str
    page: int
    score: float
    chunk_id: int

class ChatResponse(BaseModel):
    question: str
    answer: str
    processing_time: float
    context_used: bool
    context_chunks: Optional[List[ContextChunk]] = None
    num_context_chunks: Optional[int] = None

# Routes
@app.get("/")
def root():
    return {
        "message": "Welcome to Saudi Labor Law Chatbot with RAG",
        "features": [
            "Retrieval-Augmented Generation",
            "Saudi Labor Law expertise",
            "Context-aware responses"
        ],
        "endpoints": {
            "/ask": "POST - Ask questions about Saudi Labor Law",
            "/search": "POST - Search for relevant law sections",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Test RAG system connectivity
        test_result = rag_system.retrieve_context("test", )
        return {
            "status": "healthy",
            "rag_system": "operational",
            "collection": rag_system.collection_name,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/ask", response_model=ChatResponse)
def ask_question(request: QuestionRequest):
    """
    Ask a question about Saudi Labor Law with RAG-enhanced responses
    """
    start_time = time.time()
    
    try:
        # Get RAG-augmented response
        rag_response = rag_system.get_augmented_response(request.question)
        rag_response["top_k"] = request.max_context_chunks
        print('***'*20 )
        print(f"RAG retrieved {len(rag_response['context_chunks'])} chunks for question.")
        # Generate answer using LLM with context
        llm_response = model.invoke(rag_response["rag_prompt"])
        print("LLM Response:", llm_response)
        # Extract answer content (adjust based on your model's response format)
        if hasattr(llm_response, 'content'):
            if isinstance(llm_response.content, list):
                answer = llm_response.content[0] if llm_response.content else "No response generated"
            else:
                answer = llm_response.content
        else:
            answer = str(llm_response)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare context chunks for response (if requested)
        context_chunks = None
        if request.include_context:
            context_chunks = [
                ContextChunk(
                    text=chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"],
                    source_file=chunk["source_file"],
                    page=chunk["page"],
                    score=chunk["score"],
                    chunk_id=chunk["chunk_id"]
                )
                for chunk in rag_response["context_chunks"]
            ]
        
        return ChatResponse(
            question=request.question,
            answer=answer,
            processing_time=round(processing_time, 3),
            context_used=len(rag_response["context_chunks"]) > 0,
            context_chunks=context_chunks,
            num_context_chunks=len(rag_response["context_chunks"])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/search")
def search_context(request: QuestionRequest) -> Dict[str, Any]:
    """
    Search for relevant context without generating an answer
    Useful for finding specific law sections
    """
    try:
        # Get only the context
        context_chunks = rag_system.retrieve_context(request.question)
        
        return {
            "query": request.question,
            "num_results": len(context_chunks),
            "results": [
                {
                    "text": chunk["text"],
                    "source_file": chunk["source_file"],
                    "page": chunk["page"],
                    "relevance_score": chunk["score"],
                    "chunk_id": chunk["chunk_id"]
                }
                for chunk in context_chunks[:request.max_context_chunks]
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching context: {str(e)}")

@app.post("/ask-simple")
def ask_question_simple(question: str):
    """
    Simple endpoint for basic questions (backward compatibility)
    Uses RAG but returns simpler response format
    """
    try:
        # Get RAG response
        rag_response = rag_system.get_augmented_response(question)
        
        # Generate answer
        answer = model.invoke(rag_response["rag_prompt"])
        
        # Extract content
        if hasattr(answer, 'content'):
            content = answer.content[0] if isinstance(answer.content, list) else answer.content
        else:
            content = str(answer)
        
        return {
            "question": question, 
            "answer": content,
            "context_sources": len(rag_response["context_chunks"])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Additional utility endpoints
@app.get("/stats")
def get_system_stats():
    """Get system statistics"""
    try:
        client = rag_system.client
        collection_info = client.get_collection(rag_system.collection_name)
        
        return {
            "collection_name": rag_system.collection_name,
            "total_documents": collection_info.points_count,
            "vector_dimension": collection_info.config.params.vectors.size,
            "distance_metric": collection_info.config.params.vectors.distance,
            "rag_settings": {
                "default_top_k": rag_system.top_k,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7500)