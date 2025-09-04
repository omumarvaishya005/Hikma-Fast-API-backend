# test_rag.py

import sys
import os
import json

# Add the project root and services directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
services_path = os.path.join(project_root, 'services')
sys.path.insert(0, project_root)
sys.path.insert(0, services_path)


from app.services.rag import get_rag_system
from app.models.generated import model  # your LLM instance


def debug_rag_and_llm(question: str, top_k: int = 5):
    """
    Inspect RAG retrieval and LLM response for a single question.
    """
    # Step 1: Get RAG-augmented response
    rag_system = get_rag_system()  # <-- create instance
    rag_response = rag_system.get_augmented_response(question)
    rag_response["top_k"] = top_k

    # print("="*50)
    # print(f"Question: {question}")
    # print("\nStep 1: RAG Retrieved Chunks:")
    # for idx, chunk in enumerate(rag_response["context_chunks"]):
    #     print(f"{idx+1}. Source: {chunk['source_file']} | Page: {chunk['page']} | Score: {chunk['score']:.3f}")
    #     print(f"   Text (first 200 chars): {chunk['text'][:200]}...\n")
    print("****"*50)
    print("\nStep 2: Generated Prompt Sent to LLM:")
    print(rag_response["rag_prompt"][:10000] + "..." if len(rag_response["rag_prompt"]) > 10000 else rag_response["rag_prompt"])
    print("****"*50)
    # Step 3: Invoke LLM with augmented prompt
    llm_response = model.invoke(rag_response["rag_prompt"])
    print("LLM Response:", llm_response)
    # Extract final answer
    if hasattr(llm_response, 'content'):
        if isinstance(llm_response.content, list):
            answer = llm_response.content[0] if llm_response.content else "No response generated"
        else:
            answer = llm_response.content
    else:
        answer = str(llm_response)
    answer = str(llm_response)
    print("\nStep 3: LLM Generated Answer:")
    print(answer)
    print("="*50)

    return rag_response, llm_response, answer
if __name__ == "__main__":
    question = "What are the working hours in Saudi Arabia?"
    debug_rag_and_llm(question)
