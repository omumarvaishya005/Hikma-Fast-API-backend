#!/usr/bin/env python3
"""
Fixed test script for RAG system with correct import paths
"""

import sys
import os
import json

# Add the project root and services directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
services_path = os.path.join(project_root, 'services')
sys.path.insert(0, project_root)
sys.path.insert(0, services_path)

# Now import from services directory
try:
    from app.services.rag import get_rag_system, quick_rag_query, get_context_only
except ImportError:
    # If services.rag doesn't work, try direct import
    from app.services.rag import get_rag_system, quick_rag_query, get_context_only

def test_rag_retrieval():
    """Test basic RAG retrieval functionality"""
    print("🔍 Testing RAG Retrieval System...")
    print("=" * 50)
    
    # Test queries related to Saudi Labor Law
    test_queries = [
        "What are the working hours in Saudi Arabia?",
        "What is the minimum wage?",
        "How many vacation days are employees entitled to?",
        "What are the rules for overtime work?",
        "Can an employer terminate an employee without notice?"
    ]
    
    try:
        rag = get_rag_system()
        print(f"✅ RAG system initialized successfully")
        print(f"   Collection: {rag.collection_name}")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 40)
        
        try:
            # Get context only
            context_chunks = rag.retrieve_context(query)
            
            if context_chunks:
                print(f"✅ Found {len(context_chunks)} relevant chunks:")
                for j, chunk in enumerate(context_chunks[:3], 1):  # Show top 3
                    print(f"   {j}. {chunk['source_file']} (Page {chunk['page']}) - Score: {chunk['score']:.3f}")
                    print(f"      Preview: {chunk['text'][:100]}...")
            else:
                print("❌ No relevant context found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)

def test_rag_full_system():
    """Test complete RAG system with prompt generation"""
    print("\n🤖 Testing Full RAG System...")
    print("=" * 50)
    
    test_query = "What are the maximum working hours per day in Saudi Arabia?"
    print(f"Query: {test_query}")
    print("-" * 40)
    
    try:
        rag = get_rag_system()
        response = rag.get_augmented_response(test_query)
        
        print(f"✅ RAG Response Generated:")
        print(f"   - Context chunks: {response['num_context_chunks']}")
        print(f"   - Prompt length: {len(response['rag_prompt'])} characters")
        
        print(f"\n📄 Context Sources:")
        for chunk in response['context_chunks']:
            print(f"   - {chunk['source_file']} (Page {chunk['page']}, Score: {chunk['score']:.3f})")
        
        print(f"\n📝 Generated Prompt Preview:")
        prompt_preview = response['rag_prompt'][:500] + "..." if len(response['rag_prompt']) > 500 else response['rag_prompt']
        print(prompt_preview)
        
    except Exception as e:
        print(f"❌ Error: {e}")

def test_imports():
    """Test if all imports work correctly"""
    print("🔧 Testing Imports...")
    print("=" * 50)
    
    try:
        print("Testing services imports...")
        
        # Test rag imports
        from app.services.rag import RAGSystem
        print("✅ RAGSystem imported successfully")
        
        # Test embeddings
        from app.services.embeddings import get_embeddings
        print("✅ get_embeddings imported successfully")
        
        # Test qdrant
        from app.services.qdrant import get_client
        print("✅ get_client imported successfully")
        
        # Test pdf_loader
        from app.services.pdf_loader import load_pdf_to_chunks
        print("✅ load_pdf_to_chunks imported successfully")
        
        print("✅ All imports successful!")
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        print("\nTrying alternative import methods...")
        
        # Try without services prefix
        try:
            import app.services.rag
            import app.services.embeddings
            import app.services.qdrant
            import app.services.pdf_loader
            print("✅ Direct imports successful!")
        except Exception as e2:
            print(f"❌ Direct imports also failed: {e2}")

def main():
    """Main test function"""
    print("🚀 RAG System Test Suite")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Services path: {services_path}")
    print(f"Python path includes: {sys.path[:3]}...")
    
    # First test imports
    test_imports()
    
    if len(sys.argv) > 1 and sys.argv[1] == "skip-tests":
        return
    
    # Then run functionality tests
    test_rag_retrieval()
    test_rag_full_system()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()