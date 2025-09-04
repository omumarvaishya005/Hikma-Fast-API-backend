#!/usr/bin/env python3
"""
Script to load PDFs, generate embeddings, and store them in Qdrant
"""

import os
import sys
from qdrant_client.models import PointStruct
from pdf_loader import load_all_pdfs_from_directory, load_pdf_to_chunks
from embeddings import get_embeddings, get_embedding_dimension
from qdrant import get_client, create_collection, COLLECTION_NAME

def store_pdf_embeddings(pdf_directory="pdfs", collection_name=COLLECTION_NAME):
    """Main function to process PDFs and store embeddings"""
    
    print("=== Starting PDF Embedding Process ===")
    
    # Check if PDF directory exists
    if not os.path.exists(pdf_directory):
        print(f"Error: PDF directory '{pdf_directory}' not found!")
        return False
    
    # Step 1: Load PDFs and split into chunks
    print(f"\n1. Loading PDFs from '{pdf_directory}' directory...")
    docs = load_all_pdfs_from_directory(pdf_directory, chunk_size=500, chunk_overlap=50)
    
    if not docs:
        print("No documents loaded. Exiting.")
        return False
    
    # Step 2: Initialize embedding model
    print("\n2. Initializing embedding model...")
    embed_model = get_embeddings()
    vector_size = get_embedding_dimension()
    print(f"Using embedding model with dimension: {vector_size}")
    
    # Step 3: Create Qdrant collection
    print("\n3. Setting up Qdrant collection...")
    create_collection(collection_name, vector_size)
    client = get_client()
    
    # Step 4: Generate embeddings and store in batches
    print("\n4. Generating embeddings and storing to Qdrant...")
    batch_size = 100  # Process documents in batches to avoid memory issues
    
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        batch_end = min(i + batch_size, len(docs))
        
        print(f"Processing batch {i//batch_size + 1}: documents {i+1}-{batch_end}")
        
        # Generate embeddings for this batch
        texts = [doc.page_content for doc in batch_docs]
        try:
            embeddings = embed_model.embed_documents(texts)
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            continue
        
        # Prepare points for Qdrant
        points = []
        for idx, (doc, embedding) in enumerate(zip(batch_docs, embeddings)):
            point_id = i + idx  # Global ID across all batches
            payload = {
                "text": doc.page_content,
                "source_file": doc.metadata.get('source_file', 'unknown'),
                "page": doc.metadata.get('page', 0),
                "chunk_id": point_id
            }
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            ))
        
        # Store batch in Qdrant
        try:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"Successfully stored batch {i//batch_size + 1} ({len(points)} points)")
        except Exception as e:
            print(f"Error storing batch: {e}")
            continue
    
    print(f"\n=== Completed! Stored {len(docs)} document chunks in collection '{collection_name}' ===")
    
    # Verify storage
    try:
        collection_info = client.get_collection(collection_name)
        print(f"Collection info: {collection_info.points_count} points stored")
    except Exception as e:
        print(f"Error getting collection info: {e}")
    
    return True

if __name__ == "__main__":
    # You can specify custom PDF directory if needed
    pdf_dir = sys.argv[1] if len(sys.argv) > 1 else "pdfs"
    success = store_pdf_embeddings(pdf_dir)
    
    if success:
        print("✅ PDF embedding process completed successfully!")
    else:
        print("❌ PDF embedding process failed!")