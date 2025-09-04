from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import glob

def load_pdf_to_chunks(pdf_path, chunk_size=500, chunk_overlap=50):
    """Load a single PDF and split into chunks"""
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = splitter.split_documents(documents)
    return docs

def load_all_pdfs_from_directory(pdf_directory, chunk_size=500, chunk_overlap=50):
    """Load all PDFs from a directory and split into chunks"""
    all_docs = []
    
    # Find all PDF files in the directory
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}")
        return all_docs
    
    for pdf_file in pdf_files:
        print(f"Loading PDF: {pdf_file}")
        docs = load_pdf_to_chunks(pdf_file, chunk_size, chunk_overlap)
        
        # Add metadata about source file
        for doc in docs:
            doc.metadata['source_file'] = os.path.basename(pdf_file)
        
        all_docs.extend(docs)
        print(f"Loaded {len(docs)} chunks from {pdf_file}")
    
    print(f"Total chunks loaded: {len(all_docs)}")
    return all_docs