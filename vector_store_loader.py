"""
Vector Store Loader - Separate module for loading/building vector store
NO STREAMLIT IMPORTS - Can be safely imported from UI
"""
from pathlib import Path
from typing import Optional
from document_processor import DocumentProcessor
from vector_store import SafetyVectorStore
from config import (
    DATA_DIR, DOCUMENTS_DIR, VECTOR_STORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL
)

def load_or_build_vector_store(force_rebuild: bool = False) -> SafetyVectorStore:
    """
    Load or build vector store from documents
    This function does heavy work - should be called from @st.cache_resource
    NO STREAMLIT IMPORTS - safe to import from UI
    """
    vector_store = SafetyVectorStore(embedding_model=EMBEDDING_MODEL)
    vector_store_path = VECTOR_STORE_DIR / "faiss_index.bin"
    
    # Check if vector store exists
    if vector_store_path.exists() and not force_rebuild:
        try:
            print("📂 Loading existing vector store...")
            vector_store.load(VECTOR_STORE_DIR)
            print("✅ Vector store loaded successfully")
            return vector_store
        except Exception as e:
            print(f"⚠️  Error loading vector store: {e}, rebuilding...")
            force_rebuild = True
    
    # Build vector store if needed
    print("🔄 Building vector store from documents...")
    
    # Initialize document processor
    document_processor = DocumentProcessor(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Try professional data structure first, fallback to legacy
    if DATA_DIR.exists() and any(DATA_DIR.rglob("*.pdf")):
        print(f"📚 Using professional data structure: {DATA_DIR}")
        chunks = document_processor.process_directory(DATA_DIR, recursive=True)
    else:
        print(f"📚 Using legacy documents directory: {DOCUMENTS_DIR}")
        chunks = document_processor.process_directory(DOCUMENTS_DIR, recursive=False)
    
    if not chunks:
        raise ValueError(f"No documents found. Please add PDF files to:\n"
                       f"  - {DATA_DIR} (professional structure)\n"
                       f"  - {DOCUMENTS_DIR} (legacy)")
    
    print(f"📊 Processing {len(chunks)} chunks...")
    vector_store.create_index(chunks)
    
    # Try to save, but don't fail if it's a read-only filesystem (Streamlit Cloud)
    try:
        vector_store.save(VECTOR_STORE_DIR)
        print("✅ Vector store built and saved")
    except (PermissionError, OSError) as e:
        print(f"⚠️  Could not save vector store to disk (this is OK in Streamlit Cloud): {e}")
        print("✅ Vector store built (in memory)")
    
    return vector_store


