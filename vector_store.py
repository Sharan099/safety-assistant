"""
Vector Store for Safety Documents using FAISS
"""
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from document_processor import DocumentChunk
import json

class SafetyVectorStore:
    """FAISS-based vector store for safety documents"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        
    def create_index(self, chunks: List[DocumentChunk]):
        """Create FAISS index from document chunks"""
        if not chunks:
            raise ValueError("No chunks provided")
        
        self.chunks = chunks
        
        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        print(f"🔄 Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"✅ Created FAISS index with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 8, similarity_threshold: float = 0.3, domain_filter: Optional[str] = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar chunks with improved relevance
        Returns list of (chunk, similarity_score) tuples sorted by relevance
        """
        if self.index is None or len(self.chunks) == 0:
            return []
        
        # Generate query embedding
        try:
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        except TypeError:
            # Fallback if normalize_embeddings not supported
            query_embedding = self.embedding_model.encode([query])
        
        # Search with more candidates for better results
        search_k = min(top_k * 3, len(self.chunks))  # Search more, filter later
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        results = []
        seen_chunks = set()  # Avoid duplicates
        
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks) and idx >= 0:
                chunk = self.chunks[idx]
                
                # Skip duplicates
                chunk_key = (chunk.document_name, chunk.page_number, chunk.chunk_id)
                if chunk_key in seen_chunks:
                    continue
                seen_chunks.add(chunk_key)
                
                # Convert L2 distance to similarity (better normalization)
                # For normalized embeddings, cosine similarity = 1 - (distance^2 / 2)
                similarity = max(0.0, 1.0 - (distance * distance / 2.0))
                
                # Domain filtering if specified
                if domain_filter and chunk.domain:
                    if domain_filter.lower() not in chunk.domain.lower():
                        # Still include but with lower priority
                        similarity *= 0.8
                
                if similarity >= similarity_threshold:
                    results.append((chunk, similarity))
        
        # Sort by similarity (highest first) and limit to top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def save(self, save_dir: Path):
        """Save index and chunks to disk"""
        save_dir.mkdir(exist_ok=True)
        
        # Save FAISS index
        index_path = save_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        
        # Save chunks metadata
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        chunks_path = save_dir / "chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        # Save embedding model name
        config_path = save_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "embedding_model": self.embedding_model.get_sentence_embedding_dimension(),
                "dimension": self.dimension,
                "num_chunks": len(self.chunks)
            }, f, indent=2)
        
        print(f"✅ Saved vector store to {save_dir}")
    
    def load(self, save_dir: Path):
        """Load index and chunks from disk"""
        index_path = save_dir / "faiss_index.bin"
        chunks_path = save_dir / "chunks.json"
        
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"Vector store not found in {save_dir}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Load chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        self.chunks = []
        for chunk_data in chunks_data:
            chunk = DocumentChunk(
                text=chunk_data["text"],
                document_name=chunk_data["document_name"],
                page_number=chunk_data["page_number"],
                section_number=chunk_data.get("section_number"),
                chunk_id=chunk_data.get("chunk_id"),
                origin=chunk_data.get("origin"),
                domain=chunk_data.get("domain"),
                strictness=chunk_data.get("strictness"),
                method=chunk_data.get("method"),
                year=chunk_data.get("year"),
                source_type=chunk_data.get("source_type"),
                test_type=chunk_data.get("test_type"),
                metric=chunk_data.get("metric"),
                dummy_type=chunk_data.get("dummy_type")
            )
            self.chunks.append(chunk)
        
        print(f"✅ Loaded vector store: {len(self.chunks)} chunks, {self.index.ntotal} vectors")
    
    @staticmethod
    def load_or_build_store(force_rebuild: bool = False) -> 'SafetyVectorStore':
        """
        Load or build vector store from documents
        This function does heavy work - should be called from @st.cache_resource
        NO STREAMLIT IMPORTS - safe to import from UI
        """
        from pathlib import Path
        from document_processor import DocumentProcessor
        from config import (
            DATA_DIR, DOCUMENTS_DIR, VECTOR_STORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
            EMBEDDING_MODEL
        )
        
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

