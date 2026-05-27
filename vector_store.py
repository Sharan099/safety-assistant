"""
Vector Store for Safety Documents using FAISS with Hybrid Retrieval (Dense + BM25)
"""
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from document_processor import DocumentChunk
import json

# Try to import BM25 for keyword retrieval
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️ rank-bm25 not available. Install with: pip install rank-bm25")

class SafetyVectorStore:
    """FAISS-based vector store for safety documents"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.bm25_index = None  # BM25 keyword index
        self.tokenized_chunks = []  # For BM25
        
    def create_index(self, chunks: List[DocumentChunk]):
        """Create FAISS index and BM25 index from document chunks"""
        if not chunks:
            raise ValueError("No chunks provided")
        
        self.chunks = chunks
        
        # Generate embeddings for dense retrieval
        texts = [chunk.text for chunk in chunks]
        print(f"🔄 Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Create BM25 index for keyword retrieval
        if BM25_AVAILABLE:
            print(f"🔄 Building BM25 keyword index...")
            # Tokenize chunks for BM25
            self.tokenized_chunks = [text.lower().split() for text in texts]
            self.bm25_index = BM25Okapi(self.tokenized_chunks)
            print(f"✅ Created BM25 index")
        
        print(f"✅ Created FAISS index with {self.index.ntotal} vectors")
    
    def add_documents(self, chunks: List[DocumentChunk]):
        """
        Add new documents incrementally to existing vector store
        Useful for adding uploaded documents without rebuilding entire index
        """
        if not chunks:
            return
        
        # If index doesn't exist, create it
        if self.index is None:
            self.chunks = []
            self.dimension = 384  # Default for all-MiniLM-L6-v2
        
        # Generate embeddings for new chunks
        texts = [chunk.text for chunk in chunks]
        print(f"🔄 Generating embeddings for {len(texts)} new chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Set dimension if not set
        if self.dimension != embeddings.shape[1]:
            if self.index is None:
                self.dimension = embeddings.shape[1]
            else:
                raise ValueError(f"Embedding dimension mismatch: existing {self.dimension}, new {embeddings.shape[1]}")
        
        # Create index if it doesn't exist
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Add chunks to list
        self.chunks.extend(chunks)
        
        print(f"✅ Added {len(chunks)} chunks. Total: {self.index.ntotal} vectors")
    
    def hybrid_search(self, query: str, keyword_query: str, top_k_dense: int = 10, 
                     top_k_keyword: int = 10, top_k_final: int = 5, 
                     similarity_threshold: float = 0.3) -> List[Tuple[DocumentChunk, float]]:
        """
        Hybrid retrieval: Combine dense (semantic) and keyword (BM25) search
        Returns list of (chunk, combined_score) tuples sorted by relevance
        """
        if self.index is None or len(self.chunks) == 0:
            return []
        
        # 1. Dense retrieval (semantic)
        dense_results = []
        try:
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        except TypeError:
            query_embedding = self.embedding_model.encode([query])
        
        search_k = min(top_k_dense * 2, len(self.chunks))
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        dense_scores = {}
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks) and idx >= 0:
                similarity = max(0.0, 1.0 - (distance * distance / 2.0))
                if similarity >= similarity_threshold:
                    dense_scores[idx] = similarity
        
        # 2. Keyword retrieval (BM25)
        keyword_scores = {}
        if BM25_AVAILABLE and self.bm25_index and keyword_query:
            tokenized_query = keyword_query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            
            # Normalize BM25 scores to 0-1 range
            if len(bm25_scores) > 0:
                max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1
                for idx, score in enumerate(bm25_scores):
                    if score > 0:
                        normalized_score = min(1.0, score / max_score)
                        keyword_scores[idx] = normalized_score
        
        # 3. Merge and combine scores
        all_indices = set(dense_scores.keys()) | set(keyword_scores.keys())
        combined_results = []
        
        for idx in all_indices:
            chunk = self.chunks[idx]
            
            # Combine scores (weighted average: 60% dense, 40% keyword)
            dense_score = dense_scores.get(idx, 0.0)
            keyword_score = keyword_scores.get(idx, 0.0)
            
            if dense_score > 0 or keyword_score > 0:
                combined_score = (0.6 * dense_score) + (0.4 * keyword_score)
                combined_results.append((chunk, combined_score))
        
        # 4. Sort and return top_k_final
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:top_k_final]
    
    def search(self, query: str, top_k: int = 8, similarity_threshold: float = 0.3, 
               domain_filter: Optional[str] = None, use_hybrid: bool = True) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar chunks (uses hybrid search if available)
        Returns list of (chunk, similarity_score) tuples sorted by relevance
        """
        if use_hybrid and BM25_AVAILABLE:
            # Use hybrid search
            from query_rewriter import rewrite_query
            rewritten = rewrite_query(query)
            return self.hybrid_search(
                query=rewritten["semantic_query"],
                keyword_query=rewritten["keyword_query"],
                top_k_dense=10,
                top_k_keyword=10,
                top_k_final=top_k,
                similarity_threshold=similarity_threshold
            )
        else:
            # Fallback to dense-only search
            if self.index is None or len(self.chunks) == 0:
                return []
            
            try:
                query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
            except TypeError:
                query_embedding = self.embedding_model.encode([query])
            
            search_k = min(top_k * 3, len(self.chunks))
            distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
            
            results = []
            seen_chunks = set()
            
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.chunks) and idx >= 0:
                    chunk = self.chunks[idx]
                    chunk_key = (chunk.document_name, chunk.page_number, chunk.chunk_id)
                    if chunk_key in seen_chunks:
                        continue
                    seen_chunks.add(chunk_key)
                    
                    similarity = max(0.0, 1.0 - (distance * distance / 2.0))
                    
                    if domain_filter and chunk.domain:
                        if domain_filter.lower() not in chunk.domain.lower():
                            similarity *= 0.8
                    
                    if similarity >= similarity_threshold:
                        results.append((chunk, similarity))
            
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
    def load_or_build_store(force_rebuild: bool = False, 
                           regulations_dir: Optional[Path] = None,
                           user_documents: Optional[List[Path]] = None) -> 'SafetyVectorStore':
        """
        Load or build vector store from documents
        This function does heavy work - should be called from @st.cache_resource
        NO STREAMLIT IMPORTS - safe to import from UI
        """
        from pathlib import Path
        from document_processor import DocumentProcessor
        from config import (
            REGULATIONS_DIR, VECTOR_STORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
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
        print("🔄 Building vector store from regulations folder...")
        
        # Initialize document processor
        document_processor = DocumentProcessor(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Use regulations_dir if provided, otherwise use default
        if regulations_dir is None:
            regulations_dir = REGULATIONS_DIR
        
        # Load all PDFs from regulations folder (recursive to get all subfolders)
        base_chunks = []
        if regulations_dir.exists():
            # Check if there are any PDFs (recursive search)
            pdf_files = list(regulations_dir.rglob("*.pdf"))
            if pdf_files:
                print(f"📚 Loading all regulations from: {regulations_dir}")
                print(f"   Found {len(pdf_files)} PDF file(s)")
                base_chunks = document_processor.process_directory(regulations_dir, recursive=True)
            else:
                print(f"⚠️  No PDF files found in {regulations_dir}")
        
        # Add user documents if provided
        user_chunks = []
        if user_documents:
            print(f"📤 Adding {len(user_documents)} user-uploaded document(s)...")
            for doc_path in user_documents:
                chunks = document_processor.process_document(doc_path)
                user_chunks.extend(chunks)
        
        all_chunks = base_chunks + user_chunks
        
        if not all_chunks:
            raise ValueError(f"No documents found. Please add PDF files to:\n"
                           f"  - {regulations_dir} (regulations folder)\n"
                           f"  Or upload documents via the UI")
        
        print(f"📊 Processing {len(all_chunks)} chunks ({len(base_chunks)} from regulations folder + {len(user_chunks)} user-uploaded)...")
        vector_store.create_index(all_chunks)
        
        # Try to save, but don't fail if it's a read-only filesystem (Streamlit Cloud)
        try:
            vector_store.save(VECTOR_STORE_DIR)
            print("✅ Vector store built and saved")
        except (PermissionError, OSError) as e:
            print(f"⚠️  Could not save vector store to disk (this is OK in Streamlit Cloud): {e}")
            print("✅ Vector store built (in memory)")
        
        return vector_store

