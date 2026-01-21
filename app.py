"""
Main Application for Safety Copilot
Handles document processing, vector store management, and query processing
"""
from pathlib import Path
from typing import Dict, Optional, List
from document_processor import DocumentProcessor
from vector_store import SafetyVectorStore
from safety_copilot import create_safety_copilot_workflow
from config import (
    DATA_DIR, DOCUMENTS_DIR, VECTOR_STORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD
)
import json

class SafetyCopilotApp:
    """Main application class for Safety Copilot"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.vector_store = SafetyVectorStore(embedding_model=EMBEDDING_MODEL)
        self.workflow = None
        self.is_initialized = False
    
    def initialize(self, force_rebuild: bool = False):
        """Initialize vector store from documents"""
        vector_store_path = VECTOR_STORE_DIR / "faiss_index.bin"
        
        # Check if vector store exists
        if vector_store_path.exists() and not force_rebuild:
            try:
                print("📂 Loading existing vector store...")
                self.vector_store.load(VECTOR_STORE_DIR)
                self.is_initialized = True
                print("✅ Vector store loaded successfully")
            except Exception as e:
                print(f"⚠️  Error loading vector store: {e}, rebuilding...")
                import traceback
                traceback.print_exc()
                force_rebuild = True
        
        # Build vector store if needed
        if not self.is_initialized or force_rebuild:
            print("🔄 Building vector store from documents...")
            
            # Try professional data structure first, fallback to legacy
            from config import DATA_DIR
            if DATA_DIR.exists() and any(DATA_DIR.rglob("*.pdf")):
                print(f"📚 Using professional data structure: {DATA_DIR}")
                chunks = self.document_processor.process_directory(DATA_DIR, recursive=True)
            else:
                print(f"📚 Using legacy documents directory: {DOCUMENTS_DIR}")
                chunks = self.document_processor.process_directory(DOCUMENTS_DIR, recursive=False)
            
            if not chunks:
                raise ValueError(f"No documents found. Please add PDF files to:\n"
                               f"  - {DATA_DIR} (professional structure)\n"
                               f"  - {DOCUMENTS_DIR} (legacy)")
            
            self.vector_store.create_index(chunks)
            
            # Save vector store - handle file I/O errors gracefully for Streamlit Cloud
            try:
                self.vector_store.save(VECTOR_STORE_DIR)
                print("✅ Vector store built and saved")
            except Exception as e:
                print(f"⚠️  Warning: Could not save vector store to disk: {e}")
                print("   This is OK in Streamlit Cloud - vector store is in memory")
                # Continue anyway - vector store is in memory
            
            self.is_initialized = True
        
        # Create workflow
        self.workflow = create_safety_copilot_workflow(self.vector_store)
        print("✅ Safety Copilot initialized and ready")
    
    def process_query(self, question: str, conversation_history: List[Dict] = None) -> Dict:
        """
        Process a user question and return answer with sources
        """
        if not self.is_initialized:
            raise RuntimeError("Safety Copilot not initialized. Call initialize() first.")
        
        if conversation_history is None:
            conversation_history = []
        
        # Determine if scenario reasoning is needed (will be set by guardrail agent)
        # Create initial state
        initial_state = {
            "user_question": question,
            "retrieved_chunks": [],
            "answer": "",
            "sources": [],
            "confidence_score": 0.0,
            "confidence_level": "low",
            "should_refuse": False,
            "refusal_reason": "",
            "workflow_stage": "start",
            "needs_synthesis": False,
            "needs_scenario_reasoning": False,
            "synthesis_result": None,
            "conversation_history": conversation_history if conversation_history else []
        }
        
        # Run workflow
        result = self.workflow.invoke(initial_state)
        
        # Format response
        response = {
            "question": question,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "confidence_score": round(result.get("confidence_score", 0.0), 3),
            "confidence_level": result.get("confidence_level", "low"),
            "refused": result.get("should_refuse", False),
            "refusal_reason": result.get("refusal_reason", ""),
            "synthesis_result": result.get("synthesis_result")
        }
        
        return response
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        num_chunks = len(self.vector_store.chunks) if self.vector_store.chunks else 0
        num_documents = len(set(chunk.document_name for chunk in self.vector_store.chunks)) if self.vector_store.chunks else 0
        
        return {
            "status": "initialized",
            "num_chunks": num_chunks,
            "num_documents": num_documents,
            "embedding_model": EMBEDDING_MODEL
        }

# Removed global singleton - use st.session_state instead
# This prevents issues with Streamlit's rerun model

