"""
Core Application Logic for Safety Copilot
NO STREAMLIT IMPORTS - This is pure business logic
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

class SafetyCopilotCore:
    """
    Core application logic - NO Streamlit dependencies
    This class handles business logic only
    """
    
    def __init__(self):
        """Lightweight initialization - no heavy work here"""
        self.vector_store: Optional[SafetyVectorStore] = None
        self.workflow = None
        self.is_initialized = False
    
    def set_vector_store(self, vector_store: SafetyVectorStore):
        """Set vector store from cached resource"""
        self.vector_store = vector_store
        # Create workflow after vector store is set
        self.workflow = create_safety_copilot_workflow(self.vector_store)
        self.is_initialized = True
    
    def process_query(self, question: str, conversation_history: List[Dict] = None) -> Dict:
        """
        Process a user question and return answer with sources
        """
        if not self.is_initialized or self.workflow is None:
            raise RuntimeError("Safety Copilot not initialized. Call set_vector_store() first.")
        
        if conversation_history is None:
            conversation_history = []
        
        # Determine if scenario reasoning is needed
        from domain_classifier import DomainClassifier
        needs_synthesis = DomainClassifier.needs_synthesis(question)
        needs_scenario_reasoning = DomainClassifier.needs_scenario_reasoning(question)
        
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
            "needs_synthesis": needs_synthesis,
            "needs_scenario_reasoning": needs_scenario_reasoning,
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
        if not self.is_initialized or self.vector_store is None:
            return {"status": "not_initialized"}
        
        num_chunks = len(self.vector_store.chunks) if self.vector_store.chunks else 0
        num_documents = len(set(chunk.document_name for chunk in self.vector_store.chunks)) if self.vector_store.chunks else 0
        
        return {
            "status": "initialized",
            "num_chunks": num_chunks,
            "num_documents": num_documents,
            "embedding_model": EMBEDDING_MODEL
        }


