"""
Streamlit UI for Safety Copilot
"""
# Fix for Streamlit + PyTorch conflict - MUST be first
import os
import sys
import warnings

# Set environment variables BEFORE any other imports
# These must be set before streamlit is imported
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'error'

# Suppress torch-related warnings and errors
warnings.filterwarnings('ignore', message='.*torch.*')
warnings.filterwarnings('ignore', message='.*RuntimeError.*')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Suppress specific torch class errors
import logging
logging.getLogger('streamlit.watcher.local_sources_watcher').setLevel(logging.ERROR)

try:
    import streamlit as st
except RuntimeError as e:
    if 'event loop' in str(e) or 'torch' in str(e).lower():
        # If error is related to event loop/torch, try to continue anyway
        import streamlit as st
        st.warning("⚠️ Some compatibility warnings detected but continuing...")
    else:
        raise

# Page configuration MUST be first Streamlit command
st.set_page_config(
    page_title="Safety Copilot - AI-Powered Safety Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other modules after page config (delays torch import)
from pathlib import Path
import json
from config import DOCUMENTS_DIR, VECTOR_STORE_DIR, DATA_DIR

# Import core app (NO circular imports - core_app has no Streamlit)
from core_app import SafetyCopilotCore
# Import vector store loader (NO Streamlit imports)
from vector_store import SafetyVectorStore

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e88e5;
        text-align: center;
        padding: 1rem;
    }
    .confidence-high {
        color: #4caf50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
    .source-box {
        background: linear-gradient(135deg, #f0f7ff 0%, #e3f2fd 100%);
        padding: 1.2rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1e88e5;
        margin: 0.8rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .source-box:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
        border-left-color: #1976d2;
    }
    .pdf-link-button {
        background: linear-gradient(135deg, #1e88e5 0%, #1976d2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        text-decoration: none;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
        transition: all 0.2s;
    }
    .pdf-link-button:hover {
        background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
        transform: scale(1.05);
    }
    .source-link {
        color: #1e88e5;
        text-decoration: none;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.5rem 1rem;
        background-color: #e3f2fd;
        border-radius: 0.3rem;
        display: inline-block;
        margin-top: 0.5rem;
        transition: all 0.2s;
    }
    .source-link:hover {
        background-color: #bbdefb;
        text-decoration: underline;
    }
    .sources-header {
        background: linear-gradient(135deg, #1e88e5 0%, #1976d2 100%);
        color: white;
        padding: 0.8rem 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .refusal-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state (NON-NEGOTIABLE - must use session_state)
# Store ONLY references, NEVER FAISS/PyTorch objects directly
if "core" not in st.session_state:
    st.session_state.core = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Header
st.markdown('<h1 class="main-header">🛡️ Safety Copilot</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">AI-Powered Safety Engineering Assistant</p>', unsafe_allow_html=True)
st.markdown("---")

# Cached function for loading vector store (SAFE PATTERN)
# Cache ONLY the vector store - NOT the whole app
@st.cache_resource(show_spinner=True)
def load_vector_store(force_rebuild: bool = False) -> SafetyVectorStore:
    """
    Load or build vector store - cached to prevent re-running on every button click
    Returns ONLY the vector store object - NOT the app
    This is the ONLY safe way to do heavy work in Streamlit
    """
    return SafetyVectorStore.load_or_build_store(force_rebuild=force_rebuild)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Initialize button (SAFE PATTERN with defensive error handling)
    if st.button("🔄 Initialize/Reload Vector Store", type="primary"):
        try:
            with st.spinner("Initializing Safety Copilot..."):
                # Load vector store (cached - won't re-run if already loaded)
                vector_store = load_vector_store(force_rebuild=False)
                
                # Create core app instance and set vector store
                core = SafetyCopilotCore()
                core.set_vector_store(vector_store)
                
                # Store ONLY the core reference - NOT FAISS objects
                st.session_state.core = core
                st.session_state.initialized = True
                st.success("✅ Safety Copilot initialized!")
                st.rerun()
        except Exception as e:
            st.error("❌ Initialization failed")
            st.exception(e)  # Show full traceback for debugging
    
    # Force rebuild button (DISABLED on free tier - SAFE PATTERN)
    if st.button("🔨 Force Rebuild Vector Store"):
        # Disable rebuild on free tier (Streamlit Cloud / Render)
        # Rebuilds should be done locally, committed, and loaded read-only
        st.warning("⚠️ **Rebuild disabled on free tier**\n\n"
                  "For security and stability, vector store rebuilds are disabled in cloud deployments.\n\n"
                  "To rebuild:\n"
                  "1. Run locally: `python initialize.py`\n"
                  "2. Commit the updated vector store to Git\n"
                  "3. Deploy - the app will load the pre-built vector store")
        
        # Uncomment below for local development only
        # try:
        #     with st.spinner("Rebuilding vector store from documents... (This may take a few minutes)"):
        #         # Clear cache to force rebuild
        #         load_vector_store.clear()
        #         
        #         # Rebuild vector store
        #         vector_store = load_vector_store(force_rebuild=True)
        #         
        #         # Create core app instance and set vector store
        #         core = SafetyCopilotCore()
        #         core.set_vector_store(vector_store)
        #         
        #         # Store ONLY the core reference - NOT FAISS objects
        #         st.session_state.core = core
        #         st.session_state.initialized = True
        #         st.success("✅ Vector store rebuilt!")
        #         st.rerun()
        # except Exception as e:
        #     st.error("❌ Rebuild failed")
        #     st.exception(e)
    
    st.markdown("---")
    
    # Stats
    if st.session_state.initialized and st.session_state.core:
        stats = st.session_state.core.get_stats()
        st.header("📊 Statistics")
        st.write(f"**Documents:** {stats.get('num_documents', 0)}")
        st.write(f"**Chunks:** {stats.get('num_chunks', 0)}")
        st.write(f"**Model:** {stats.get('embedding_model', 'N/A')}")
    
    st.markdown("---")
    
    # Document management
    st.header("📄 Documents")
    st.write(f"**Professional Data Structure:** `{DATA_DIR}`")
    st.write(f"**Legacy Directory:** `{DOCUMENTS_DIR}`")
    
    # List documents from professional structure
    pdf_files = list(DATA_DIR.rglob("*.pdf")) if DATA_DIR.exists() else []
    legacy_pdfs = list(DOCUMENTS_DIR.glob("*.pdf")) if DOCUMENTS_DIR.exists() else []
    
    if pdf_files or legacy_pdfs:
        total = len(pdf_files) + len(legacy_pdfs)
        st.write(f"**Found {total} PDF(s):**")
        
        if pdf_files:
            st.write("**Professional Structure:**")
            for pdf_file in pdf_files:
                relative_path = pdf_file.relative_to(DATA_DIR)
                st.write(f"- `{relative_path}`")
        
        if legacy_pdfs:
            st.write("**Legacy Directory:**")
            for pdf_file in legacy_pdfs:
                st.write(f"- {pdf_file.name}")
    else:
        st.warning("⚠️ No PDF files found. Add PDFs to:")
        st.code(f"{DATA_DIR}/unece_regulations/\n{DATA_DIR}/nhtsa_guidelines/\n{DATA_DIR}/functional_safety_concepts/\n{DATA_DIR}/validation_testing/")
    
    st.markdown("---")
    
    # Clear chat
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    
    # Disclaimer
    st.header("⚠️ Disclaimer")
    st.warning("""
    This AI assistant provides information from safety documents for **decision support only**.
    
    - Not a substitute for professional safety engineering
    - Cannot provide legal interpretations
    - Cannot approve designs or certify compliance
    - Always consult qualified safety engineers
    """)

# Main content
if not st.session_state.initialized:
    st.info("👈 Please initialize the Safety Copilot from the sidebar first.")
    st.markdown("""
    ### Getting Started:
    1. Add PDF documents (ISO 26262, OEM manuals, etc.) to the `documents` folder
    2. Click "Initialize/Reload Vector Store" in the sidebar
    3. Start asking safety questions!
    
    ### Example Questions:
    - "What are the safety goals related to driver monitoring systems in ISO 26262?"
    - "Explain ASIL classification levels"
    - "What are the requirements for functional safety management?"
    """)
else:
    # Chat interface
    st.header("💬 Ask a Safety Question")
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        
        with st.chat_message("assistant"):
            # Check if refused
            if chat.get("refused"):
                st.markdown(f'<div class="refusal-box">⚠️ <strong>Request Refused</strong><br>{chat.get("refusal_reason", "")}</div>', unsafe_allow_html=True)
            else:
                # Display answer
                st.markdown(chat["answer"])
                
                # Confidence score
                confidence = chat.get("confidence_score", 0.0)
                confidence_level = chat.get("confidence_level", "low")
                
                if confidence_level == "high":
                    conf_class = "confidence-high"
                    conf_icon = "✅"
                elif confidence_level == "medium":
                    conf_class = "confidence-medium"
                    conf_icon = "⚠️"
                else:
                    conf_class = "confidence-low"
                    conf_icon = "❌"
                
                st.markdown(f'<p class="{conf_class}">{conf_icon} Confidence: {confidence:.1%} ({confidence_level})</p>', unsafe_allow_html=True)
                
                # Sources - Small clickable links
                sources = chat.get("sources", [])
                if sources:
                    from pdf_linker import find_pdf_path
                    # Collect unique PDF files for bottom display
                    unique_pdfs = set()
                    source_links = []
                    for source in sources:
                        pdf_path = find_pdf_path(source.get('document_name', ''))
                        if pdf_path:
                            unique_pdfs.add(pdf_path.name)
                            # Create small clickable link
                            file_url = pdf_path.as_uri()
                            anchor = f"#page={source.get('page_number', 1)}"
                            full_url = f"{file_url}{anchor}"
                            link_text = f"📄 {source.get('document_name', 'Document')} (p.{source.get('page_number', 1)})"
                            source_links.append(f'<a href="{full_url}" target="_blank" style="color: #1e88e5; text-decoration: none; font-size: 0.9rem; margin-right: 0.8rem; display: inline-block;">{link_text}</a>')
                    
                    # Display small links inline
                    if source_links:
                        st.markdown(f'<div style="margin-top: 0.5rem; padding: 0.5rem; background-color: #f5f5f5; border-radius: 0.3rem;">{" ".join(source_links)}</div>', unsafe_allow_html=True)
    
    # Chat input
    user_question = st.chat_input("Ask about ISO 26262, safety goals, ASIL levels, etc...")
    
    if user_question:
        # Add user message to history
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": "",
            "sources": [],
            "confidence_score": 0.0,
            "confidence_level": "low",
            "refused": False
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_question)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching safety documents and generating answer..."):
                try:
                    # Ensure core is initialized (defensive check)
                    if st.session_state.core is None or not st.session_state.initialized:
                        st.warning("⚠️ App not initialized. Please initialize from sidebar first.")
                        st.session_state.chat_history[-1]["answer"] = "App not initialized. Please click 'Initialize/Reload Vector Store' in the sidebar."
                    else:
                        # Get conversation history for synthesis agent
                        conv_history = []
                        for chat in st.session_state.chat_history:
                            if chat.get("question"):
                                conv_history.append({"role": "user", "content": chat["question"]})
                            if chat.get("answer"):
                                conv_history.append({"role": "assistant", "content": chat["answer"]})
                        
                        response = st.session_state.core.process_query(user_question, conversation_history=conv_history)
                        
                        # Update last chat entry
                        st.session_state.chat_history[-1].update(response)
                    
                    # Check if refused
                    if response.get("refused"):
                        st.markdown(f'<div class="refusal-box">⚠️ <strong>Request Refused</strong><br>{response.get("refusal_reason", "")}</div>', unsafe_allow_html=True)
                    else:
                        # Display answer
                        st.markdown(response["answer"])
                        
                        # Confidence score
                        confidence = response.get("confidence_score", 0.0)
                        confidence_level = response.get("confidence_level", "low")
                        
                        if confidence_level == "high":
                            conf_class = "confidence-high"
                            conf_icon = "✅"
                        elif confidence_level == "medium":
                            conf_class = "confidence-medium"
                            conf_icon = "⚠️"
                        else:
                            conf_class = "confidence-low"
                            conf_icon = "❌"
                        
                        st.markdown(f'<p class="{conf_class}">{conf_icon} Confidence: {confidence:.1%} ({confidence_level})</p>', unsafe_allow_html=True)
                        
                        # Show synthesis indicators if synthesis was used
                        synthesis_result = response.get("synthesis_result")
                        if synthesis_result:
                            num_sources = synthesis_result.get("num_sources", 0)
                            if num_sources > 1:
                                st.info(f"🔄 **Synthesis Mode**: Analyzed {num_sources} sources across multiple documents")
                            
                            if synthesis_result.get("tables"):
                                st.success(f"📊 **Tables Interpreted**: {len(synthesis_result['tables'])} technical table(s) found and analyzed")
                            
                            if synthesis_result.get("conflicts"):
                                st.warning(f"⚠️ **Conflicts Detected**: {len(synthesis_result['conflicts'])} potential standard conflict(s) identified")
                        
                        # Sources - Small clickable links
                        sources = response.get("sources", [])
                        if sources:
                            from pdf_linker import find_pdf_path
                            # Collect unique PDF files for bottom display
                            unique_pdfs = set()
                            source_links = []
                            for source in sources:
                                pdf_path = find_pdf_path(source.get('document_name', ''))
                                if pdf_path:
                                    unique_pdfs.add(pdf_path.name)
                                    # Create small clickable link
                                    file_url = pdf_path.as_uri()
                                    anchor = f"#page={source.get('page_number', 1)}"
                                    full_url = f"{file_url}{anchor}"
                                    link_text = f"📄 {source.get('document_name', 'Document')} (p.{source.get('page_number', 1)})"
                                    source_links.append(f'<a href="{full_url}" target="_blank" style="color: #1e88e5; text-decoration: none; font-size: 0.9rem; margin-right: 0.8rem; display: inline-block;">{link_text}</a>')
                            
                            # Display small links inline
                            if source_links:
                                st.markdown(f'<div style="margin-top: 0.5rem; padding: 0.5rem; background-color: #f5f5f5; border-radius: 0.3rem;">{" ".join(source_links)}</div>', unsafe_allow_html=True)
                            
                            # Store PDF names for bottom display
                            if 'pdf_files_used' not in st.session_state:
                                st.session_state.pdf_files_used = set()
                            st.session_state.pdf_files_used.update(unique_pdfs)
                    
                except Exception as e:
                    st.error("❌ Error processing query")
                    st.exception(e)  # Show full traceback for debugging
                    st.session_state.chat_history[-1]["answer"] = f"Error processing query: {str(e)}"
        
        st.rerun()

# Footer with PDF files used
st.markdown("---")
if 'pdf_files_used' in st.session_state and st.session_state.pdf_files_used:
    pdf_list = ", ".join(sorted(st.session_state.pdf_files_used))
    st.markdown(f"""
    <div style='color: #666; padding: 0.5rem; font-size: 0.85rem;'>
        <strong>📚 Documents Referenced:</strong> {pdf_list}
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Safety Copilot</strong> - AI-Powered Safety Engineering Assistant</p>
    <p>Built with RAG, LangGraph, FAISS, and SentenceTransformers</p>
    <p>⚠️ For decision support only. Always consult qualified safety engineers.</p>
</div>
""", unsafe_allow_html=True)

