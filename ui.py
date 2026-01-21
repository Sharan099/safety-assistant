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
from config import DOCUMENTS_DIR, VECTOR_STORE_DIR, DATA_DIR, EMBEDDING_MODEL

# Import core app (NO circular imports - core_app has no Streamlit)
from core_app import SafetyCopilotCore
# Import vector store loader (NO Streamlit imports)
from vector_store import SafetyVectorStore

# Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts for better typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Main content styling */
    .main-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        line-height: 1.7;
        color: #2c3e50;
    }
    
    /* Answer text styling */
    .stMarkdown {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        line-height: 1.8;
        color: #2c3e50;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #1e88e5;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .stMarkdown p {
        margin-bottom: 1rem;
        text-align: justify;
    }
    
    .stMarkdown ul, .stMarkdown ol {
        margin-left: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .stMarkdown li {
        margin-bottom: 0.5rem;
    }
    
    .stMarkdown strong {
        color: #1976d2;
        font-weight: 600;
    }
    
    .stMarkdown code {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        background-color: #f5f5f5;
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        font-size: 0.9em;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e88e5;
        text-align: center;
        padding: 1rem;
        font-family: 'Inter', sans-serif;
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
    
    /* Chat message styling */
    .stChatMessage {
        font-family: 'Inter', sans-serif;
    }
    
    /* Better spacing for chat */
    [data-testid="stChatMessage"] {
        margin-bottom: 1.5rem;
    }
    
    /* Answer container - Human-friendly, no white box */
    .answer-container {
        background: transparent;
        padding: 1rem 0;
        margin: 0.5rem 0;
        line-height: 1.8;
    }
    
    /* Chat message styling - More human-friendly */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 1rem 0 !important;
    }
    
    /* Answer text - Better readability with spacing */
    .answer-text {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        line-height: 1.9;
        color: #2c3e50;
        text-align: left;
        padding: 0.5rem 0;
    }
    
    .answer-text p {
        margin-bottom: 1rem;
        text-align: left;
    }
    
    /* Section styling */
    .answer-section {
        margin: 1.2rem 0;
        padding: 0.8rem 0;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .answer-section:last-child {
        border-bottom: none;
    }
    
    /* Source highlighting - Different colors */
    .source-highlight {
        display: inline-block;
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        color: #2e7d32;
        padding: 0.3rem 0.6rem;
        border-radius: 0.4rem;
        font-weight: 500;
        font-size: 0.95rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        border-left: 3px solid #4caf50;
    }
    
    .clause-highlight {
        display: inline-block;
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        color: #e65100;
        padding: 0.3rem 0.6rem;
        border-radius: 0.4rem;
        font-weight: 500;
        font-size: 0.95rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        border-left: 3px solid #ff9800;
    }
    
    /* Source links - Professional styling */
    .source-links-container {
        margin-top: 1.5rem;
        padding: 1rem;
        background: linear-gradient(135deg, #f5f5f5 0%, #eeeeee 100%);
        border-radius: 0.5rem;
        border-left: 4px solid #1e88e5;
    }
    
    .source-link-item {
        display: inline-block;
        background: #ffffff;
        color: #1e88e5;
        padding: 0.5rem 1rem;
        border-radius: 0.4rem;
        margin: 0.3rem 0.5rem 0.3rem 0;
        text-decoration: none;
        font-size: 0.9rem;
        font-weight: 500;
        border: 1px solid #e0e0e0;
        transition: all 0.2s;
    }
    
    .source-link-item:hover {
        background: #1e88e5;
        color: white;
        border-color: #1e88e5;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(30, 136, 229, 0.2);
    }
    
    /* Disclaimer styling */
    .disclaimer-box {
        margin-top: 2rem;
        padding: 1rem 1.2rem;
        background: linear-gradient(135deg, #fff9e6 0%, #fff3cd 100%);
        border-left: 4px solid #ffc107;
        border-radius: 0.5rem;
        font-size: 0.9rem;
        color: #856404;
        line-height: 1.6;
    }
    
    /* Remove white background from chat messages */
    .stChatMessage {
        background: transparent !important;
    }
    
    /* Better paragraph spacing */
    .answer-text p + p {
        margin-top: 0.8rem;
    }
    
    /* Improved readability */
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
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
if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []
if "show_rag_process" not in st.session_state:
    st.session_state.show_rag_process = False

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
    # Load base regulations from /data/regulations/ at startup
    from config import REGULATIONS_DIR
    return SafetyVectorStore.load_or_build_store(
        force_rebuild=force_rebuild,
        regulations_dir=REGULATIONS_DIR,
        user_documents=None  # User documents added separately via upload
    )

# Sidebar
with st.sidebar:
    st.header("📤 Upload Documents")
    st.markdown("Upload PDF documents to build your knowledge base")
    
    # Document uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF documents to process and add to the vector store"
    )
    
    if uploaded_files:
        st.info(f"📄 {len(uploaded_files)} file(s) selected")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")
    
    # Process uploaded documents button
    if uploaded_files and st.button("🔄 Process & Add Documents", type="primary"):
        try:
            from document_processor import DocumentProcessor
            from config import CHUNK_SIZE, CHUNK_OVERLAP
            import tempfile
            import os
            
            with st.spinner("Processing uploaded documents..."):
                # Initialize document processor
                processor = DocumentProcessor(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                
                # Get or create vector store
                if st.session_state.initialized and st.session_state.core:
                    vector_store = st.session_state.core.vector_store
                else:
                    # Create new vector store
                    vector_store = SafetyVectorStore(embedding_model=EMBEDDING_MODEL)
                    core = SafetyCopilotCore()
                    core.set_vector_store(vector_store)
                    st.session_state.core = core
                
                processed_count = 0
                total_chunks = 0
                
                # Process each uploaded file
                for uploaded_file in uploaded_files:
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            tmp_path = Path(tmp_file.name)
                        
                        # Process document
                        chunks = processor.process_document(tmp_path)
                        
                        if chunks:
                            # Add to vector store
                            vector_store.add_documents(chunks)
                            processed_count += 1
                            total_chunks += len(chunks)
                            
                            # Track uploaded documents
                            if uploaded_file.name not in st.session_state.uploaded_documents:
                                st.session_state.uploaded_documents.append(uploaded_file.name)
                            
                            st.success(f"✅ Processed {uploaded_file.name}: {len(chunks)} chunks")
                        else:
                            st.warning(f"⚠️ No content extracted from {uploaded_file.name}")
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                        if tmp_path.exists():
                            os.unlink(tmp_path)
                
                if processed_count > 0:
                    # Recreate workflow with updated vector store
                    st.session_state.core.set_vector_store(vector_store)
                    st.session_state.initialized = True
                    st.success(f"🎉 Successfully processed {processed_count} document(s) with {total_chunks} total chunks!")
                    st.rerun()
                else:
                    st.error("❌ No documents were successfully processed")
                    
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")
            st.exception(e)
    
    st.markdown("---")
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
    st.header("📄 Documents in Knowledge Base")
    
    # Show all regulations from /regulations/ folder (recursive)
    from config import REGULATIONS_DIR
    all_regulations = list(REGULATIONS_DIR.rglob("*.pdf")) if REGULATIONS_DIR.exists() else []
    
    if all_regulations:
        st.write(f"**Regulations Folder ({len(all_regulations)} PDFs):**")
        # Group by subfolder for better display
        by_folder = {}
        for pdf_file in all_regulations:
            folder = pdf_file.parent.relative_to(REGULATIONS_DIR)
            folder_name = str(folder) if str(folder) != "." else "root"
            if folder_name not in by_folder:
                by_folder[folder_name] = []
            by_folder[folder_name].append(pdf_file)
        
        # Show first 15 PDFs total
        shown_count = 0
        for folder_name, pdfs in sorted(by_folder.items()):
            if shown_count >= 15:
                break
            for pdf_file in pdfs[:15 - shown_count]:
                if folder_name == "root":
                    st.write(f"- 📄 {pdf_file.name}")
                else:
                    st.write(f"- 📄 `{folder_name}/{pdf_file.name}`")
                shown_count += 1
                if shown_count >= 15:
                    break
        
        if len(all_regulations) > 15:
            st.write(f"  ... and {len(all_regulations) - 15} more")
    
    # Show uploaded documents
    if st.session_state.uploaded_documents:
        st.write(f"**User Uploaded Documents ({len(st.session_state.uploaded_documents)}):**")
        for doc in st.session_state.uploaded_documents:
            st.write(f"- 📄 {doc}")
    
    if not all_regulations and not st.session_state.uploaded_documents:
        st.info("📤 Add PDFs to `/regulations/` folder or upload documents above to get started!")
    
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
    # Welcome screen with RAG explanation
    st.markdown("""
    ## 🛡️ Welcome to Safety Copilot!
    
    **AI-Powered Safety Engineering Assistant using RAG (Retrieval-Augmented Generation)**
    
    ### 🚀 How It Works (RAG Process):
    
    1. **📤 Upload Documents**: Upload your PDF safety documents (ISO 26262, regulations, manuals, etc.)
    2. **🔄 Processing**: 
       - Documents are parsed and split into chunks
       - Each chunk is converted to embeddings (vector representations)
       - Embeddings are stored in a FAISS vector database
    3. **🔍 Query Processing**:
       - Your question is converted to an embedding
       - Similar document chunks are retrieved from the vector store
       - Relevant context is passed to the AI model
    4. **💬 Answer Generation**: 
       - AI generates answers based on retrieved context
       - Sources are cited with page numbers and document references
    
    ### 📋 Getting Started:
    
    **Option 1: Upload Documents (Recommended)**
    1. Go to the sidebar → Upload PDF files
    2. Click "Process & Add Documents"
    3. Start asking questions!
    
    **Option 2: Use Pre-loaded Documents**
    1. Click "Initialize/Reload Vector Store" in the sidebar
    2. Start asking questions!
    
    ### 💡 Example Questions:
    - "What are the safety goals related to driver monitoring systems in ISO 26262?"
    - "Explain ASIL classification levels"
    - "What are the requirements for functional safety management?"
    - "Compare UNECE R155 and R156 requirements"
    """)
    
    # Toggle to show/hide RAG process details
    with st.expander("🔬 Learn More About RAG (Retrieval-Augmented Generation)"):
        st.markdown("""
        ### What is RAG?
        
        **Retrieval-Augmented Generation (RAG)** combines the power of:
        - **Vector Search**: Fast similarity search using embeddings
        - **Large Language Models**: Advanced text generation
        
        ### The RAG Pipeline:
        
        ```
        User Question
            ↓
        [Embedding Model] → Query Vector
            ↓
        [FAISS Vector Store] → Similar Document Chunks
            ↓
        [Context Assembly] → Relevant Context
            ↓
        [LLM (Claude/GPT)] → Generated Answer + Sources
        ```
        
        ### Benefits:
        - ✅ **No Hallucination**: Answers are grounded in your documents
        - ✅ **Source Traceability**: Every answer cites its sources
        - ✅ **Up-to-date**: Add new documents anytime
        - ✅ **Domain-Specific**: Trained on your safety documents
        
        ### Technical Details:
        - **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
        - **Vector Database**: FAISS (Facebook AI Similarity Search)
        - **Chunking Strategy**: 600 characters with 100 character overlap
        - **Retrieval**: Top-K similarity search with threshold filtering
        """)
else:
    # RAG Process Visualization Toggle
    st.session_state.show_rag_process = st.checkbox("🔬 Show RAG Process Details", value=st.session_state.show_rag_process)
    
    # Chat interface
    st.header("💬 Ask a Safety Question")
    
    # Show RAG process if enabled
    if st.session_state.show_rag_process:
        with st.expander("🔍 RAG Process Flow (Click to see details)", expanded=True):
            st.markdown("""
            ### Current RAG Pipeline:
            
            1. **📤 Document Upload** → PDFs are parsed and chunked
            2. **🔄 Embedding Generation** → Chunks converted to 384-dimensional vectors
            3. **💾 Vector Storage** → Stored in FAISS index for fast similarity search
            4. **🔍 Query Processing** → Your question → embedding → similarity search
            5. **📚 Context Retrieval** → Top-K most relevant chunks retrieved
            6. **🤖 Answer Generation** → LLM generates answer using retrieved context
            7. **📎 Source Citation** → Answer includes document references and page numbers
            
            **Current Stats:**
            """)
            if st.session_state.core:
                stats = st.session_state.core.get_stats()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents", stats.get('num_documents', 0))
                with col2:
                    st.metric("Chunks", stats.get('num_chunks', 0))
                with col3:
                    st.metric("Embedding Model", stats.get('embedding_model', 'N/A').split('/')[-1])
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        
        with st.chat_message("assistant"):
            # Check if refused
            if chat.get("refused"):
                st.markdown(f'<div class="refusal-box">⚠️ <strong>Request Refused</strong><br>{chat.get("refusal_reason", "")}</div>', unsafe_allow_html=True)
            else:
                # Display answer with better formatting and expandable sections
                answer = chat["answer"]
                
                # Check if answer has 4-section format
                has_sections = "### 1. Simple Explanation" in answer or "## Simple Explanation" in answer
                
                # Process answer text - remove disclaimer for separate display
                answer_clean = answer
                disclaimer_text = None
                if "⚠️" in answer or "Disclaimer" in answer:
                    # Extract disclaimer
                    lines = answer.split('\n')
                    answer_lines = []
                    disclaimer_lines = []
                    in_disclaimer = False
                    for line in lines:
                        if "⚠️" in line or ("Disclaimer" in line and "disclaimer" in line.lower()):
                            in_disclaimer = True
                        if in_disclaimer:
                            disclaimer_lines.append(line)
                        else:
                            answer_lines.append(line)
                    answer_clean = '\n'.join(answer_lines).strip()
                    if disclaimer_lines:
                        disclaimer_text = '\n'.join(disclaimer_lines).replace('⚠️ **Disclaimer**:', '').replace('⚠️ **Disclaimer**', '').strip()
                
                # Display answer with better formatting
                if has_sections:
                    # Parse and display sections separately with better styling
                    sections = {}
                    current_section = None
                    current_text = []
                    
                    lines = answer_clean.split('\n')
                    for line in lines:
                        if line.startswith('###') or line.startswith('##'):
                            if current_section:
                                sections[current_section] = '\n'.join(current_text).strip()
                            # Extract section name
                            section_name = line.replace('#', '').strip()
                            current_section = section_name
                            current_text = []
                        else:
                            if current_section:
                                current_text.append(line)
                    if current_section:
                        sections[current_section] = '\n'.join(current_text).strip()
                    
                    # Display sections with professional styling
                    for section_name, section_text in sections.items():
                        if "Reference" in section_name or "References" in section_name:
                            # Skip - will be shown separately with sources
                            continue
                        elif "Calculation" in section_name or "Analysis" in section_name:
                            st.markdown(f'<div class="answer-section" style="background: linear-gradient(135deg, #f0f7ff 0%, #e3f2fd 100%); padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1e88e5;">', unsafe_allow_html=True)
                            st.markdown(f'<h4 style="color: #1e88e5; margin-top: 0; margin-bottom: 0.8rem;">{section_name}</h4>', unsafe_allow_html=True)
                            st.markdown(f'<div class="answer-text">{section_text}</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="answer-section">', unsafe_allow_html=True)
                            st.markdown(f'<h4 style="color: #2c3e50; margin-top: 0; margin-bottom: 0.8rem; font-weight: 600;">{section_name}</h4>', unsafe_allow_html=True)
                            st.markdown(f'<div class="answer-text">{section_text}</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    # Display as regular answer with better formatting
                    st.markdown(f'<div class="answer-container">', unsafe_allow_html=True)
                    st.markdown(f'<div class="answer-text">{answer_clean}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Sources - Highlighted with different colors
                sources = chat.get("sources", [])
                if sources:
                    from pdf_linker import find_pdf_path
                    # Collect unique PDF files for bottom display
                    unique_pdfs = set()
                    source_links_html = []
                    
                    for source in sources:
                        pdf_path = find_pdf_path(source.get('document_name', ''))
                        if pdf_path:
                            unique_pdfs.add(pdf_path.name)
                            
                            # Extract regulation and clause info
                            regulation = source.get('regulation') or source.get('method') or source.get('document_name', 'Document')
                            clause = source.get('clause') or source.get('section_number', '')
                            page = source.get('page_number', 1)
                            
                            # Create styled link
                            file_url = pdf_path.as_uri()
                            anchor = f"#page={page}"
                            full_url = f"{file_url}{anchor}"
                            
                            # Build link text with highlights
                            link_parts = []
                            if regulation:
                                link_parts.append(f'<span class="source-highlight">{regulation}</span>')
                            if clause:
                                link_parts.append(f'<span class="clause-highlight">Clause {clause}</span>')
                            link_parts.append(f'<a href="{full_url}" target="_blank" class="source-link-item">📄 Page {page}</a>')
                            
                            source_links_html.append(' '.join(link_parts))
                    
                    # Display sources in styled container
                    if source_links_html:
                        st.markdown('<div class="source-links-container">', unsafe_allow_html=True)
                        st.markdown('<strong style="color: #1e88e5; font-size: 0.95rem; margin-bottom: 0.5rem; display: block;">📚 Sources:</strong>', unsafe_allow_html=True)
                        st.markdown('<div>' + '<br>'.join(source_links_html) + '</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Store for footer
                    if 'pdf_files_used' not in st.session_state:
                        st.session_state.pdf_files_used = set()
                    st.session_state.pdf_files_used.update(unique_pdfs)
                
                # Show disclaimer separately at the end with spacing
                if disclaimer_text:
                    st.markdown(f'<div class="disclaimer-box"><strong>⚠️ Disclaimer:</strong> {disclaimer_text}</div>', unsafe_allow_html=True)
    
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
                        # Process answer - remove disclaimer for separate display
                        answer_clean = response.get("answer", "")
                        disclaimer_text = None
                        if "⚠️" in answer_clean or "Disclaimer" in answer_clean:
                            lines = answer_clean.split('\n')
                            answer_lines = []
                            disclaimer_lines = []
                            in_disclaimer = False
                            for line in lines:
                                if "⚠️" in line or ("Disclaimer" in line and "disclaimer" in line.lower()):
                                    in_disclaimer = True
                                if in_disclaimer:
                                    disclaimer_lines.append(line)
                                else:
                                    answer_lines.append(line)
                            answer_clean = '\n'.join(answer_lines).strip()
                            if disclaimer_lines:
                                disclaimer_text = '\n'.join(disclaimer_lines).replace('⚠️ **Disclaimer**:', '').replace('⚠️ **Disclaimer**', '').strip()
                        
                        # Display answer with better formatting
                        st.markdown(f'<div class="answer-container">', unsafe_allow_html=True)
                        st.markdown(f'<div class="answer-text">{answer_clean}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Sources - Highlighted
                        sources = response.get("sources", [])
                        if sources:
                            from pdf_linker import find_pdf_path
                            unique_pdfs = set()
                            source_links_html = []
                            
                            for source in sources:
                                pdf_path = find_pdf_path(source.get('document_name', ''))
                                if pdf_path:
                                    unique_pdfs.add(pdf_path.name)
                                    regulation = source.get('regulation') or source.get('method') or source.get('document_name', 'Document')
                                    clause = source.get('clause') or source.get('section_number', '')
                                    page = source.get('page_number', 1)
                                    
                                    file_url = pdf_path.as_uri()
                                    anchor = f"#page={page}"
                                    full_url = f"{file_url}{anchor}"
                                    
                                    link_parts = []
                                    if regulation:
                                        link_parts.append(f'<span class="source-highlight">{regulation}</span>')
                                    if clause:
                                        link_parts.append(f'<span class="clause-highlight">Clause {clause}</span>')
                                    link_parts.append(f'<a href="{full_url}" target="_blank" class="source-link-item">📄 Page {page}</a>')
                                    source_links_html.append(' '.join(link_parts))
                            
                            if source_links_html:
                                st.markdown('<div class="source-links-container">', unsafe_allow_html=True)
                                st.markdown('<strong style="color: #1e88e5; font-size: 0.95rem; margin-bottom: 0.5rem; display: block;">📚 Sources:</strong>', unsafe_allow_html=True)
                                st.markdown('<div>' + '<br>'.join(source_links_html) + '</div>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            if 'pdf_files_used' not in st.session_state:
                                st.session_state.pdf_files_used = set()
                            st.session_state.pdf_files_used.update(unique_pdfs)
                        
                        # Disclaimer at the end
                        if disclaimer_text:
                            st.markdown(f'<div class="disclaimer-box"><strong>⚠️ Disclaimer:</strong> {disclaimer_text}</div>', unsafe_allow_html=True)
                        
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

