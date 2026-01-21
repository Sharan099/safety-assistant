"""
Streamlit UI for Safety Copilot - Clean & Simple Design with Sky Blue/Teal Theme
"""
# Fix for Streamlit + PyTorch conflict - MUST be first
import os
import sys
import warnings

os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'error'

warnings.filterwarnings('ignore', message='.*torch.*')
warnings.filterwarnings('ignore', message='.*RuntimeError.*')
warnings.filterwarnings('ignore', category=RuntimeWarning)

import logging
logging.getLogger('streamlit.watcher.local_sources_watcher').setLevel(logging.ERROR)

try:
    import streamlit as st
except RuntimeError as e:
    if 'event loop' in str(e) or 'torch' in str(e).lower():
        import streamlit as st
        st.warning("⚠️ Some compatibility warnings detected but continuing...")
    else:
        raise

# Page configuration
st.set_page_config(
    page_title="Safety Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules
from pathlib import Path
from config import REGULATIONS_DIR, VECTOR_STORE_DIR, EMBEDDING_MODEL
from core_app import SafetyCopilotCore
from vector_store import SafetyVectorStore

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'core' not in st.session_state:
    st.session_state.core = None

# Custom CSS with Sky Blue/Teal Theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Color Palette - Sky Blue & Teal */
    :root {
        --primary-teal: #14B8A6;
        --primary-sky: #0EA5E9;
        --accent-blue: #3B82F6;
        --light-teal: #E0F2FE;
        --light-sky: #F0F9FF;
        --dark-teal: #0F766E;
        --text-dark: #1E293B;
        --text-light: #FFFFFF;
        --border-color: #CBD5E1;
        --bg-light: #F8FAFC;
        --shadow: 0 2px 8px rgba(20, 184, 166, 0.1);
    }
    
    /* Main App Background */
    .stApp {
        background: var(--bg-light) !important;
        font-family: 'Inter', 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F766E 0%, #14B8A6 100%) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: var(--text-light) !important;
    }
    
    /* Main Content Area */
    .main .block-container {
        max-width: 1000px;
        padding: 2rem 1rem;
        background: var(--bg-light);
    }
    
    /* Chat Messages - Clean White Cards */
    [data-testid="stChatMessage"] {
        background: white !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        margin: 1rem 0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* User Message */
    [data-testid="stChatMessage"][data-message-author="user"] {
        background: var(--light-sky) !important;
        border-left: 4px solid var(--primary-sky) !important;
    }
    
    /* Assistant Message */
    [data-testid="stChatMessage"][data-message-author="assistant"] {
        background: white !important;
        border-left: 4px solid var(--primary-teal) !important;
    }
    
    /* Markdown Styling - Clean & Readable */
    .stMarkdown {
        color: var(--text-dark) !important;
        font-family: 'Inter', 'Poppins', sans-serif !important;
        font-size: 1rem !important;
        line-height: 1.7 !important;
    }
    
    .stMarkdown p {
        margin-bottom: 0.8rem !important;
        color: var(--text-dark) !important;
        line-height: 1.7 !important;
    }
    
    /* Headings - Teal Color */
    .stMarkdown h3, .stMarkdown h4 {
        color: var(--primary-teal) !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', 'Poppins', sans-serif !important;
    }
    
    /* Bullet Points */
    .stMarkdown ul, .stMarkdown ol {
        margin: 1rem 0 !important;
        padding-left: 1.5rem !important;
    }
    
    .stMarkdown li {
        margin-bottom: 0.5rem !important;
        color: var(--text-dark) !important;
        line-height: 1.7 !important;
    }
    
    /* Code Blocks */
    .stMarkdown code {
        background: var(--light-teal) !important;
        color: var(--dark-teal) !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stMarkdown pre {
        background: var(--light-teal) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Buttons - Teal */
    .stButton > button {
        background: var(--primary-teal) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', 'Poppins', sans-serif !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        background: var(--dark-teal) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(20, 184, 166, 0.3) !important;
    }
    
    /* Chat Input */
    .stChatInput {
        background: white !important;
        border-radius: 12px !important;
        border: 2px solid var(--border-color) !important;
        box-shadow: var(--shadow) !important;
    }
    
    .stChatInput:focus {
        border-color: var(--primary-teal) !important;
    }
    
    /* Source Links */
    .source-links {
        margin-top: 1.5rem !important;
        padding: 1rem !important;
        background: var(--light-teal) !important;
        border-radius: 8px !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .source-link {
        color: var(--primary-teal) !important;
        text-decoration: none !important;
        margin-right: 1rem !important;
        font-weight: 500 !important;
    }
    
    .source-link:hover {
        text-decoration: underline !important;
        color: var(--dark-teal) !important;
    }
    
    /* Disclaimer */
    .disclaimer {
        margin-top: 1.5rem !important;
        padding: 1rem 1.5rem !important;
        background: #FEF3C7 !important;
        border-left: 4px solid #F59E0B !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
        color: #92400E !important;
    }
    
    /* Title */
    h1 {
        color: var(--primary-teal) !important;
        font-family: 'Inter', 'Poppins', sans-serif !important;
        font-weight: 700 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Status indicators */
    .status-ready {
        color: var(--primary-teal) !important;
        font-weight: 600 !important;
    }
    
    .status-waiting {
        color: #64748B !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='color: #FFFFFF; font-size: 1.8rem; margin: 0; font-weight: 700;'>🛡️ Safety Copilot</h1>
        <p style='color: #FFFFFF; opacity: 0.9; font-size: 0.9rem; margin-top: 0.5rem;'>AI-Powered Safety Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Initialize vector store
    @st.cache_resource
    def load_vector_store():
        return SafetyVectorStore.load_or_build_store(
            force_rebuild=False,
            regulations_dir=REGULATIONS_DIR,
            user_documents=None
        )
    
    # Initialize button
    if st.button("🚀 Initialize Safety Copilot", use_container_width=True):
        with st.spinner("Initializing..."):
            try:
                vector_store = load_vector_store()
                core = SafetyCopilotCore()
                core.set_vector_store(vector_store)
                st.session_state.core = core
                st.session_state.initialized = True
                st.success("✅ Initialized successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.exception(e)
    
    # Status indicator
    st.markdown("---")
    if st.session_state.initialized:
        st.markdown('<p class="status-ready">✅ Ready</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-waiting">⏳ Not initialized</p>', unsafe_allow_html=True)

# Main Content Area
st.title("💬 Safety Assistant")

# Initialize if not done
if not st.session_state.initialized:
    st.info("👈 Please initialize the Safety Copilot from the sidebar to get started.")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            answer = message.get("content", "") or ""
            
            if message["role"] == "user":
                st.write(message["content"])
            else:
                # Display answer - SIMPLIFIED: Just show markdown directly
                if answer and answer.strip():
                    st.markdown(answer)
                else:
                    st.warning("⚠️ No answer generated. Please try again.")
                
                # Sources
                sources = message.get("sources", [])
                if sources:
                    from pdf_linker import find_pdf_path
                    source_links = []
                    for source in sources:
                        pdf_path = find_pdf_path(source.get('document_name', ''))
                        if pdf_path:
                            page = source.get('page_number', 1)
                            file_url = pdf_path.as_uri()
                            full_url = f"{file_url}#page={page}"
                            regulation = source.get('regulation') or source.get('document_name', 'Document')
                            clause = source.get('clause', '')
                            if clause:
                                source_links.append(f'<a href="{full_url}" target="_blank" class="source-link">📄 {regulation} - Clause {clause} (p.{page})</a>')
                            else:
                                source_links.append(f'<a href="{full_url}" target="_blank" class="source-link">📄 {regulation} (p.{page})</a>')
                    
                    if source_links:
                        st.markdown(f'<div class="source-links"><strong>📚 Sources:</strong><br>{" ".join(source_links)}</div>', unsafe_allow_html=True)
                
                # Disclaimer
                if "Disclaimer" in answer or "⚠️" in answer:
                    st.markdown('<div class="disclaimer"><strong>⚠️ Disclaimer:</strong> This information is for decision support only. Always consult qualified safety engineers and follow your organization\'s safety processes.</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("💬 Ask a Safety Question"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            if st.session_state.core:
                # Show spinner while processing
                with st.spinner("Thinking..."):
                    # Get conversation history
                    conv_history = []
                    for msg in st.session_state.messages[:-1]:
                        conv_history.append({"role": msg["role"], "content": msg["content"]})
                    
                    # Process query
                    response = st.session_state.core.process_query(prompt, conversation_history=conv_history)
                
                # Get answer from response
                answer = response.get("answer", "") or ""
                sources = response.get("sources", [])
                
                # Debug output
                print(f"🔍 DEBUG: Answer length: {len(answer)}, Preview: {answer[:200] if answer else 'EMPTY'}")
                
                # If answer is empty, show error
                if not answer or answer.strip() == "":
                    answer = "### ✅ Simple Answer\n\n- I couldn't generate an answer. Please check if the LLM service is available and try again."
                    st.warning("⚠️ No answer generated. Check LLM service availability.")
                else:
                    # Minimal cleaning - only for non-structured answers
                    if "###" not in answer and "**" not in answer:
                        import re
                        # Remove garbled patterns like "F z F z"
                        answer = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\s+\1\s+\2\b', '', answer)
                        answer = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\b(?=\s+[A-Za-z])', r'\1\2', answer)
                
                # Display answer immediately - SIMPLIFIED
                if answer and answer.strip():
                    st.markdown(answer)
                else:
                    st.error("Failed to generate answer.")
                
                # Show sources if available
                if sources:
                    from pdf_linker import find_pdf_path
                    source_links = []
                    for source in sources:
                        pdf_path = find_pdf_path(source.get('document_name', ''))
                        if pdf_path:
                            page = source.get('page_number', 1)
                            file_url = pdf_path.as_uri()
                            full_url = f"{file_url}#page={page}"
                            regulation = source.get('regulation') or source.get('document_name', 'Document')
                            clause = source.get('clause', '')
                            if clause:
                                source_links.append(f'<a href="{full_url}" target="_blank" class="source-link">📄 {regulation} - Clause {clause} (p.{page})</a>')
                            else:
                                source_links.append(f'<a href="{full_url}" target="_blank" class="source-link">📄 {regulation} (p.{page})</a>')
                    
                    if source_links:
                        st.markdown(f'<div class="source-links"><strong>📚 Sources:</strong><br>{" ".join(source_links)}</div>', unsafe_allow_html=True)
                
                # Show disclaimer
                if "Disclaimer" in answer or "⚠️" in answer:
                    st.markdown('<div class="disclaimer"><strong>⚠️ Disclaimer:</strong> This information is for decision support only. Always consult qualified safety engineers and follow your organization\'s safety processes.</div>', unsafe_allow_html=True)
                
                # Save to session state AFTER displaying
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
                # Rerun to show in history
                st.rerun()
            else:
                st.error("❌ Safety Copilot not initialized. Please initialize from the sidebar.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)
