"""
Streamlit UI for Safety Copilot - ChatGPT Style with Dark/Light Mode
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
    layout="centered",
    initial_sidebar_state="collapsed"
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
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ChatGPT-style CSS with Dark/Light Mode
dark_mode = st.session_state.dark_mode
st.markdown(f"""
<style>
    /* ChatGPT-style Interface */
    .stApp {{
        background: {'#343541' if dark_mode else '#ffffff'} !important;
    }}
    
    .main .block-container {{
        max-width: 800px;
        padding: 0;
        margin: 0 auto;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Chat messages */
    [data-testid="stChatMessage"] {{
        background: transparent !important;
        padding: 1.5rem 0 !important;
    }}
    
    /* User message */
    [data-testid="stChatMessage"][data-message-author="user"] {{
        background: {'#444654' if dark_mode else '#f7f7f8'} !important;
        padding: 1rem 1.5rem !important;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }}
    
    /* Assistant message */
    [data-testid="stChatMessage"][data-message-author="assistant"] {{
        background: transparent !important;
        padding: 1rem 0 !important;
    }}
    
    /* Markdown styling */
    .stMarkdown {{
        color: {'#ececf1' if dark_mode else '#353740'} !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        font-size: 1rem;
        line-height: 1.75;
    }}
    
    .stMarkdown p {{
        margin-bottom: 0.8rem;
        color: {'#ececf1' if dark_mode else '#353740'} !important;
    }}
    
    .stMarkdown h3, .stMarkdown h4 {{
        color: {'#ececf1' if dark_mode else '#1e88e5'} !important;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }}
    
    .stMarkdown ul, .stMarkdown ol {{
        margin: 0.8rem 0;
        padding-left: 1.5rem;
    }}
    
    .stMarkdown li {{
        margin-bottom: 0.5rem;
        color: {'#ececf1' if dark_mode else '#353740'} !important;
    }}
    
    .stMarkdown code {{
        background: {'#40414f' if dark_mode else '#f0f0f0'} !important;
        color: {'#ececf1' if dark_mode else '#353740'} !important;
        padding: 0.2rem 0.4rem;
        border-radius: 0.25rem;
    }}
    
    /* Chat input */
    .stChatInput {{
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 800px;
        background: {'#40414f' if dark_mode else '#ffffff'} !important;
        border-top: 1px solid {'#565869' if dark_mode else '#e5e5e5'} !important;
        padding: 1rem;
        z-index: 100;
    }}
    
    /* Section styling */
    .answer-section {{
        margin: 1.5rem 0;
        padding: 1rem;
        background: {'#40414f' if dark_mode else '#f7f7f8'} !important;
        border-radius: 0.5rem;
        border-left: 4px solid {'#10a37f' if dark_mode else '#1e88e5'};
    }}
    
    .regulation-section {{
        background: {'#2d4a2d' if dark_mode else '#e8f5e9'} !important;
        border-left-color: {'#4caf50' if dark_mode else '#2e7d32'};
    }}
    
    .calculation-section {{
        background: {'#2d3a4a' if dark_mode else '#e3f2fd'} !important;
        border-left-color: {'#1e88e5' if dark_mode else '#1976d2'};
    }}
    
    /* Source links */
    .source-links {{
        margin-top: 1rem;
        padding: 0.75rem;
        background: {'#40414f' if dark_mode else '#f5f5f5'} !important;
        border-radius: 0.5rem;
        font-size: 0.9rem;
    }}
    
    .source-link {{
        color: {'#10a37f' if dark_mode else '#1e88e5'} !important;
        text-decoration: none;
        margin-right: 1rem;
    }}
    
    .source-link:hover {{
        text-decoration: underline;
    }}
    
    /* Disclaimer */
    .disclaimer {{
        margin-top: 2rem;
        padding: 1rem;
        background: {'#3d3d3d' if dark_mode else '#fff9e6'} !important;
        border-left: 4px solid {'#ffc107' if dark_mode else '#ff9800'};
        border-radius: 0.5rem;
        font-size: 0.9rem;
        color: {'#ececf1' if dark_mode else '#856404'} !important;
    }}
    
    /* Dark mode toggle */
    .dark-mode-toggle {{
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 1000;
    }}
</style>
""", unsafe_allow_html=True)

# Dark mode toggle
col1, col2, col3 = st.columns([1, 1, 1])
with col3:
    if st.button("🌙" if dark_mode else "☀️", key="dark_mode_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Title
st.title("🛡️ Safety Copilot")

# Initialize vector store
@st.cache_resource
def load_vector_store():
    return SafetyVectorStore.load_or_build_store(
        force_rebuild=False,
        regulations_dir=REGULATIONS_DIR,
        user_documents=None
    )

# Initialize
if not st.session_state.initialized:
    with st.spinner("Initializing Safety Copilot..."):
        try:
            vector_store = load_vector_store()
            core = SafetyCopilotCore()
            core.set_vector_store(vector_store)
            st.session_state.core = core
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Initialization failed: {e}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            # Parse and display structured answer
            answer = message.get("content", "")
            
            # Check for structured format
            has_sections = "### ✅" in answer or "### 📘" in answer or "### 🧮" in answer
            
            if has_sections:
                # Parse sections
                sections = {}
                current_section = None
                current_text = []
                
                for line in answer.split('\n'):
                    if line.startswith('###'):
                        if current_section:
                            sections[current_section] = '\n'.join(current_text).strip()
                        current_section = line.replace('#', '').strip()
                        current_text = []
                    else:
                        if current_section:
                            current_text.append(line)
                if current_section:
                    sections[current_section] = '\n'.join(current_text).strip()
                
                # Display sections
                for section_name, section_text in sections.items():
                    if "🔗" in section_name or "Reference" in section_name:
                        continue
                    elif "📘" in section_name or "Regulation" in section_name:
                        st.markdown(f'<div class="answer-section regulation-section">', unsafe_allow_html=True)
                        st.markdown(f"**{section_name}**")
                        st.markdown(section_text)
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif "🧮" in section_name or "Calculation" in section_name or "Analysis" in section_name:
                        st.markdown(f'<div class="answer-section calculation-section">', unsafe_allow_html=True)
                        st.markdown(f"**{section_name}**")
                        st.markdown(section_text)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{section_name}**")
                        st.markdown(section_text)
            else:
                # Display as regular markdown
                st.markdown(answer)
            
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
                        source_links.append(f'<a href="{full_url}" target="_blank" class="source-link">📄 {regulation} (p.{page})</a>')
                
                if source_links:
                    st.markdown(f'<div class="source-links"><strong>Sources:</strong> {" ".join(source_links)}</div>', unsafe_allow_html=True)
            
            # Disclaimer
            if "Disclaimer" in answer or "⚠️" in answer:
                st.markdown('<div class="disclaimer"><strong>⚠️ Disclaimer:</strong> This information is for decision support only. Always consult qualified safety engineers.</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("💬 Ask a Safety Question"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if st.session_state.core:
                    # Get conversation history
                    conv_history = []
                    for msg in st.session_state.messages[:-1]:
                        conv_history.append({"role": msg["role"], "content": msg["content"]})
                    
                    # Process query
                    response = st.session_state.core.process_query(prompt, conversation_history=conv_history)
                    
                    # Clean answer - remove garbled patterns
                    answer = response.get("answer", "")
                    
                    # Remove patterns like "F z F z"
                    import re
                    answer = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\s+\1\s+\2\b', '', answer)
                    answer = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\b(?=\s+[A-Za-z])', r'\1\2', answer)
                    
                    # Remove isolated single letters (except a, I)
                    words = answer.split()
                    clean_words = []
                    for i, word in enumerate(words):
                        if len(word) == 1 and word.isalpha() and word.lower() not in ['a', 'i']:
                            if i > 0 and i < len(words) - 1:
                                continue
                        clean_words.append(word)
                    answer = ' '.join(clean_words)
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": response.get("sources", [])
                    })
                    
                    st.rerun()
                else:
                    st.error("Safety Copilot not initialized. Please refresh the page.")
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)


