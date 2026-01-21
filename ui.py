"""
Streamlit UI for Safety Copilot - ChatGPT Style with Custom Design
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
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Custom CSS with Design System
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Color Palette */
    :root {
        --primary-orange: #EF6A36;
        --dark-sidebar: #1E1E21;
        --light-bg: #F7F7F7;
        --text-dark: #000000;
        --text-light: #FFFFFF;
        --border-gray: #E5E5E5;
        --shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Main App Background */
    .stApp {
        background: var(--light-bg) !important;
        font-family: 'Inter', 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Sidebar Styling - Dark Background */
    [data-testid="stSidebar"] {
        background: var(--dark-sidebar) !important;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background: var(--dark-sidebar) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: var(--text-light) !important;
    }
    
    /* Sidebar Navigation Cards */
    [data-testid="stSidebar"] .css-1lcbmhc {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
        margin: 0.5rem 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* PRO Badge */
    .pro-badge {
        background: var(--primary-orange) !important;
        color: white !important;
        padding: 0.2rem 0.5rem !important;
        border-radius: 4px !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        margin-left: 0.5rem !important;
    }
    
    /* Main Content Area */
    .main .block-container {
        max-width: 1200px;
        padding: 2rem;
        background: var(--light-bg);
    }
    
    /* Chat Container - Large Border Radius */
    .chat-container {
        background: white !important;
        border-radius: 20px !important;
        box-shadow: var(--shadow) !important;
        padding: 2rem !important;
        margin: 1rem 0 !important;
        border: 1px solid var(--border-gray) !important;
    }
    
    /* Chat Messages */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 1rem 0 !important;
        margin: 0.5rem 0 !important;
    }
    
    /* User Message - White Rounded Container */
    [data-testid="stChatMessage"][data-message-author="user"] {
        background: white !important;
        padding: 1rem 1.5rem !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-gray) !important;
        margin: 1rem 0 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    
    /* Assistant Message */
    [data-testid="stChatMessage"][data-message-author="assistant"] {
        background: transparent !important;
        padding: 1rem 0 !important;
    }
    
    /* Markdown Styling - Line by Line */
    .stMarkdown {
        color: var(--text-dark) !important;
        font-family: 'Inter', 'Poppins', sans-serif !important;
        font-size: 1rem !important;
        line-height: 1.8 !important;
    }
    
    .stMarkdown p {
        margin-bottom: 0.8rem !important;
        color: var(--text-dark) !important;
        line-height: 1.8 !important;
    }
    
    /* Headings */
    .stMarkdown h3, .stMarkdown h4 {
        color: var(--primary-orange) !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', 'Poppins', sans-serif !important;
    }
    
    /* Bullet Points - Line by Line */
    .stMarkdown ul, .stMarkdown ol {
        margin: 1rem 0 !important;
        padding-left: 1.5rem !important;
    }
    
    .stMarkdown li {
        margin-bottom: 0.6rem !important;
        color: var(--text-dark) !important;
        line-height: 1.8 !important;
    }
    
    /* Code Blocks - Syntax Highlighting */
    .stMarkdown code {
        background: #f5f5f5 !important;
        color: var(--text-dark) !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    }
    
    .stMarkdown pre {
        background: #f5f5f5 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        border: 1px solid var(--border-gray) !important;
    }
    
    .stMarkdown pre code {
        background: transparent !important;
        color: #d73a49 !important; /* Red for keywords */
    }
    
    /* Section Styling */
    .answer-section {
        margin: 1.5rem 0 !important;
        padding: 1.5rem !important;
        background: white !important;
        border-radius: 12px !important;
        border-left: 4px solid var(--primary-orange) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    
    .regulation-section {
        border-left-color: #4caf50 !important;
        background: #f1f8f4 !important;
    }
    
    .calculation-section {
        border-left-color: #1e88e5 !important;
        background: #e3f2fd !important;
    }
    
    /* Buttons - Orange Rounded */
    .stButton > button {
        background: var(--primary-orange) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', 'Poppins', sans-serif !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        background: #d85a26 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(239, 106, 54, 0.3) !important;
    }
    
    /* Chat Input */
    .stChatInput {
        background: white !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-gray) !important;
        box-shadow: var(--shadow) !important;
    }
    
    /* Source Links */
    .source-links {
        margin-top: 1.5rem !important;
        padding: 1rem !important;
        background: #f9f9f9 !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-gray) !important;
    }
    
    .source-link {
        color: var(--primary-orange) !important;
        text-decoration: none !important;
        margin-right: 1rem !important;
        font-weight: 500 !important;
    }
    
    .source-link:hover {
        text-decoration: underline !important;
    }
    
    /* Disclaimer */
    .disclaimer {
        margin-top: 2rem !important;
        padding: 1rem 1.5rem !important;
        background: #fff9e6 !important;
        border-left: 4px solid #ffc107 !important;
        border-radius: 12px !important;
        font-size: 0.9rem !important;
        color: #856404 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* History Sidebar on Right */
    .history-sidebar {
        position: fixed;
        right: 0;
        top: 0;
        width: 300px;
        height: 100vh;
        background: white;
        border-left: 1px solid var(--border-gray);
        padding: 1rem;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='color: #EF6A36; font-size: 1.8rem; margin: 0;'>🛡️ Safety Copilot</h1>
        <p style='color: #FFFFFF; opacity: 0.8; font-size: 0.9rem; margin-top: 0.5rem;'>AI-Powered Safety Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation Items as Cards
    st.markdown("### Navigation")
    
    nav_items = [
        ("💬 Chat", "chat", True),
        ("📚 Documents", "documents", False),
        ("⚙️ Settings", "settings", False),
        ("📊 Analytics", "analytics", True),  # PRO feature
    ]
    
    for item, key, is_pro in nav_items:
        badge = '<span class="pro-badge">PRO</span>' if is_pro else ''
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.05); border-radius: 12px; padding: 0.75rem; margin: 0.5rem 0; border: 1px solid rgba(255,255,255,0.1);'>
            {item} {badge}
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
    if st.button("🚀 Initialize", use_container_width=True):
        with st.spinner("Initializing..."):
            try:
                vector_store = load_vector_store()
                core = SafetyCopilotCore()
                core.set_vector_store(vector_store)
                st.session_state.core = core
                st.session_state.initialized = True
                st.success("✅ Initialized!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

# Main Content Area
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Title
st.title("💬 Safety Assistant")

# Initialize if not done
if not st.session_state.initialized:
    st.info("👈 Please initialize the Safety Copilot from the sidebar to get started.")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                # Parse and display structured answer - LINE BY LINE
                answer = message.get("content", "") or ""
                
                # If answer is empty, show error
                if not answer or answer.strip() == "":
                    st.warning("⚠️ No answer generated. Please try again or check if the Safety Copilot is properly initialized.")
                    continue
                
                # Check for structured format
                has_sections = "### ✅" in answer or "### 📘" in answer or "### 🧮" in answer or "### Simple Answer" in answer
                
                if has_sections:
                    # Parse sections
                    sections = {}
                    current_section = None
                    current_text = []
                    
                    for line in answer.split('\n'):
                        line = line.strip()
                        if not line:
                            if current_section:
                                current_text.append('')  # Preserve blank lines
                            continue
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
                    
                    # Display sections LINE BY LINE with proper formatting
                    for section_name, section_text in sections.items():
                        if "🔗" in section_name or "Reference" in section_name:
                            continue
                        elif "📘" in section_name or "Regulation" in section_name:
                            st.markdown(f'<div class="answer-section regulation-section">', unsafe_allow_html=True)
                            st.markdown(f"**{section_name}**")
                            # Render line by line
                            for line in section_text.split('\n'):
                                if line.strip():
                                    if line.strip().startswith('-'):
                                        st.markdown(line)
                                    else:
                                        st.markdown(f"- {line}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        elif "🧮" in section_name or "Calculation" in section_name or "Analysis" in section_name:
                            st.markdown(f'<div class="answer-section calculation-section">', unsafe_allow_html=True)
                            st.markdown(f"**{section_name}**")
                            # Render line by line
                            for line in section_text.split('\n'):
                                if line.strip():
                                    if line.strip().startswith('-') or line.strip().startswith('Step'):
                                        st.markdown(line)
                                    else:
                                        st.markdown(f"- {line}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            # Simple Answer or other sections
                            st.markdown(f'<div class="answer-section">', unsafe_allow_html=True)
                            st.markdown(f"**{section_name}**")
                            # Render line by line with bullet points
                            for line in section_text.split('\n'):
                                if line.strip():
                                    if line.strip().startswith('-'):
                                        st.markdown(line)
                                    else:
                                        # Convert to bullet point if not already
                                        st.markdown(f"- {line}")
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    # Display as regular markdown - convert to line-by-line
                    lines = answer.split('\n')
                    for line in lines:
                        if line.strip():
                            if line.strip().startswith('-') or line.strip().startswith('*'):
                                st.markdown(line)
                            elif line.strip().startswith('#'):
                                st.markdown(line)
                            else:
                                # Convert to bullet point for readability
                                st.markdown(f"- {line}")
                
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
                    
                    # Get answer from response
                    answer = response.get("answer", "") or ""
                    
                    # If answer is empty, show error message
                    if not answer or answer.strip() == "":
                        answer = "### ✅ Simple Answer\n\n- I couldn't generate an answer. Please check if the LLM service is available and try again.\n- Make sure the Safety Copilot is properly initialized."
                        st.warning("⚠️ No answer generated. Check LLM service availability.")
                    else:
                        # Clean answer - remove garbled patterns
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
                        
                        # Ensure structured format if not present
                        if "### ✅" not in answer and "### Simple Answer" not in answer and "### 📘" not in answer:
                            # Format as structured markdown with line-by-line bullet points
                            lines = [l.strip() for l in answer.split('\n') if l.strip() and len(l.strip()) > 5]
                            if lines:
                                formatted = "### ✅ Simple Answer\n\n"
                                # First 2-3 meaningful lines as simple answer
                                count = 0
                                for line in lines:
                                    if count >= 3:
                                        break
                                    if len(line) > 10 and not line.startswith('⚠️') and 'Disclaimer' not in line:
                                        formatted += f"- {line}\n"
                                        count += 1
                                formatted += "\n### 📘 Regulation Requirement\n\n"
                                # Next lines as regulation requirement
                                count = 0
                                for line in lines[3:]:
                                    if count >= 3:
                                        break
                                    if len(line) > 10 and not line.startswith('⚠️') and 'Disclaimer' not in line:
                                        formatted += f"- {line}\n"
                                        count += 1
                                formatted += "\n### 🔗 References\n\n"
                                formatted += "- See sources below\n"
                                answer = formatted
                    
                    # Display the answer immediately using the same parsing logic as history
                    # Check for structured format
                    has_sections = "### ✅" in answer or "### 📘" in answer or "### 🧮" in answer or "### Simple Answer" in answer
                    
                    if has_sections:
                        # Parse sections
                        sections = {}
                        current_section = None
                        current_text = []
                        
                        for line in answer.split('\n'):
                            line = line.strip()
                            if not line:
                                if current_section:
                                    current_text.append('')
                                continue
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
                        
                        # Display sections LINE BY LINE with proper formatting
                        for section_name, section_text in sections.items():
                            if "🔗" in section_name or "Reference" in section_name:
                                continue
                            elif "📘" in section_name or "Regulation" in section_name:
                                st.markdown(f'<div class="answer-section regulation-section">', unsafe_allow_html=True)
                                st.markdown(f"**{section_name}**")
                                for line in section_text.split('\n'):
                                    if line.strip():
                                        if line.strip().startswith('-'):
                                            st.markdown(line)
                                        else:
                                            st.markdown(f"- {line}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            elif "🧮" in section_name or "Calculation" in section_name or "Analysis" in section_name:
                                st.markdown(f'<div class="answer-section calculation-section">', unsafe_allow_html=True)
                                st.markdown(f"**{section_name}**")
                                for line in section_text.split('\n'):
                                    if line.strip():
                                        if line.strip().startswith('-') or line.strip().startswith('Step'):
                                            st.markdown(line)
                                        else:
                                            st.markdown(f"- {line}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="answer-section">', unsafe_allow_html=True)
                                st.markdown(f"**{section_name}**")
                                for line in section_text.split('\n'):
                                    if line.strip():
                                        if line.strip().startswith('-'):
                                            st.markdown(line)
                                        else:
                                            st.markdown(f"- {line}")
                                st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        # Display as regular markdown - convert to line-by-line
                        if answer.strip():
                            lines = answer.split('\n')
                            for line in lines:
                                if line.strip():
                                    if line.strip().startswith('-') or line.strip().startswith('*'):
                                        st.markdown(line)
                                    elif line.strip().startswith('#'):
                                        st.markdown(line)
                                    else:
                                        st.markdown(f"- {line}")
                    
                    # Show sources if available
                    sources = response.get("sources", [])
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
                    
                    # Show disclaimer if present
                    if "Disclaimer" in answer or "⚠️" in answer:
                        st.markdown('<div class="disclaimer"><strong>⚠️ Disclaimer:</strong> This information is for decision support only. Always consult qualified safety engineers and follow your organization\'s safety processes.</div>', unsafe_allow_html=True)
                    
                    # Add assistant message to history AFTER displaying
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                    st.rerun()
                else:
                    st.error("Safety Copilot not initialized. Please initialize from the sidebar.")
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

st.markdown('</div>', unsafe_allow_html=True)
