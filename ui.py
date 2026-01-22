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
    
    /* Markdown Styling - Clean & Readable with Black Text */
    .stMarkdown {
        color: #000000 !important;
        font-family: 'Inter', 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        font-size: 1rem !important;
        line-height: 1.8 !important;
    }
    
    /* Assistant message markdown - Black text, better font */
    [data-testid="stChatMessage"][data-message-author="assistant"] .stMarkdown {
        color: #000000 !important;
        font-family: 'Inter', 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        font-weight: 400 !important;
        font-size: 1rem !important;
        line-height: 1.8 !important;
    }
    
    .stMarkdown p {
        margin-bottom: 0.8rem !important;
        color: #000000 !important;
        line-height: 1.8 !important;
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
        margin-bottom: 0.6rem !important;
        color: #000000 !important;
        line-height: 1.8 !important;
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
    
    /* Sidebar spinner and info boxes - Make visible */
    [data-testid="stSidebar"] .stSpinner,
    [data-testid="stSidebar"] .stSpinner > div {
        color: #FFFFFF !important;
        background: rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="stSidebar"] .stInfo,
    [data-testid="stSidebar"] .stSuccess,
    [data-testid="stSidebar"] .stWarning,
    [data-testid="stSidebar"] .stError {
        background: rgba(255, 255, 255, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
    }
    
    [data-testid="stSidebar"] .stInfo > div,
    [data-testid="stSidebar"] .stSuccess > div,
    [data-testid="stSidebar"] .stWarning > div,
    [data-testid="stSidebar"] .stError > div {
        color: #FFFFFF !important;
    }
    
    /* Structured sections styling */
    .answer-section {
        margin: 1.5rem 0 !important;
        padding: 1.25rem !important;
        background: white !important;
        border-radius: 10px !important;
        border-left: 4px solid var(--primary-teal) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        color: #1F2937 !important;
    }
    
    .answer-section * {
        color: #1F2937 !important;
    }
    
    .simple-answer-section {
        border-left-color: var(--primary-teal) !important;
        background: white !important;
    }
    
    .regulation-section {
        border-left-color: #10B981 !important;
        background: white !important;
        border-left-width: 4px !important;
    }
    
    .calculation-section {
        border-left-color: var(--primary-sky) !important;
        background: white !important;
        border-left-width: 4px !important;
    }
    
    .references-section {
        border-left-color: #8B5CF6 !important;
        background: white !important;
        border-left-width: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

def parse_and_display_sections(answer: str):
    """
    Parse structured answer into sections and display with proper formatting.
    Handles cases where sections might be in a continuous paragraph.
    """
    import re
    
    if not answer or not answer.strip():
        return False
    
    # Check for structured format
    has_sections = (
        "### ✅" in answer or "### 📘" in answer or "### 🧮" in answer or "### 🔗" in answer or 
        "### Simple Answer" in answer or "✅ Simple Answer" in answer or 
        "📘 Regulation Requirement" in answer or "🧮 Analysis" in answer or "🔗 References" in answer
    )
    
    if not has_sections:
        st.markdown(answer)
        return False
    
    # Parse sections - handle both line-by-line and paragraph formats
    sections = {}
    
    # Pattern to match section headers (with or without ###, with or without emoji)
    section_pattern = r'(###?\s*[✅📘🧮🔗]?\s*(?:Simple Answer|Regulation Requirement|Analysis(?:\s*/\s*Calculation)?|Calculation|References))'
    
    # Also match emoji directly followed by section name (for paragraph format)
    emoji_section_pattern = r'([✅📘🧮🔗]\s*(?:Simple Answer|Regulation Requirement|Analysis(?:\s*/\s*Calculation)?|Calculation|References))'
    
    # Try splitting by both patterns
    parts = re.split(f'{section_pattern}|{emoji_section_pattern}', answer, flags=re.IGNORECASE)
    
    current_section = None
    current_text = []
    
    i = 0
    while i < len(parts):
        part = parts[i]
        if part is None:
            i += 1
            continue
        part = part.strip()
        if not part:
            i += 1
            continue
        
        # Check if this part is a section header
        is_header = False
        normalized_header = None
        
        if re.match(section_pattern, part, re.IGNORECASE) or re.match(emoji_section_pattern, part):
            is_header = True
            normalized_header = part.replace('#', '').strip()
            # Normalize section names
            if 'Simple Answer' in normalized_header:
                normalized_header = '✅ Simple Answer'
            elif 'Regulation Requirement' in normalized_header:
                normalized_header = '📘 Regulation Requirement'
            elif 'Analysis' in normalized_header or 'Calculation' in normalized_header:
                normalized_header = '🧮 Analysis / Calculation'
            elif 'References' in normalized_header:
                normalized_header = '🔗 References'
        
        if is_header:
            # Save previous section
            if current_section and current_text:
                sections[current_section] = '\n'.join(current_text).strip()
            # Start new section
            current_section = normalized_header
            current_text = []
        else:
            # This is content - add it to current section
            if current_section:
                current_text.append(part)
        i += 1
    
    # Save last section
    if current_section and current_text:
        sections[current_section] = '\n'.join(current_text).strip()
    
    # Also parse line by line for proper formatting (fallback)
    if not sections:
        current_section_line = None
        current_text_line = []
        
        for line in answer.split('\n'):
            if line is None:
                continue
            line_stripped = line.strip()
            if not line_stripped:
                if current_section_line:
                    current_text_line.append('')
                continue
            
            if line_stripped.startswith('###') or re.match(r'[✅📘🧮🔗]\s*(?:Simple Answer|Regulation Requirement|Analysis|Calculation|References)', line_stripped):
                # Save previous section
                if current_section_line:
                    sections[current_section_line] = '\n'.join(current_text_line).strip()
                # Start new section
                current_section_line = line_stripped.replace('#', '').strip()
                # Normalize
                if 'Simple Answer' in current_section_line:
                    current_section_line = '✅ Simple Answer'
                elif 'Regulation Requirement' in current_section_line:
                    current_section_line = '📘 Regulation Requirement'
                elif 'Analysis' in current_section_line or 'Calculation' in current_section_line:
                    current_section_line = '🧮 Analysis / Calculation'
                elif 'References' in current_section_line:
                    current_section_line = '🔗 References'
                current_text_line = []
            else:
                if current_section_line:
                    current_text_line.append(line)
        
        # Save last section
        if current_section_line:
            sections[current_section_line] = '\n'.join(current_text_line).strip()
    
    if not sections:
        st.markdown(answer)
        return False
    
    # Display sections in proper order
    section_order = [
        ("✅ Simple Answer", "simple-answer-section"),
        ("📘 Regulation Requirement", "regulation-section"),
        ("🧮 Analysis / Calculation", "calculation-section"),
        ("🧮 Analysis", "calculation-section"),
        ("🧮 Calculation", "calculation-section"),
        ("🔗 References", "references-section")
    ]
    
    displayed_sections = set()
    for section_key, section_class in section_order:
        for section_name in sections.keys():
            if section_key in section_name and section_name not in displayed_sections:
                section_text = sections[section_name]
                if section_text and section_text.strip():
                    st.markdown(f'<div class="answer-section {section_class}">', unsafe_allow_html=True)
                    st.markdown(f"**{section_name}**")
                    
                    # Break down section text into individual points
                    # Handle both line-by-line format and continuous paragraph format
                    text_parts = []
                    
                    # First, check if this is a continuous paragraph (no newlines or few newlines)
                    if '\n' not in section_text or section_text.count('\n') < 2:
                        # This is likely a continuous paragraph - split by " - " pattern
                        # But be careful - " - " might appear in the middle of sentences
                        # Split by " - " but preserve context
                        parts = re.split(r'\s+-\s+', section_text)
                        for i, part in enumerate(parts):
                            if part is None:
                                continue
                            part = part.strip()
                            if not part:
                                continue
                            
                            # Remove leading/trailing punctuation that might be from splitting
                            part = re.sub(r'^[.,;:\s]+|[.,;:\s]+$', '', part)
                            
                            if part:
                                # Check if this part looks like a complete sentence or point
                                # If it starts with a capital or is a keyword pattern, it's likely a new point
                                if i == 0 or part[0].isupper() or re.match(r'^(Keyword|Limit|Condition|Regulation|Clause|Page):', part, re.IGNORECASE):
                                    text_parts.append(f"- {part}")
                                else:
                                    # This might be a continuation - append to previous or make new
                                    if text_parts and not text_parts[-1].endswith('.'):
                                        text_parts[-1] += f" {part}"
                                    else:
                                        text_parts.append(f"- {part}")
                    else:
                        # Line-by-line format
                        lines = section_text.split('\n')
                        for line in lines:
                            if line is None:
                                continue
                            line = line.strip()
                            if not line:
                                text_parts.append('')
                                continue
                            
                            # If line already starts with bullet, keep it
                            if line.startswith('-') or line.startswith('*'):
                                text_parts.append(line)
                            # If line contains " - " pattern, split it intelligently
                            elif ' - ' in line:
                                # Check if this is a keyword pattern (Keyword:, Limit:, etc.)
                                if re.search(r'(Keyword|Limit|Condition|Regulation|Clause|Page):', line, re.IGNORECASE):
                                    # Split by keyword patterns first
                                    sub_parts = re.split(r'(Keyword:|Limit:|Condition:|Regulation:|Clause:|Page:)', line)
                                    current_item = ""
                                    for j, sub_part in enumerate(sub_parts):
                                        if sub_part is None:
                                            continue
                                        sub_part = sub_part.strip()
                                        if not sub_part:
                                            continue
                                        if sub_part in ['Keyword:', 'Limit:', 'Condition:', 'Regulation:', 'Clause:', 'Page:']:
                                            if current_item:
                                                text_parts.append(f"- {current_item}")
                                            current_item = f"**{sub_part}** "
                                        else:
                                            current_item += sub_part
                                    if current_item:
                                        text_parts.append(f"- {current_item}")
                                else:
                                    # Regular split by " - "
                                    sub_parts = line.split(' - ')
                                    for j, sub_part in enumerate(sub_parts):
                                        sub_part = sub_part.strip()
                                        if sub_part:
                                            text_parts.append(f"- {sub_part}")
                            # If line is long and contains periods, try to split by sentences
                            elif len(line) > 80 and '. ' in line:
                                sentences = re.split(r'(?<=\.)\s+', line)
                                for sentence in sentences:
                                    if sentence is None:
                                        continue
                                    sentence = sentence.strip()
                                    if sentence:
                                        text_parts.append(f"- {sentence}")
                            else:
                                text_parts.append(f"- {line}")
                    
                    # Render each part
                    for text_part in text_parts:
                        if text_part is None or not text_part:
                            st.markdown("")  # Empty line
                            continue
                        
                        text_part = text_part.strip()
                        if not text_part:
                            continue
                        
                        # If already a bullet point, render as is
                        if text_part.startswith('-') or text_part.startswith('*'):
                            # Check for special formatting keywords
                            if any(keyword in text_part for keyword in ['Keyword:', 'Limit:', 'Condition:', 'Regulation:', 'Clause:', 'Page:']):
                                # Format keywords in bold
                                formatted = re.sub(r'(Keyword:|Limit:|Condition:|Regulation:|Clause:|Page:)(\s*)', r'**\1**\2', text_part)
                                st.markdown(formatted)
                            elif text_part.startswith('Step') or re.match(r'Step\s+\d+:', text_part, re.IGNORECASE):
                                st.markdown(f"**{text_part}**")
                            else:
                                st.markdown(text_part)
                        else:
                            st.markdown(f"- {text_part}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    displayed_sections.add(section_name)
    
    # Display any remaining sections
    for section_name, section_text in sections.items():
        if section_name not in displayed_sections and section_text and section_text.strip():
            st.markdown(f'<div class="answer-section">', unsafe_allow_html=True)
            st.markdown(f"**{section_name}**")
            for text_line in section_text.split('\n'):
                if text_line is None:
                    continue
                text_line = text_line.strip()
                if text_line:
                    if text_line.startswith('-') or text_line.startswith('*'):
                        st.markdown(text_line)
                    else:
                        st.markdown(f"- {text_line}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    return True

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
    
    # About RAG Section
    st.markdown("---")
    st.markdown("""
    <div style='padding: 1rem 0;'>
        <h3 style='color: #FFFFFF; font-size: 1.1rem; margin-bottom: 0.8rem; font-weight: 600;'>📚 About This RAG</h3>
        <p style='color: #FFFFFF; opacity: 0.9; font-size: 0.85rem; line-height: 1.5; margin-bottom: 1rem;'>
            This AI-powered Safety Assistant uses Retrieval-Augmented Generation (RAG) to answer questions based on verified automotive safety regulations and standards. It searches through a comprehensive database of safety documents to provide accurate, source-backed answers.
        </p>
        <h4 style='color: #FFFFFF; font-size: 0.95rem; margin-bottom: 0.6rem; font-weight: 600;'>📋 Regulations Included:</h4>
        <div style='color: #FFFFFF; opacity: 0.9; font-size: 0.8rem; line-height: 1.6;'>
            <p style='margin: 0.3rem 0;'><strong>UNECE Regulations:</strong></p>
            <p style='margin: 0.2rem 0; padding-left: 0.5rem;'>• R16 (Seat Belts)</p>
            <p style='margin: 0.2rem 0; padding-left: 0.5rem;'>• R17 (Seat Strength)</p>
            <p style='margin: 0.2rem 0; padding-left: 0.5rem;'>• R29 (Head Restraints)</p>
            <p style='margin: 0.2rem 0; padding-left: 0.5rem;'>• R94 (Frontal Impact)</p>
            <p style='margin: 0.2rem 0; padding-left: 0.5rem;'>• R127 (Pedestrian Safety)</p>
            <p style='margin: 0.2rem 0; padding-left: 0.5rem;'>• R137 (Side Impact)</p>
            <p style='margin: 0.2rem 0; padding-left: 0.5rem;'>• R152 (Advanced Emergency Braking)</p>
            <p style='margin: 0.2rem 0; padding-left: 0.5rem;'>• R155 (Cybersecurity)</p>
            <p style='margin: 0.2rem 0; padding-left: 0.5rem;'>• R156 (Software Updates)</p>
            <p style='margin: 0.3rem 0; margin-top: 0.8rem;'><strong>EU Regulations:</strong></p>
            <p style='margin: 0.2rem 0; padding-left: 0.5rem;'>• General Safety Regulation (GSR)</p>
            <p style='margin: 0.2rem 0; padding-left: 0.5rem;'>• Type Approval Directives</p>
            <p style='margin: 0.3rem 0; margin-top: 0.8rem;'><strong>Standards & Protocols:</strong></p>
            <p style='margin: 0.2rem 0; padding-left: 0.5rem;'>• Euro NCAP Protocols</p>
            <p style='margin: 0.2rem 0; padding-left: 0.5rem;'>• Functional Safety (ISO 26262)</p>
            <p style='margin: 0.2rem 0; padding-left: 0.5rem;'>• ADAS Guidelines</p>
            <p style='margin: 0.2rem 0; padding-left: 0.5rem;'>• Passive Safety Standards</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main Content Area
st.title("💬 Safety Assistant")

# Initialize if not done
if not st.session_state.initialized:
    st.info("👈 Please initialize the Safety Copilot from the sidebar to get started.")
    
    # Instructions section
    st.markdown("---")
    st.markdown("""
    <div style='padding: 1.5rem; background: #F8F9FA; border-radius: 10px; border-left: 4px solid #14B8A6; margin: 1.5rem 0;'>
        <h2 style='color: #14B8A6; font-size: 1.5rem; margin-bottom: 1rem;'>🚀 Getting Started</h2>
        
        <h3 style='color: #1F2937; font-size: 1.2rem; margin-top: 1.5rem; margin-bottom: 0.8rem;'>Step 1: Initialize</h3>
        <p style='color: #4B5563; font-size: 1rem; line-height: 1.6; margin-bottom: 1rem;'>
            Click the <strong>"🚀 Initialize Safety Copilot"</strong> button in the left sidebar to load the safety regulations database. 
            This may take a few moments on first use.
        </p>
        
        <h3 style='color: #1F2937; font-size: 1.2rem; margin-top: 1.5rem; margin-bottom: 0.8rem;'>Step 2: Ask Your Question</h3>
        <p style='color: #4B5563; font-size: 1rem; line-height: 1.6; margin-bottom: 1rem;'>
            Type your safety-related question in the chat input at the bottom of the page. The AI will search through 
            verified safety documents and provide you with accurate, source-backed answers.
        </p>
        
        <h3 style='color: #1F2937; font-size: 1.2rem; margin-top: 1.5rem; margin-bottom: 0.8rem;'>💡 What You Can Ask</h3>
        <div style='color: #4B5563; font-size: 1rem; line-height: 1.8;'>
            <p style='margin: 0.5rem 0;'><strong>📋 Definition Questions:</strong></p>
            <ul style='margin: 0.5rem 0; padding-left: 1.5rem;'>
                <li>"What is the wrap-around distance (WAD) in UN R127?"</li>
                <li>"Define HIC (Head Injury Criterion)"</li>
                <li>"What does ASIL mean in functional safety?"</li>
            </ul>
            
            <p style='margin: 0.5rem 0; margin-top: 1rem;'><strong>📊 Requirement Questions:</strong></p>
            <ul style='margin: 0.5rem 0; padding-left: 1.5rem;'>
                <li>"What is the maximum allowable HIC value for a 50th percentile male dummy in UN R94?"</li>
                <li>"What are the injury criteria limits for frontal impact tests?"</li>
                <li>"What are the requirements for cybersecurity in UN R155?"</li>
            </ul>
            
            <p style='margin: 0.5rem 0; margin-top: 1rem;'><strong>🔍 Comparison Questions:</strong></p>
            <ul style='margin: 0.5rem 0; padding-left: 1.5rem;'>
                <li>"Compare UN R94 and Euro NCAP frontal impact requirements"</li>
                <li>"What's the difference between R155 and R156?"</li>
            </ul>
            
            <p style='margin: 0.5rem 0; margin-top: 1rem;'><strong>🧮 Calculation/Scenario Questions:</strong></p>
            <ul style='margin: 0.5rem 0; padding-left: 1.5rem;'>
                <li>"If a vehicle has a HIC value of 850, does it meet UN R94 requirements?"</li>
                <li>"Calculate the wrap-around distance for a vehicle with bonnet length of 1200mm"</li>
            </ul>
            
            <p style='margin: 0.5rem 0; margin-top: 1rem;'><strong>✅ Compliance Questions:</strong></p>
            <ul style='margin: 0.5rem 0; padding-left: 1.5rem;'>
                <li>"Does this test result meet UN R94 requirements?"</li>
                <li>"Is this vehicle compliant with pedestrian safety regulations?"</li>
            </ul>
        </div>
        
        <div style='background: #FEF3C7; border-left: 4px solid #F59E0B; padding: 1rem; border-radius: 5px; margin-top: 1.5rem;'>
            <p style='color: #92400E; font-size: 0.9rem; margin: 0;'>
                <strong>⚠️ Important:</strong> All answers are based on the documents in the database. The AI will clearly indicate 
                when information is not available in the source documents. Always verify critical safety decisions with qualified engineers.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            answer = message.get("content", "") or ""
            
            if message["role"] == "user":
                st.write(message["content"])
            else:
                # Display answer with structured format parsing
                if answer and answer.strip():
                    parse_and_display_sections(answer)
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
                
                # Display answer with structured format parsing
                if answer and answer.strip():
                    parse_and_display_sections(answer)
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
