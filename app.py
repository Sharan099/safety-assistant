# app.py — PSA AI
# ChatGPT-Style Passive Safety Assistant

import re
import time
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PSA AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =========================================================
# LOAD PIPELINE
# =========================================================

@st.cache_resource(show_spinner=False)
def load_pipeline():

    from pipeline import query as pipeline_query

    return pipeline_query

# =========================================================
# CSS
# =========================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

/* ===================================================== */
/* GLOBAL */
/* ===================================================== */

.stApp {

    background:
        radial-gradient(circle at top left,
        rgba(239,68,68,0.05),
        transparent 20%),

        radial-gradient(circle at bottom right,
        rgba(239,68,68,0.03),
        transparent 18%),

        #0a0f1f;

    color: white;
}

/* ===================================================== */
/* REMOVE STREAMLIT DEFAULTS */
/* ===================================================== */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* ===================================================== */
/* SIDEBAR */
/* ===================================================== */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #0b1020 0%,
            #111827 100%
        );

    border-right:
        1px solid rgba(255,255,255,0.05);
}

.sidebar-header {

    background:
        linear-gradient(
            135deg,
            #991b1b,
            #ef4444
        );

    border-radius: 22px;

    padding: 22px;

    margin-bottom: 24px;

    box-shadow:
        0 12px 35px rgba(239,68,68,0.2);
}

.sidebar-title {

    font-size: 2rem;

    font-weight: 800;

    color: white;
}

.sidebar-sub {

    color: rgba(255,255,255,0.82);

    margin-top: 6px;

    font-size: 0.92rem;
}

/* ===================================================== */
/* SIDEBAR BUTTONS */
/* ===================================================== */

.stButton > button {

    width: 100%;

    border-radius: 18px;

    background:
        rgba(255,255,255,0.03);

    border:
        1px solid rgba(255,255,255,0.05);

    color: white;

    text-align: left;

    padding: 16px;

    transition: 0.25s ease;
}

.stButton > button:hover {

    border:
        1px solid rgba(239,68,68,0.45);

    background:
        rgba(239,68,68,0.08);

    transform: translateY(-2px);
}

/* ===================================================== */
/* CHATGPT STYLE LAYOUT */
/* ===================================================== */

.chat-wrapper {

    max-width: 920px;

    margin: auto;

    padding-bottom: 140px;
}

/* ===================================================== */
/* TOP BAR */
/* ===================================================== */

.topbar {

    display: flex;

    align-items: center;

    justify-content: space-between;

    padding: 18px 24px;

    margin-bottom: 28px;

    border-radius: 22px;

    background:
        rgba(15,23,42,0.72);

    backdrop-filter: blur(12px);

    border:
        1px solid rgba(255,255,255,0.05);
}

.logo {

    display: flex;

    align-items: center;

    gap: 14px;
}

.logo-icon {

    width: 54px;

    height: 54px;

    border-radius: 16px;

    background:
        linear-gradient(
            135deg,
            #991b1b,
            #ef4444
        );

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 1.7rem;

    box-shadow:
        0 10px 30px rgba(239,68,68,0.2);
}

.logo-title {

    font-size: 2rem;

    font-weight: 800;

    color: white;
}

.logo-red {
    color: #ef4444;
}

/* ===================================================== */
/* STICKY CHAT INPUT */
/* ===================================================== */

.stChatInput {

    position: fixed !important;

    bottom: 18px;

    left: 50%;

    transform: translateX(-50%);

    width: min(920px, 72vw);

    z-index: 999;
}

.stChatInput textarea {

    background:
        rgba(15,23,42,0.96) !important;

    border:
        1px solid rgba(255,255,255,0.08) !important;

    border-radius: 26px !important;

    color: white !important;

    padding: 20px !important;

    font-size: 1rem !important;

    box-shadow:
        0 12px 35px rgba(0,0,0,0.28);

    min-height: 62px !important;
}

/* ===================================================== */
/* USER MESSAGE */
/* ===================================================== */

.user-container {

    display: flex;

    justify-content: flex-end;

    margin-top: 24px;

    animation: fadeIn 0.25s ease;
}

.user-bubble {

    max-width: 72%;

    background:
        linear-gradient(
            135deg,
            #dc2626,
            #ef4444
        );

    color: white;

    padding: 18px 22px;

    border-radius: 24px 24px 10px 24px;

    line-height: 1.7;

    font-size: 1rem;

    box-shadow:
        0 10px 30px rgba(239,68,68,0.18);
}

/* ===================================================== */
/* ASSISTANT MESSAGE */
/* ===================================================== */

.assistant-container {

    display: flex;

    justify-content: flex-start;

    margin-top: 30px;

    animation: fadeIn 0.35s ease;
}

.assistant-bubble {

    width: 100%;

    background: transparent;

    padding: 0;

    color: #f3f4f6;

    line-height: 1.9;

    font-size: 1rem;

    overflow-wrap: break-word;
}

/* ===================================================== */
/* ANSWER TYPOGRAPHY */
/* ===================================================== */

.assistant-bubble h1 {

    font-size: 1.8rem;

    margin-bottom: 22px;

    color: white;

    font-weight: 800;
}

.assistant-bubble h2 {

    margin-top: 34px;

    margin-bottom: 16px;

    font-size: 1.2rem;

    color: #ef4444;

    font-weight: 700;
}

.assistant-bubble h3 {

    margin-top: 24px;

    margin-bottom: 10px;

    color: #fca5a5;
}

.assistant-bubble p {

    margin-bottom: 14px;

    line-height: 1.9;
}

.assistant-bubble ul {

    padding-left: 24px;

    margin-top: 12px;

    margin-bottom: 18px;
}

.assistant-bubble li {

    margin-bottom: 12px;

    line-height: 1.85;
}

.assistant-bubble strong {

    color: white;
}

.assistant-bubble em {

    color: #94a3b8;
}

.assistant-bubble code {

    background:
        rgba(255,255,255,0.06);

    padding: 3px 7px;

    border-radius: 7px;

    font-size: 0.92rem;
}

.assistant-bubble blockquote {

    border-left:
        3px solid #ef4444;

    padding-left: 14px;

    color: #cbd5e1;

    margin-left: 0;
}

/* ===================================================== */
/* ANIMATION */
/* ===================================================== */

@keyframes fadeIn {

    from {

        opacity: 0;

        transform: translateY(8px);
    }

    to {

        opacity: 1;

        transform: translateY(0);
    }
}

</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# QUESTIONS
# =========================================================

QUESTIONS = [

    "What are the UN R14 requirements for seat belt anchorages?",

    "What test loads are required for UN R16 approval?",

    "Explain UN R14 anchorage geometry requirements.",

    "Calculate belt load for 75kg occupant at 20g",

    "Calculate kinetic energy for 1500kg at 56 km/h",

    "Summarize UN R16 restraint system approval procedures.",
]

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
<div class="sidebar-header">

<div class="sidebar-title">
🛡️ PSA AI
</div>

<div class="sidebar-sub">
Passive Safety Assistant
</div>

</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("### Suggested Questions")

    for i, q in enumerate(QUESTIONS):

        if st.button(q, key=f"q_{i}"):

            st.session_state.pending_query = q

            st.rerun()

    st.markdown("---")

    st.markdown("### Models & Methods")

    st.markdown(
        """
- Hybrid RAG  
- BM25 Retrieval  
- Semantic Search  
- Llama 3.1-8B  
- MiniLM Embeddings  
- Engineering Reasoning  
"""
    )

# =========================================================
# MAIN CHAT AREA
# =========================================================

st.markdown(
    '<div class="chat-wrapper">',
    unsafe_allow_html=True
)

# =========================================================
# TOP BAR
# =========================================================

st.markdown(
    """
<div class="topbar">

<div class="logo">

<div class="logo-icon">
🛡️
</div>

<div class="logo-title">
PSA <span class="logo-red">AI</span>
</div>

</div>

</div>
""",
    unsafe_allow_html=True
)

# =========================================================
# CHAT HISTORY
# =========================================================

for msg in st.session_state.messages:

    if msg["role"] == "user":

        st.markdown(
            f"""
<div class="user-container">

<div class="user-bubble">

{msg["content"]}

</div>

</div>
""",
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
<div class="assistant-container">

<div class="assistant-bubble">

{msg["content"]}

</div>

</div>
""",
            unsafe_allow_html=True
        )

# =========================================================
# CHAT INPUT
# =========================================================

user_input = st.chat_input(
    "Message PSA AI..."
)

active_query = user_input

if (
    not active_query
    and
    st.session_state.pending_query
):

    active_query = (
        st.session_state.pending_query
    )

    st.session_state.pending_query = None

# =========================================================
# USER MESSAGE
# =========================================================

if active_query:

    st.session_state.messages.append({

        "role": "user",

        "content": active_query
    })

    st.rerun()

# =========================================================
# ASSISTANT RESPONSE
# =========================================================

if (
    st.session_state.messages
    and
    st.session_state.messages[-1]["role"] == "user"
):

    query = st.session_state.messages[-1]["content"]

    try:

        pipeline_fn = load_pipeline()

        result = pipeline_fn(query)

        answer = result.get(
            "answer",
            "No answer returned."
        )

        # =================================================
        # CLEANUP
        # =================================================

        formatted_answer = answer

        formatted_answer = re.sub(
            r"#+",
            "",
            formatted_answer
        )

        formatted_answer = re.sub(
            r"(\d)\s*\.\s*#+\s*(\d)",
            r"\1.\2",
            formatted_answer
        )

        formatted_answer = re.sub(
            r"(\d)\s*#+\s*(\d)",
            r"\1.\2",
            formatted_answer
        )

        formatted_answer = re.sub(
            r"\n{3,}",
            "\n\n",
            formatted_answer
        )

        formatted_answer = formatted_answer.replace(
            "Reference:",
            "\n\n*Reference:*"
        )

        # =================================================
        # STREAM RESPONSE
        # =================================================

        container = st.empty()

        streamed = ""

        lines = formatted_answer.split("\n")

        for line in lines:

            streamed += line + "\n"

            container.markdown(

                f"""
<div class="assistant-container">

<div class="assistant-bubble">

{streamed}

</div>

</div>
""",

                unsafe_allow_html=True
            )

            time.sleep(0.02)

        st.session_state.messages.append({

            "role": "assistant",

            "content": formatted_answer
        })

        st.session_state.last_result = result

        st.rerun()

    except Exception as e:

        st.error(str(e))

st.markdown(
    "</div>",
    unsafe_allow_html=True
)