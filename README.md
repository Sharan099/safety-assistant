# 🛡️ PSA AI — Passive Safety Assistant

PSA AI is an AI-powered engineering assistant designed for passive safety regulations, homologation engineering, and crash safety analysis.

The system combines:
- Hybrid Retrieval
- Semantic Search
- BM25 Ranking
- Engineering Reasoning
- LLM-based Response Generation

to provide grounded answers for:
- UN R14
- UN R16
- Seat Belt Anchorage Requirements
- Occupant Restraint Systems
- Crashworthiness
- Homologation Engineering

---

# 🚀 Features

✅ ChatGPT-style interface  
✅ Regulation-aware retrieval  
✅ Engineering calculations  
✅ Hybrid semantic retrieval  
✅ BM25 + embedding search  
✅ Streamlit web application  
✅ Groq-powered LLM inference  
✅ Engineering-focused response formatting  
✅ Fast inference and retrieval pipeline  

---

# 🧠 Models & Methods

| Component | Technology |
|---|---|
| LLM | Groq API |
| Model | Llama 3.3 70B |
| Embeddings | MiniLM-L6-v2 |
| Retrieval | Hybrid RAG |
| Search | BM25 + Semantic Search |
| UI | Streamlit |
| Vector Similarity | Sentence Transformers |

---

# 📂 Project Structure

```text
PSA-AI/
│
├── app.py
├── pipeline.py
├── config.py
├── requirements.txt
├── README.md
│
├── llm/
├── retrieval/
├── graph/
├── data/
└── .streamlit/
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/PSA-AI.git

cd PSA-AI
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv rag_env

rag_env\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv rag_env

source rag_env/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

---

# ▶️ Run Application

```bash
python -m streamlit run app.py
```

---

# 💬 Example Queries

- What are the UN R14 requirements for seat belt anchorages?
- Explain UN R16 dynamic testing requirements.
- Calculate belt load for 75kg occupant at 20g.
- Calculate crash energy for 1500kg at 56 km/h.
- Explain UN R14 anchorage geometry requirements.

---

# 🧮 Engineering Capabilities

PSA AI supports:
- Crash energy calculations
- Belt load calculations
- Unit conversions
- Engineering formula reasoning
- Regulation-grounded responses

---

# 🌐 Deployment

The application can be deployed using:
- Hugging Face Spaces
- Streamlit Community Cloud
- Docker
- Azure App Services

---

# 📸 UI

The interface is inspired by ChatGPT and optimized for:
- engineering readability,
- conversational interactions,
- structured regulation responses.

---

# ⚠️ Disclaimer

This project is intended for educational and engineering-assistance purposes only.

Always validate regulatory requirements using official homologation documents and engineering standards.

---

# 👨‍💻 Author

Sharan  
Aspiring AI Engineer | Passive Safety AI Systems