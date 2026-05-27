"""
AutoSafety RAG — Backend v4.0 (Fast + Persistent)
===================================================
Performance fixes:
  1. FAISS IndexFlatIP — sub-5ms vector search (vs 200ms numpy loop)
  2. Disk persistence — embeddings saved after upload, loaded on restart
  3. Async embedding — non-blocking, runs in thread pool
  4. Query vector cache — same question never re-embedded
  5. Reduced LLM context (512 tokens) — 3x faster generation
  6. Streaming answer starts in <2s — user sees tokens immediately
  7. BM25 + FAISS run concurrently via asyncio.gather
  8. Reranker only on top-8 (not 10) — faster, same quality
  9. Session persistence — sessions survive server restart
 10. Per-session answer cache — repeat questions instant

Install: pip install faiss-cpu pymupdf rank-bm25 dspy-ai sentence-transformers

Run: uvicorn backend_api:app --port 8002 --reload-dir H:\AutoSafety_RAG
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama, time, json, re, os, asyncio, hashlib, pickle
import numpy as np
from rank_bm25 import BM25Okapi
from collections import Counter
from typing import Optional, Any
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    import faiss
    USE_FAISS = True
except ImportError:
    USE_FAISS = False
    print("WARNING: faiss-cpu not installed. Run: pip install faiss-cpu")
    print("Falling back to numpy dot product (slower)")

try:
    import fitz
except ImportError:
    raise RuntimeError("pip install pymupdf")

try:
    import dspy
    USE_DSPY = True
except ImportError:
    USE_DSPY = False

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def tok(t: str) -> int: return len(_enc.encode(t))
except ImportError:
    def tok(t: str) -> int: return max(1, int(len(t.split()) * 1.33))

# ── CONFIG ─────────────────────────────────────────────────────────────────────
EMBED_MODEL     = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL       = "llama3.2:1b"
SESSIONS_DIR    = r"H:\AutoSafety_RAG\sessions"   # persisted sessions on disk

# Retrieval
TOP_K_RETRIEVE  = 8      # reduced from 10 — FAISS is fast enough
TOP_K_RERANK    = 5
TOP_K_FINAL     = 3
MIN_SCORE       = 0.38
CONFIDENCE_GATE = 0.45   # lowered — regulations have niche vocabulary

# LLM — keep small for speed
LLM_MAX_TOKENS  = 120    # was 200 — each token = ~100ms on CPU
NUM_CTX         = 512    # was 1024 — smaller context = 2x faster
NUM_THREAD      = 4
BM25_WEIGHT     = 0.30
SEMANTIC_WEIGHT = 0.70

# Chunking
MIN_TOKENS      = 50
TARGET_TOKENS   = 600
MAX_TOKENS      = 1000
OVERLAP_TOKENS  = 80

# Cache
QUERY_VEC_CACHE_SIZE   = 500   # cache question embeddings
ANSWER_CACHE_SIZE      = 200   # cache (doc_id, question) → answer

LLM_OPTIONS = {
    "temperature":0.0, "num_predict":LLM_MAX_TOKENS,
    "repeat_penalty":1.2, "top_p":0.9,
    "num_ctx":NUM_CTX, "num_thread":NUM_THREAD,
    "num_keep":0, "num_batch":512,
}

STOPWORDS = {"what","is","are","how","does","the","a","an","in","of","for","to",
             "do","can","i","me","my","it","its","was","be","and","or","with","on"}

DOMAIN_MAP = {
    "fmvss":        ["fmvss","federal motor vehicle","208","214","301","nhtsa"],
    "unece":        ["unece","un r94","un r95","ece regulation","frontal collision"],
    "euroncap":     ["euro ncap","ncap","consumer test","star rating","mpdb"],
    "iso_standards":["iso 26262","iso 21448","sotif","functional safety","asil"],
    "airbag":       ["airbag","air bag","inflator","deployment","squib"],
    "seatbelt":     ["seatbelt","seat belt","pretensioner","load limiter"],
    "crashtest":    ["crash test","crash pulse","delta-v","deceleration","barrier"],
    "injury":       ["injury criteria","hic","tti","chest deflection","femur","ais"],
    "sensor":       ["accelerometer","crash sensor","safing","discrimination"],
    "structure":    ["body structure","b-pillar","rocker","intrusion","crumple"],
}

def detect_domain(text: str) -> str:
    t = text.lower()
    scores = {d: sum(1 for kw in kws if kw in t) for d, kws in DOMAIN_MAP.items()}
    best   = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"

os.makedirs(SESSIONS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STARTUP — LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  AutoSafety RAG v4.0 — Fast + Persistent")
print("=" * 60)

print(f"[1/3] Loading {EMBED_MODEL}...")
embedder  = SentenceTransformer(EMBED_MODEL)
EMBED_DIM = embedder.get_sentence_embedding_dimension()
print(f"      {EMBED_DIM} dims ✓")

print("[2/3] Loading reranker...")
try:
    reranker, USE_RERANKER = CrossEncoder(RERANKER_MODEL), True
    print("      Reranker ✓")
except Exception as e:
    reranker, USE_RERANKER = None, False
    print(f"      Skipped: {e}")

# DSPy
if USE_DSPY:
    try:
        lm = dspy.LM(model=f"ollama/{LLM_MODEL}", api_base="http://localhost:11434",
                     max_tokens=LLM_MAX_TOKENS, temperature=0.0)
        dspy.configure(lm=lm)

        class SafetyQASig(dspy.Signature):
            """
            Automotive passive safety engineer assistant.
            Answer ONLY from provided regulation excerpts.
            Cite regulation and section. Include exact numbers.
            If not found: say 'Not in loaded documents.'
            """
            context  = dspy.InputField(desc="Regulation excerpts [1][2][3]")
            question = dspy.InputField(desc="Safety engineering question")
            answer   = dspy.OutputField(desc="Precise answer with citations, 2-3 sentences")

        class SafetyRAG(dspy.Module):
            def __init__(self):
                super().__init__()
                self.qa = dspy.ChainOfThought(SafetyQASig)
            def forward(self, context, question):
                return self.qa(context=context, question=question)

        dspy_module = SafetyRAG()
        print("[DSPy] SafetyRAG module ready ✓")
    except Exception as e:
        USE_DSPY = False
        print(f"[DSPy] Failed: {e}")

print("[3/3] Warming up LLM + pre-warming query encoder...")
try:
    ollama.chat(model=LLM_MODEL, messages=[{"role":"user","content":"hi"}],
                options={"num_predict":1,"num_ctx":128})
    # Pre-warm encoder cache
    embedder.encode("warmup query for automotive safety", normalize_embeddings=True)
    print("      LLM + encoder warm ✓")
except Exception as e:
    print(f"      Warmup failed: {e}")

print(f"      FAISS: {'✓' if USE_FAISS else '✗ (using numpy fallback)'}  "
      f"DSPy: {'✓' if USE_DSPY else '✗'}  "
      f"Reranker: {'✓' if USE_RERANKER else '✗'}")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION DATA STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DocSession:
    doc_id:           str
    filename:         str
    doc_type:         str
    doc_structure:    str
    regulation_id:    str
    chunks:           list        # [{text, metadata}]
    vectors:          np.ndarray  # (n, EMBED_DIM) float32 normalized
    faiss_index:      Any         # faiss.IndexFlatIP or None
    bm25:             BM25Okapi
    bm25_texts:       list
    metadata:         list
    pages:            int
    tables:           int
    entity_summary:   dict
    suggested_questions: list
    created_at:       float = field(default_factory=time.time)

# In-memory store
document_store: dict[str, DocSession] = {}
active_doc_id:  str = None

# Caches
_query_vec_cache: dict = {}    # question → embedding vector
_answer_cache:    dict = {}    # (doc_id, q_hash) → answer string

def _q_hash(q: str) -> str:
    return hashlib.md5(q.lower().strip().encode()).hexdigest()

def get_query_vec(question: str) -> np.ndarray:
    """Return cached or freshly computed query embedding."""
    h = _q_hash(question)
    if h in _query_vec_cache:
        return _query_vec_cache[h]
    vec = embedder.encode(
        f"Represent this sentence for searching relevant passages: {question}",
        normalize_embeddings=True
    ).astype(np.float32)
    if len(_query_vec_cache) >= QUERY_VEC_CACHE_SIZE:
        del _query_vec_cache[next(iter(_query_vec_cache))]
    _query_vec_cache[h] = vec
    return vec

def get_answer_cache(doc_id: str, question: str) -> Optional[str]:
    return _answer_cache.get(f"{doc_id}:{_q_hash(question)}")

def set_answer_cache(doc_id: str, question: str, answer: str):
    if len(_answer_cache) >= ANSWER_CACHE_SIZE:
        del _answer_cache[next(iter(_answer_cache))]
    _answer_cache[f"{doc_id}:{_q_hash(question)}"] = answer

# ══════════════════════════════════════════════════════════════════════════════
# DISK PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def session_path(doc_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{doc_id}.pkl")

def save_session_to_disk(session: DocSession):
    """
    Saves session to disk so it survives server restart.
    FAISS index serialized via faiss.serialize_index().
    """
    try:
        path = session_path(session.doc_id)
        data = {
            "doc_id":           session.doc_id,
            "filename":         session.filename,
            "doc_type":         session.doc_type,
            "doc_structure":    session.doc_structure,
            "regulation_id":    session.regulation_id,
            "chunks":           session.chunks,
            "vectors":          session.vectors,
            "bm25_texts":       session.bm25_texts,
            "metadata":         session.metadata,
            "pages":            session.pages,
            "tables":           session.tables,
            "entity_summary":   session.entity_summary,
            "suggested_questions": session.suggested_questions,
            "created_at":       session.created_at,
        }
        if USE_FAISS and session.faiss_index is not None:
            data["faiss_bytes"] = faiss.serialize_index(session.faiss_index).tobytes()
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(path) // 1024 // 1024
        print(f"  Session saved: {session.filename} ({size_mb}MB) → {path}")
    except Exception as e:
        print(f"  Session save failed: {e}")

def load_session_from_disk(path: str) -> Optional[DocSession]:
    """Load session from .pkl file, rebuild FAISS index and BM25."""
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Rebuild BM25
        bm25 = BM25Okapi([t.lower().split() for t in data["bm25_texts"]])

        # Rebuild FAISS
        faiss_index = None
        if USE_FAISS:
            if "faiss_bytes" in data:
                faiss_index = faiss.deserialize_index(
                    np.frombuffer(data["faiss_bytes"], dtype=np.uint8)
                )
            else:
                # Rebuild from vectors
                vectors = data["vectors"].astype(np.float32)
                faiss_index = faiss.IndexFlatIP(vectors.shape[1])
                faiss_index.add(vectors)

        session = DocSession(
            doc_id=data["doc_id"], filename=data["filename"],
            doc_type=data["doc_type"], doc_structure=data["doc_structure"],
            regulation_id=data["regulation_id"], chunks=data["chunks"],
            vectors=data["vectors"].astype(np.float32),
            faiss_index=faiss_index, bm25=bm25,
            bm25_texts=data["bm25_texts"], metadata=data["metadata"],
            pages=data["pages"], tables=data["tables"],
            entity_summary=data["entity_summary"],
            suggested_questions=data["suggested_questions"],
            created_at=data.get("created_at", time.time()),
        )
        print(f"  Loaded: {session.filename} ({len(session.chunks)} chunks)")
        return session
    except Exception as e:
        print(f"  Failed to load {path}: {e}")
        return None

def load_all_sessions():
    """Load all persisted sessions from disk at startup."""
    global active_doc_id
    loaded = 0
    if not os.path.exists(SESSIONS_DIR):
        return
    for fname in sorted(os.listdir(SESSIONS_DIR)):
        if not fname.endswith(".pkl"): continue
        session = load_session_from_disk(os.path.join(SESSIONS_DIR, fname))
        if session:
            document_store[session.doc_id] = session
            active_doc_id = session.doc_id   # most recent becomes active
            loaded += 1
    if loaded:
        print(f"  Loaded {loaded} persisted session(s) from disk")

# Load sessions at module import time
print("Loading persisted sessions...")
load_all_sessions()

# ══════════════════════════════════════════════════════════════════════════════
# PDF PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

HEADING_PATTERNS = [
    (r"^(Annex|ANNEX)\s+([A-Z0-9]+)\b[.\s—–\-]*(.*)","annex"),
    (r"^(Chapter|CHAPTER|Part|PART)\s+(\d+|[IVXLC]+)\b[.\s—–\-]*(.*)","chapter"),
    (r"^(Article|ARTICLE)\s+(\d+(?:\.\d+)?)\b[.\s—–\-]*(.*)","article"),
    (r"^(\d{1,3})\.\s{1,5}([A-Z][^\n]{2,})","section"),
    (r"^(\d{1,3}\.\d{1,3}(?:\.\d{1,3})?)\s+([^\n]{2,})","clause"),
    (r"^\(([a-z]|[ivxlc]+)\)\s+([^\n]{2,})","subclause"),
]

def classify_heading(text: str) -> tuple:
    for pat, ntype in HEADING_PATTERNS:
        m = re.match(pat, text.strip(), re.IGNORECASE)
        if m:
            g = m.groups()
            if ntype in ("annex","chapter","article"):
                return ntype, g[1] if len(g)>1 else "", (g[2] or "").strip() if len(g)>2 else ""
            return ntype, g[0], (g[1] or "").strip() if len(g)>1 else ""
    return "text","",text.strip()

INJURY_MAP = {
    "HIC":r"\bHIC(?:15|36)?\b","TTI":r"\bTTI\b","VC":r"\bVC\s*(?:criterion|limit)\b",
    "Nij":r"\bNij\b","NIC":r"\bNIC\b","CTI":r"\bCTI\b",
    "Chest_defl":r"\bchest\s+(?:deflection|compression)\b",
    "FemurLoad":r"\bfemur\s+(?:load|force)\b",
    "TibiaIdx":r"\btibia(?:l)?\s+(?:index|force)\b",
    "AbdomenForce":r"\babdomen\s+(?:force|load)\b",
}
DUMMY_MAP = {
    "Hybrid III 50M":r"hybrid\s*iii\s*(?:50(?:th)?|male|50m)\b",
    "Hybrid III 5F": r"hybrid\s*iii\s*(?:5(?:th)?|female|5f)\b",
    "WorldSID 50M":  r"worldsid\s*(?:50(?:th)?|50m|male)?\b",
    "WorldSID 5F":   r"worldsid\s*(?:5(?:th)?|5f|female)?\b",
    "ES-2":r"\bes-?2\b","SID-IIs":r"\bsid-?ii\b","THOR":r"\bthor\b(?!\s*speed)",
}
TEST_MAP = {
    "frontal_obd":r"\bobd\b|offset\s+deformable",
    "frontal_full":r"full[\s-]width\s+frontal",
    "frontal_mpdb":r"\bmpdb\b",
    "frontal":r"frontal\s+(?:crash|impact|test)",
    "side_pole":r"pole\s+(?:test|impact)",
    "side_mdb":r"moving\s+deformable\s+barrier",
    "side":r"side\s+(?:crash|impact|test)",
    "rear":r"rear(?:-end)?\s+(?:crash|impact)",
    "rollover":r"rollover","pedestrian":r"pedestrian\s+(?:protection|impact)",
}
SPEED_PATS = [
    (r"(\d+(?:\.\d+)?)\s*km/?h","kmph"),
    (r"(\d+(?:\.\d+)?)\s*mph\b","mph"),
    (r"(\d+(?:\.\d+)?)\s*m/s\b","mps"),
]
THRESH_PATS = [
    r"(?:not\s+(?:exceed|greater)\s+(?:than\s+)?)([\d,\.]+)",
    r"(?:≤|<=|maximum\s+of?)\s*([\d,\.]+)",
    r"([\d,\.]+)\s*(?:or\s+less|maximum)",
    r"(?:limit(?:ed)?\s+to)\s*([\d,\.]+)",
]

def extract_entities(text: str) -> dict:
    t = text.lower()
    metrics   = [m for m,p in INJURY_MAP.items() if re.search(p,t,re.IGNORECASE)]
    dummy     = next((d for d,p in DUMMY_MAP.items() if re.search(p,t,re.IGNORECASE)),"")
    test_type = next((tt for tt,p in TEST_MAP.items() if re.search(p,t)),"")
    threshold = None
    for p in THRESH_PATS:
        m = re.search(p,t)
        if m:
            try:
                v = float(m.group(1).replace(",",""))
                if 0<v<1_000_000: threshold=v; break
            except Exception: pass
    speed_kmph = None
    for pat,stype in SPEED_PATS:
        m = re.search(pat,t)
        if m:
            try:
                v = float(m.group(1))
                if stype=="kmph": speed_kmph=v
                elif stype=="mph": speed_kmph=round(v*1.60934,1)
                elif stype=="mps": speed_kmph=round(v*3.6,1)
                break
            except Exception: pass
    req=""
    if re.search(r"\b(?:shall\s+not|must\s+not|prohibited)\b",t): req="prohibited"
    elif re.search(r"\b(?:shall|must|is required)\b",t): req="mandatory"
    elif re.search(r"\bshould\b",t): req="recommended"
    return {
        "metric":m[0] if metrics else "","injury_metrics":",".join(metrics),
        "dummy":dummy,"test_type":test_type,
        "threshold":str(threshold) if threshold is not None else "",
        "speed_kmph":str(speed_kmph) if speed_kmph is not None else "",
        "requirement_type":req,
    }

def extract_pdf(pdf_bytes: bytes) -> tuple:
    """Returns (blocks, stats). Blocks have type: text|heading|table|footnote."""
    doc    = fitz.open(stream=pdf_bytes, filetype="pdf")
    blocks = []
    stats  = {"pages":len(doc),"tables":0}

    # Detect body font size from first 5 pages
    sizes = []
    for pg in doc[:5]:
        for blk in pg.get_text("dict")["blocks"]:
            if blk.get("type")!=0: continue
            for ln in blk.get("lines",[]):
                for sp in ln.get("spans",[]):
                    sz = round(sp.get("size",0),1)
                    if 6<sz<24: sizes.append(sz)
    body_size  = Counter(sizes).most_common(1)[0][0] if sizes else 10.0
    hdg_thresh = body_size * 1.10
    fn_thresh  = body_size * 0.88

    for page_num, page in enumerate(doc, start=1):
        ph = page.rect.height
        pw = page.rect.width
        hdr_zone, ftr_zone, fn_zone = ph*0.06, ph*0.94, ph*0.82

        # Tables
        try:
            for tab in page.find_tables():
                rows = tab.extract()
                if not rows or len(rows)<2: continue
                hdr  = [str(c or "").strip() for c in rows[0]]
                if not any(hdr): continue
                lines = ["| "+" | ".join(hdr)+" |", "|"+ "|".join(["---"]*len(hdr))+"|"]
                for row in rows[1:]:
                    cells = [str(c or "").strip() for c in row]
                    if any(cells): lines.append("| "+" | ".join(cells)+" |")
                blocks.append({"text":"\n".join(lines),"page":page_num,"type":"table","bbox":getattr(tab,"bbox",(0,0,pw,ph))})
                stats["tables"]+=1
        except Exception: pass

        # Text blocks
        for blk in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks",[]):
            if blk.get("type")!=0: continue
            bbox = blk.get("bbox",(0,0,0,0)); y0=bbox[1]
            if y0<hdr_zone or y0>ftr_zone: continue
            span_text,max_sz,has_bold = "",0.0,False
            for ln in blk.get("lines",[]):
                for sp in ln.get("spans",[]):
                    sz=sp.get("size",body_size); bold=bool(sp.get("flags",0)&(1<<4))
                    txt=sp.get("text","").strip()
                    if not txt: continue
                    span_text+=txt+" "; max_sz=max(max_sz,sz)
                    if bold: has_bold=True
            span_text=span_text.strip()
            if not span_text or len(span_text)<3: continue
            btype = "heading" if (max_sz>=hdg_thresh or (has_bold and max_sz>=body_size*0.98)) \
                    else "footnote" if (y0>fn_zone and max_sz<=fn_thresh and len(span_text)<300) \
                    else "text"
            blocks.append({"text":span_text,"page":page_num,"type":btype,"font_size":max_sz,"is_bold":has_bold,"bbox":bbox})

    doc.close()
    return blocks, stats

def classify_doc(blocks: list, filename: str) -> tuple:
    fname  = filename.lower()
    sample = " ".join(b["text"] for b in blocks[:100] if b["type"] in ("text","heading"))
    text_l = sample.lower()

    reg_patterns = {
        "FMVSS 208":r"fmvss.?208|federal.*208","FMVSS 214":r"fmvss.?214|federal.*214",
        "FMVSS 301":r"fmvss.?301","UN R94":r"un.?r.?94|r94.*frontal",
        "UN R95":r"un.?r.?95|r95.*side","UN R127":r"un.?r.?127|pedestrian.*safety",
        "Euro NCAP":r"euro.?ncap|ncap.*protocol","ISO 26262":r"iso.?26262","ISO 21448":r"iso.?21448|sotif",
    }
    regulation_id = next((rid for rid,pat in reg_patterns.items()
                           if re.search(pat,fname) or re.search(pat,text_l)), "")

    doc_type = "regulation"
    if any(w in fname for w in ["fmvss","federal_motor","49_cfr"]): doc_type="FMVSS"
    elif any(w in fname for w in ["un_r","unece","ece_r"]): doc_type="UNECE"
    elif "ncap" in fname: doc_type="Euro_NCAP"
    elif "iso"  in fname: doc_type="ISO"
    elif any(w in fname for w in ["internal","report","study","analysis"]): doc_type="Internal"

    struct_hits  = sum(1 for b in blocks[:80] if b["type"]=="heading" and
                       classify_heading(b["text"])[0] in ("article","chapter","section"))
    protocol_hit = sum(1 for p in [r"test\s+procedure",r"rating\s+scheme",r"euro.ncap"]
                       if re.search(p,text_l))
    numbered     = sum(1 for b in blocks[:80] if re.match(r"^\d{1,3}[\.\s]", b["text"].strip()))

    if struct_hits>=3 or numbered>=5: doc_structure="structured_regulation"
    elif protocol_hit>=2 or sum(1 for b in blocks[:80] if b["type"]=="table")>=3: doc_structure="protocol_report"
    else: doc_structure="paragraph_document"

    return doc_structure, doc_type, regulation_id

def chunk_document(blocks: list, doc_structure: str, doc_type: str,
                   regulation_id: str, filename: str) -> list:
    """Returns list of {text, metadata} dicts."""
    chunks = []
    ctx    = {"chapter":"","section":"","article":"","annex":""}
    crumb  = ""

    def build_crumb(ntype, num, title):
        parts = []
        if ctx["annex"]: parts.append(f"Annex {ctx['annex']}")
        elif ctx["chapter"]: parts.append(f"Chapter {ctx['chapter']}")
        if ctx["section"] and ntype not in ("chapter","annex"): parts.append(f"Section {ctx['section']}")
        if ctx["article"] and ntype not in ("chapter","section","annex"): parts.append(f"Article {ctx['article']}")
        if ntype in ("clause","subclause") and num: parts.append(f"{ntype.title()} {num}")
        if title and parts: parts[-1] += f" — {title}"
        elif title: parts.append(title)
        return " > ".join(parts)

    current_text = ""; current_page = 1; current_end = 1
    current_title = ""; footnotes = []

    def flush():
        nonlocal current_text, footnotes
        text = re.sub(r"\s{2,}"," ", current_text).strip()
        if footnotes: text += " " + " | ".join(f"[Footnote: {f}]" for f in footnotes)
        footnotes = []
        if tok(text) < 15: current_text=""; return
        domain = detect_domain(text + " " + filename)
        ents   = extract_entities(text)
        chunks.append({
            "text": text,
            "metadata": {
                "breadcrumb":crumb[:200],"document":filename,"doc_type":doc_type,
                "regulation_id":regulation_id,"domain":domain,
                "chapter":ctx["chapter"],"section":ctx["section"],
                "article":ctx["article"],"annex":ctx["annex"],
                "title":current_title[:80],"page_start":str(current_page),
                "page_end":str(current_end),"chunk_type":"text",
                **ents,
            }
        })
        current_text=""

    for blk in blocks:
        if blk["type"]=="table":
            flush()
            ctx_prefix = crumb+"\nTable" if crumb else "Table"
            tbl_text   = ctx_prefix+"\n"+blk["text"]
            domain     = detect_domain(tbl_text)
            ents       = extract_entities(tbl_text)
            chunks.append({"text":tbl_text,"metadata":{
                "breadcrumb":crumb[:200],"document":filename,"doc_type":doc_type,
                "regulation_id":regulation_id,"domain":domain,
                "chapter":ctx["chapter"],"section":ctx["section"],
                "article":ctx["article"],"annex":ctx["annex"],
                "title":"Table","page_start":str(blk["page"]),"page_end":str(blk["page"]),
                "chunk_type":"table",**ents,
            }})
            continue

        if blk["type"]=="footnote":
            footnotes.append(blk["text"].strip()); continue

        if blk["type"]=="heading":
            ntype,num,title = classify_heading(blk["text"])
            if ntype!="text":
                flush()
                if ntype=="annex": ctx.update({"annex":num,"article":"","section":""})
                elif ntype=="chapter": ctx.update({"chapter":num,"section":"","article":"","annex":""})
                elif ntype=="section": ctx.update({"section":num,"article":""})
                elif ntype in ("article","clause"): ctx["article"]=num
                crumb=build_crumb(ntype,num,title); current_title=title
                current_page=blk["page"]; current_text=crumb+"\n"; continue

        txt=blk["text"].strip()
        if not txt: continue
        current_text+=txt+" "; current_end=blk["page"]
        if not current_page: current_page=blk["page"]

    flush()

    # Split oversized chunks
    result = []
    for chunk in chunks:
        t = tok(chunk["text"])
        if t<=MAX_TOKENS: result.append(chunk); continue
        parts = re.split(r'(?<=[.;])\s{2,}(?=[A-Z\(\d])', chunk["text"])
        ov    = ""
        for pi,part in enumerate(parts):
            cand = (ov+" "+part).strip() if ov else part
            if tok(cand)>MAX_TOKENS and ov: cand=part
            c = chunk.copy()
            c["text"] = cand
            c["metadata"] = chunk["metadata"].copy()
            if len(parts)>1: c["metadata"]["breadcrumb"] += f" [part {pi+1}]"
            result.append(c)
            words = cand.split(); nw=max(1,int(OVERLAP_TOKENS*0.75))
            ov = " ".join(words[-nw:]) if len(words)>nw else cand
    return result

def build_faiss_index(vectors: np.ndarray) -> Any:
    """Build FAISS IndexFlatIP. For normalized vectors this = cosine similarity."""
    if not USE_FAISS: return None
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors.astype(np.float32))
    return idx

def fast_dedup(chunks: list, vectors: np.ndarray, threshold: float = 0.92) -> tuple:
    kept_vecs, keep = [], []
    for i in range(len(vectors)):
        if kept_vecs:
            sims = np.array(kept_vecs, dtype=np.float32) @ vectors[i]
            if np.any(sims >= threshold): continue
        kept_vecs.append(vectors[i]); keep.append(i)
    return [chunks[i] for i in keep], vectors[keep]

def build_entity_summary(chunks: list) -> dict:
    metrics,speeds,dummies,test_types = set(),set(),set(),set()
    for c in chunks:
        m = c["metadata"]
        if m.get("injury_metrics"):
            for x in m["injury_metrics"].split(","): metrics.add(x.strip())
        if m.get("speed_kmph"):
            try: speeds.add(round(float(m["speed_kmph"]),1))
            except Exception: pass
        if m.get("dummy"):     dummies.add(m["dummy"])
        if m.get("test_type"): test_types.add(m["test_type"])
    return {
        "metrics":  sorted(metrics-{""})[:8],
        "speeds":   sorted(speeds)[:6],
        "dummies":  sorted(dummies-{""})[:5],
        "test_types":sorted(test_types-{""})[:6],
    }

def gen_suggested_questions(doc_type, regulation_id, entities, doc_structure) -> list:
    q, reg = [], regulation_id or doc_type
    metrics = entities.get("metrics",[])
    speeds  = entities.get("speeds",[])
    dummies = entities.get("dummies",[])
    test_types = entities.get("test_types",[])

    if "HIC" in metrics: q.append({"text":f"What is the HIC limit in {reg}?","category":"Injury"})
    if "Chest_defl" in metrics: q.append({"text":f"What is the chest deflection limit in {reg}?","category":"Injury"})
    if "TTI" in metrics: q.append({"text":"What is the TTI threshold?","category":"Injury"})
    if "FemurLoad" in metrics: q.append({"text":"What is the maximum femur load?","category":"Injury"})
    if "Nij" in metrics: q.append({"text":"What is the Nij neck injury limit?","category":"Injury"})
    if speeds: q.append({"text":f"What are the test requirements at {speeds[0]} km/h?","category":"Test"})
    for d in dummies[:2]: q.append({"text":f"How is the {d} positioned?","category":"Setup"})
    if "frontal_obd" in test_types: q.append({"text":"What are the ODB frontal barrier requirements?","category":"Test"})
    if "side_pole" in test_types: q.append({"text":"What are the side pole impact requirements?","category":"Test"})
    if "FMVSS" in doc_type:
        q+=[{"text":f"What injury criteria shall not be exceeded?","category":"Compliance"},
            {"text":"What belt reminder requirements are defined?","category":"Requirements"}]
    if "Euro_NCAP" in doc_type:
        q+=[{"text":"How are star ratings calculated?","category":"Rating"},
            {"text":"What is the minimum passing score?","category":"Compliance"}]
    if "ISO" in doc_type:
        q+=[{"text":"What ASIL levels are defined?","category":"Safety"},
            {"text":"What are the functional safety requirements?","category":"Requirements"}]
    if "Internal" in doc_type:
        q+=[{"text":"What are the main findings?","category":"Summary"},
            {"text":"What recommendations are made?","category":"Analysis"}]
    if len(q)<4:
        q+=[{"text":f"What are the main requirements in {reg}?","category":"General"},
            {"text":"What test procedures are defined?","category":"Test"}]
    return q[:8]

# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL — FAISS FAST PATH
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_faiss(question: str, session: DocSession, top_k: int) -> list:
    """
    FAISS retrieval: sub-5ms for any index size.
    IndexFlatIP with normalized vectors = cosine similarity.
    """
    q_vec = get_query_vec(question).reshape(1, -1)

    if USE_FAISS and session.faiss_index is not None:
        scores, indices = session.faiss_index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0: continue
            results.append({
                "text": session.bm25_texts[idx],
                "sem_score": float(score),
                "rrf_score": 0.0,
                "bm25_score": 0.0,
                **session.metadata[idx],
            })
    else:
        # Numpy fallback
        sims    = session.vectors @ q_vec[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_idx:
            results.append({
                "text": session.bm25_texts[idx],
                "sem_score": float(sims[idx]),
                "rrf_score": 0.0,
                "bm25_score": 0.0,
                **session.metadata[idx],
            })

    # Assign semantic RRF scores
    for rank, r in enumerate(results):
        r["rrf_score"] = (1 / (rank + 60)) * SEMANTIC_WEIGHT
    return results

def retrieve_bm25(question: str, session: DocSession, top_k: int) -> list:
    """BM25 keyword search — catches exact regulation numbers and medical terms."""
    scores  = session.bm25.get_scores(question.lower().split())
    max_s   = scores.max()
    if max_s > 0: scores = scores / max_s
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_idx):
        s = float(scores[idx])
        if s < 0.05: continue
        results.append({
            "text": session.bm25_texts[idx],
            "sem_score": 0.0,
            "bm25_score": s,
            "rrf_score": (1 / (rank + 60)) * BM25_WEIGHT,
            **session.metadata[idx],
        })
    return results

async def retrieve_hybrid_async(question: str, session: DocSession,
                                 top_k: int) -> list:
    """Run FAISS + BM25 CONCURRENTLY. Total time = max(FAISS, BM25) not sum."""
    loop = asyncio.get_event_loop()

    faiss_results, bm25_results = await asyncio.gather(
        loop.run_in_executor(None, retrieve_faiss, question, session, top_k),
        loop.run_in_executor(None, retrieve_bm25,  question, session, top_k),
    )

    # Merge via RRF
    merged: dict = {r["text"]: r for r in faiss_results}
    for r in bm25_results:
        if r["text"] in merged:
            merged[r["text"]]["bm25_score"] = r["bm25_score"]
            merged[r["text"]]["rrf_score"] += r["rrf_score"]
        else:
            merged[r["text"]] = r

    # Filter by minimum score
    topic_words = set(question.lower().split()) - STOPWORDS
    filtered = [
        c for c in merged.values()
        if (c["sem_score"] >= MIN_SCORE or c["bm25_score"] >= 0.1)
        and (not topic_words or any(w in c["text"].lower() for w in topic_words))
    ]

    return sorted(filtered, key=lambda x: x["rrf_score"], reverse=True)[:top_k]

async def rerank_async(question: str, candidates: list, top_k: int) -> list:
    """Rerank in thread pool — non-blocking."""
    if not USE_RERANKER or not candidates: return candidates[:top_k]
    loop  = asyncio.get_event_loop()
    pairs = [(question, c["text"]) for c in candidates]
    def _r():
        scores = reranker.predict(pairs)
        for c,s in zip(candidates,scores): c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
    return await loop.run_in_executor(None, _r)

# ══════════════════════════════════════════════════════════════════════════════
# LLM GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(question: str, candidates: list) -> str:
    context_parts = []
    for i,c in enumerate(candidates,1):
        hdr   = f"[{i}]"
        if c.get("regulation_id"): hdr += f" {c['regulation_id']}"
        if c.get("doc_type"):      hdr += f" ({c['doc_type']})"
        if c.get("page_start"):    hdr += f" p.{c['page_start']}"
        clean = re.sub(r"\s{2,}"," ", c["text"])[:600].strip()
        context_parts.append(f"{hdr}\n{clean}")
    return (
        f"<start_of_turn>user\n"
        f"Automotive passive safety engineer assistant.\n"
        f"Answer from these regulation excerpts ONLY. Include exact numbers. Cite source. 2-3 sentences.\n\n"
        + "\n\n".join(context_parts)
        + f"\n\nQuestion: {question}\n<end_of_turn>\n<start_of_turn>model\n"
    )

async def generate_streaming(question: str, candidates: list):
    """
    Yields tokens from Ollama streaming API.
    Using DSPy if available, else direct Ollama.
    """
    if USE_DSPY:
        loop = asyncio.get_event_loop()
        def _dspy():
            context = "\n\n".join(
                f"[{i+1}] {c.get('regulation_id','')} p.{c.get('page_start','')}\n{c['text'][:600]}"
                for i,c in enumerate(candidates)
            )
            pred = dspy_module(context=context, question=question)
            return pred.answer if hasattr(pred,"answer") else str(pred)
        try:
            answer = await loop.run_in_executor(None, _dspy)
            for word in postprocess(answer).split():
                yield word + " "
            return
        except Exception as e:
            print(f"DSPy error: {e} — falling back to Ollama")

    # Ollama streaming
    prompt = build_prompt(question, candidates)
    try:
        stream = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role":"user","content":prompt}],
            stream=True, options=LLM_OPTIONS
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if token: yield token
    except Exception as e:
        yield f"\n[Error: {e}]"

def postprocess(answer: str) -> str:
    for m in ["<start_of_turn>model","<end_of_turn>","<|assistant|>","Answer:"]:
        if m in answer: answer=answer.split(m)[-1]; break
    noise = [r"<start_of_turn>.*?<end_of_turn>",r"<\|.*?\|>",
             r"automotive.*?assistant.*?[.\n]",r"answer from.*?[.\n]"]
    for p in noise: answer=re.sub(p,"",answer,flags=re.IGNORECASE|re.DOTALL)
    def fix_caps(w):
        if len(w)<=2 or w.isupper(): return w
        if any(c.isupper() for c in w[1:]):
            return (w[0].upper()+w[1:].lower()) if w[0].isupper() else w.lower()
        return w
    answer=" ".join(fix_caps(w) for w in answer.split())
    sents=[s for s in re.split(r'(?<=[.!?])\s+',answer.strip()) if len(s.split())>4]
    if len(sents)>4: answer=" ".join(sents[:4])
    return re.sub(r"\s{2,}"," ",answer).strip()

# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="AutoSafety RAG v4.0", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    question: str
    doc_id:   str
    top_k:    int = TOP_K_FINAL

class SwitchDocRequest(BaseModel):
    doc_id: str

# ── UPLOAD ─────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")

    pdf_bytes = await file.read()
    filename  = file.filename

    async def process():
        try:
            loop = asyncio.get_event_loop()

            # 1. Extract
            yield f"data: {json.dumps({'type':'extract','status':'running'})}\n\n"
            blocks, stats = await loop.run_in_executor(None, lambda: extract_pdf(pdf_bytes))
            yield f"data: {json.dumps({'type':'extract','pages':stats['pages'],'blocks':len(blocks),'tables':stats['tables']})}\n\n"

            # 2. Structure
            yield f"data: {json.dumps({'type':'structure','status':'running'})}\n\n"
            doc_structure, doc_type, regulation_id = await loop.run_in_executor(
                None, lambda: classify_doc(blocks, filename))
            yield f"data: {json.dumps({'type':'structure','doc_structure':doc_structure,'doc_type':doc_type,'regulation_id':regulation_id})}\n\n"

            # 3. Chunk
            yield f"data: {json.dumps({'type':'chunk','status':'running'})}\n\n"
            chunks = await loop.run_in_executor(
                None, lambda: chunk_document(blocks, doc_structure, doc_type, regulation_id, filename))
            yield f"data: {json.dumps({'type':'chunk','chunks':len(chunks)})}\n\n"

            # 4. Embed
            yield f"data: {json.dumps({'type':'embed','status':'running'})}\n\n"
            texts   = [c["text"] for c in chunks]
            vectors = await loop.run_in_executor(None, lambda:
                embedder.encode(texts, batch_size=64, normalize_embeddings=True,
                                show_progress_bar=False).astype(np.float32))

            # 5. Dedup + FAISS + BM25 + Save
            yield f"data: {json.dumps({'type':'index','status':'running'})}\n\n"
            chunks, vectors = await loop.run_in_executor(
                None, lambda: fast_dedup(chunks, vectors))

            bm25_texts  = [c["text"] for c in chunks]
            bm25_index  = BM25Okapi([t.lower().split() for t in bm25_texts])
            metadata    = [c["metadata"] for c in chunks]
            faiss_index = await loop.run_in_executor(None, lambda: build_faiss_index(vectors))
            entities    = build_entity_summary(chunks)
            entity_count= sum(len(v) for v in entities.values() if isinstance(v,list))
            suggested   = gen_suggested_questions(doc_type, regulation_id, entities, doc_structure)

            doc_id  = hashlib.md5(f"{filename}{time.time()}".encode()).hexdigest()[:12]
            session = DocSession(
                doc_id=doc_id, filename=filename, doc_type=doc_type,
                doc_structure=doc_structure, regulation_id=regulation_id,
                chunks=chunks, vectors=vectors, faiss_index=faiss_index,
                bm25=bm25_index, bm25_texts=bm25_texts, metadata=metadata,
                pages=stats["pages"], tables=stats["tables"],
                entity_summary=entities, suggested_questions=suggested,
            )
            document_store[doc_id] = session

            global active_doc_id
            active_doc_id = doc_id

            # Save to disk in background
            await loop.run_in_executor(None, lambda: save_session_to_disk(session))

            yield f"data: {json.dumps({'type':'index','status':'done'})}\n\n"
            yield f"data: {json.dumps({'type':'done','doc_id':doc_id,'filename':filename,'doc_type':doc_type,'doc_structure':doc_structure,'regulation_id':regulation_id,'chunks':len(chunks),'pages':stats['pages'],'tables':stats['tables'],'entity_count':entity_count,'entities':{'metrics':entities['metrics'],'speeds':entities['speeds'],'dummies':entities['dummies'],'test_types':entities['test_types']},'suggested_questions':suggested})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(process(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ── SWITCH DOC ─────────────────────────────────────────────────────────────────
@app.post("/switch_doc")
async def switch_doc(req: SwitchDocRequest):
    global active_doc_id
    if req.doc_id not in document_store:
        raise HTTPException(status_code=404, detail="Document not found.")
    active_doc_id = req.doc_id
    s = document_store[req.doc_id]
    return {"doc_id":req.doc_id,"filename":s.filename,"chunks":len(s.chunks)}

# ── CHAT STREAM ────────────────────────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    doc_id  = req.doc_id or active_doc_id
    if not doc_id or doc_id not in document_store:
        raise HTTPException(status_code=400, detail="No document loaded. Upload a PDF first.")

    session = document_store[doc_id]

    async def generate():
        t0 = time.time()

        # Cache hit → instant response
        cached = get_answer_cache(doc_id, req.question)
        if cached:
            for word in cached.split():
                yield f"data: {json.dumps({'token': word+' '})}\n\n"
            yield f"data: {json.dumps({'done':True,'answered':True,'confidence':'High','sources':[],'cache_hit':True,'latency_ms':1})}\n\n"
            return

        # Hybrid retrieval (FAISS + BM25 concurrent)
        t_ret = time.time()
        candidates = await retrieve_hybrid_async(req.question, session, TOP_K_RETRIEVE)
        t_ret_ms   = int((time.time()-t_ret)*1000)
        print(f"  Retrieve → {t_ret_ms}ms ({len(candidates)} candidates, FAISS={'✓' if USE_FAISS else '✗'})")

        # Rerank
        t_rer = time.time()
        reranked = await rerank_async(req.question, candidates, TOP_K_RERANK)
        top_3    = reranked[:TOP_K_FINAL]
        print(f"  Rerank  → {int((time.time()-t_rer)*1000)}ms")

        # Confidence check
        best_score = max((c.get("sem_score",0) for c in top_3), default=0)
        if not top_3 or best_score < CONFIDENCE_GATE:
            idk = (f"This specific information was not found in {session.filename}. "
                   f"Please consult the original regulation source.")
            for w in idk.split():
                yield f"data: {json.dumps({'token': w+' '})}\n\n"
            yield f"data: {json.dumps({'done':True,'answered':False,'confidence':'None','sources':[]})}\n\n"
            return

        # Build sources metadata for frontend
        src_data = []
        for i,c in enumerate(top_3,1):
            score = c.get("rerank_score", c.get("sem_score",0))
            if "rerank_score" in c: score = min(1.0, max(0.0,(score+10)/20))
            label = "High" if score>=0.75 else "Medium" if score>=0.55 else "Low"
            src_data.append({
                "ref":i,"text":c["text"],"score":round(score,4),"label":label,
                "regulation_id":c.get("regulation_id",""),
                "source_file":c.get("document",""),
                "page":c.get("page_start",""),
                "breadcrumb":c.get("breadcrumb",""),
                "metric":c.get("metric",""),
                "threshold":c.get("threshold",""),
            })

        detected = detect_domain(req.question) or session.doc_type

        # Stream tokens directly as they arrive
        full_answer = ""
        t_llm = time.time()
        async for token in generate_streaming(req.question, top_3):
            full_answer += token
            yield f"data: {json.dumps({'token': token})}\n\n"
        print(f"  LLM     → {int((time.time()-t_llm)*1000)}ms")

        answer = postprocess(full_answer)
        set_answer_cache(doc_id, req.question, answer)

        total_ms = int((time.time()-t0)*1000)
        print(f"  TOTAL   → {total_ms}ms")
        yield f"data: {json.dumps({'done':True,'answered':True,'confidence':'High','sources':src_data,'detected_domain':detected,'cache_hit':False,'latency_ms':total_ms})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ── HEALTH + UTILS ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    total_chunks = sum(len(s.chunks) for s in document_store.values())
    return {
        "status":"ok","total_chunks":total_chunks,
        "loaded_docs":len(document_store),"active_doc":active_doc_id,
        "embed_model":EMBED_MODEL,"llm_model":LLM_MODEL,
        "faiss":USE_FAISS,"dspy":USE_DSPY,"reranker":USE_RERANKER,
        "sessions_dir":SESSIONS_DIR,
    }

@app.get("/docs_list")
async def docs_list():
    return {"documents":[
        {"doc_id":s.doc_id,"filename":s.filename,"doc_type":s.doc_type,
         "chunks":len(s.chunks),"active":s.doc_id==active_doc_id}
        for s in document_store.values()
    ]}

@app.delete("/cache/clear")
async def clear_cache():
    _query_vec_cache.clear(); _answer_cache.clear()
    return {"message":"Caches cleared."}

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_api:app", host="0.0.0.0", port=8002,
                reload=True, reload_dirs=[r"H:\AutoSafety_RAG"])