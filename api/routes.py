import os
import re
import shutil
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session
from loguru import logger

from database.connection import get_db
from database.models import Regulation, Document, Test, TestResult, Chunk
from app.config import settings
from registry.auth import AuthenticatedUser, get_current_user, get_optional_user
from registry.search import RegulationSearchEngine
from registry.coverage import build_coverage_report
from registry.chat_intent import (
    classify_chat_query,
    corpus_meta_chat_result,
    identity_chat_result,
    injection_chat_result,
)
from registry.margin_query import try_margin_query_response
from scheduler.tasks import ingest_document_task, reindex_task

router = APIRouter()

# Instantiate search engine lazily or globally
search_engine = None

def get_search_engine():
    global search_engine
    if search_engine is None:
        search_engine = RegulationSearchEngine()
    return search_engine


# Pydantic schemas
class IngestRequest(BaseModel):
    file_path: str
    manual_metadata: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    query: str
    filter: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = 5
    rerank: Optional[bool] = True


class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 8
    rerank: Optional[bool] = True


_ready_cache: dict[str, Any] | None = None


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF document to the staging uploads directory."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    dest_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    logger.info(f"Saving uploaded file {file.filename} to {dest_path}")
    
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {
            "file_path": dest_path,
            "filename": file.filename,
            "size": os.path.getsize(dest_path)
        }
    except Exception as e:
        logger.error(f"Failed to upload document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/documents/ingest")
async def ingest_document(request: IngestRequest):
    """Enqueues document ingestion asynchronously in Celery."""
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="File path does not exist.")
        
    try:
        # Enqueue Celery task
        task = ingest_document_task.delay(request.file_path, request.manual_metadata)
        return {
            "task_id": task.id,
            "status": "queued"
        }
    except Exception as e:
        logger.error(f"Failed to enqueue ingestion task: {e}")
        raise HTTPException(status_code=500, detail=f"Ingest queuing failed: {str(e)}")


@router.get("/regulations")
async def list_regulations(
    regulation_code: Optional[str] = None,
    status: Optional[str] = None,
    market: Optional[str] = None,
    source_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Lists registered regulations, supporting filtering."""
    query = db.query(Regulation)
    if regulation_code:
        query = query.filter(Regulation.regulation_code == regulation_code)
    if status:
        query = query.filter(Regulation.status == status)
    if market:
        query = query.filter(Regulation.market == market)
    if source_type:
        query = query.filter(Regulation.source_type == source_type)
        
    results = query.all()
    return results


@router.get("/regulations/{id}")
async def get_regulation(id: int, db: Session = Depends(get_db)):
    """Retrieves metadata for a specific regulation."""
    reg = db.query(Regulation).filter(Regulation.id == id).first()
    if not reg:
        raise HTTPException(status_code=404, detail="Regulation not found.")
    return reg


@router.get("/documents")
async def list_documents(db: Session = Depends(get_db)):
    """Lists registered document files in the repository."""
    docs = db.query(Document).all()
    return docs


@router.get("/documents/{id}")
async def get_document(id: int, db: Session = Depends(get_db)):
    """Retrieves document file metadata by ID."""
    doc = db.query(Document).filter(Document.id == id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@router.get("/versions")
async def trace_versions(regulation_code: Optional[str] = None, db: Session = Depends(get_db)):
    """Traces regulation version history, displaying active and superseded entries."""
    query = db.query(Regulation)
    if regulation_code:
        query = query.filter(Regulation.regulation_code == regulation_code)
        
    # Order by regulation code, effective date, and status
    results = query.order_by(Regulation.regulation_code, Regulation.effective_date.desc(), Regulation.status).all()
    
    # Group versions by regulation code
    history = {}
    for reg in results:
        code = reg.regulation_code
        if code not in history:
            history[code] = []
        history[code].append({
            "id": reg.id,
            "title": reg.title,
            "amendment": reg.amendment,
            "supplement": reg.supplement,
            "corrigendum": reg.corrigendum,
            "effective_date": str(reg.effective_date) if reg.effective_date else None,
            "status": reg.status,
            "checksum": reg.checksum
        })
    return history


@router.post("/search/timing")
async def search_timing(
    request: SearchRequest,
    db: Session = Depends(get_db),
    engine: RegulationSearchEngine = Depends(get_search_engine),
):
    """Return per-step timing breakdown for a query (same path as /search, timing-focused)."""
    try:
        results = engine.search(
            db=db,
            query=request.query,
            filters=request.filter,
            top_k=request.top_k or 5,
            rerank=request.rerank if request.rerank is not None else True,
        )
        return {
            "query": request.query,
            "timing": results.get("timing") or results.get("metadata", {}).get("timing"),
            "total_ms": (results.get("timing") or {}).get("total_ms"),
            "slowest_step": (results.get("timing") or {}).get("slowest_step"),
            "source_count": len(results.get("sources") or []),
        }
    except Exception as e:
        logger.error(f"Search timing endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Search timing failed: {str(e)}")


@router.post("/search")
async def search_registry(request: SearchRequest, db: Session = Depends(get_db), engine: RegulationSearchEngine = Depends(get_search_engine)):
    """Performs hybrid dense+sparse retrieval followed by cross-encoder reranking and optional answer synthesis."""
    try:
        results = engine.search(
            db=db, 
            query=request.query, 
            filters=request.filter, 
            top_k=request.top_k or 5, 
            rerank=request.rerank if request.rerank is not None else True
        )
        return results
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/reindex")
async def trigger_reindexing():
    """Asynchronously triggers vector database reindexing (regenerates embeddings)."""
    try:
        task = reindex_task.delay()
        return {
            "task_id": task.id,
            "status": "queued"
        }
    except Exception as e:
        logger.error(f"Failed to enqueue reindexing task: {e}")
        raise HTTPException(status_code=500, detail=f"Reindexing queuing failed: {str(e)}")


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Database, Redis broker, and Celery worker health (FR-4)."""
    health = {
        "status": "healthy",
        "database": "unknown",
        "redis": "unknown",
        "celery_worker": "unknown",
        "scheduler_mock_mode": os.getenv("CRAWL_MOCK", "false"),
    }

    try:
        db.execute(text("SELECT 1"))
        health["database"] = "up"
    except Exception as e:
        logger.error(f"Healthcheck: DB connection down: {e}")
        health["database"] = "down"
        health["status"] = "unhealthy"

    try:
        from scheduler.celery_app import celery_app

        celery_app.control.ping(timeout=2.0)
        health["redis"] = "up"
        health["celery_worker"] = "up"
    except Exception as e:
        logger.error(f"Healthcheck: Celery worker unavailable: {e}")
        health["redis"] = "down"
        health["celery_worker"] = "down"
        health["status"] = "unhealthy"
        health["worker_error"] = str(e)

    return health


@router.get("/coverage")
async def coverage_report(db: Session = Depends(get_db)):
    """FR-17: Europe passive-safety coverage gap report vs coverage_expected.yaml."""
    return build_coverage_report(db)


def _search_to_chat_response(query: str, results: dict[str, Any]) -> dict[str, Any]:
    routing = results.get("metadata", {}).get("routing") or {}
    evidence_only = bool(routing.get("evidence_only"))
    sources = results.get("sources") or []
    citations = []
    for idx, src in enumerate(sources, start=1):
        chunk_type = src.get("chunk_type") or ""
        source_kind = src.get("source_kind")
        if not source_kind:
            source_kind = "harness_test" if chunk_type == "test_record" else "regulation_corpus"
        citations.append(
            {
                "id": f"S{idx}",
                "source_kind": source_kind,
                "regulation_code": src.get("regulation_code"),
                "document_name": src.get("document_name"),
                "page_number": src.get("page_number"),
                "section": src.get("section") or src.get("heading_path"),
                "amendment": src.get("amendment"),
                "snippet": (src.get("chunk_text") or "")[:400],
                "test_id": src.get("test_id"),
                "upload_id": src.get("upload_id"),
                "confidential_tier": src.get("confidential_tier"),
            }
        )
    return {
        "query": query,
        "answer": results.get("answer") or "",
        "citations": citations,
        "sources": sources,
        "gateway": routing,
        "timing": results.get("timing") or {},
        "evidence_only": evidence_only,
        "generation_failed": evidence_only and bool(sources),
    }


@router.get("/ready")
async def readiness(
    db: Session = Depends(get_db),
    engine: RegulationSearchEngine = Depends(get_search_engine),
):
    """Self-test gate — frontend enables chat once `ready` is true."""
    global _ready_cache
    if _ready_cache and _ready_cache.get("ready"):
        return _ready_cache

    health = await health_check(db)
    if health.get("status") != "healthy":
        return {"ready": False, "reason": "health_unhealthy", "health": health}

    try:
        probe = engine.search(db=db, query="UN R94 frontal impact", top_k=3, rerank=False)
        source_count = len(probe.get("sources") or [])
        _ready_cache = {
            "ready": source_count > 0,
            "probe_sources": source_count,
            "chunks_in_corpus": db.execute(text("SELECT COUNT(*) FROM chunks")).scalar(),
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
        }
        return _ready_cache
    except Exception as exc:
        logger.error(f"Readiness probe failed: {exc}")
        return {"ready": False, "reason": str(exc)}


class HarnessQueryRequest(BaseModel):
    query: str
    model_key: Optional[str] = "groq"
    model_id: Optional[str] = "llama-3.3-70b-versatile"


@router.post("/harness/query")
async def query_harness(
    request: HarnessQueryRequest,
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Structured query endpoint for the harness MVP.
    Finds test results matching the query, verifies access, and returns them
    joined with the linked regulation clause side by side.
    """
    from registry.harness_security import check_harness_access
    
    query_upper = request.query.upper()
    
    # 1. Query matching test events
    test_query = db.query(Test)
    
    has_filter = False
    if "FRONTAL" in query_upper:
        test_query = test_query.filter(Test.impact_mode == "FRONTAL_OFFSET")
        has_filter = True
    elif "SIDE" in query_upper:
        test_query = test_query.filter(Test.impact_mode.in_(["SIDE_MDB", "POLE_SIDE"]))
        has_filter = True
        
    if "THCC" in query_upper:
        test_query = test_query.join(Test.results).filter(TestResult.injury_criterion == "ThCC")
        has_filter = True
    elif "HIC" in query_upper:
        test_query = test_query.join(Test.results).filter(TestResult.injury_criterion == "HIC36")
        has_filter = True
        
    m_test_id = re.search(r"\b(TEST-\d{4}-\d{2}-\d{3})\b", query_upper)
    if m_test_id:
        test_query = test_query.filter(Test.test_id == m_test_id.group(1))
        has_filter = True
        
    tests = test_query.all()
    from registry.harness_security import check_harness_access, filter_tests_for_user

    tests = filter_tests_for_user(tests, current_user.user_id)
    
    if not tests:
        return {
            "query": request.query,
            "results": [],
            "message": "No test records matched the query criteria."
        }
        
    # 2. Check authorization for all matched tests
    test_ids = [t.test_id for t in tests]
    check_harness_access(
        db=db,
        model_key=request.model_key,
        model_id=request.model_id,
        user_id=current_user.user_id,
        test_ids=test_ids
    )
    
    # 3. Retrieve linked regulation clauses and construct response
    results = []
    for t in tests:
        for tr in t.results:
            if "THCC" in query_upper and tr.injury_criterion != "ThCC":
                continue
            if "HIC" in query_upper and tr.injury_criterion != "HIC36":
                continue
                
            linked_clause = tr.linked_regulation_clause
            chunk_text = None
            chunk_section = None
            
            if linked_clause and "#" in linked_clause:
                reg_code, sec = linked_clause.split("#", 1)
                chunk = db.query(Chunk).join(Document).join(Regulation).filter(
                    Regulation.regulation_code == reg_code,
                    Chunk.section == sec
                ).first()
                if chunk:
                    chunk_text = chunk.chunk_text
                    chunk_section = chunk.section
                    
            results.append({
                "test_record": {
                    "test_id": t.test_id,
                    "program": t.program,
                    "date": str(t.date),
                    "test_type": t.test_type,
                    "impact_mode": t.impact_mode,
                    "dummy": t.dummy,
                    "injury_criterion": tr.injury_criterion,
                    "value": tr.value,
                    "pass_fail": tr.pass_fail,
                    "linked_regulation_clause": tr.linked_regulation_clause,
                    "confidential_tier": t.confidential_tier
                },
                "linked_clause": {
                    "clause_reference": tr.linked_regulation_clause,
                    "section": chunk_section,
                    "text": chunk_text
                }
            })
            
    return {
        "query": request.query,
        "results": results
    }


@router.post("/chat")
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    engine: RegulationSearchEngine = Depends(get_search_engine),
    current_user: AuthenticatedUser | None = Depends(get_optional_user),
):
    """Chat wrapper over hybrid search + grounded generation with harness fallback."""
    try:
        intent = classify_chat_query(request.query)
        if intent == "identity":
            return _search_to_chat_response(request.query, identity_chat_result(request.query))
        if intent == "injection_blocked":
            return _search_to_chat_response(request.query, injection_chat_result(request.query))
        if intent == "corpus_meta":
            return _search_to_chat_response(request.query, corpus_meta_chat_result(request.query, db))

        from registry.margin_query import margin_requires_harness_auth, try_margin_query_response

        if margin_requires_harness_auth(request.query) and current_user is None:
            raise HTTPException(
                status_code=401,
                detail="Login required for harness margin comparisons (confidential test data).",
            )

        margin_payload = try_margin_query_response(
            db,
            request.query,
            user_id=current_user.user_id if current_user else None,
        )
        if margin_payload:
            return _search_to_chat_response(request.query, margin_payload)

        query_upper = request.query.upper()
        if "TEST" in query_upper and ("RESULT" in query_upper or "VS" in query_upper or "LIMIT" in query_upper):
            primary_model = settings.DEFAULT_PRIMARY if hasattr(settings, "DEFAULT_PRIMARY") else "groq"

            test_query = db.query(Test)
            if "FRONTAL" in query_upper:
                test_query = test_query.filter(Test.impact_mode == "FRONTAL_OFFSET")
            elif "SIDE" in query_upper:
                test_query = test_query.filter(Test.impact_mode.in_(["SIDE_MDB", "POLE_SIDE"]))

            tests = test_query.all()

            if tests:
                from registry.harness_security import check_harness_access, filter_tests_for_user

                confidential = [t for t in tests if t.confidential_tier]
                if confidential and current_user is None:
                    raise HTTPException(
                        status_code=401,
                        detail="Login required to access confidential test data.",
                    )

                user_id = current_user.user_id if current_user else None
                accessible = filter_tests_for_user(tests, user_id)
                confidential_accessible = [t for t in accessible if t.confidential_tier]

                if confidential_accessible:
                    check_harness_access(
                        db=db,
                        model_key=primary_model,
                        model_id="",
                        user_id=user_id,
                        test_ids=[t.test_id for t in confidential_accessible],
                    )

                if accessible:
                    results = engine.search(
                        db=db,
                        query=request.query,
                        top_k=request.top_k or 8,
                        rerank=request.rerank if request.rerank is not None else True,
                    )

                    for t in accessible:
                        for tr in t.results:
                            if "THCC" in query_upper and tr.injury_criterion != "ThCC":
                                continue
                            if "HIC" in query_upper and tr.injury_criterion != "HIC36":
                                continue

                            results["sources"].append({
                                "chunk_id": 99999,
                                "source_kind": "harness_test",
                                "test_id": t.test_id,
                                "confidential_tier": t.confidential_tier,
                                "chunk_text": (
                                    f"CONFIDENTIAL TEST RESULT: Our physical test {t.test_id} for program {t.program} "
                                    f"measured a peak {tr.injury_criterion} value of {tr.value} mm, which is a {tr.pass_fail} "
                                    f"against the limits."
                                ),
                                "page_number": None,
                                "section": "Test Record",
                                "paragraph": None,
                                "document_id": 99999,
                                "document_name": f"{t.test_id}_report.json",
                                "file_path": f"/confidential/{t.test_id}",
                                "regulation_id": 99999,
                                "regulation_code": "TestRecord",
                                "title": f"Confidential Test {t.test_id}",
                                "amendment": t.setup_revision,
                                "effective_date": str(t.date),
                                "source_type": "INTERNAL",
                                "source_url": None,
                                "market": "GLOBAL",
                                "chunk_type": "test_record",
                                "heading_path": "Harness Data",
                                "parent_chunk_id": None,
                                "content_hash": t.test_id,
                                "search_score": 1.0
                            })

                    answer, routing = engine._generate_grounded_answer(request.query, results["sources"])
                    results["answer"] = answer
                    results["metadata"]["routing"] = routing

                    return _search_to_chat_response(request.query, results)
                
        results = engine.search(
            db=db,
            query=request.query,
            top_k=request.top_k or 8,
            rerank=request.rerank if request.rerank is not None else True,
        )
        return _search_to_chat_response(request.query, results)
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
