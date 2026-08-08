"""GET /pdf, /citation, /regulations."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.catalog import lookup_citation, resolve_pdf
from retrieval.retrieve import get_indexed_regulations

router = APIRouter()


@router.get("/regulations")
def list_regulations():
    """Live catalog of indexed regulation_id + revision + chunk counts."""
    rows = get_indexed_regulations()
    return {
        "regulations": [
            {
                "regulation_id": r.regulation_id,
                "revision": r.revision,
                "chunk_count": r.chunk_count,
                "label": r.label,
            }
            for r in rows
        ],
        "count": len(rows),
    }


@router.get("/pdf/{regulation_id}")
def get_pdf(regulation_id: str):
    """Serve the source regulation PDF for the viewer pane."""
    try:
        path = resolve_pdf(regulation_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=path.name,
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )


@router.get("/citation/{chunk_id:path}")
def get_citation(chunk_id: str):
    """Return page_number + bounding_box for PDF highlighting."""
    data = lookup_citation(chunk_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Unknown chunk_id={chunk_id!r}")
    return data
