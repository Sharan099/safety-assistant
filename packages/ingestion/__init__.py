"""Knowledge-source ingestion pipeline — TRD.md §13-18, ENVIRONMENT_SETUP.md §13.

Register -> SHA-256 -> extract (PyMuPDF) -> quality gate -> sections ->
chunks -> (embeddings: packages/retrieval). Tables/figures/equations and
OCR/VLM enrichment are NOT implemented in V1 — see `packages/ingestion/extract.py`
module docstring for why, rather than faking extraction we don't do.
"""
