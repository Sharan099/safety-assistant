"""Mode-aware hard filters and soft boosts for retrieval."""

from __future__ import annotations

from typing import Any

from backend.app.core.modes import ModeConfig, get_mode


def chunk_passes_mode_filter(chunk: dict[str, Any], mode: ModeConfig) -> bool:
  """Hard exclude — returns False when chunk must not appear for this mode."""
  hf = mode.hard_filters
  doc_type = chunk.get("doc_type", "")
  if hf.get("doc_type") and doc_type not in hf["doc_type"]:
    return False
  if hf.get("is_synthetic") is False and chunk.get("is_synthetic"):
    return False
  if not mode.allow_synthetic and chunk.get("is_synthetic"):
    return False
  impact = chunk.get("impact_mode") or chunk.get("test_type")
  if hf.get("impact_mode") and impact not in hf["impact_mode"]:
    return False
  if hf.get("region") and chunk.get("region") not in hf["region"]:
    return False
  if hf.get("authority_tier"):
    tier = chunk.get("authority_tier")
    if not tier:
      from backend.app.core.authority_tier import chunk_authority_tier
      tier = chunk_authority_tier(chunk)
    if tier not in hf["authority_tier"]:
      return False
  return True


def mode_soft_boost(chunk: dict[str, Any], mode: ModeConfig, query: str = "") -> float:
  """Rank multiplier — never excludes."""
  boost = 1.0
  sb = mode.soft_boosts
  ct = chunk.get("chunk_type", "")
  if sb.get("chunk_type_table") and ct == "table":
    boost *= sb["chunk_type_table"]
  if sb.get("chunk_type_procedure_step") and ct == "procedure_step":
    boost *= sb["chunk_type_procedure_step"]
  if sb.get("has_loads") and chunk.get("has_loads"):
    boost *= sb["has_loads"]
  if sb.get("has_test_procedure") and chunk.get("has_test_procedure"):
    boost *= sb["has_test_procedure"]
  if sb.get("has_requirements") and chunk.get("has_requirements"):
    boost *= sb["has_requirements"]
  if sb.get("value_type_measured") and chunk.get("value_type") == "measured":
    boost *= sb["value_type_measured"]
  if sb.get("doc_type_internal") and chunk.get("doc_type") == "internal":
    boost *= sb["doc_type_internal"]
  if sb.get("vehicle_program_match"):
    prog = (chunk.get("vehicle_program") or "").upper()
    if prog and prog in query.upper():
      boost *= sb["vehicle_program_match"]
  return boost


def filter_docs_by_mode(docs: list[dict], chunk_by_id: dict, mode_name: str | None) -> list[dict]:
  mode = get_mode(mode_name)
  out = []
  for d in docs:
    chunk = chunk_by_id.get(d.get("id", ""), d)
    if chunk_passes_mode_filter(chunk, mode):
      out.append(d)
  return out
