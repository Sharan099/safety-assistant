"""Gold dataset schema + loader (evals/datasets/*.yaml)."""

from __future__ import annotations

import datetime
import pathlib
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Answerability = Literal["answerable", "partial", "unanswerable_out_of_scope", "unanswerable_not_in_corpus", "ambiguous"]
# synthetic: written by tooling/engineers from corpus inspection, not by the product owner and not
# by an LLM; human_reviewed stays False until the owner marks it.
Source = Literal["human", "llm_generated", "llm_generated_reviewed", "synthetic"]


class GoldCase(BaseModel):
    case_id: str
    query: str
    query_type: str
    difficulty: str = "medium"
    jurisdiction: str | None = None
    as_of_date: datetime.date | None = None
    expected_regulation_key: str | None = None
    expected_regulation_keys: list[str] = Field(default_factory=list)
    expected_version_label: str | None = None
    # Documents a chunk-only retriever confuses with the expected one (DRM benchmark); reporting only.
    hard_negative_regulation_keys: list[str] = Field(default_factory=list)
    expected_section_paths: list[str] = Field(default_factory=list)
    expected_section_prefixes: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    acceptable_citations: list[str] = Field(default_factory=list)
    answerability: Answerability = "answerable"
    review_status: str = "DRAFT"
    notes: str | None = None
    # Provenance. `source` is derived from review_status when omitted so older files stay valid:
    # REVIEWED → human, AUTO_GROUNDED → llm_generated. Never claim human review that did not happen.
    source: Source | None = None
    human_reviewed: bool | None = None
    reviewer_notes: str | None = None

    @model_validator(mode="after")
    def _derive_provenance(self) -> GoldCase:
        if self.source is None:
            self.source = "llm_generated" if self.review_status == "AUTO_GROUNDED" else "human"
        if self.human_reviewed is None:
            self.human_reviewed = self.review_status == "REVIEWED"
        return self

    @property
    def regulation_keys(self) -> set[str]:
        keys = set(self.expected_regulation_keys)
        if self.expected_regulation_key:
            keys.add(self.expected_regulation_key)
        return keys

    @property
    def has_section_truth(self) -> bool:
        return bool(self.expected_section_paths or self.expected_section_prefixes)

    def section_matches(self, path: str, merged_paths: list[str] | None = None) -> bool:
        candidates = {path, *(merged_paths or [])}
        if candidates & set(self.expected_section_paths):
            return True
        return any(c.startswith(p) for c in candidates for p in self.expected_section_prefixes)


class GoldDataset(BaseModel):
    dataset_version: str
    cases: list[GoldCase]

    def slices(self) -> dict[str, list[GoldCase]]:
        out: dict[str, list[GoldCase]] = {}
        for c in self.cases:
            out.setdefault(c.query_type, []).append(c)
        return out


def load_dataset(
    path: pathlib.Path,
    *,
    source: Source | None = None,
    query_types: list[str] | None = None,
    human_reviewed_only: bool = False,
) -> GoldDataset:
    """Load a gold set, optionally restricted to a provenance class or query types.
    Filtering keeps `dataset_version` so reports remain attributable."""
    with path.open("r", encoding="utf-8") as f:
        ds = GoldDataset.model_validate(yaml.safe_load(f))
    ids = [c.case_id for c in ds.cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case_id in dataset")
    cases = ds.cases
    if source:
        cases = [c for c in cases if c.source == source]
    if human_reviewed_only:
        cases = [c for c in cases if c.human_reviewed]
    if query_types:
        cases = [c for c in cases if c.query_type in query_types]
    return ds.model_copy(update={"cases": cases})
