"""Build `knowledge/00_registry/sources.yaml` from the curated source manifest below.

    uv run python scripts/maintenance/build_registry.py [--source-dir "Knowledge source"]

For every manifest entry the script: merges split scans into one PDF where needed, moves the
file to its descriptive name under knowledge/, computes size + SHA-256, reads the UNECE cover
page (document symbol, revision, series, dates) and writes the registry. Titles, keys, kinds
and authority levels are the human-reviewed part of the manifest; cover fields are parsed and
never guessed (absent = null). Re-running against knowledge/ is a no-op.

Curation rules applied to the 2026-09-14 delivery (52 files → 42 sources):
- exact duplicates (same bytes) kept once: R16, R17, R129, R135, R137;
- two consolidated texts of one regulation: only the newest kept (R34 Rev.4 replaces Rev.3);
- an amendment sheet already incorporated in the consolidated text is dropped (R95 Rev.2/Amend.5 ⊂ Rev.3);
- amendment sheets newer than the consolidated text are separate sources (R11, R25, R32, R33, R42, R94):
  they supplement the full text and must not supersede it;
- scanned split texts are merged (R21, R32, R33 base) and need OCR_PROVIDER=tesseract.
"""

# ruff: noqa: E501 — the manifest is a table; one source per line reads better than wrapped dicts
# mypy: ignore-errors
# (pymupdf has no type stubs)
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import pathlib
import re
import shutil
import sys

import pymupdf
import yaml

from safety_assistant.ingestion.normalize.cover import _parse_date, parse_cover

UNECE = dict(authority="UNECE", jurisdiction="UNECE-1958-AGREEMENT", authority_level="AUTHORITATIVE", publisher="UNECE",
             license="PUBLIC_OFFICIAL_DOCUMENT", source_uri="https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations")  # fmt: skip
NCAP = dict(authority="Euro NCAP", jurisdiction="EURO-NCAP", authority_level="OFFICIAL_DOCUMENTATION", publisher="Euro NCAP",
            license="EURO_NCAP_COPYRIGHT_NO_REPUBLISH", source_uri="https://www.euroncap.com/en/for-engineers/protocols/")  # fmt: skip
ANSYS = dict(authority="ANSYS-LST", jurisdiction="VENDOR", authority_level="REFERENCE", publisher="ANSYS / Synopsys", license="VENDOR_DOCUMENTATION")  # fmt: skip
GNS = dict(authority="GNS mbH", jurisdiction="VENDOR", authority_level="REFERENCE", publisher="GNS mbH", license="VENDOR_DOCUMENTATION")  # fmt: skip


@dataclasses.dataclass(frozen=True)
class Src:
    files: tuple[str, ...]  # original file name(s); several = merged in this order
    key: str
    kind: str
    title: str
    folder: str
    name: str  # descriptive file name (without .pdf)
    org: dict[str, str]
    label: str | None = None  # version label when not readable from a UNECE cover
    published: str | None = None
    valid_from: str | None = None


def un(
    files: str | tuple[str, ...],
    no: str,
    subject: str,
    name: str,
    *,
    key_suffix: str = "",
    kind_note: str = "",
    label: str | None = None,
) -> Src:
    files = (files,) if isinstance(files, str) else files
    key = f"UN-R{no}{key_suffix}"
    title = f"UN Regulation No. {no} — {subject}" + (f" ({kind_note})" if kind_note else "")
    return Src(files, key, "REGULATION", title, "unece", name, UNECE, label=label)


MANIFEST: list[Src] = [
    # ---- UNECE consolidated texts (newest revision available)
    un("UN_R11_base.pdf", "11", "Door latches and door retention components", "UN_R11_Rev3_04series_2015_door_latches"),
    un(
        "UN_R11_03Series.pdf",
        "11",
        "Door latches and door retention components",
        "UN_R11_Amend2_Suppl2_to_04series_2019_amendment_sheet",
        key_suffix="-SUPPL2-04",
        kind_note="Supplement 2 to the 04 series, amendment sheet",
    ),
    un(
        "UN_R12_base.pdf",
        "12",
        "Protection of the driver against the steering mechanism in the event of impact",
        "UN_R12_Rev4_04series_2012_steering_mechanism_impact",
    ),
    un(
        "UN_R14_base.pdf",
        "14",
        "Safety-belt anchorages, ISOFIX anchorages systems and ISOFIX top tether anchorages",
        "UN_R14_Rev7_09series_2023_safety_belt_anchorages",
    ),
    un(
        "UN_R16_base.pdf",
        "16",
        "Safety-belts, restraint systems, child restraint systems and ISOFIX child restraint systems",
        "UN_R16_Rev7_06series_2012_safety_belts_restraint_systems",
        label="Rev.7 (06 series, Suppl. 2)",
    ),
    un(
        "UN_R17_base.pdf",
        "17",
        "Seats, their anchorages and any head restraints",
        "UN_R17_Rev7_10series_2023_seats_anchorages_head_restraints",
    ),
    un(
        ("UN_R21_base.pdf", "UN_R21_base_part2.pdf"),
        "21",
        "Interior fittings",
        "UN_R21_consolidated_scanned_interior_fittings",
    ),
    un(
        "UN_R25_base.pdf",
        "25",
        "Head restraints (headrests), whether or not incorporated in vehicle seats",
        "UN_R25_Rev1_03series_1990_head_restraints",
    ),
    un(
        "UN_R25_04Series.pdf",
        "25",
        "Head restraints (headrests), whether or not incorporated in vehicle seats",
        "UN_R25_Amend4_Suppl2_to_04series_2025_amendment_sheet",
        key_suffix="-SUPPL2-04",
        kind_note="Supplement 2 to the 04 series, amendment sheet",
    ),
    un(
        "UN_R29_base.pdf",
        "29",
        "Protection of the occupants of the cab of a commercial vehicle",
        "UN_R29_Rev2_03series_2012_commercial_vehicle_cab",
    ),
    un(
        ("UN_R32_base_part1.pdf", "UN_R32_base_part2.pdf"),
        "32",
        "Behaviour of the structure of the impacted vehicle in a rear-end collision",
        "UN_R32_consolidated_scanned_rear_end_collision_structure",
    ),
    un(
        "UN_R32_02Series.pdf",
        "32",
        "Behaviour of the structure of the impacted vehicle in a rear-end collision",
        "UN_R32_Amend2_Suppl2_2025_amendment_sheet",
        key_suffix="-SUPPL2",
        kind_note="Supplement 2 to the original version, amendment sheet",
    ),
    un(
        ("UN_R33_base_part1.pdf", "UN_R33_base_part2.pdf"),
        "33",
        "Behaviour of the structure of the impacted vehicle in a head-on collision",
        "UN_R33_consolidated_scanned_head_on_collision_structure",
    ),
    un(
        "UN_R33_03Series.pdf",
        "33",
        "Behaviour of the structure of the impacted vehicle in a head-on collision",
        "UN_R33_Amend3_Suppl3_2025_amendment_sheet",
        key_suffix="-SUPPL3",
        kind_note="Supplement 3 to the original version, amendment sheet",
    ),
    un(
        "UN_R34_base.pdf",
        "34",
        "Prevention of fire risks (fuel tanks, fuel system, electric power train)",
        "UN_R34_Rev4_04series_2023_fire_risks",
    ),
    un("UN_R42_base.pdf", "42", "Front and rear protective devices (bumpers, etc.)", "UN_R42_original_1980_bumpers"),
    un(
        "UN_R42_02Series.pdf",
        "42",
        "Front and rear protective devices (bumpers, etc.)",
        "UN_R42_Amend2_Suppl2_2021_amendment_sheet",
        key_suffix="-SUPPL2",
        kind_note="Supplement 2 to the original version, amendment sheet",
    ),
    un(
        "UN_R44_base.pdf",
        "44",
        "Restraining devices for child occupants of power-driven vehicles (child restraint systems)",
        "UN_R44_Rev3_04series_2013_child_restraint_systems",
    ),
    un(
        "UN_R94.pdf",
        "94",
        "Protection of the occupants in the event of a frontal collision",
        "UN_R94_Rev4_04series_2021_frontal_collision",
    ),
    un(
        "UN_R94_04Series.pdf",
        "94",
        "Protection of the occupants in the event of a frontal collision",
        "UN_R94_Amend3_05series_2024_amendment_sheet",
        key_suffix="-AMEND-05",
        kind_note="05 series of amendments, amendment sheet",
    ),
    un(
        "UN_R95_base.pdf",
        "95",
        "Protection of the occupants in the event of a lateral collision",
        "UN_R95_Rev3_04series_2021_lateral_collision",
    ),
    un(
        "UN_R100_base.pdf",
        "100",
        "Approval of vehicles with regard to specific requirements for the electric power train",
        "UN_R100_Rev1_01series_2010_electric_power_train",
    ),
    un("UN_R127_base.pdf", "127", "Pedestrian safety performance", "UN_R127_Rev2_02series_2016_pedestrian_safety"),
    un(
        "UN_R129_base.pdf",
        "129",
        "Enhanced Child Restraint Systems used on board of motor vehicles (i-Size)",
        "UN_R129_Rev3_02series_2018_enhanced_child_restraint_systems",
        label="Rev.3 (02 series, Suppl. 2)",
    ),
    un("UN_R135_base.pdf", "135", "Pole side impact performance", "UN_R135_original_2015_pole_side_impact"),
    un(
        "UN_R137_base.pdf",
        "137",
        "Frontal collision with focus on the restraint system (full-width frontal impact)",
        "UN_R137_original_2016_full_width_frontal_collision",
    ),
    un(
        "UN_R153_base.pdf",
        "153",
        "Fuel system integrity and safety of electric power train in the event of a rear-end collision",
        "UN_R153_original_2021_rear_end_collision_fuel_system",
    ),
    # ---- other regulations / protocols / standards
    Src(
        ("FMVSS_208.pdf.pdf",),
        "US-49CFR571",
        "REGULATION",
        "49 CFR Part 571 — Federal Motor Vehicle Safety Standards (complete part incl. FMVSS 208 occupant crash protection), as of 2026-05-07",
        "us_fmvss",
        "US_49CFR571_FMVSS_all_standards_2026-05-07",
        dict(
            authority="NHTSA",
            jurisdiction="US-FMVSS",
            authority_level="AUTHORITATIVE",
            publisher="US GPO / eCFR",
            license="US_GOVERNMENT_WORK",
            source_uri="https://www.ecfr.gov/current/title-49/subtitle-B/chapter-V/part-571",
        ),
        label="eCFR 2026-05-07",
        published="2026-05-07",
        valid_from="2026-05-07",
    ),
    Src(
        ("EURO_NCAP_FRONTAL.pdf.pdf",),
        "EURONCAP-FRONTAL",
        "STANDARD",
        "Euro NCAP Crash Protection — Frontal Impact Testing Protocol v1.1 (September 2025, implementation January 2026)",
        "euro_ncap",
        "EuroNCAP_Frontal_Impact_Protocol_v1.1_2025-09",
        NCAP,
        label="v1.1 (2025-09)",
        published="2025-09-01",
        valid_from="2026-01-01",
    ),
    Src(
        ("EURO_NCAP_SIDE.pdf.pdf",),
        "EURONCAP-SIDE",
        "STANDARD",
        "Euro NCAP Crash Protection — Side Impact Testing Protocol v1.1 (October 2025, implementation January 2026)",
        "euro_ncap",
        "EuroNCAP_Side_Impact_Protocol_v1.1_2025-10",
        NCAP,
        label="v1.1 (2025-10)",
        published="2025-10-01",
        valid_from="2026-01-01",
    ),
    Src(
        ("EURO_NCAP_REAR.pdf.pdf",),
        "EURONCAP-REAR",
        "STANDARD",
        "Euro NCAP Crash Protection — Rear Impact (Whiplash) Testing Protocol v1.1 (September 2025, implementation January 2026)",
        "euro_ncap",
        "EuroNCAP_Rear_Impact_Protocol_v1.1_2025-09",
        NCAP,
        label="v1.1 (2025-09)",
        published="2025-09-01",
        valid_from="2026-01-01",
    ),
    Src(
        ("EURO_NCAP_VRU.pdf.pdf",),
        "EURONCAP-VRU",
        "STANDARD",
        "Euro NCAP Crash Protection — Vulnerable Road User Impacts Testing Protocol v1.1 (September 2025, implementation January 2026)",
        "euro_ncap",
        "EuroNCAP_VRU_Impacts_Protocol_v1.1_2025-09",
        NCAP,
        label="v1.1 (2025-09)",
        published="2025-09-01",
        valid_from="2026-01-01",
    ),
    Src(
        ("ISO_26262.pdf.pdf",),
        "ISO-26262-OVERVIEW",
        "TECHNICAL_REPORT",
        "ISO 26262:2018 Road vehicles — Functional safety: overview article (ISO Library web page; NOT the standard's text)",
        "standards",
        "ISO_26262_2018_functional_safety_overview_article",
        dict(
            authority="ISO Library (third-party)",
            jurisdiction="ISO",
            authority_level="REFERENCE",
            publisher="ISO Library web page",
            license="THIRD_PARTY_WEB_PAGE",
        ),
        label="web capture 2026",
        published=None,
        valid_from=None,
    ),
    # ---- CAE manuals and reference handbooks
    Src(
        ("LS-DYNA_Manual_Theory_R17.pdf",),
        "LSDYNA-R17-THEORY",
        "MANUAL",
        "LS-DYNA Theory Manual R17",
        "cae_manuals",
        "LS-DYNA_R17_Theory_Manual",
        ANSYS,
        label="R17",
    ),
    Src(
        ("LS-DYNA_Manual_Vol_I_R17.pdf",),
        "LSDYNA-R17-VOL-I",
        "MANUAL",
        "LS-DYNA Keyword User's Manual Vol. I R17",
        "cae_manuals",
        "LS-DYNA_R17_Keyword_Manual_Vol_I",
        ANSYS,
        label="R17",
    ),
    Src(
        ("LS-DYNA_Manual_Vol_II_R17.pdf",),
        "LSDYNA-R17-VOL-II",
        "MANUAL",
        "LS-DYNA Keyword User's Manual Vol. II (Material Models) R17",
        "cae_manuals",
        "LS-DYNA_R17_Keyword_Manual_Vol_II_Material_Models",
        ANSYS,
        label="R17",
    ),
    Src(
        ("LS-DYNA_Manual_Vol_III_R17.pdf",),
        "LSDYNA-R17-VOL-III",
        "MANUAL",
        "LS-DYNA Keyword User's Manual Vol. III (Multi-Physics) R17",
        "cae_manuals",
        "LS-DYNA_R17_Keyword_Manual_Vol_III_Multiphysics",
        ANSYS,
        label="R17",
    ),
    Src(
        ("LS-DYNA_Users_Guide.pdf",),
        "LSDYNA-USERS-GUIDE",
        "MANUAL",
        "LS-DYNA User's Guide (Ansys Workbench) Release 2025 R2",
        "cae_manuals",
        "LS-DYNA_Users_Guide_2025_R2",
        ANSYS,
        label="2025 R2",
        published="2025-07-01",
    ),
    Src(
        ("ls-dyna-examples-manual.pdf",),
        "LSDYNA-EXAMPLES",
        "MANUAL",
        "LS-DYNA Examples Manual (LSTC, March 1998)",
        "cae_manuals",
        "LS-DYNA_Examples_Manual_1998",
        ANSYS,
        label="1998-03",
        published="1998-03-01",
    ),
    Src(
        ("cc-PAM-Crash-Spec-Sheet.pdf",),
        "PAMCRASH-ISIGHT-SPEC",
        "MANUAL",
        "Isight PAM-CRASH component specification sheet (ESI PAM-CRASH interface)",
        "cae_manuals",
        "PAM-CRASH_Isight_Component_Spec_Sheet",
        dict(
            authority="ESI / Dassault Systèmes",
            jurisdiction="VENDOR",
            authority_level="REFERENCE",
            publisher="Dassault Systèmes",
            license="VENDOR_DOCUMENTATION",
        ),
        label="undated",
    ),
    Src(
        ("SAFETY_COMPANION.pdf.pdf",),
        "GNS-SAFETY-COMPANION-2026",
        "TECHNICAL_REPORT",
        "GNS SafetyCompanion 2026 v1.6 — passive-safety load-case and regulation compendium (vendor reference)",
        "reference",
        "GNS_SafetyCompanion_2026_v1.6",
        GNS,
        label="2026 v1.6",
    ),
    Src(
        ("CAE_COMPANION.pdf.pdf",),
        "GNS-CAE-COMPANION-2026",
        "TECHNICAL_REPORT",
        "GNS CAE Companion 2026/2027 — CAE methods and load-case compendium (vendor reference)",
        "reference",
        "GNS_CAE_Companion_2026-2027",
        GNS,
        label="2026/2027",
    ),
]

DROPPED = {  # original file → why it is not a source (recorded here, not silently)
    "UN_R16.pdf": "byte-identical duplicate of UN_R16_base.pdf",
    "UN_R17.pdf": "byte-identical duplicate of UN_R17_base.pdf",
    "UN_R129.pdf": "byte-identical duplicate of UN_R129_base.pdf",
    "UN_R135.pdf": "byte-identical duplicate of UN_R135_base.pdf",
    "UN_R137.pdf": "byte-identical duplicate of UN_R137_base.pdf",
    "UN_R34_03Series.pdf": "older consolidated text (Rev.3, 2015); Rev.4 (2023) kept",
    "UN_R95.pdf": "Rev.2/Amend.5 amendment sheet (04 series) already incorporated in UN_R95_base.pdf (Rev.3)",
}

_SERIES_RE = re.compile(r"(\d{2})\s+series", re.IGNORECASE)
# Cover phrasings the strict parser skips ("…: Date of entry into force: 15 June 2015", "as an annex
# to the 1958 Agreement: 9 June 2016"); the latest such date is the text's entry into force.
_EIF_RE = re.compile(r"entry into force.{0,60}?(\d{1,2}\s+[A-Z][a-z]+\s+(?:19|20)\d\d)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", default="Knowledge source")
    ap.add_argument("--out", default="knowledge/00_registry/sources.yaml")
    args = ap.parse_args(argv)
    src_dir, root = pathlib.Path(args.source_dir), pathlib.Path("knowledge")
    entries = []
    for s in MANIFEST:
        target = root / s.folder / f"{s.name}.pdf"
        if not target.exists():
            _place(src_dir, s, target)
        data = target.read_bytes()
        version: dict[str, object] = {"label": s.label, "published_at": s.published, "valid_from": s.valid_from}
        if s.org is UNECE:
            version.update(_unece_version(data, s))
        entries.append(
            {
                "source_key": s.key.lower().replace("_", "-"),
                "regulation_key": s.key,
                "kind": s.kind,
                "title": s.title,
                **{k: s.org[k] for k in ("authority", "jurisdiction", "authority_level", "publisher")},
                "data_class": "PUBLIC",
                "source_uri": s.org.get("source_uri"),
                "source_uri_status": "LANDING_PAGE_UNVERIFIED" if s.org.get("source_uri") else "NONE",
                "local_path": target.as_posix(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
                "media_type": "application/pdf",
                "license": s.org["license"],
                "version": {k: v for k, v in version.items() if v is not None},
            }
        )
        print(f"{s.key:<28} {target.name}  {version.get('label')}")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Source registry — the ONLY sources ingestion may touch (allowlist). schema_version 2.\n"
        "# Generated by scripts/maintenance/build_registry.py from the curated manifest there; titles, keys and\n"
        "# authority levels are human-reviewed, cover fields (symbol, revision, series, dates) are parsed from\n"
        "# the PDF cover page and left null when absent. Edit the manifest, not this file.\n"
    )
    out.write_text(
        header
        + yaml.safe_dump({"schema_version": 2, "sources": entries}, sort_keys=False, allow_unicode=True, width=200),
        encoding="utf-8",
    )
    leftovers = sorted(p.name for p in src_dir.glob("*.pdf")) if src_dir.exists() else []
    print(f"written {out} ({len(entries)} sources); dropped {len(DROPPED)}; left in {src_dir}: {leftovers}")
    return 0


def _place(src_dir: pathlib.Path, s: Src, target: pathlib.Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if len(s.files) == 1:
        shutil.move(src_dir / s.files[0], target)
        return
    merged = pymupdf.open()
    for f in s.files:
        with pymupdf.open(src_dir / f) as part:
            merged.insert_pdf(part)
    merged.save(target, deflate=True, garbage=3)
    merged.close()
    for f in s.files:
        (src_dir / f).unlink()


def _unece_version(data: bytes, s: Src) -> dict[str, object]:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        pages = [str(doc[i].get_text()) for i in range(min(3, len(doc)))]
    text = "\n".join(pages)
    cover = parse_cover(pages)
    series = cover.latest_series
    if series is None:
        m = _SERIES_RE.search(s.name.replace("series", " series"))
        series = m.group(1) if m else None
    # Amendment sheets state "Revision N - Amendment M" (the consolidated text they amend); consolidated
    # texts state "Revision N" on its own line — which pymupdf may join with neighbours, so the manifest
    # name (typed from the cover) is the fallback and the parser's reading the cross-check.
    sheet = re.search(r"Revision\s+(\d+)\s*-\s*Amendment\s+(\d+)", text)
    named = re.search(r"_Rev(\d+)_", s.name)
    revision = f"Rev.{sheet.group(1)}" if sheet else cover.revision or (f"Rev.{named.group(1)}" if named else None)
    base = revision or "original"
    label = f"{base} ({series} series)" if series else base
    if sheet:
        label = f"{base} Amend.{sheet.group(2)}" + (f" ({series} series)" if series else "")
    eif = cover.latest_entry_into_force
    if eif is None:
        found = [_parse_date(m.group(1)) for m in _EIF_RE.finditer(" ".join(text.split()))]
        eif = max((d for d in found if d), default=None)
    return {
        "label": s.label or label,
        "series": series,
        "revision": revision,
        "document_symbol": cover.document_symbol,
        "published_at": cover.document_date.isoformat() if cover.document_date else s.published,
        "valid_from": eif.isoformat() if eif else s.valid_from,
    }


if __name__ == "__main__":
    sys.exit(main())
