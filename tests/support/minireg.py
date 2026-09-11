"""Synthetic UNECE-style regulation PDFs for tests.

Two consolidated versions of a fictional "UN Regulation No. 999":
- Rev.1 (01 series, in force 2019-01-01): 42 mm thorax limit;
- Rev.2 (02 series, in force 2022-01-01): 45 mm limit + one new clause 3.2.4.

Everything else is byte-for-byte the same text, so the v1→v2 update path
can prove changed-section-only re-embedding. Layout mirrors the real PDFs:
document-symbol running header, page number, clause numbers on their own
line, an Annex page whose numbering restarts.
"""

from __future__ import annotations

import datetime
import hashlib
import pathlib

import pymupdf

from safety_assistant.ingestion.sources.registry import SourceEntry, SourceRegistry, VersionInfo

REG_KEY = "UN-R999"

_COMMON_HEAD = [
    "Agreement",
    "Concerning the Adoption of Harmonized Technical United Nations Regulations",
    "Addendum 998: UN Regulation No. 999",
]


def _pages(revision: int) -> list[list[str]]:
    series = "01" if revision == 1 else "02"
    eif = "1 January 2019" if revision == 1 else "1 January 2022"
    limit = "42 mm" if revision == 1 else "45 mm"
    symbol = f"E/ECE/TRANS/505/Rev.3/Add.998/Rev.{revision}"
    cover = [
        *_COMMON_HEAD,
        f"Revision {revision}",
        "Incorporating all valid text up to:",
        f"{series} series of amendments - Date of entry into force: {eif}",
        "Uniform provisions concerning the approval of vehicles with regard to",
        "the protection of the occupants in the event of a synthetic collision",
        f"{symbol}",
        "2 March 2023" if revision == 2 else "3 March 2019",
    ]
    contents = [
        symbol,
        "2",
        "Contents",
        "1. Scope ........................................ 3",
        "2. Definitions .................................. 3",
        "3. Specifications ............................... 3",
        "Annex 3 Test procedure .......................... 4",
    ]
    body = [
        symbol,
        "3",
        "1.",
        "Scope",
        "This Regulation applies to vehicles of category M1 of a total permissible mass",
        "not exceeding 3,500 kg.",
        "2.",
        "Definitions",
        "2.1.",
        '"Protective system" means interior fittings and devices intended to restrain the',
        "occupants.",
        "2.2.",
        '"Synthetic barrier" means the deformable face described in Annex 3.',
        "3.",
        "Specifications",
        "3.1.",
        "All vehicles shall be tested in accordance with Annex 3, paragraph 1.2.",
        "3.2.",
        "Performance criteria",
        "3.2.1.",
        "The head performance criterion (HPC) shall not exceed 1,000.",
        "3.2.2.",
        f"The thorax compression criterion (ThCC) shall not exceed {limit}.",
        "3.2.3.",
        "The viscous criterion (V * C) for the thorax shall not exceed 1,0 m/s.",
    ]
    if revision == 2:
        body += ["3.2.4.", "The tibia index (TI) shall not exceed 1,3 at either location."]
    annex = [
        symbol,
        "Annex 3",
        "4",
        "Test procedure",
        "1.",
        "Installation and preparation of the vehicle",
        "1.1.",
        "The vehicle shall be at its normal attitude.",
        "1.2.",
        "The vehicle shall overlap the barrier face by 40 per cent +/- 20 mm.",
        "2.",
        "Dummies",
        "2.1.",
        "A Hybrid III 50th percentile male dummy shall be installed in each front outboard seat.",
    ]
    return [cover, contents, body, annex]


def build_pdf(path: pathlib.Path, revision: int) -> bytes:
    doc = pymupdf.open()
    for lines in _pages(revision):
        page = doc.new_page(width=595, height=842)
        y = 60
        for line in lines:
            page.insert_text((60, y), line, fontsize=10)
            y += 16
    data = doc.tobytes(deflate=True, garbage=0)
    doc.close()
    path.write_bytes(data)
    return data


def build_r998_pdf(path: pathlib.Path) -> bytes:
    """A second, single-version fictional regulation (lateral) for comparison routes."""
    pages = [
        [
            "Agreement",
            "Addendum 997: UN Regulation No. 998",
            "Revision 1",
            "Incorporating all valid text up to:",
            "01 series of amendments - Date of entry into force: 1 June 2020",
            "E/ECE/TRANS/505/Rev.3/Add.997/Rev.1",
            "4 June 2020",
        ],
        [
            "E/ECE/TRANS/505/Rev.3/Add.997/Rev.1",
            "2",
            "1.",
            "Scope",
            "This Regulation applies to vehicles of category M1 in a lateral collision.",
            "2.",
            "Definitions",
            "2.1.",
            '"R point" means the seating reference point specified by the manufacturer.',
            "3.",
            "Specifications",
            "3.1.",
            "Performance criteria",
            "3.1.1.",
            "The head performance criterion (HPC) shall be less than or equal to 1,000.",
            "3.1.2.",
            "The rib deflection criterion (RDC) shall be less than or equal to 42 mm.",
            "3.1.3.",
            "The pubic symphysis peak force (PSPF) shall be less than or equal to 6 kN.",
        ],
    ]
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page(width=595, height=842)
        y = 60
        for line in lines:
            page.insert_text((60, y), line, fontsize=10)
            y += 16
    data = doc.tobytes(deflate=True, garbage=0)
    doc.close()
    path.write_bytes(data)
    return data


def registry_for(
    tmp: pathlib.Path, revisions: tuple[int, ...] = (1, 2), *, include_r998: bool = False
) -> SourceRegistry:
    entries = []
    if include_r998:
        p = tmp / "UN_R998_rev1.pdf"
        data = build_r998_pdf(p)
        entries.append(
            SourceEntry(
                source_key="test-un-r998-rev1",
                regulation_key="UN-R998",
                kind="REGULATION",
                title="UN Regulation No. 998 — Synthetic lateral collision protection (test fixture)",
                authority="UNECE",
                jurisdiction="UNECE-1958-AGREEMENT",
                authority_level="AUTHORITATIVE",
                publisher="test",
                source_uri="https://example.invalid/r998",
                source_uri_status="TEST",
                local_path=p.name,
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
                license="TEST_FIXTURE",
                version=VersionInfo(
                    label="Rev.1 (01 series)",
                    series="01",
                    revision="Rev.1",
                    document_symbol="E/ECE/TRANS/505/Rev.3/Add.997/Rev.1",
                    published_at=datetime.date(2020, 6, 4),
                    valid_from=datetime.date(2020, 6, 1),
                ),
            )
        )
    for rev in revisions:
        p = tmp / f"UN_R999_rev{rev}.pdf"
        data = build_pdf(p, rev)
        entries.append(
            SourceEntry(
                source_key=f"test-un-r999-rev{rev}",
                regulation_key=REG_KEY,
                kind="REGULATION",
                title="UN Regulation No. 999 — Synthetic collision protection (test fixture)",
                authority="UNECE",
                jurisdiction="UNECE-1958-AGREEMENT",
                authority_level="AUTHORITATIVE",
                publisher="test",
                source_uri="https://example.invalid/r999",
                source_uri_status="TEST",
                local_path=p.name,
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
                license="TEST_FIXTURE",
                version=VersionInfo(
                    label=f"Rev.{rev} (0{rev} series)",
                    series=f"0{rev}",
                    revision=f"Rev.{rev}",
                    document_symbol=f"E/ECE/TRANS/505/Rev.3/Add.998/Rev.{rev}",
                    published_at=datetime.date(2019 if rev == 1 else 2023, 3, 2),
                    valid_from=datetime.date(2019 if rev == 1 else 2022, 1, 1),
                ),
            )
        )
    return SourceRegistry(schema_version=2, sources=entries)
