#!/usr/bin/env python3
"""
Generate SYNTHETIC internal documents for pilot program PROG_X.

All outputs land in data/corpus/synthetic/ as markdown with is_synthetic=true.
Delete this folder + re-run to swap in real data later.

Usage:
  python scripts/generate_synthetic_docs.py
  python scripts/generate_synthetic_docs.py --output data/corpus/synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SYNTHETIC_HEADER = (
    "> **SYNTHETIC TEST DATA — NOT A REAL TEST RESULT**\n"
    "> Program: PROG_X | is_synthetic: true | vehicle_program: PROG_X\n\n"
)

FRONTMATTER = """---
source_type: synthetic
is_synthetic: true
vehicle_program: PROG_X
doc_type: internal
authority: internal
region: EU
dummy_type: Hybrid III
regulation: {regulation}
source_pdf: {source_pdf}
---

"""


def _wrap(title: str, body: str, *, regulation: str, source_pdf: str) -> str:
    fm = FRONTMATTER.format(regulation=regulation, source_pdf=source_pdf)
    return f"{fm}{SYNTHETIC_HEADER}# {title}\n\n{body.strip()}\n"


def frontal_crash_test_01() -> str:
    return _wrap(
        "PROG_X Frontal Crash Test Report — FT-PROG-X-001",
        """
## Test identification
| Field | Value |
| --- | --- |
| test_id | FT-PROG-X-001 |
| impact_mode | frontal |
| dummy_type | Hybrid III 50th percentile male |
| vehicle_program | PROG_X |
| date | 2025-11-14 |

## Configuration
ODB 40 % offset deformable barrier, 56 km/h. Driver and passenger three-point belts.

## Measurements (fabricated)
| Channel | Measured | Unit |
| --- | --- | --- |
| Upper anchorage resultant (driver) | 1,218 | daN |
| Lower anchorage resultant (driver) | 1,305 | daN |
| Belt webbing force (shoulder) | 4.2 | kN |
| HIC15 (driver) | 412 | — |
| Chest deflection (driver) | 38.4 | mm |

## Regulatory assessment
| Requirement | Cited clause | Limit | Measured | Result |
| --- | --- | --- | --- | --- |
| Anchorage strength M1/N1 | UN R14 §6.4.2 | 1,350 ± 20 daN | 1,305 daN (lower) | **PASS** |
| Anchorage strength M1/N1 | UN R14 §6.4.2 | 1,350 ± 20 daN | 1,218 daN (upper) | **PASS** |
| Belt assembly performance | UN R16 §7.7 (dynamic) | per approval | no separation | **PASS** |

## Observation
All anchorage loads remained below UN R14 Rev.7 limits. Belt routing per UN R16 geometry checks satisfied.
""",
        regulation="PROG_X_FT_001",
        source_pdf="PROG_X_FT_001_frontal_crash_test.md",
    )


def frontal_crash_test_02() -> str:
    return _wrap(
        "PROG_X Frontal Crash Test Report — FT-PROG-X-002",
        """
## Test identification
| Field | Value |
| --- | --- |
| test_id | FT-PROG-X-002 |
| impact_mode | frontal |
| dummy_type | Hybrid III 50th percentile male |
| vehicle_program | PROG_X |
| date | 2026-01-22 |

## Configuration
Full-width rigid barrier, 50 km/h, driver only, pretensioner + load limiter active.

## Measurements (fabricated)
| Channel | Measured | Unit |
| --- | --- | --- |
| Lower anchorage resultant | 1,428 | daN |
| Upper anchorage resultant | 1,367 | daN |
| Lap belt force peak | 5.1 | kN |
| Chest deflection | 41.2 | mm |
| Femur force (left) | 3.8 | kN |

## Regulatory assessment
| Requirement | Cited clause | Limit | Measured | Result |
| --- | --- | --- | --- | --- |
| Lower anchorage M1 | UN R14 §6.4.1 | 1,350 ± 20 daN | 1,428 daN | **MARGINAL — within tolerance band** |
| Upper anchorage M1 | UN R14 §6.4.2 | 1,350 ± 20 daN | 1,367 daN | **PASS** |
| Dynamic belt test | UN R16 §7.7 | no release | no release | **PASS** |

## Observation
Lower anchorage approached upper tolerance; review bracket stiffness for next build.
""",
        regulation="PROG_X_FT_002",
        source_pdf="PROG_X_FT_002_frontal_crash_test.md",
    )


def cae_correlation_01() -> str:
    return _wrap(
        "PROG_X CAE Correlation Report — CAE-PROG-X-001 vs FT-PROG-X-001",
        """
## Correlation summary
| Field | Value |
| --- | --- |
| CAE_model_version | PROG_X_v2.3.1_20251110 |
| reference_test | FT-PROG-X-001 |
| vehicle_program | PROG_X |

## Delta table (sim vs test)
| Metric | Test (FT-PROG-X-001) | CAE (v2.3.1) | Delta | % |
| --- | --- | --- | --- | --- |
| Lower anchorage load | 1,305 daN | 1,278 daN | −27 daN | −2.1 % |
| Upper anchorage load | 1,218 daN | 1,241 daN | +23 daN | +1.9 % |
| Chest deflection | 38.4 mm | 36.9 mm | −1.5 mm | −3.9 % |
| HIC15 | 412 | 398 | −14 | −3.4 % |

## Conclusion
Model within ±5 % on primary restraint metrics. Acceptable for design iteration per internal gate.
""",
        regulation="PROG_X_CAE_001",
        source_pdf="PROG_X_CAE_001_correlation.md",
    )


def cae_correlation_02() -> str:
    return _wrap(
        "PROG_X CAE Correlation Report — CAE-PROG-X-002 vs FT-PROG-X-002",
        """
## Correlation summary
| Field | Value |
| --- | --- |
| CAE_model_version | PROG_X_v2.4.0_20260115 |
| reference_test | FT-PROG-X-002 |
| vehicle_program | PROG_X |

## Delta table (sim vs test)
| Metric | Test (FT-PROG-X-002) | CAE (v2.4.0) | Delta | % |
| --- | --- | --- | --- | --- |
| Lower anchorage load | 1,428 daN | 1,362 daN | −66 daN | −4.6 % |
| Upper anchorage load | 1,367 daN | 1,351 daN | −16 daN | −1.2 % |
| Chest deflection | 41.2 mm | 43.8 mm | +2.6 mm | +6.3 % |

## Conclusion
Lower anchorage under-predicted in CAE; stiffen B-pillar bracket mesh before sign-off.
""",
        regulation="PROG_X_CAE_002",
        source_pdf="PROG_X_CAE_002_correlation.md",
    )


def rca_report() -> str:
    return _wrap(
        "PROG_X Root Cause Analysis — RCA-PROG-X-001 Anchorage Margin",
        """
## Problem statement
FT-PROG-X-002 lower anchorage load 1,428 daN approached UN R14 §6.4.1 limit (1,350 ± 20 daN).

## Causal chain (fabricated)
1. **Observation**: Lower anchorage resultant 1,428 daN on FT-PROG-X-002 [test report FT-PROG-X-002].
2. **Evidence**: Bracket strain 4.2 % vs 2.8 % on FT-PROG-X-001; lap belt angle 41° vs 38°.
3. **Reasoning**: Increased bracket compliance redistributed load to lower anchorage per UN R14 load path.
4. **Conclusion**: Stiffen seat bracket (drawing BRK-PROG-X-12) to restore margin vs UN R14 §6.4.1.

## Linked sources
- Test: FT-PROG-X-002
- Regulation: UN R14 Rev.7 §6.4.1 (M1 lower anchorage 1,350 ± 20 daN)
""",
        regulation="PROG_X_RCA_001",
        source_pdf="PROG_X_RCA_001_anchorage.md",
    )


def design_review_minutes() -> str:
    return _wrap(
        "PROG_X Design Review Minutes — DR-PROG-X-2026-02",
        """
## Requirement traceability
| ID | Requirement | Cited clause | Status | Notes |
| --- | --- | --- | --- | --- |
| DR-01 | Anchorage strength M1 | UN R14 §6.4.2 | **Met** | FT-PROG-X-001 margin 45 daN |
| DR-02 | Anchorage strength M1 lower | UN R14 §6.4.1 | **Needs review** | FT-PROG-X-002 at 1,428 daN |
| DR-03 | Belt dynamic performance | UN R16 §7.7 | **Met** | No release both tests |
| DR-04 | Belt geometry | UN R16 §6.2 | **Met** | CAD check 2026-01-05 |
| DR-05 | Retractor endurance | UN R16 §6.2.2.4 | **Not met** | Lot B failed cycle 8,200 / 10,000 |
| DR-06 | ISOFIX anchorage (if fitted) | UN R14 Annex 3 | **N/A** | Not on PROG_X scope |

## Actions
- DR-02: Implement bracket stiffening per RCA-PROG-X-001.
- DR-05: Replace retractor lot B before SOP.
""",
        regulation="PROG_X_DR",
        source_pdf="PROG_X_DR_2026_02_minutes.md",
    )


def project_status() -> str:
    return _wrap(
        "PROG_X Project Status Summary — February 2026",
        """
## Executive overview
PROG_X restraint package is **conditionally on track** for EU homologation. Two frontal tests complete;
one anchorage margin item open; retractor lot issue blocks final sign-off.

## Key metrics (fabricated)
| Area | Status | Risk |
| --- | --- | --- |
| UN R14 anchorage | Amber | Lower anchorage margin on FT-PROG-X-002 |
| UN R16 belt approval | Green | Dynamic tests pass |
| CAE correlation | Green | v2.4.0 within internal ±5 % gate |
| Retractor endurance | Red | Lot B failure — replacement required |

## Milestones
- **Mar 2026**: Retest FT-PROG-X-003 with stiffened bracket
- **Apr 2026**: Submit UN R16 extension dossier
- **May 2026**: Management gate for series approval

## Management ask
Approve €42k for bracket tooling and expedited retractor re-qualification.
""",
        regulation="PROG_X_STATUS",
        source_pdf="PROG_X_status_2026_02.md",
    )


DOCUMENTS: dict[str, str] = {
    "PROG_X_FT_001_frontal_crash_test.md": frontal_crash_test_01(),
    "PROG_X_FT_002_frontal_crash_test.md": frontal_crash_test_02(),
    "PROG_X_CAE_001_correlation.md": cae_correlation_01(),
    "PROG_X_CAE_002_correlation.md": cae_correlation_02(),
    "PROG_X_RCA_001_anchorage.md": rca_report(),
    "PROG_X_DR_2026_02_minutes.md": design_review_minutes(),
    "PROG_X_status_2026_02.md": project_status(),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic PROG_X internal docs")
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "corpus" / "synthetic",
        help="Output directory for markdown files",
    )
    args = ap.parse_args()
    out: Path = args.output
    out.mkdir(parents=True, exist_ok=True)

    for name, content in DOCUMENTS.items():
        path = out / name
        path.write_text(content, encoding="utf-8")
        # Mirror into markdown dir for ingestion pipeline
        md_copy = ROOT / "output" / "markdown" / name
        md_copy.parent.mkdir(parents=True, exist_ok=True)
        md_copy.write_text(content, encoding="utf-8")
        print(f"Wrote {path.relative_to(ROOT)} (+ markdown mirror)")

    print(f"\nGenerated {len(DOCUMENTS)} synthetic documents in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
