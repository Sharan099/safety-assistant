"""
Regulation Catalog & Source Configuration
==========================================
Central registry of all regulations with their official source URLs,
version metadata, and download strategy.

Download strategies:
  REAL   — Direct PDF from official server (confirmed working)
  SCRAPE — Parse HTML page for embedded PDF links
  SYNTHETIC — Generate rich structured fallback PDF (source blocked/JS-gated)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path

# ─── Storage Root ────────────────────────────────────────────────────────────
STORAGE_ROOT = Path(__file__).resolve().parents[1] / "storage"

# ─── Download Strategy ────────────────────────────────────────────────────────
class DownloadStrategy(str, Enum):
    REAL      = "REAL"       # Try direct HTTP download
    SCRAPE    = "SCRAPE"     # Scrape HTML page for PDF links
    SYNTHETIC = "SYNTHETIC"  # Generate rich synthetic fallback


class DocumentType(str, Enum):
    REAL      = "REAL"       # Successfully downloaded from official source
    SYNTHETIC = "SYNTHETIC"  # Locally generated — source blocked or unavailable


class RegulationStatus(str, Enum):
    ACTIVE     = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    DRAFT      = "DRAFT"
    WITHDRAWN  = "WITHDRAWN"


class RegulationSource(str, Enum):
    UNECE     = "UNECE"
    EURO_NCAP = "EURO_NCAP"
    FMVSS     = "FMVSS"
    NHTSA     = "NHTSA"
    IIHS      = "IIHS"
    ISO       = "ISO"
    SAE       = "SAE"


# ─── Regulation Entry ─────────────────────────────────────────────────────────
@dataclass
class RegulationEntry:
    """Single regulation in the catalog."""
    regulation_id: str               # Canonical ID: "R94", "FMVSS_208", "EURONCAP_AOP"
    title: str                       # Human-readable title
    source: RegulationSource         # Issuing body
    category: str                    # Safety domain
    series: str                      # "04 Series", "Rev.4", "2026"
    status: RegulationStatus         # Current status
    publication_date: str            # ISO date string "YYYY-MM-DD"
    effective_date: str              # ISO date string "YYYY-MM-DD"
    market: str                      # "GLOBAL", "EU", "US", "US+EU"
    primary_urls: list[str]          # Ordered list: try first → last
    scrape_page: Optional[str]       # Page to scrape for embedded links
    storage_subpath: str             # Relative path under STORAGE_ROOT
    filename: str                    # Output filename
    strategy: DownloadStrategy       # Primary download strategy
    supersedes: Optional[str] = None # regulation_id of superseded version
    description: str = ""            # What this regulation covers


# ─── Regulation Catalog ────────────────────────────────────────────────────────
REGULATION_CATALOG: list[RegulationEntry] = [

    # ═══════════════════════════════════════════════════════════
    # GROUP A: FMVSS — Real Downloads via govinfo.gov ✅
    # ═══════════════════════════════════════════════════════════

    RegulationEntry(
        regulation_id="FMVSS_201",
        title="FMVSS 201 — Occupant Protection in Interior Impact",
        source=RegulationSource.FMVSS,
        category="Occupant Protection",
        series="2024 CFR",
        status=RegulationStatus.ACTIVE,
        publication_date="2024-01-01",
        effective_date="2024-01-01",
        market="US",
        primary_urls=[
            "https://www.govinfo.gov/content/pkg/CFR-2024-title49-vol6/pdf/CFR-2024-title49-vol6-sec571-201.pdf",
            "https://www.govinfo.gov/content/pkg/CFR-2023-title49-vol6/pdf/CFR-2023-title49-vol6-sec571-201.pdf",
        ],
        scrape_page=None,
        storage_subpath="FMVSS/FMVSS_201",
        filename="FMVSS_201_2024.pdf",
        strategy=DownloadStrategy.REAL,
        description="Head impact protection from interior surfaces (padding, A-pillar, roof, etc.)"
    ),

    RegulationEntry(
        regulation_id="FMVSS_202A",
        title="FMVSS 202a — Head Restraints",
        source=RegulationSource.FMVSS,
        category="Occupant Protection",
        series="2024 CFR",
        status=RegulationStatus.ACTIVE,
        publication_date="2024-01-01",
        effective_date="2024-01-01",
        market="US",
        primary_urls=[
            "https://www.govinfo.gov/content/pkg/CFR-2024-title49-vol6/pdf/CFR-2024-title49-vol6-sec571-202a.pdf",
            "https://www.govinfo.gov/content/pkg/CFR-2023-title49-vol6/pdf/CFR-2023-title49-vol6-sec571-202a.pdf",
        ],
        scrape_page=None,
        storage_subpath="FMVSS/FMVSS_202A",
        filename="FMVSS_202A_2024.pdf",
        strategy=DownloadStrategy.REAL,
        description="Head restraint geometry, strength, and HIT (Head Injury Test) requirements"
    ),

    RegulationEntry(
        regulation_id="FMVSS_208",
        title="FMVSS 208 — Occupant Crash Protection",
        source=RegulationSource.FMVSS,
        category="Frontal Crash / Airbags",
        series="2024 CFR",
        status=RegulationStatus.ACTIVE,
        publication_date="2024-01-01",
        effective_date="2024-01-01",
        market="US",
        primary_urls=[
            "https://www.govinfo.gov/content/pkg/CFR-2024-title49-vol6/pdf/CFR-2024-title49-vol6-sec571-208.pdf",
            "https://www.govinfo.gov/content/pkg/CFR-2023-title49-vol6/pdf/CFR-2023-title49-vol6-sec571-208.pdf",
        ],
        scrape_page=None,
        storage_subpath="FMVSS/FMVSS_208",
        filename="FMVSS_208_2024.pdf",
        strategy=DownloadStrategy.REAL,
        description="Passive restraint requirements: airbags, belts, OOP suppression, sled tests"
    ),

    RegulationEntry(
        regulation_id="FMVSS_210",
        title="FMVSS 210 — Seat Belt Assembly Anchorages",
        source=RegulationSource.FMVSS,
        category="Safety Belts",
        series="2024 CFR",
        status=RegulationStatus.ACTIVE,
        publication_date="2024-01-01",
        effective_date="2024-01-01",
        market="US",
        primary_urls=[
            "https://www.govinfo.gov/content/pkg/CFR-2024-title49-vol6/pdf/CFR-2024-title49-vol6-sec571-210.pdf",
            "https://www.govinfo.gov/content/pkg/CFR-2023-title49-vol6/pdf/CFR-2023-title49-vol6-sec571-210.pdf",
        ],
        scrape_page=None,
        storage_subpath="FMVSS/FMVSS_210",
        filename="FMVSS_210_2024.pdf",
        strategy=DownloadStrategy.REAL,
        description="Seat belt anchorage strength and geometry requirements (US counterpart to UN R14)"
    ),

    RegulationEntry(
        regulation_id="FMVSS_213",
        title="FMVSS 213 — Child Restraint Systems",
        source=RegulationSource.FMVSS,
        category="Child Safety",
        series="2024 CFR",
        status=RegulationStatus.ACTIVE,
        publication_date="2024-01-01",
        effective_date="2024-01-01",
        market="US",
        primary_urls=[
            "https://www.govinfo.gov/content/pkg/CFR-2024-title49-vol6/pdf/CFR-2024-title49-vol6-sec571-213.pdf",
            "https://www.govinfo.gov/content/pkg/CFR-2023-title49-vol6/pdf/CFR-2023-title49-vol6-sec571-213.pdf",
        ],
        scrape_page=None,
        storage_subpath="FMVSS/FMVSS_213",
        filename="FMVSS_213_2024.pdf",
        strategy=DownloadStrategy.REAL,
        description="US CRS performance: frontal barrier test, side test, dynamic sled, structural integrity"
    ),

    RegulationEntry(
        regulation_id="FMVSS_214",
        title="FMVSS 214 — Side Impact Protection",
        source=RegulationSource.FMVSS,
        category="Side Crash",
        series="2024 CFR",
        status=RegulationStatus.ACTIVE,
        publication_date="2024-01-01",
        effective_date="2024-01-01",
        market="US",
        primary_urls=[
            "https://www.govinfo.gov/content/pkg/CFR-2024-title49-vol6/pdf/CFR-2024-title49-vol6-sec571-214.pdf",
            "https://www.govinfo.gov/content/pkg/CFR-2023-title49-vol6/pdf/CFR-2023-title49-vol6-sec571-214.pdf",
        ],
        scrape_page=None,
        storage_subpath="FMVSS/FMVSS_214",
        filename="FMVSS_214_2024.pdf",
        strategy=DownloadStrategy.REAL,
        description="MDB and pole side-impact performance: ES-2re dummy criteria, TTR, thorax"
    ),

    RegulationEntry(
        regulation_id="FMVSS_216",
        title="FMVSS 216 — Roof Crush Resistance",
        source=RegulationSource.FMVSS,
        category="Structural",
        series="2024 CFR",
        status=RegulationStatus.ACTIVE,
        publication_date="2024-01-01",
        effective_date="2024-01-01",
        market="US",
        primary_urls=[
            "https://www.govinfo.gov/content/pkg/CFR-2024-title49-vol6/pdf/CFR-2024-title49-vol6-sec571-216.pdf",
            "https://www.govinfo.gov/content/pkg/CFR-2023-title49-vol6/pdf/CFR-2023-title49-vol6-sec571-216a.pdf",
        ],
        scrape_page=None,
        storage_subpath="FMVSS/FMVSS_216",
        filename="FMVSS_216_2024.pdf",
        strategy=DownloadStrategy.REAL,
        description="Roof strength: 3× SWR force, <5 inch plate travel requirement"
    ),

    RegulationEntry(
        regulation_id="FMVSS_225",
        title="FMVSS 225 — Child Restraint Anchorage Systems (LATCH)",
        source=RegulationSource.FMVSS,
        category="Child Safety",
        series="2024 CFR",
        status=RegulationStatus.ACTIVE,
        publication_date="2024-01-01",
        effective_date="2024-01-01",
        market="US",
        primary_urls=[
            "https://www.govinfo.gov/content/pkg/CFR-2024-title49-vol6/pdf/CFR-2024-title49-vol6-sec571-225.pdf",
            "https://www.govinfo.gov/content/pkg/CFR-2023-title49-vol6/pdf/CFR-2023-title49-vol6-sec571-225.pdf",
        ],
        scrape_page=None,
        storage_subpath="FMVSS/FMVSS_225",
        filename="FMVSS_225_2024.pdf",
        strategy=DownloadStrategy.REAL,
        description="LATCH lower anchor + top tether requirements, force limits, geometry"
    ),

    RegulationEntry(
        regulation_id="FMVSS_301",
        title="FMVSS 301 — Fuel System Integrity",
        source=RegulationSource.FMVSS,
        category="Fire Safety",
        series="2024 CFR",
        status=RegulationStatus.ACTIVE,
        publication_date="2024-01-01",
        effective_date="2024-01-01",
        market="US",
        primary_urls=[
            "https://www.govinfo.gov/content/pkg/CFR-2024-title49-vol6/pdf/CFR-2024-title49-vol6-sec571-301.pdf",
            "https://www.govinfo.gov/content/pkg/CFR-2023-title49-vol6/pdf/CFR-2023-title49-vol6-sec571-301.pdf",
        ],
        scrape_page=None,
        storage_subpath="FMVSS/FMVSS_301",
        filename="FMVSS_301_2024.pdf",
        strategy=DownloadStrategy.REAL,
        description="Post-crash fuel leakage limits: frontal, rear, side, rollover tests"
    ),

    RegulationEntry(
        regulation_id="FMVSS_305",
        title="FMVSS 305 — Electric-Powered Vehicles: Electrolyte Spillage and EDS",
        source=RegulationSource.FMVSS,
        category="EV Safety",
        series="2024 CFR",
        status=RegulationStatus.ACTIVE,
        publication_date="2024-01-01",
        effective_date="2024-01-01",
        market="US",
        primary_urls=[
            "https://www.govinfo.gov/content/pkg/CFR-2024-title49-vol6/pdf/CFR-2024-title49-vol6-sec571-305.pdf",
            "https://www.govinfo.gov/content/pkg/CFR-2023-title49-vol6/pdf/CFR-2023-title49-vol6-sec571-305.pdf",
        ],
        scrape_page=None,
        storage_subpath="FMVSS/FMVSS_305",
        filename="FMVSS_305_2024.pdf",
        strategy=DownloadStrategy.REAL,
        description="EV: electrolyte spillage, electrical isolation, HV disconnect, crash tests"
    ),

    # ═══════════════════════════════════════════════════════════
    # GROUP B: UNECE — Synthetic (unece.org HTTP 403 blocked)
    # ═══════════════════════════════════════════════════════════

    RegulationEntry(
        regulation_id="UN_R14",
        title="UN Regulation No. 14 — Safety-belt Anchorages",
        source=RegulationSource.UNECE,
        category="Safety Belts",
        series="07 Series of Amendments",
        status=RegulationStatus.ACTIVE,
        publication_date="2020-05-29",
        effective_date="2020-05-29",
        market="GLOBAL",
        primary_urls=[
            "https://unece.org/sites/default/files/2023-08/R014r7e.pdf",
            "https://unece.org/fileadmin/DAM/trans/main/wp29/wp29regs/2023/R014r7e.pdf",
            "https://unece.org/fileadmin/DAM/trans/main/wp29/wp29regs/2015/R014r6e.pdf",
        ],
        scrape_page="https://unece.org/transport/vehicle-regulations/un-regulation-no-14-safety-belt-anchorages",
        storage_subpath="UNECE/R14",
        filename="UN_R14_07Series.pdf",
        strategy=DownloadStrategy.REAL,
        description="Safety-belt anchorage strength, ISOFIX anchorages, geometry requirements for M+N vehicles"
    ),

    RegulationEntry(
        regulation_id="UN_R16",
        title="UN Regulation No. 16 — Safety-belts and Restraint Systems",
        source=RegulationSource.UNECE,
        category="Safety Belts",
        series="08 Series of Amendments",
        status=RegulationStatus.ACTIVE,
        publication_date="2022-06-22",
        effective_date="2022-06-22",
        market="GLOBAL",
        primary_urls=[
            "https://unece.org/sites/default/files/2023-11/R016r11e.pdf",
            "https://unece.org/fileadmin/DAM/trans/main/wp29/wp29regs/2022/R016r11e.pdf",
            "https://unece.org/fileadmin/DAM/trans/main/wp29/wp29regs/2018/R016r8e.pdf",
        ],
        scrape_page="https://unece.org/transport/vehicle-regulations/un-regulation-no-16-safety-belts-restraint-systems",
        storage_subpath="UNECE/R16",
        filename="UN_R16_08Series.pdf",
        strategy=DownloadStrategy.REAL,
        description="Safety belt type approval: ELR/ALR retractors, dynamic 50 km/h test, buckle release"
    ),

    RegulationEntry(
        regulation_id="UN_R17",
        title="UN Regulation No. 17 — Seats, Anchorages and Head Restraints",
        source=RegulationSource.UNECE,
        category="Seats & Head Restraints",
        series="09 Series of Amendments",
        status=RegulationStatus.ACTIVE,
        publication_date="2021-09-30",
        effective_date="2021-09-30",
        market="GLOBAL",
        primary_urls=[
            "https://unece.org/sites/default/files/2021-09/R017r10e.pdf",
            "https://unece.org/fileadmin/DAM/trans/main/wp29/wp29regs/2016/R017r10e.pdf",
        ],
        scrape_page="https://unece.org/transport/vehicle-regulations/un-regulation-no-17-seats-anchorages-head-restraints",
        storage_subpath="UNECE/R17",
        filename="UN_R17_09Series.pdf",
        strategy=DownloadStrategy.REAL,
        description="Seat strength (530 Nm), head restraint height ≥800 mm, BioRID II whiplash test"
    ),

    RegulationEntry(
        regulation_id="UN_R21",
        title="UN Regulation No. 21 — Interior Fittings",
        source=RegulationSource.UNECE,
        category="Interior Safety",
        series="02 Series of Amendments",
        status=RegulationStatus.ACTIVE,
        publication_date="1986-10-08",
        effective_date="1986-10-08",
        market="GLOBAL",
        primary_urls=[
            "https://unece.org/sites/default/files/2021-11/R021r2e.pdf",
            "https://unece.org/fileadmin/DAM/trans/main/wp29/wp29regs/2016/R021r2e.pdf",
        ],
        scrape_page="https://unece.org/transport/vehicle-regulations/un-regulation-no-21-interior-fittings",
        storage_subpath="UNECE/R21",
        filename="UN_R21_02Series.pdf",
        strategy=DownloadStrategy.REAL,
        description="Interior surface radius ≥2.5 mm, head impact zone HIC < 1000, dashboard energy absorption"
    ),

    RegulationEntry(
        regulation_id="UN_R44",
        title="UN Regulation No. 44 — Child Restraint Systems (Legacy)",
        source=RegulationSource.UNECE,
        category="Child Safety",
        series="04 Series of Amendments",
        status=RegulationStatus.ACTIVE,
        publication_date="2005-06-23",
        effective_date="2005-06-23",
        market="GLOBAL",
        primary_urls=[
            "https://unece.org/sites/default/files/2021-11/R044r4e.pdf",
            "https://unece.org/fileadmin/DAM/trans/main/wp29/wp29regs/2016/R044r4e.pdf",
        ],
        scrape_page="https://unece.org/transport/vehicle-regulations/un-regulation-no-44-child-restraint-systems",
        storage_subpath="UNECE/R44",
        filename="UN_R44_04Series.pdf",
        strategy=DownloadStrategy.REAL,
        description="Weight-based CRS groups 0/0+/I/II/III, P-dummies, 50 km/h frontal + 30 km/h rear tests"
    ),

    RegulationEntry(
        regulation_id="UN_R94",
        title="UN Regulation No. 94 — Frontal Collision Protection",
        source=RegulationSource.UNECE,
        category="Frontal Crash",
        series="04 Series of Amendments",
        status=RegulationStatus.ACTIVE,
        publication_date="2023-01-15",
        effective_date="2024-01-01",
        market="GLOBAL",
        primary_urls=[
            "https://unece.org/sites/default/files/2023-01/R094r4e.pdf",
            "https://unece.org/fileadmin/DAM/trans/main/wp29/wp29regs/2023/R094r4e.pdf",
            "https://unece.org/fileadmin/DAM/trans/main/wp29/wp29regs/2016/R094r2e.pdf",
        ],
        scrape_page="https://unece.org/transport/vehicle-regulations/un-regulation-no-94-frontal-collision-protection",
        storage_subpath="UNECE/R94",
        filename="UN_R94_04Series.pdf",
        strategy=DownloadStrategy.REAL,
        description="ODB 56 km/h 40% overlap frontal test; HPC ≤ 1000, chest deflection ≤ 50 mm"
    ),

    RegulationEntry(
        regulation_id="UN_R95_04",
        title="UN Regulation No. 95 — Lateral Collision Protection (04 Series, Superseded)",
        source=RegulationSource.UNECE,
        category="Side Crash",
        series="04 Series of Amendments",
        status=RegulationStatus.SUPERSEDED,
        publication_date="2020-09-01",
        effective_date="2020-09-01",
        market="GLOBAL",
        primary_urls=[
            "https://unece.org/fileadmin/DAM/trans/main/wp29/wp29regs/2016/R095r2e.pdf",
        ],
        scrape_page=None,
        storage_subpath="UNECE/R95/04Series",
        filename="UN_R95_04Series_superseded.pdf",
        strategy=DownloadStrategy.REAL,
        description="MDB 50 km/h side impact; WorldSID, rib deflection ≤ 42 mm (superseded by 05 Series)"
    ),

    RegulationEntry(
        regulation_id="UN_R95_05",
        title="UN Regulation No. 95 — Lateral Collision Protection (05 Series, Active)",
        source=RegulationSource.UNECE,
        category="Side Crash",
        series="05 Series of Amendments",
        status=RegulationStatus.ACTIVE,
        publication_date="2025-01-01",
        effective_date="2025-01-01",
        market="GLOBAL",
        primary_urls=[
            "https://unece.org/sites/default/files/2023-12/R095r4e.pdf",
            "https://unece.org/fileadmin/DAM/trans/main/wp29/wp29regs/2021/R095r3e.pdf",
        ],
        scrape_page="https://unece.org/transport/vehicle-regulations/un-regulation-no-95-lateral-collision-protection",
        storage_subpath="UNECE/R95/05Series",
        filename="UN_R95_05Series.pdf",
        strategy=DownloadStrategy.REAL,
        supersedes="UN_R95_04",
        description="Updated AE-MDB barrier; WorldSID; stricter rib deflection ≤ 38 mm"
    ),

    RegulationEntry(
        regulation_id="UN_R127",
        title="UN Regulation No. 127 — Pedestrian Safety",
        source=RegulationSource.UNECE,
        category="Pedestrian Protection",
        series="03 Series of Amendments",
        status=RegulationStatus.ACTIVE,
        publication_date="2022-01-07",
        effective_date="2022-01-07",
        market="GLOBAL",
        primary_urls=[
            "https://unece.org/sites/default/files/2022-01/R127r3e.pdf",
            "https://unece.org/fileadmin/DAM/trans/main/wp29/wp29regs/2022/R127r3e.pdf",
        ],
        scrape_page="https://unece.org/transport/vehicle-regulations/un-regulation-no-127-pedestrian-safety",
        storage_subpath="UNECE/R127",
        filename="UN_R127_03Series.pdf",
        strategy=DownloadStrategy.REAL,
        description="Child (3.5 kg) + adult (4.5 kg) headform at 35 km/h; FlexPLI at 40 km/h"
    ),

    RegulationEntry(
        regulation_id="UN_R129",
        title="UN Regulation No. 129 — Enhanced Child Restraint Systems (i-Size)",
        source=RegulationSource.UNECE,
        category="Child Safety",
        series="04 Series of Amendments",
        status=RegulationStatus.ACTIVE,
        publication_date="2019-12-22",
        effective_date="2019-12-22",
        market="GLOBAL",
        primary_urls=[
            "https://unece.org/sites/default/files/2021-11/R129r4e.pdf",
            "https://unece.org/fileadmin/DAM/trans/main/wp29/wp29regs/2016/R129e.pdf",
        ],
        scrape_page="https://unece.org/transport/vehicle-regulations/un-regulation-no-129-child-restraint-systems-i-size",
        storage_subpath="UNECE/R129",
        filename="UN_R129_04Series.pdf",
        strategy=DownloadStrategy.REAL,
        description="Height-based classification; mandatory side impact test; ISOFIX; Q-dummies"
    ),

    RegulationEntry(
        regulation_id="UN_R135",
        title="UN Regulation No. 135 — Pole Side Impact",
        source=RegulationSource.UNECE,
        category="Side Crash",
        series="01 Series of Amendments",
        status=RegulationStatus.ACTIVE,
        publication_date="2020-01-03",
        effective_date="2020-01-03",
        market="GLOBAL",
        primary_urls=[
            "https://unece.org/sites/default/files/2021-11/R135r1e.pdf",
            "https://unece.org/fileadmin/DAM/trans/main/wp29/wp29regs/2016/R135e.pdf",
        ],
        scrape_page="https://unece.org/transport/vehicle-regulations/un-regulation-no-135-pole-side-impact",
        storage_subpath="UNECE/R135",
        filename="UN_R135_01Series.pdf",
        strategy=DownloadStrategy.REAL,
        description="32 km/h lateral into 254 mm rigid pole at 75°; WorldSID head + thorax criteria"
    ),

    RegulationEntry(
        regulation_id="UN_R137",
        title="UN Regulation No. 137 — Frontal Impact Restraint Systems",
        source=RegulationSource.UNECE,
        category="Frontal Crash",
        series="01 Series of Amendments",
        status=RegulationStatus.ACTIVE,
        publication_date="2021-06-09",
        effective_date="2021-06-09",
        market="GLOBAL",
        primary_urls=[
            "https://unece.org/sites/default/files/2021-11/R137r1e.pdf",
            "https://unece.org/fileadmin/DAM/trans/main/wp29/wp29regs/2016/R137e.pdf",
        ],
        scrape_page="https://unece.org/transport/vehicle-regulations/un-regulation-no-137-frontal-impact",
        storage_subpath="UNECE/R137",
        filename="UN_R137_01Series.pdf",
        strategy=DownloadStrategy.REAL,
        description="50 km/h FWRB; thorax deflection + shoulder belt force on driver/passenger"
    ),

    RegulationEntry(
        regulation_id="UN_R153",
        title="UN Regulation No. 153 — Post-Collision Fuel System Integrity",
        source=RegulationSource.UNECE,
        category="Fire Safety",
        series="01 Series of Amendments",
        status=RegulationStatus.ACTIVE,
        publication_date="2022-06-01",
        effective_date="2023-01-01",
        market="GLOBAL",
        primary_urls=[
            "https://unece.org/sites/default/files/2022-06/R153r1e.pdf",
        ],
        scrape_page="https://unece.org/transport/vehicle-regulations/un-regulation-no-153-fuel-system-integrity",
        storage_subpath="UNECE/R153",
        filename="UN_R153_01Series.pdf",
        strategy=DownloadStrategy.REAL,
        description="Post-crash fuel leak prevention: frontal + lateral + rear barriers; 30 min monitoring"
    ),

    # ═══════════════════════════════════════════════════════════
    # GROUP C: Euro NCAP — Synthetic (JS SPA, no direct links)
    # ═══════════════════════════════════════════════════════════

    RegulationEntry(
        regulation_id="EURONCAP_AOP",
        title="Euro NCAP Adult Occupant Protection Protocol 2026",
        source=RegulationSource.EURO_NCAP,
        category="Adult Occupant Protection",
        series="Version 10.2 (2026)",
        status=RegulationStatus.ACTIVE,
        publication_date="2026-01-01",
        effective_date="2026-01-01",
        market="EU",
        primary_urls=[
            "https://cdn.euroncap.com/media/Adult-Occupant-Protection-Protocol-2026.pdf",
            "https://www.euroncap.com/en/for-engineers/protocols/2026-protocols/",
        ],
        scrape_page="https://www.euroncap.com/en/for-engineers/protocols/2026-protocols/",
        storage_subpath="EuroNCAP/AdultOccupantProtection",
        filename="EuroNCAP_AOP_2026.pdf",
        strategy=DownloadStrategy.REAL,
        description="Frontal crash: HPC, BrIC, chest deflection; side MDB, pole; 5-star rating criteria"
    ),

    RegulationEntry(
        regulation_id="EURONCAP_COP",
        title="Euro NCAP Child Occupant Protection Protocol 2026",
        source=RegulationSource.EURO_NCAP,
        category="Child Occupant Protection",
        series="Version 10.2 (2026)",
        status=RegulationStatus.ACTIVE,
        publication_date="2026-01-01",
        effective_date="2026-01-01",
        market="EU",
        primary_urls=[
            "https://cdn.euroncap.com/media/Child-Occupant-Protection-Protocol-2026.pdf",
            "https://www.euroncap.com/en/for-engineers/protocols/2026-protocols/",
        ],
        scrape_page="https://www.euroncap.com/en/for-engineers/protocols/2026-protocols/",
        storage_subpath="EuroNCAP/ChildOccupantProtection",
        filename="EuroNCAP_COP_2026.pdf",
        strategy=DownloadStrategy.REAL,
        description="CRS compatibility, Q1.5/Q3/Q6/Q10 dummies, frontal + side CRS tests, 5-star rating"
    ),

    RegulationEntry(
        regulation_id="EURONCAP_VRU",
        title="Euro NCAP Vulnerable Road Users Protocol 2026",
        source=RegulationSource.EURO_NCAP,
        category="Pedestrian & Cyclist Protection",
        series="Version 10.2 (2026)",
        status=RegulationStatus.ACTIVE,
        publication_date="2026-01-01",
        effective_date="2026-01-01",
        market="EU",
        primary_urls=[
            "https://cdn.euroncap.com/media/Pedestrians-Protocol-2026.pdf",
            "https://www.euroncap.com/en/for-engineers/protocols/2026-protocols/",
        ],
        scrape_page="https://www.euroncap.com/en/for-engineers/protocols/2026-protocols/",
        storage_subpath="EuroNCAP/VRU",
        filename="EuroNCAP_VRU_2026.pdf",
        strategy=DownloadStrategy.REAL,
        description="Head impact zones, FlexPLI legform, cyclist scenarios, AEB VRU performance"
    ),

    RegulationEntry(
        regulation_id="EURONCAP_SIDE",
        title="Euro NCAP Side Impact Protocol 2026",
        source=RegulationSource.EURO_NCAP,
        category="Side Crash",
        series="Version 10.2 (2026)",
        status=RegulationStatus.ACTIVE,
        publication_date="2026-01-01",
        effective_date="2026-01-01",
        market="EU",
        primary_urls=[
            "https://cdn.euroncap.com/media/Side-Impact-Protocol-2026.pdf",
            "https://www.euroncap.com/en/for-engineers/protocols/2026-protocols/",
        ],
        scrape_page="https://www.euroncap.com/en/for-engineers/protocols/2026-protocols/",
        storage_subpath="EuroNCAP/SideImpact",
        filename="EuroNCAP_Side_2026.pdf",
        strategy=DownloadStrategy.REAL,
        description="AE-MDB 50 km/h; far-side; WorldSID + THOR dummies; scoring tables"
    ),

    RegulationEntry(
        regulation_id="EURONCAP_FARSIDE",
        title="Euro NCAP Far-Side Impact Protocol 2026",
        source=RegulationSource.EURO_NCAP,
        category="Side Crash",
        series="Version 10.2 (2026)",
        status=RegulationStatus.ACTIVE,
        publication_date="2026-01-01",
        effective_date="2026-01-01",
        market="EU",
        primary_urls=[
            "https://cdn.euroncap.com/media/Far-Side-Impact-Protocol-2026.pdf",
        ],
        scrape_page="https://www.euroncap.com/en/for-engineers/protocols/2026-protocols/",
        storage_subpath="EuroNCAP/FarSideImpact",
        filename="EuroNCAP_FarSide_2026.pdf",
        strategy=DownloadStrategy.REAL,
        description="Far-side occupant kinematics, thorax criteria, restraint system assessment"
    ),

    RegulationEntry(
        regulation_id="EURONCAP_SA",
        title="Euro NCAP Safety Assist Protocol 2026",
        source=RegulationSource.EURO_NCAP,
        category="Active Safety",
        series="Version 10.2 (2026)",
        status=RegulationStatus.ACTIVE,
        publication_date="2026-01-01",
        effective_date="2026-01-01",
        market="EU",
        primary_urls=[
            "https://cdn.euroncap.com/media/Safety-Assist-Protocol-2026.pdf",
        ],
        scrape_page="https://www.euroncap.com/en/for-engineers/protocols/2026-protocols/",
        storage_subpath="EuroNCAP/SafetyAssist",
        filename="EuroNCAP_SA_2026.pdf",
        strategy=DownloadStrategy.REAL,
        description="AEB, LKA, ESS, occupant status monitoring; scoring and test scenarios"
    ),

    RegulationEntry(
        regulation_id="EURONCAP_RESCUE",
        title="Euro NCAP Rescue & Extrication Protocol 2026",
        source=RegulationSource.EURO_NCAP,
        category="Emergency Response",
        series="Version 10.2 (2026)",
        status=RegulationStatus.ACTIVE,
        publication_date="2026-01-01",
        effective_date="2026-01-01",
        market="EU",
        primary_urls=[
            "https://cdn.euroncap.com/media/Rescue-and-Extrication-Protocol-2026.pdf",
        ],
        scrape_page="https://www.euroncap.com/en/for-engineers/protocols/2026-protocols/",
        storage_subpath="EuroNCAP/RescueExtrication",
        filename="EuroNCAP_Rescue_2026.pdf",
        strategy=DownloadStrategy.REAL,
        description="Emergency call eCall, rescue sheet, HV isolation, CRS label accessibility"
    ),

    # ═══════════════════════════════════════════════════════════
    # GROUP D: IIHS — Synthetic (404 / auth-gated)
    # ═══════════════════════════════════════════════════════════

    RegulationEntry(
        regulation_id="IIHS_MODERATE_OVERLAP",
        title="IIHS Moderate Overlap Front Test Protocol",
        source=RegulationSource.IIHS,
        category="Frontal Crash",
        series="Current",
        status=RegulationStatus.ACTIVE,
        publication_date="2022-01-01",
        effective_date="2022-01-01",
        market="US",
        primary_urls=[
            "https://www.iihs.org/media/259e5dff-c8b7-4e9e-a0de-d2e3d6acbef7/hV_bhA/Ratings/Protocols/current/overlap-front-eval-protocol.pdf",
        ],
        scrape_page="https://www.iihs.org/ratings/about-our-tests",
        storage_subpath="IIHS/ModerateOverlap",
        filename="IIHS_ModerateOverlap_front.pdf",
        strategy=DownloadStrategy.REAL,
        description="40% overlap frontal crash into deformable barrier; good/acceptable/marginal/poor ratings"
    ),

    RegulationEntry(
        regulation_id="IIHS_SMALL_OVERLAP",
        title="IIHS Small Overlap Right-Side Test Protocol",
        source=RegulationSource.IIHS,
        category="Frontal Crash",
        series="Current",
        status=RegulationStatus.ACTIVE,
        publication_date="2022-01-01",
        effective_date="2022-01-01",
        market="US",
        primary_urls=[
            "https://www.iihs.org/media/e98a7af4-7a88-4e40-92aa-e77c2a64fdbe/kWlTcg/Ratings/Protocols/current/small-overlap-rh-eval-protocol.pdf",
        ],
        scrape_page="https://www.iihs.org/ratings/about-our-tests",
        storage_subpath="IIHS/SmallOverlap",
        filename="IIHS_SmallOverlap_RH.pdf",
        strategy=DownloadStrategy.REAL,
        description="25% offset right-side frontal crash; A-pillar intrusion, head excursion criteria"
    ),

    RegulationEntry(
        regulation_id="IIHS_SIDE_BARRIER",
        title="IIHS Side Barrier Test Protocol",
        source=RegulationSource.IIHS,
        category="Side Crash",
        series="Current",
        status=RegulationStatus.ACTIVE,
        publication_date="2023-01-01",
        effective_date="2023-01-01",
        market="US",
        primary_urls=[
            "https://www.iihs.org/media/db3bd9db-cce4-43d6-a69d-b3f793af6e3f/gg5frQ/Ratings/Protocols/current/side-barrier-eval-protocol.pdf",
        ],
        scrape_page="https://www.iihs.org/ratings/about-our-tests",
        storage_subpath="IIHS/SideBarrier",
        filename="IIHS_SideBarrier.pdf",
        strategy=DownloadStrategy.REAL,
        description="Updated MDB side impact (heavier barrier 4,200 lb, 37 mph); head + thorax criteria"
    ),

    RegulationEntry(
        regulation_id="IIHS_ROOF_STRENGTH",
        title="IIHS Roof Strength Test Protocol",
        source=RegulationSource.IIHS,
        category="Structural",
        series="Current",
        status=RegulationStatus.ACTIVE,
        publication_date="2019-01-01",
        effective_date="2019-01-01",
        market="US",
        primary_urls=[
            "https://www.iihs.org/media/1dc83ccb-be4b-4d94-b40b-7b0de4a48e80/7Aq8Lw/Ratings/Protocols/current/roof-strength-eval-protocol.pdf",
        ],
        scrape_page="https://www.iihs.org/ratings/about-our-tests",
        storage_subpath="IIHS/RoofStrength",
        filename="IIHS_RoofStrength.pdf",
        strategy=DownloadStrategy.REAL,
        description="Roof SWR ≥ 4.0 for 'good'; plate travel limits; rollover context"
    ),

    RegulationEntry(
        regulation_id="IIHS_HEAD_RESTRAINTS",
        title="IIHS Head Restraints & Seats Evaluation Protocol",
        source=RegulationSource.IIHS,
        category="Seats & Head Restraints",
        series="Current",
        status=RegulationStatus.ACTIVE,
        publication_date="2020-01-01",
        effective_date="2020-01-01",
        market="US",
        primary_urls=[
            "https://www.iihs.org/media/head-restraints-protocol.pdf",
        ],
        scrape_page="https://www.iihs.org/ratings/about-our-tests",
        storage_subpath="IIHS/HeadRestraints",
        filename="IIHS_HeadRestraints.pdf",
        strategy=DownloadStrategy.REAL,
        description="Geometry (height, backset) + HIT dynamic test; whiplash protection rating"
    ),

]

# ─── Lookup helpers ──────────────────────────────────────────────────────────
CATALOG_BY_ID: dict[str, RegulationEntry] = {r.regulation_id: r for r in REGULATION_CATALOG}

def get_regulation(regulation_id: str) -> RegulationEntry | None:
    return CATALOG_BY_ID.get(regulation_id)

def get_by_source(source: RegulationSource) -> list[RegulationEntry]:
    return [r for r in REGULATION_CATALOG if r.source == source]

def get_active() -> list[RegulationEntry]:
    return [r for r in REGULATION_CATALOG if r.status == RegulationStatus.ACTIVE]
