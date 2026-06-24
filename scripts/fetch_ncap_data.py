#!/usr/bin/env python3
"""
Fetch NHTSA NCAP star ratings (free API, no auth) into data/corpus/historical/.

These are consumer star ratings + crash-test media URLs — NOT raw biomechanical
channels (HIC, chest deflection, femur load). Use PROG_X synthetic docs or
optional --with-reports for injury-channel PDFs.

Usage:
  python scripts/fetch_ncap_data.py
  python scripts/fetch_ncap_data.py --limit 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.document_registry import register_ncap_document  # noqa: E402

HISTORICAL_DIR = ROOT / "data" / "corpus" / "historical"
MARKDOWN_DIR = ROOT / "output" / "markdown"
MANIFEST_PATH = ROOT / "data" / "manifest" / "corpus_manifest.json"

# Popular US-market vehicles (year, make, model) — first variant per query is used.
DEFAULT_VEHICLES: list[tuple[int, str, str]] = [
    (2024, "toyota", "camry"),
    (2024, "hyundai", "elantra"),
    (2023, "nissan", "altima"),
    (2024, "bmw", "x3"),
    (2024, "honda", "civic"),
    (2023, "mazda", "cx-5"),
    (2024, "kia", "telluride"),
    (2023, "volkswagen", "jetta"),
    (2024, "subaru", "outback"),
]

API_BASE = "https://api.nhtsa.gov/SafetyRatings"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _api_get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "AutoSafety-RAG/1.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_vehicle_ids(year: int, make: str, model: str) -> list[dict]:
    url = (
        f"{API_BASE}/modelyear/{year}/make/{quote(make)}/model/{quote(model)}"
    )
    data = _api_get(url)
    return data.get("Results") or []


def fetch_vehicle_ratings(vehicle_id: int) -> dict:
    url = f"{API_BASE}/VehicleId/{vehicle_id}"
    data = _api_get(url)
    results = data.get("Results") or []
    if not results:
        raise ValueError(f"No ratings for VehicleId {vehicle_id}")
    return results[0]


def _impact_mode(ratings: dict) -> str:
    frontal = ratings.get("OverallFrontCrashRating")
    side = ratings.get("SideCrashDriversideRating") or ratings.get("SideCrashPassengersideRating")
    if frontal and not side:
        return "frontal"
    if side and not frontal:
        return "side"
    return "general"


def _ratings_table(r: dict) -> str:
    rows = [
        ("Overall", r.get("OverallRating")),
        ("Overall frontal", r.get("OverallFrontCrashRating")),
        ("Front driver", r.get("FrontCrashDriversideRating")),
        ("Front passenger", r.get("FrontCrashPassengersideRating")),
        ("Side driver", r.get("SideCrashDriversideRating")),
        ("Side passenger", r.get("SideCrashPassengersideRating")),
        ("Side pole", r.get("SidePoleCrashRating")),
        ("Rollover", r.get("RolloverRating")),
        ("Rollover risk (%)", r.get("RolloverPossibility")),
    ]
    lines = ["| Rating category | Stars |", "| --- | --- |"]
    for label, val in rows:
        if val is not None and str(val).strip():
            lines.append(f"| {label} | {val} |")
    return "\n".join(lines)


def _media_section(r: dict) -> str:
    media = [
        ("Vehicle picture", r.get("VehiclePicture")),
        ("Front crash picture", r.get("FrontCrashPicture")),
        ("Front crash video", r.get("FrontCrashVideo")),
        ("Side crash picture", r.get("SideCrashPicture")),
        ("Side crash video", r.get("SideCrashVideo")),
        ("Side pole picture", r.get("SidePoleCrashPicture")),
        ("Rollover picture", r.get("RolloverPicture")),
    ]
    lines = ["## Crash test media", ""]
    for label, url in media:
        if url:
            lines.append(f"- **{label}:** {url}")
    if len(lines) == 2:
        lines.append("_No media URLs returned by NHTSA for this variant._")
    return "\n".join(lines)


def build_markdown(
    *,
    year: int,
    make: str,
    model: str,
    vehicle_id: int,
    ratings: dict,
    reg_code: str,
) -> str:
    desc = ratings.get("VehicleDescription") or f"{year} {make} {model}"
    impact = _impact_mode(ratings)
    fm = f"""---
source_type: nhtsa_ncap
is_synthetic: false
doc_type: internal
authority_tier: historical_data
authority: NHTSA
region: US
impact_mode: {impact}
license_status: public
regulation: {reg_code}
source_pdf: NCAP_{year}_{_slug(make)}_{_slug(model)}_{vehicle_id}.md
vehicle_program: NCAP_{year}_{_slug(make)}_{_slug(model)}
revision: "{year}"
document_kind: test_report
---

> **NHTSA NCAP HISTORICAL DATA — star ratings and media only**
> These are NHTSA consumer crash-test star ratings and publicly released photos/videos.
> They are **not** raw biomechanical injury channels (HIC, chest deflection, femur load).
> Authority tier: `historical_data` | Region: US | Impact mode: {impact}

# NHTSA NCAP — {desc}

| Field | Value |
| --- | --- |
| VehicleId | {vehicle_id} |
| Model year | {year} |
| Make | {make} |
| Model | {model} |
| Data source | [NHTSA Safety Ratings API](https://www.nhtsa.gov/ratings) |

## Crash ratings (star scale 1–5)

{_ratings_table(ratings)}

{_media_section(ratings)}

## Notes for crash-development crew

Use this document for **historical NCAP context** (star ratings, test configuration hints from media).
For biomechanical pass/fail against regulatory limits, cite `legal_binding` sources (FMVSS, UN/ECE).
"""
    return fm


def _update_manifest(entry: dict) -> None:
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        manifest = {"corpus_version": 1, "documents": []}
    docs = manifest.setdefault("documents", [])
    path_key = entry["path"].replace("/", "\\")
    docs = [d for d in docs if d.get("regulation") != entry["regulation"]]
    docs.append(entry)
    manifest["documents"] = docs
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def fetch_all(*, limit: int | None = None, vehicles: list[tuple[int, str, str]] | None = None) -> list[Path]:
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    targets = (vehicles or DEFAULT_VEHICLES)[: limit or None]

    for year, make, model in targets:
        try:
            variants = list_vehicle_ids(year, make, model)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            print(f"SKIP {year} {make} {model}: list failed ({exc})")
            continue
        if not variants:
            print(f"SKIP {year} {make} {model}: no variants")
            continue

        variant = variants[0]
        vehicle_id = int(variant["VehicleId"])
        reg_code = f"NCAP_{year}_{_slug(make)}_{_slug(model)}_{vehicle_id}".upper()

        try:
            ratings = fetch_vehicle_ratings(vehicle_id)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            print(f"SKIP VehicleId {vehicle_id}: {exc}")
            continue

        impact = _impact_mode(ratings)
        display = ratings.get("VehicleDescription") or f"{year} {make} {model}"
        register_ncap_document(
            reg_code,
            display_name=f"NCAP {display}",
            year=year,
            make=make,
            model=model,
            vehicle_id=vehicle_id,
            impact_mode=impact,
        )

        md = build_markdown(
            year=year,
            make=make,
            model=model,
            vehicle_id=vehicle_id,
            ratings=ratings,
            reg_code=reg_code,
        )
        fname = f"NCAP_{year}_{_slug(make)}_{_slug(model)}_{vehicle_id}.md"
        hist_path = HISTORICAL_DIR / fname
        hist_path.write_text(md, encoding="utf-8")
        (MARKDOWN_DIR / fname).write_text(md, encoding="utf-8")
        written.append(hist_path)

        rel = f"data\\corpus\\historical\\{fname}"
        _update_manifest({
            "path": rel,
            "name": fname,
            "category": "historical",
            "regulation": reg_code,
            "doc_type": "internal",
            "authority_tier": "historical_data",
            "impact_mode": impact,
            "license_status": "public",
            "region": "US",
        })
        print(f"Wrote {hist_path}")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NHTSA NCAP ratings into corpus")
    parser.add_argument("--limit", type=int, default=None, help="Max vehicles to fetch")
    args = parser.parse_args()
    paths = fetch_all(limit=args.limit)
    print(f"Done: {len(paths)} NCAP markdown file(s)")
    if not paths:
        sys.exit(1)


if __name__ == "__main__":
    main()
