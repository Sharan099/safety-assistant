import re
import os
from typing import Dict, Any, Optional

def parse_document_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """
    Parses a regulation filename and determines:
      - document_role (BASE_REGULATION, AMENDMENT, SUPPLEMENT, CORRIGENDUM, TECHNICAL_BULLETIN)
      - is_complete_regulation (bool)
      - revision_number (int or None)
      - series_number (int or None)
      - supplement_number (int or None)
      - corrigendum_number (int or None)
      - regulation_code (str)
    """
    name = os.path.splitext(filename)[0]
    name_lower = name.lower()

    # Default values
    result = {
        "document_role": "BASE_REGULATION",
        "is_complete_regulation": True,
        "revision_number": None,
        "series_number": None,
        "supplement_number": None,
        "corrigendum_number": None,
        "regulation_code": "UNKNOWN"
    }

    # 1. Determine Role
    if "amend" in name_lower or "amendment" in name_lower:
        result["document_role"] = "AMENDMENT"
        result["is_complete_regulation"] = False
    elif "supp" in name_lower or "supplement" in name_lower:
        result["document_role"] = "SUPPLEMENT"
        result["is_complete_regulation"] = False
    elif "corr" in name_lower or "corrigendum" in name_lower:
        result["document_role"] = "CORRIGENDUM"
        result["is_complete_regulation"] = False
    elif "tb_" in name_lower or "_tb" in name_lower or "technical_bulletin" in name_lower:
        result["document_role"] = "TECHNICAL_BULLETIN"
        result["is_complete_regulation"] = False

    # 2. Extract Numbers (Series, Rev, Supp, Corr)
    # Series (e.g. 05Series, 03_Series, 04_amend, 08Series)
    series_match = re.search(r"\b(\d+)\s*(?:series|series_of_amendments|series\s+of\s+amendments)\b", name, re.IGNORECASE)
    if not series_match:
        series_match = re.search(r"(\d+)Series", name, re.IGNORECASE)
    if not series_match:
        series_match = re.search(r"_(\d+)Series", name, re.IGNORECASE)
    if not series_match:
        # UNECE official filenames: R016r6am8e.pdf -> 08 series
        am_series = re.search(r"am(\d+)", name_lower)
        if am_series:
            result["series_number"] = int(am_series.group(1))
            series_match = am_series
    if series_match and result.get("series_number") is None:
        result["series_number"] = int(series_match.group(1))

    # Revision (e.g. Rev4, Revision_3, Rev.7)
    rev_match = re.search(r"\b(?:rev|revision)\.?\s*(\d+)\b", name, re.IGNORECASE)
    if not rev_match:
        rev_match = re.search(r"Rev(\d+)", name, re.IGNORECASE)
    if rev_match:
        result["revision_number"] = int(rev_match.group(1))

    # Supplement (e.g. Supp2, Supplement_1)
    supp_match = re.search(r"\b(?:supp|supplement)\.?\s*(\d+)\b", name, re.IGNORECASE)
    if not supp_match:
        supp_match = re.search(r"Supp(\d+)", name, re.IGNORECASE)
    if supp_match:
        result["supplement_number"] = int(supp_match.group(1))

    # Corrigendum (e.g. Corr1, Corrigendum_2)
    corr_match = re.search(r"\b(?:corr|corrigendum)\.?\s*(\d+)\b", name, re.IGNORECASE)
    if not corr_match:
        corr_match = re.search(r"Corr(\d+)", name, re.IGNORECASE)
    if corr_match:
        result["corrigendum_number"] = int(corr_match.group(1))

    # 3. Detect Regulation Code
    # UNECE (e.g. UN_R95_05Series -> R95)
    un_match = re.search(r"\bUN_R(\d+)", name, re.IGNORECASE)
    if not un_match:
        un_match = re.search(r"\bR(\d+)", name, re.IGNORECASE)
    if un_match:
        result["regulation_code"] = f"UN_R{un_match.group(1)}"

    # FMVSS (e.g. FMVSS_208_2024 -> FMVSS 208)
    elif "fmvss" in name_lower:
        fmvss_match = re.search(r"fmvss_(\d+)", name_lower)
        if fmvss_match:
            result["regulation_code"] = f"FMVSS {fmvss_match.group(1)}"
        else:
            result["regulation_code"] = "FMVSS"

    # Euro NCAP (e.g. EuroNCAP_AOP_2026 -> Euro NCAP AOP)
    elif "euroncap" in name_lower:
        if "_aop" in name_lower:
            result["regulation_code"] = "Euro NCAP AOP"
        elif "_cop" in name_lower:
            result["regulation_code"] = "Euro NCAP COP"
        elif "_vru" in name_lower:
            result["regulation_code"] = "Euro NCAP VRU"
        elif "_sa_" in name_lower or name_lower.endswith("_sa"):
            result["regulation_code"] = "Euro NCAP Safety Assist"
        elif "_side" in name_lower:
            result["regulation_code"] = "Euro NCAP Side"
        elif "_farside" in name_lower:
            result["regulation_code"] = "Euro NCAP Far-Side"
        elif "_rescue" in name_lower:
            result["regulation_code"] = "Euro NCAP Rescue"
        elif "_postcrash" in name_lower:
            result["regulation_code"] = "Euro NCAP Post-Crash"
        elif "_overall" in name_lower:
            result["regulation_code"] = "Euro NCAP Overall Assessment"
        elif "_scoring" in name_lower:
            result["regulation_code"] = "Euro NCAP Scoring"
        elif "_generalguidance" in name_lower:
            result["regulation_code"] = "Euro NCAP General Guidance"
        elif "tb_thor" in name_lower:
            result["regulation_code"] = "Euro NCAP THOR Dummy"
        elif "tb_aemdb" in name_lower:
            result["regulation_code"] = "Euro NCAP AE-MDB"
        elif "tb_farside" in name_lower:
            result["regulation_code"] = "Euro NCAP Far-Side Bulletin"
        else:
            result["regulation_code"] = "Euro NCAP"

    # EU Regulations
    elif "eu_" in name_lower or "_gsr" in name_lower or "_wvta" in name_lower:
        if "2019_2144" in name_lower or "gsr2" in name_lower or "gsr_amendment" in name_lower:
            result["regulation_code"] = "EU GSR2"
        elif "2018_858" in name_lower or "wvta" in name_lower:
            result["regulation_code"] = "EU WVTA"
        else:
            result["regulation_code"] = "EU Regulation"

    # IIHS
    elif "iihs" in name_lower:
        if "moderateoverlap" in name_lower:
            result["regulation_code"] = "IIHS Moderate Overlap"
        elif "smalloverlap" in name_lower:
            result["regulation_code"] = "IIHS Small Overlap"
        elif "sidebarrier" in name_lower:
            result["regulation_code"] = "IIHS Side Barrier"
        elif "roofstrength" in name_lower:
            result["regulation_code"] = "IIHS Roof Strength"
        elif "headrestraints" in name_lower:
            result["regulation_code"] = "IIHS Head Restraints"
        else:
            result["regulation_code"] = "IIHS Protocol"

    # NHTSA NCAP
    elif "nhtsa_ncap" in name_lower:
        if "frontal" in name_lower:
            result["regulation_code"] = "NHTSA NCAP Frontal"
        elif "pole" in name_lower:
            result["regulation_code"] = "NHTSA NCAP Pole"
        elif "side" in name_lower:
            result["regulation_code"] = "NHTSA NCAP Side"
        else:
            result["regulation_code"] = "NHTSA NCAP"

    # CNCAP
    elif "cncap" in name_lower:
        result["regulation_code"] = "C-NCAP 2024"

    return result
