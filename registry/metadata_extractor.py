import os
import re
from datetime import datetime, date
from typing import Dict, Any, Optional
from loguru import logger

class MetadataExtractor:
    """
    Extracts structured regulatory metadata from the text content of safety regulations.
    Supports UNECE, Euro NCAP, FMVSS, NHTSA, and IIHS document formats.
    """

    MONTH_MAP = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }

    @classmethod
    def parse_date(cls, date_str: str) -> Optional[date]:
        """Utility to parse date string to a datetime.date object."""
        if not date_str:
            return None
        date_str = date_str.lower().strip()
        
        # 1. Check YYYY-MM-DD
        m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", date_str)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
                
        # 2. Check DD.MM.YYYY
        m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", date_str)
        if m:
            try:
                return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            except ValueError:
                pass

        # 3. Check "DD Month YYYY" (e.g. 23 June 2011) or "Month DD, YYYY"
        # Match day and month name
        m_dmy = re.search(r"(\d{1,2})\s+([a-z]+)\s+(\d{4})", date_str)
        if m_dmy:
            day = int(m_dmy.group(1))
            month_name = m_dmy.group(2)
            year = int(m_dmy.group(3))
            month = cls.MONTH_MAP.get(month_name, 1)
            try:
                return date(year, month, day)
            except ValueError:
                pass
                
        m_mdy = re.search(r"([a-z]+)\s+(\d{1,2}),?\s+(\d{4})", date_str)
        if m_mdy:
            day = int(m_mdy.group(2))
            month_name = m_mdy.group(1)
            year = int(m_mdy.group(3))
            month = cls.MONTH_MAP.get(month_name, 1)
            try:
                return date(year, month, day)
            except ValueError:
                pass

        # 4. Check Year only (e.g. 2026) -> default to Jan 1st of that year
        m_y = re.match(r"^(\d{4})$", date_str)
        if m_y:
            return date(int(m_y.group(1)), 1, 1)

        return None

    def extract(self, text_sample: str, filename: str = "") -> Dict[str, Any]:
        """
        Scans a text sample (e.g., first 3 pages) and extracts regulation registry fields.
        """
        # Determine source type based on content or filename
        source_type = self._detect_source_type(text_sample, filename)
        
        # Initialize default response
        metadata = {
            "regulation_code": "UNKNOWN",
            "title": "Unknown Regulation Document",
            "source_type": source_type,
            "amendment": None,
            "revision": None,
            "supplement": None,
            "corrigendum": None,
            "publication_date": None,
            "effective_date": None,
            "market": "GLOBAL",
            "status": "ACTIVE"
        }

        # Apply specific parser based on detected source type
        if source_type == "UNECE":
            self._extract_unece(text_sample, metadata)
        elif source_type == "Euro NCAP":
            self._extract_euroncap(text_sample, metadata, filename)
        elif source_type == "FMVSS":
            self._extract_fmvss(text_sample, metadata)
        elif source_type == "NHTSA":
            self._extract_nhtsa(text_sample, metadata, filename)
        elif source_type == "IIHS":
            self._extract_iihs(text_sample, metadata, filename)
        else:
            self._extract_generic(text_sample, metadata, filename)

        # Basic fallback for title
        if metadata["title"] == "Unknown Regulation Document" and filename:
            # Clean filename to form a readable title
            clean_name = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
            metadata["title"] = clean_name
            
        return metadata

    def _detect_source_type(self, text: str, filename: str) -> str:
        text_upper = text.upper()
        file_upper = filename.upper()
        
        if "UNECE" in text_upper or "E/ECE" in text_upper or "WP.29" in text_upper or re.search(r"\bUN\s+R\d+", text_upper) or "ADDENDUM" in text_upper and "REGULATION NO." in text_upper:
            return "UNECE"
        elif "EURO NCAP" in text_upper or "EURONCAP" in text_upper:
            return "Euro NCAP"
        elif "FMVSS" in text_upper or "49 CFR" in text_upper:
            return "FMVSS"
        elif "NHTSA" in text_upper or "ODI" in text_upper or "RECALL" in text_upper:
            return "NHTSA"
        elif "IIHS" in text_upper or "HIGHWAY SAFETY" in text_upper:
            return "IIHS"
        
        # Filename based detection fallback
        if "UNECE" in file_upper or re.match(r"^R\d+", file_upper):
            return "UNECE"
        elif "NCAP" in file_upper:
            return "Euro NCAP"
        elif "FMVSS" in file_upper:
            return "FMVSS"
        elif "NHTSA" in file_upper:
            return "NHTSA"
        elif "IIHS" in file_upper:
            return "IIHS"
            
        return "INTERNAL"

    def _extract_unece(self, text: str, meta: Dict[str, Any]) -> None:
        meta["market"] = "GLOBAL"
        
        # 1. Regulation Code (e.g. UN Regulation No. 95, R95)
        m_code = re.search(r"Regulation\s+No\.\s*(\d+)", text, re.IGNORECASE)
        if m_code:
            meta["regulation_code"] = f"R{m_code.group(1)}"
        else:
            m_code_alt = re.search(r"\bUN\s+R(\d+)\b", text, re.IGNORECASE)
            if m_code_alt:
                meta["regulation_code"] = f"R{m_code_alt.group(1)}"

        # 2. Amendment Series (e.g. 05 series of amendments, 03 series)
        m_amend = re.search(r"(\d+)\s*(?:series of amendments|series\s+of\s+amendments)", text, re.IGNORECASE)
        if m_amend:
            val = int(m_amend.group(1))
            meta["amendment"] = f"{val:02d} Series"
        else:
            # Look for "05 series" or similar
            m_amend_short = re.search(r"\b(0\d+)\s+series\b", text, re.IGNORECASE)
            if m_amend_short:
                meta["amendment"] = f"{m_amend_short.group(1)} Series"

        # 3. Supplement (e.g. Supplement 3 to the 03 series)
        m_supp = re.search(r"Supplement\s+(\d+)\s+to\s+the\s+(\d+)?", text, re.IGNORECASE)
        if m_supp:
            meta["supplement"] = f"Supplement {m_supp.group(1)}"

        # 4. Corrigendum (e.g. Corrigendum 1 to...)
        m_corr = re.search(r"Corrigendum\s+(\d+)\b", text, re.IGNORECASE)
        if m_corr:
            meta["corrigendum"] = f"Corrigendum {m_corr.group(1)}"

        # 5. Effective Date / Date of entry into force
        m_eff = re.search(r"Date\s+of\s+entry\s+into\s+force:\s*([^\n\r]+)", text, re.IGNORECASE)
        if m_eff:
            meta["effective_date"] = self.parse_date(m_eff.group(1))
            meta["publication_date"] = meta["effective_date"]  # Fallback

        # 6. Extract Title (often the text after the regulation number on page 1)
        # Uniform provisions concerning the approval of...
        m_title = re.search(r"concerning the approval of(.*?)(?=uniform|agreement|regulation|issues|revision|\n{4,}|$)", text, re.DOTALL | re.IGNORECASE)
        if m_title:
            clean_title = re.sub(r"\s+", " ", m_title.group(1)).strip()
            # Trim prefix / suffix junk
            if clean_title:
                meta["title"] = f"UN Regulation concerning: {clean_title[:200]}"

    def _extract_euroncap(self, text: str, meta: Dict[str, Any], filename: str) -> None:
        meta["market"] = "EU"
        
        # 1. Regulation Code (Euro NCAP is organized by years and protocols)
        # Category: e.g. Adult Occupant Protection (AOP)
        category = "General"
        if "ADULT OCCUPANT" in text.upper() or "AOP" in text.upper():
            category = "Adult Occupant Protection"
        elif "CHILD OCCUPANT" in text.upper() or "COP" in text.upper():
            category = "Child Occupant Protection"
        elif "VULNERABLE ROAD USERS" in text.upper() or "VRU" in text.upper() or "PEDESTRIAN" in text.upper():
            category = "Vulnerable Road Users"
        elif "SAFETY ASSIST" in text.upper() or "SAD" in text.upper():
            category = "Safety Assist"

        # Year detection (e.g., 2026, 2024, etc.)
        year = "2026"  # Future default
        m_year = re.search(r"\b(20\d{2})\b", text)
        if m_year:
            year = m_year.group(1)
        else:
            m_year_file = re.search(r"\b(20\d{2})\b", filename)
            if m_year_file:
                year = m_year_file.group(1)
                
        meta["regulation_code"] = f"Euro NCAP {year} - {category}"
        meta["amendment"] = f"Protocol Year {year}"
        
        # Title
        m_title = re.search(r"EURO\s+NCAP\s+(.*?)\s+(?:Protocol|Assessment|Methodology)", text, re.IGNORECASE | re.DOTALL)
        if m_title:
            meta["title"] = f"Euro NCAP Protocol: {m_title.group(1).strip()}"
        else:
            meta["title"] = f"Euro NCAP {category} Assessment Protocol"

        # Effective date (Jan 1 of that protocol year)
        meta["effective_date"] = date(int(year), 1, 1)
        meta["publication_date"] = date(int(year) - 1, 11, 1)  # Usually published late previous year

    def _extract_fmvss(self, text: str, meta: Dict[str, Any]) -> None:
        meta["market"] = "US"
        
        # Code: e.g. FMVSS 214
        m_code = re.search(r"\bFMVSS\s+(\d+)\b", text, re.IGNORECASE)
        if m_code:
            meta["regulation_code"] = f"FMVSS {m_code.group(1)}"
        else:
            # Try CFR Title 49 Part 571
            m_cfr = re.search(r"49\s+CFR\s+(?:§\s*)?571\.(\d+)", text, re.IGNORECASE)
            if m_cfr:
                meta["regulation_code"] = f"FMVSS {m_cfr.group(1)}"
                
        # Title
        meta["title"] = f"Federal Motor Vehicle Safety Standard: {meta['regulation_code']}"
        m_title = re.search(r"Standard\s+No\.\s*\d+\s*[-—:\s]+(.*?)(?=\n|$)", text, re.IGNORECASE)
        if m_title and m_title.group(1).strip():
            meta["title"] = f"FMVSS: {m_title.group(1).strip()}"
            
        meta["effective_date"] = date(2024, 1, 1)  # Default/fallback

    def _extract_nhtsa(self, text: str, meta: Dict[str, Any], filename: str) -> None:
        meta["market"] = "US"
        
        # Check recall ID e.g. 23V-123 or ODI ID e.g. PE23-004
        m_odi = re.search(r"\b(PE|EA|RQ|DP)\d{2}-\d{3}\b", text)
        m_recall = re.search(r"\b\d{2}V-\d{3}\b", text)
        
        if m_odi:
            meta["regulation_code"] = f"NHTSA ODI {m_odi.group(1)}"
            meta["title"] = f"NHTSA Investigation: {m_odi.group(0)}"
        elif m_recall:
            meta["regulation_code"] = f"NHTSA Recall {m_recall.group(0)}"
            meta["title"] = f"NHTSA Recall Campaign: {m_recall.group(0)}"
        else:
            meta["regulation_code"] = "NHTSA Standard"
            
        # Try to find a date
        m_date = re.search(r"(?:date|opened|closed|issued):\s*([^\n\r]+)", text, re.IGNORECASE)
        if m_date:
            meta["effective_date"] = self.parse_date(m_date.group(1))

    def _extract_iihs(self, text: str, meta: Dict[str, Any], filename: str) -> None:
        meta["market"] = "US"
        
        # IIHS Test Protocols
        meta["regulation_code"] = "IIHS Protocol"
        if "SIDE" in text.upper():
            meta["regulation_code"] = "IIHS Side Impact"
            meta["title"] = "IIHS Side Impact Test Procedure"
        elif "FRONT" in text.upper():
            meta["regulation_code"] = "IIHS Frontal Impact"
            meta["title"] = "IIHS Frontal Impact Test Procedure"
            
        # Extract date
        m_date = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b", text, re.IGNORECASE)
        if m_date:
            meta["effective_date"] = self.parse_date(f"1 {m_date.group(1)} {m_date.group(2)}")

    def _extract_generic(self, text: str, meta: Dict[str, Any], filename: str) -> None:
        # Fallback parser
        meta["market"] = "GLOBAL"
        meta["regulation_code"] = "INTERNAL"
        
        # Check if there are years in text
        m_year = re.search(r"\b(20\d{2})\b", text)
        if m_year:
            meta["effective_date"] = date(int(m_year.group(1)), 1, 1)
