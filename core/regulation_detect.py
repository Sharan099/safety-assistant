"""Extract UNECE regulation codes referenced in a query."""

from __future__ import annotations

import re

# UN R94, UN_R94, R94
_R_CODE = re.compile(r"\b(?:UN[\s_-]?)?R\s*(\d{2,3})\b", re.I)
# UN Regulation No. 94 / Regulation No. 129
_REG_NO = re.compile(r"\bRegulation\s+No\.?\s*(\d{2,3})\b", re.I)


def extract_regulation_numbers(query: str) -> list[str]:
    """Return sorted unique regulation numbers as strings, e.g. ['94', '129']."""
    found: set[str] = set()
    for pattern in (_R_CODE, _REG_NO):
        for m in pattern.finditer(query):
            found.add(m.group(1))
    return sorted(found, key=int)


def regulation_codes(query: str) -> list[str]:
    return [f"UN_R{n}" for n in extract_regulation_numbers(query)]
