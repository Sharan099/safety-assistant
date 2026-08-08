"""UNECE page header/footer boilerplate cleanup (Phase 1).

Strips recurring ECE document identifiers and orphan page-number lines that
appear on nearly every PDF page and inflate every chunk's token count.
"""

from __future__ import annotations

import re

# Document revision identifiers printed on every page, e.g.
#   E/ECE/324/Rev.2/Add.128/Rev.3
#   E/ECE/TRANS/505/Rev.2/Add.128/Rev.3
_ECE_DOC_LINE = re.compile(r"(?m)^E/ECE/(?:324|TRANS)/\S+[ \t]*\r?\n?")

# Orphan page numbers left after header removal (standalone 1–3 digit lines).
_ORPHAN_PAGE_NUM = re.compile(r"(?m)^[ \t]*\d{1,3}[ \t]*\r?\n")


def strip_unece_boilerplate(text: str) -> str:
    """Remove recurring UNECE PDF headers/footers; keep clause content intact."""
    if not text:
        return text
    cleaned = _ECE_DOC_LINE.sub("", text)
    cleaned = _ORPHAN_PAGE_NUM.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def boilerplate_char_count(text: str) -> int:
    """Chars that strip_unece_boilerplate would remove (for measurement)."""
    if not text:
        return 0
    return max(0, len(text) - len(strip_unece_boilerplate(text)))
