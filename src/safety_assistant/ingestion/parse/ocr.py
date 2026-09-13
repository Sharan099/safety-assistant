"""Optional OCR for pages without a usable text layer.

The parser decides *which* pages need OCR (`ParsedPage.needs_ocr`); this module decides *how*.
The default is no OCR: such pages stay flagged and the extraction report says NEEDS_REVIEW, which is
the honest outcome on a machine without an OCR engine. `TesseractCli` shells out to a locally
installed `tesseract` binary (Apache-2.0) when `OCR_PROVIDER=tesseract`; nothing is downloaded.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Protocol


class OcrUnavailable(RuntimeError):
    pass


class OcrAdapter(Protocol):
    name: str

    def ocr_png(self, png: bytes) -> str: ...


class NoOcr:
    name = "none"

    def ocr_png(self, png: bytes) -> str:
        raise OcrUnavailable("OCR is not configured (OCR_PROVIDER=none)")


class TesseractCli:
    name = "tesseract"

    def __init__(self, binary: str = "tesseract", lang: str = "eng", timeout_s: float = 60.0) -> None:
        self.binary = shutil.which(binary) or binary
        self.lang = lang
        self.timeout_s = timeout_s
        if shutil.which(self.binary) is None:
            raise OcrUnavailable(f"tesseract binary not found: {binary}")

    def ocr_png(self, png: bytes) -> str:
        try:
            out = subprocess.run(  # noqa: S603 — fixed argv, image via stdin, no shell
                [self.binary, "stdin", "stdout", "-l", self.lang, "--psm", "6"],
                input=png,
                capture_output=True,
                timeout=self.timeout_s,
                check=True,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise OcrUnavailable(f"tesseract failed: {type(exc).__name__}") from exc
        return out.stdout.decode("utf-8", errors="replace")


def ocr_from_settings(provider: str, *, lang: str = "eng") -> OcrAdapter:
    if provider == "tesseract":
        return TesseractCli(lang=lang)
    return NoOcr()
