"""Minimal nntplib stub for Python 3.13+ (stdlib module removed).

DeepTeam still imports ``nntplib.NNTPDataError`` at import time. We only need
the exception type present so the package loads; nothing in this eval path
uses NNTP.
"""

from __future__ import annotations

import sys
import types


def ensure_nntplib_shim() -> None:
    if "nntplib" in sys.modules:
        return
    mod = types.ModuleType("nntplib")

    class NNTPError(Exception):
        pass

    class NNTPReplyError(NNTPError):
        pass

    class NNTPTemporaryError(NNTPError):
        pass

    class NNTPPermanentError(NNTPError):
        pass

    class NNTPProtocolError(NNTPError):
        pass

    class NNTPDataError(NNTPError):
        pass

    mod.NNTPError = NNTPError
    mod.NNTPReplyError = NNTPReplyError
    mod.NNTPTemporaryError = NNTPTemporaryError
    mod.NNTPPermanentError = NNTPPermanentError
    mod.NNTPProtocolError = NNTPProtocolError
    mod.NNTPDataError = NNTPDataError
    sys.modules["nntplib"] = mod
