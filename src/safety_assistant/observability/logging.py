"""Structured JSON logging. One line per event; `extra=` fields are merged in."""

from __future__ import annotations

import datetime
import json
import logging
import re
import sys

_STD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


# Secrets that can appear inside exception text or messages: connection-string passwords and bearer
# tokens. Redacted at the formatter so every handler and every logger benefits.
_REDACT = (
    (re.compile(r"(://[^/:@\s]+:)[^@\s]+@"), r"\1***@"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"), r"\1***"),
    (re.compile(r"(?i)((?:api[_-]?key|secret|password|token)['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]{6,}"), r"\1***"),
)


def redact(text: str) -> str:
    for pattern, repl in _REDACT:
        text = pattern.sub(repl, text)
    return text


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.datetime.fromtimestamp(record.created, datetime.UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _STD and not k.startswith("_")})
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info)[-2000:])
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    root.setLevel(level.upper())
    logging.getLogger("uvicorn.access").disabled = True  # replaced by RequestIdMiddleware line
    logging.getLogger("httpx").setLevel(logging.WARNING)
